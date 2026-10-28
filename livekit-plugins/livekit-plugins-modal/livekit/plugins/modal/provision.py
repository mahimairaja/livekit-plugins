# Copyright 2026 Mahimai Raja J
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deploy an STT server to Modal once, and find it again at runtime by name.

`provision_stt` is idempotent: it looks the App up and deploys only when it is missing (or
when asked to redeploy). The agent never deploys; it resolves the endpoint URL by name with
`lookup_web_url`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from types import ModuleType

import modal
from modal.exception import NotFoundError

from .log import logger
from .models import (
    REGION_MULTIPLIER,
    ModalSTTModel,
    app_name,
    get_model,
    hourly_cost,
    provision_command,
)

CONFIG_ENV = "LIVEKIT_MODAL_SERVER_CONFIG"


class DeploymentNotFoundError(RuntimeError):
    """The model's Modal App is not deployed in the environment. The message has the fix."""


@dataclass(frozen=True)
class ServerConfig:
    model: str
    environment: str | None = None
    gpu: str | None = None
    region: str | None = None
    min_containers: int = 0
    scaledown_window: int = 300

    @property
    def spec(self) -> ModalSTTModel:
        return get_model(self.model)

    @property
    def app_name(self) -> str:
        return app_name(self.model)

    def to_env(self) -> str:
        spec = self.spec
        return json.dumps(
            {
                "app_name": self.app_name,
                "engine_model": spec.engine_model,
                "gpu": self.gpu or spec.gpu,
                "region": self.region,
                "min_containers": self.min_containers,
                "scaledown_window": self.scaledown_window,
                "cpu": spec.cpu_cores,
                "memory_mib": int(spec.memory_gib * 1024),
            }
        )


@dataclass(frozen=True)
class ProvisionResult:
    app_name: str
    environment: str | None
    url: str
    created: bool
    """True when this call deployed the App; False when it was already deployed."""
    gpu: str
    min_containers: int
    warm_cost_per_hour: tuple[float, float]
    """Cost of the warm floor (`min_containers` idle containers), low and high estimate."""


def load_server_app(config: ServerConfig) -> modal.App:
    """Import the model's server template as a top-level module and return its App.

    It must be a top-level module, not part of this package: Modal mounts the defining file
    into the container and imports it by its file name.
    """
    path = config.spec.template_path
    os.environ[CONFIG_ENV] = config.to_env()
    try:
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        assert module_spec is not None and module_spec.loader is not None
        module: ModuleType = importlib.util.module_from_spec(module_spec)
        # Modal finds the defining module through sys.modules.
        sys.modules[path.stem] = module
        module_spec.loader.exec_module(module)
    finally:
        del os.environ[CONFIG_ENV]
    app = module.app
    assert isinstance(app, modal.App)
    return app


def lookup_web_url(model: str, environment: str | None = None) -> str:
    """URL of the deployed server for `model`. Raises DeploymentNotFoundError if missing."""
    spec = get_model(model)
    try:
        cls = modal.Cls.from_name(app_name(model), spec.cls_name, environment_name=environment)
        url: str | None = cls().web.get_web_url()
    except NotFoundError:
        raise DeploymentNotFoundError(
            f"no Modal deployment {app_name(model)!r} in environment "
            f"{environment or '(default)'!r}. Deploy it once with:\n\n"
            f"    {provision_command(model, environment)}\n"
        ) from None
    if not url:
        raise RuntimeError(f"{app_name(model)} has no web endpoint; redeploy it")
    return url


def warm_floor_cost(config: ServerConfig) -> tuple[float, float]:
    """USD per hour to keep `min_containers` warm, as a (low, high) range.

    Pinning a region costs 1.15x to 1.75x; without one, low and high are the same.
    """
    base = config.min_containers * hourly_cost(config.spec, config.gpu)
    if config.region:
        return base * REGION_MULTIPLIER[0], base * REGION_MULTIPLIER[1]
    return base, base


def provision_stt(config: ServerConfig, *, redeploy: bool = False) -> ProvisionResult:
    """Deploy the model's server unless it is already deployed. Safe to run repeatedly."""
    created = False
    url: str | None = None
    if not redeploy:
        try:
            url = lookup_web_url(config.model, config.environment)
        except DeploymentNotFoundError:
            pass
    if url is None:
        logger.info("deploying %s", config.app_name)
        app = load_server_app(config)
        with modal.enable_output():
            app.deploy(name=config.app_name, environment_name=config.environment)
        created = True
        url = lookup_web_url(config.model, config.environment)
    return ProvisionResult(
        app_name=config.app_name,
        environment=config.environment,
        url=url,
        created=created,
        gpu=config.gpu or config.spec.gpu,
        min_containers=config.min_containers,
        warm_cost_per_hour=warm_floor_cost(config),
    )


def create_proxy_token(environment: str | None) -> tuple[str, str]:
    """Create a workspace proxy token and return (token_id, token_secret).

    On workspaces with RBAC, tokens are scoped to environments, so the new token is also
    allowed into `environment`.
    """
    ws = modal.Workspace.from_context()
    token = ws.proxy_tokens.create()
    if environment:
        ws.proxy_tokens.allow(token.token_id, environment)
    return token.token_id, token.token_secret
