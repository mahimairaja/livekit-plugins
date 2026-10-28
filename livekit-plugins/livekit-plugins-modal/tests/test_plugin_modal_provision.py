"""Provisioning: server template loading, lookup-or-deploy, cost math and the CLI.

Nothing here talks to Modal: lookups and deploys are replaced, the template is only imported.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import modal as modal_sdk
import pytest

from livekit.plugins.modal import __main__ as cli, provision
from livekit.plugins.modal.models import STT_MODELS, hourly_cost

pytestmark = pytest.mark.unit

WHISPER = "whisper-large-v3-turbo"


def test_server_template_loads_as_a_top_level_module() -> None:
    app = provision.load_server_app(
        provision.ServerConfig(model=WHISPER, region="us-east", min_containers=1)
    )

    assert app.name == "livekit-stt-whisper-large-v3-turbo"
    # Modal mounts a module without a package as a single file and imports it by file name.
    module = sys.modules["whisper_server"]
    assert not module.__package__
    assert module.app is app
    assert isinstance(module.WhisperServer, modal_sdk.Cls)
    assert module.REGION == "us-east"
    assert module.MIN_CONTAINERS == 1
    assert module.ENGINE_MODEL == "large-v3-turbo"
    assert provision.CONFIG_ENV not in os.environ


def test_warm_floor_cost() -> None:
    per_container = hourly_cost(STT_MODELS[WHISPER])

    assert provision.warm_floor_cost(provision.ServerConfig(model=WHISPER)) == (0.0, 0.0)
    assert provision.warm_floor_cost(
        provision.ServerConfig(model=WHISPER, min_containers=2)
    ) == pytest.approx((2 * per_container, 2 * per_container))
    assert provision.warm_floor_cost(
        provision.ServerConfig(model=WHISPER, min_containers=1, region="us-east")
    ) == pytest.approx((1.15 * per_container, 1.75 * per_container))
    # An L4 with 2 cores and 8 GiB is a little under a dollar an hour at list price.
    assert 0.9 < per_container < 1.0


def test_unknown_gpu_has_no_price() -> None:
    with pytest.raises(ValueError, match="no price"):
        hourly_cost(STT_MODELS[WHISPER], "TPU")


class FakeApp:
    def __init__(self) -> None:
        self.deploys: list[dict[str, Any]] = []

    def deploy(self, **kwargs: Any) -> FakeApp:
        self.deploys.append(kwargs)
        return self


def test_provision_reuses_an_existing_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(config: provision.ServerConfig) -> Any:
        raise AssertionError("must not deploy when the App exists")

    monkeypatch.setattr(provision, "lookup_web_url", lambda model, env: "https://existing")
    monkeypatch.setattr(provision, "load_server_app", load)

    result = provision.provision_stt(provision.ServerConfig(model=WHISPER, environment="main"))

    assert (result.url, result.created) == ("https://existing", False)


def test_provision_deploys_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FakeApp()
    deployed = False

    def lookup(model: str, environment: str | None) -> str:
        if not deployed:
            raise provision.DeploymentNotFoundError(model)
        return "https://new"

    def load(config: provision.ServerConfig) -> Any:
        nonlocal deployed
        deployed = True
        return app

    monkeypatch.setattr(provision, "lookup_web_url", lookup)
    monkeypatch.setattr(provision, "load_server_app", load)

    result = provision.provision_stt(
        provision.ServerConfig(model=WHISPER, environment="main", gpu="A10", min_containers=1)
    )

    assert app.deploys == [
        {"name": "livekit-stt-whisper-large-v3-turbo", "environment_name": "main"}
    ]
    assert (result.url, result.created, result.gpu) == ("https://new", True, "A10")
    assert result.warm_cost_per_hour[0] > 0


def test_redeploy_skips_the_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FakeApp()
    monkeypatch.setattr(provision, "lookup_web_url", lambda model, env: "https://url")
    monkeypatch.setattr(provision, "load_server_app", lambda config: app)

    result = provision.provision_stt(provision.ServerConfig(model=WHISPER), redeploy=True)

    assert len(app.deploys) == 1
    assert result.created


def _fake_provision(calls: list[provision.ServerConfig]) -> Any:
    def provision_stt(config: provision.ServerConfig, **_: Any) -> provision.ProvisionResult:
        calls.append(config)
        return provision.ProvisionResult(
            app_name=config.app_name,
            environment=config.environment,
            url="https://url",
            created=True,
            gpu="L4",
            min_containers=config.min_containers,
            warm_cost_per_hour=provision.warm_floor_cost(config),
        )

    return provision_stt


def test_cli_provisions_and_creates_a_proxy_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[provision.ServerConfig] = []
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.setattr(provision, "provision_stt", _fake_provision(calls))
    monkeypatch.setattr(provision, "create_proxy_token", lambda env: ("wk-new", "ws-new"))

    code = cli.main(
        ["provision", "stt", "--model", WHISPER, "--env", "prod", "--region", "us-east",
         "--min-containers", "1", "--scaledown-window", "60"]
    )  # fmt: skip

    out = capsys.readouterr().out
    assert code == 0
    assert calls == [
        provision.ServerConfig(
            model=WHISPER, environment="prod", region="us-east", min_containers=1,
            scaledown_window=60,
        )
    ]  # fmt: skip
    assert "https://url" in out
    assert "(region pricing)" in out
    assert "export MODAL_PROXY_TOKEN_ID=wk-new" in out
    assert "export MODAL_PROXY_TOKEN_SECRET=ws-new" in out


def test_cli_keeps_an_existing_proxy_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def create(env: str | None) -> tuple[str, str]:
        raise AssertionError("must not create a token when one is set")

    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-set")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-set")
    monkeypatch.setattr(provision, "provision_stt", _fake_provision([]))
    monkeypatch.setattr(provision, "create_proxy_token", create)

    assert cli.main(["provision", "stt"]) == 0
    assert "using MODAL_PROXY_TOKEN_ID" in capsys.readouterr().out


def test_modal_sdk_pin_has_programmatic_proxy_tokens() -> None:
    # provision creates tokens through this API instead of shelling out to the CLI.
    assert hasattr(modal_sdk.Workspace.from_context(), "proxy_tokens")


@dataclass
class FakeToken:
    token_id: str
    token_secret: str = "ws-secret"
    scoped: bool = False


class FakeProxyTokens:
    def __init__(self, *, scoped: bool, fail_allow: bool = False) -> None:
        self.scoped = scoped
        self.fail_allow = fail_allow
        self.tokens: dict[str, FakeToken] = {}
        self.allowed: list[tuple[str, str]] = []

    def create(self) -> FakeToken:
        token = FakeToken(f"wk-{len(self.tokens)}", scoped=self.scoped)
        self.tokens[token.token_id] = token
        return token

    def list(self) -> list[FakeToken]:
        return list(self.tokens.values())

    def allow(self, token_id: str, environment: str) -> None:
        if self.fail_allow:
            raise RuntimeError("allow failed")
        self.allowed.append((token_id, environment))

    def delete(self, token_id: str) -> None:
        del self.tokens[token_id]


def _workspace(monkeypatch: pytest.MonkeyPatch, tokens: FakeProxyTokens) -> None:
    workspace = type("FakeWorkspace", (), {"proxy_tokens": tokens})()
    monkeypatch.setattr(modal_sdk.Workspace, "from_context", lambda: workspace)


def test_proxy_token_is_not_scoped_without_rbac(monkeypatch: pytest.MonkeyPatch) -> None:
    # Modal rejects `allow` for unscoped tokens: "Token is not environment-scoped".
    tokens = FakeProxyTokens(scoped=False)
    _workspace(monkeypatch, tokens)

    assert provision.create_proxy_token("main") == ("wk-0", "ws-secret")
    assert tokens.allowed == []


def test_proxy_token_is_allowed_into_the_environment_with_rbac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = FakeProxyTokens(scoped=True)
    _workspace(monkeypatch, tokens)

    provision.create_proxy_token("prod")

    assert tokens.allowed == [("wk-0", "prod")]


def test_failed_proxy_token_setup_does_not_leak_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = FakeProxyTokens(scoped=True, fail_allow=True)
    _workspace(monkeypatch, tokens)

    with pytest.raises(RuntimeError, match="allow failed"):
        provision.create_proxy_token("prod")

    assert tokens.tokens == {}


def test_whisper_routes_resolve_their_fastapi_types() -> None:
    # With string annotations FastAPI cannot see `Request` (imported inside `web()`) and
    # asks for it as a query parameter: every request then fails with a 422.
    provision.load_server_app(provision.ServerConfig(model=WHISPER))
    api = _web_app("whisper_server", "WhisperServer")

    route = next(r for r in api.routes if getattr(r, "path", None) == "/v1/transcribe")
    assert [p.name for p in route.dependant.query_params] == ["language"]


def _web_app(module: str, cls_name: str) -> Any:
    """Build a server's FastAPI app without loading its model."""
    from modal._utils.async_utils import synchronizer

    user_cls = synchronizer._translate_in(getattr(sys.modules[module], cls_name))._user_cls
    return user_cls.__dict__["web"]._get_raw_f()(user_cls.__new__(user_cls))
