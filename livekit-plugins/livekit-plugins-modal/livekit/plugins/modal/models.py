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

"""The STT models this plugin can deploy, and what each one costs to keep warm.

A model is a server template in `server/` plus the settings it is deployed with. Adding a
model is one `ModalSTTModel` entry here, and a template if no existing one fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

STTModels = Literal["parakeet-tdt-0.6b-v2", "whisper-large-v3-turbo"]

SERVER_DIR = Path(__file__).parent / "server"

# Modal list prices, USD per GPU per second (modal.com/pricing, October 2026).
GPU_PRICE_PER_S: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
}
CPU_PRICE_PER_CORE_S = 0.0000131
MEMORY_PRICE_PER_GIB_S = 0.00000222
# Pinning a region bills at 1.15x to 1.75x the base price, depending on the region.
REGION_MULTIPLIER = (1.15, 1.75)


@dataclass(frozen=True)
class ModalSTTModel:
    template: str
    """Server template file in `server/`, deployed as its own Modal App."""
    cls_name: str
    """Modal class in the template that serves the web endpoint."""
    engine_model: str
    """Model id passed to the inference engine inside the container."""
    gpu: str
    language: str | None
    """Language of a monolingual model; None for multilingual models."""
    streaming: bool
    cpu_cores: float = 2.0
    memory_gib: float = 8.0

    @property
    def template_path(self) -> Path:
        return SERVER_DIR / self.template


STT_MODELS: dict[str, ModalSTTModel] = {
    "parakeet-tdt-0.6b-v2": ModalSTTModel(
        template="parakeet_server.py",
        cls_name="ParakeetServer",
        engine_model="nvidia/parakeet-tdt-0.6b-v2",
        gpu="L4",
        language="en",
        streaming=True,
    ),
    "whisper-large-v3-turbo": ModalSTTModel(
        template="whisper_server.py",
        cls_name="WhisperServer",
        engine_model="large-v3-turbo",
        gpu="L4",
        language=None,
        streaming=False,
    ),
}

DEFAULT_MODEL: STTModels = "parakeet-tdt-0.6b-v2"


def get_model(name: str) -> ModalSTTModel:
    try:
        return STT_MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; known: {list(STT_MODELS)}") from None


def app_name(model: str) -> str:
    """Name of the Modal App that serves `model`. One App per model per environment."""
    return f"livekit-stt-{model}"


def hourly_cost(spec: ModalSTTModel, gpu: str | None = None) -> float:
    """List price, in USD, of one container of `spec` running for an hour."""
    gpu = gpu or spec.gpu
    if gpu not in GPU_PRICE_PER_S:
        raise ValueError(f"no price for GPU {gpu!r}; known: {list(GPU_PRICE_PER_S)}")
    per_s = (
        GPU_PRICE_PER_S[gpu]
        + spec.cpu_cores * CPU_PRICE_PER_CORE_S
        + spec.memory_gib * MEMORY_PRICE_PER_GIB_S
    )
    return per_s * 3600


def provision_command(model: str, environment: str | None) -> str:
    """The shell command that deploys `model`. Error messages quote it verbatim."""
    cmd = f"python -m livekit.plugins.modal provision stt --model {model}"
    return f"{cmd} --env {environment}" if environment else cmd
