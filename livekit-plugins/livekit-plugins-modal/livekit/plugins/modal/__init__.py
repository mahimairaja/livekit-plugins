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

"""Speech-to-text for LiveKit Agents, served from your own Modal GPUs.

Deploy a model once with `python -m livekit.plugins.modal provision stt`, then use
`modal.STT(...)` in the agent. The agent finds the deployment by name; it never deploys.
"""

from livekit.agents import Plugin

from .log import logger
from .models import STT_MODELS, ModalSTTModel, STTModels
from .provision import DeploymentNotFoundError
from .stt import STT, SpeechStream
from .version import __version__

__all__ = [
    "STT",
    "STT_MODELS",
    "DeploymentNotFoundError",
    "ModalSTTModel",
    "STTModels",
    "SpeechStream",
    "__version__",
]


class ModalPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(ModalPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
