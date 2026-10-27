<p align="center">
  <img src="docs/assets/hero.gif" alt="livekit-plugins: local STT, TTS and LLM for LiveKit Agents, installable from GitHub today" width="100%">
</p>

<h3 align="center">Local STT, TTS and LLM plugins for LiveKit Agents, installable from GitHub ahead of their PyPI release.</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20to%203.14-79c0ff?style=flat-square&labelColor=0d1117" alt="Python 3.12 to 3.14">
  <img src="https://img.shields.io/badge/livekit--agents-1.8.6-d2a8ff?style=flat-square&labelColor=0d1117" alt="livekit-agents 1.8.6">
  <img src="https://img.shields.io/badge/API%20keys-0-7ee787?style=flat-square&labelColor=0d1117" alt="0 API keys">
  <img src="https://img.shields.io/badge/install-from%20GitHub-f2cc60?style=flat-square&labelColor=0d1117" alt="install from GitHub">
</p>

This repository packages plugins that are under review in [livekit/agents](https://github.com/livekit/agents) so they can be installed and used before they are released. All models run inside the agent process: no API keys, no network calls at inference time, no per-minute cost.

<p align="center">
  <img src="docs/assets/pipeline.png" alt="What's inside: livekit-plugins-sherpa (STT and TTS) and livekit-plugins-hf (LLM), imported by your agent" width="100%">
</p>

## Plugins

| Package | Provides | Description |
| --- | --- | --- |
| [`livekit-plugins-sherpa`](livekit-plugins/livekit-plugins-sherpa) | STT, TTS | Speech-to-text (streaming Zipformer; batch Whisper, Moonshine, SenseVoice, Parakeet) and text-to-speech (Kokoro, Kitten, Piper) on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). ONNX Runtime only: no PyTorch, no GPU required. |
| [`livekit-plugins-hf`](livekit-plugins/livekit-plugins-hf) | LLM | Any Hugging Face chat model, run in process with `transformers`. Supports streaming and function tools. |

## Upstream status

Each plugin tracks an open pull request in livekit/agents. When a PR merges, the plugin ships on PyPI as an official LiveKit package and the copy here is retired. `livekit-plugins-sherpa` is retired once both its STT and TTS PRs have merged.

| Plugin | Component | Upstream PR | Status |
| --- | --- | --- | --- |
| `livekit-plugins-hf` | LLM | [livekit/agents#7731](https://github.com/livekit/agents/pull/7731) | Open, in review |
| `livekit-plugins-sherpa` | STT | [livekit/agents#7748](https://github.com/livekit/agents/pull/7748) | Open, in review |
| `livekit-plugins-sherpa` | TTS | [livekit/agents#7749](https://github.com/livekit/agents/pull/7749) | Open, in review |

Package and import names match upstream (`from livekit.plugins import sherpa, hf`). Once a plugin is on PyPI, replace the GitHub install with `pip install <package>`; no code changes are needed.

## Install

```bash
pip install "livekit-plugins-sherpa @ git+https://github.com/mahimairaja/livekit-plugins#subdirectory=livekit-plugins/livekit-plugins-sherpa"
pip install "livekit-plugins-hf @ git+https://github.com/mahimairaja/livekit-plugins#subdirectory=livekit-plugins/livekit-plugins-hf"
```

With uv, use `uv add` with the same string. To pin a commit, append `@<commit>` to the repository URL.

Requires Python 3.12 to 3.14. Dependencies are pinned exactly, including `livekit-agents==1.8.6`.

## Use

```python
from livekit.agents import AgentSession
from livekit.plugins import hf, sherpa, silero

session = AgentSession(
    stt=sherpa.STT(),  # streaming zipformer, English
    llm=hf.LLM(model="Qwen/Qwen2.5-1.5B-Instruct"),
    tts=sherpa.TTS(voice="af_heart"),  # Kokoro
    vad=silero.VAD.load(),  # pip install livekit-plugins-silero
)
```

Download the models once while online. The agent does not download anything at runtime:

```bash
python my_agent.py download-files
```

Configuration options, supported models and tool calling are documented in each plugin's README.

## Example

[local-livekit](https://github.com/mahimairaja/local-livekit) combines these plugins into a single voice agent that runs fully offline. Write-up: [A LiveKit voice agent that runs offline](https://mahimai.ca/blog/local-livekit-voice-agent-offline).

## Issues

Bugs that also apply to the upstream code are best reported on the corresponding livekit/agents PR. Issues specific to this packaging belong here.

## Develop

```bash
uv sync
uv run pytest                       # unit tests, plus real-model tests (download once, skip offline)
uv run pre-commit run --all-files   # ruff, mypy --strict, uv lock --check
```

## License

Plugin packages are Apache-2.0, matching livekit/agents. The rest of the repo is MIT.
