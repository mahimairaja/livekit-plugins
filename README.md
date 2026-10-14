<p align="center">
  <img src="docs/assets/hero.gif" alt="livekit-plugins: local STT, TTS and LLM for LiveKit Agents, installable from GitHub today" width="100%">
</p>

<h3 align="center">Local STT, TTS and LLM plugins for LiveKit Agents. Not on PyPI yet. Install them today.</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20to%203.14-79c0ff?style=flat-square&labelColor=0d1117" alt="Python 3.12 to 3.14">
  <img src="https://img.shields.io/badge/livekit--agents-1.8.6-d2a8ff?style=flat-square&labelColor=0d1117" alt="livekit-agents 1.8.6">
  <img src="https://img.shields.io/badge/API%20keys-0-7ee787?style=flat-square&labelColor=0d1117" alt="0 API keys">
  <img src="https://img.shields.io/badge/install-from%20GitHub-f2cc60?style=flat-square&labelColor=0d1117" alt="install from GitHub">
</p>

Every model runs inside your agent process. No API keys, no cloud, no per-minute bill.
Each plugin is the code from an open [livekit/agents](https://github.com/livekit/agents) PR, packaged so you can `pip install` it from this repo while review continues.

<p align="center">
  <img src="docs/assets/pipeline.png" alt="What's inside: livekit-plugins-sherpa (STT and TTS) and livekit-plugins-hf (LLM), imported by your agent" width="100%">
</p>

| Package | What it does | Upstream PR |
| --- | --- | --- |
| [`livekit-plugins-sherpa`](livekit-plugins/livekit-plugins-sherpa) | Speech-to-text (streaming zipformer, or batch Whisper, Moonshine, SenseVoice, Parakeet) and text-to-speech (Kokoro, Kitten, Piper) on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). ONNX Runtime only, no torch, no GPU. | STT [#7748](https://github.com/livekit/agents/pull/7748), TTS [#7749](https://github.com/livekit/agents/pull/7749) |
| [`livekit-plugins-hf`](livekit-plugins/livekit-plugins-hf) | LLM: any Hugging Face chat model, run in process with `transformers`. Streaming and function tools. | [#7731](https://github.com/livekit/agents/pull/7731) |

Package and import names match upstream (`from livekit.plugins import sherpa, hf`). When a PR merges, switch to the PyPI release and change nothing else.

## Install

```bash
pip install "livekit-plugins-sherpa @ git+https://github.com/mahimairaja/livekit-plugins#subdirectory=livekit-plugins/livekit-plugins-sherpa"
pip install "livekit-plugins-hf @ git+https://github.com/mahimairaja/livekit-plugins#subdirectory=livekit-plugins/livekit-plugins-hf"
```

With uv, use `uv add` with the same string. Add `@<commit>` after the repo URL to pin a commit.

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

Fetch the models once while online. After that the agent never downloads at runtime:

```bash
python my_agent.py download-files
```

Options, model lists and tool calling are in each plugin's README.

## See it running

[local-livekit](https://github.com/mahimairaja/local-livekit) wires these plugins into one voice agent that runs with the Wi-Fi off. The write-up: [A LiveKit voice agent that runs offline](https://mahimai.ca/blog/local-livekit-voice-agent-offline).

## Status

Temporary. Each plugin here is the code from its upstream PR. Once a PR merges and ships on PyPI, its package leaves this repo. Bugs that also apply upstream go to the PR.

## Develop

```bash
uv sync
uv run pytest                       # unit tests, plus real-model tests (download once, skip offline)
uv run pre-commit run --all-files   # ruff, mypy --strict, uv lock --check
```

## License

Plugin packages are Apache-2.0, matching livekit/agents. The rest of the repo is MIT.
