# Modal plugin for LiveKit Agents

Speech-to-text on your own [Modal](https://modal.com) GPUs. Deploy an open model once, then point your agent at it by name. You pay Modal per second of GPU time; there is no per-minute STT bill and audio never leaves your Modal workspace.

## Installation

> Not on PyPI. Install from GitHub.

```bash
pip install "livekit-plugins-modal @ git+https://github.com/mahimairaja/livekit-plugins#subdirectory=livekit-plugins/livekit-plugins-modal"
# or
uv add "livekit-plugins-modal @ git+https://github.com/mahimairaja/livekit-plugins#subdirectory=livekit-plugins/livekit-plugins-modal"
```

Requires Python 3.12 to 3.14, pins `livekit-agents==1.8.6` and `modal==1.6.1`, and a Modal account (`modal setup`).

## Usage

Two steps. Provision once, from your laptop or CI:

```bash
python -m livekit.plugins.modal provision stt --model parakeet-tdt-0.6b-v2 --env main
```

It deploys the server if it is not there yet (rerunning it is safe), prints the endpoint URL and what the warm floor costs per hour, and creates a proxy token for the agent:

```
app      livekit-stt-parakeet-tdt-0.6b-v2 (env: main)
status   deployed now
url      https://<workspace>--livekit-stt-parakeet-tdt-0-6b-v2-parakeetse-<hash>.modal.run
gpu      L4
warm     0 container(s), $0.00/h while idle
         scales to zero: the first request after idle waits for a cold start

Created a proxy token. The secret is shown once; set these for your agent:
    export MODAL_PROXY_TOKEN_ID=wk-...
    export MODAL_PROXY_TOKEN_SECRET=ws-...
```

Then use it in the agent:

```python
from livekit.agents import AgentSession
from livekit.plugins import modal, silero

session = AgentSession(
    stt=modal.STT(model="parakeet-tdt-0.6b-v2", env="main"),
    vad=silero.VAD.load(),
    # llm=..., tts=...
)
```

The agent looks the deployment up by name on first use and caches the URL. It never deploys. If the deployment is missing, `modal.STT` raises `modal.DeploymentNotFoundError` with the exact `provision` command to run.

Parakeet streams over a WebSocket: interim transcripts while the user speaks, a final one when they pause. The final always precedes `END_OF_SPEECH` for the same utterance. Whisper is a batch model (`streaming=False`): `AgentSession` wraps it in `stt.StreamAdapter`, which sends each utterance the session VAD detects.

## Models

| Model | Engine | GPU | Languages |
| --- | --- | --- | --- |
| `parakeet-tdt-0.6b-v2` (default) | [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), streaming | L4 | English |
| `whisper-large-v3-turbo` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), float16, batch | L4 | 99, auto-detected |

Parakeet TDT is an offline model, so the server streams it: Silero VAD finds utterance boundaries (500 ms of silence ends one), and the open utterance is decoded again every 0.5 s of new speech for the interim transcript. Each decode takes tens of milliseconds on an L4.

Weights download once into the `livekit-stt-cache` Modal Volume, so later cold starts skip the download.

## Provision flags

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `parakeet-tdt-0.6b-v2` | Model to deploy |
| `--env` | your profile's | Modal environment |
| `--gpu` | the model's | GPU type, e.g. `T4`, `L4`, `A10`, `H100` |
| `--region` | any | Pin containers to a region, e.g. `us-east`, close to your LiveKit server. Costs 1.15x to 1.75x |
| `--min-containers` | `0` | Containers kept warm. Each one bills while idle (L4: about $0.96/h) |
| `--scaledown-window` | `300` | Seconds an idle container stays up after its last request |
| `--redeploy` | off | Deploy even if the App exists, to apply changed flags |
| `--no-proxy-token` | off | Do not create a proxy token |

With `--min-containers 0` the deployment costs nothing while unused, but the first utterance after an idle period waits for a cold start. Keep one container warm for production traffic.

## STT options

| Option | Default | Description |
| --- | --- | --- |
| `model` | `parakeet-tdt-0.6b-v2` | Deployed model |
| `env` | your profile's | Modal environment of the deployment |
| `language` | model's, or auto-detect | Language code, e.g. `"en"`. Whisper detects it when unset |
| `proxy_token_id` / `proxy_token_secret` | `$MODAL_PROXY_TOKEN_ID` / `$MODAL_PROXY_TOKEN_SECRET` | Modal proxy token. The endpoint rejects requests without one |
| `base_url` | looked up | Use this endpoint URL instead of looking it up |
| `auto_provision` | `False` | Deploy the model if it is missing. Takes minutes; development only |

The agent process needs Modal credentials (`modal token` or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`) for the lookup, unless you pass `base_url`.

## Endpoint

The server is a plain HTTP API behind [Modal proxy auth](https://modal.com/docs/guide/webhook-proxy-auth), so anything can call it:

```bash
curl -H "Modal-Key: $MODAL_PROXY_TOKEN_ID" -H "Modal-Secret: $MODAL_PROXY_TOKEN_SECRET" \
  --data-binary @clip.wav "$URL/v1/transcribe?language=en"
```

The body is 16 kHz mono 16-bit WAV; the response has `text`, `language`, `duration` and `segments`. Parakeet also serves `WS /v1/stream`: send binary 16 kHz mono 16-bit PCM and `{"type": "flush"}` or `{"type": "close"}`; receive `start_of_speech`, `interim`, `final` and `end_of_speech` messages with times in seconds since the stream started.

## Remove

```bash
modal app stop livekit-stt-parakeet-tdt-0.6b-v2 --env main
```
