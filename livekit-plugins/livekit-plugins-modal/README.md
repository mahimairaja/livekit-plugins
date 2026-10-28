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
python -m livekit.plugins.modal provision stt --model whisper-large-v3-turbo --env main
```

It deploys the server if it is not there yet (rerunning it is safe), prints the endpoint URL and what the warm floor costs per hour, and creates a proxy token for the agent:

```
app      livekit-stt-whisper-large-v3-turbo (env: main)
status   deployed now
url      https://<workspace>-main--livekit-stt-whisper-large-v3-turbo-whisperserver-web.modal.run
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
    stt=modal.STT(model="whisper-large-v3-turbo", env="main"),
    vad=silero.VAD.load(),
    # llm=..., tts=...
)
```

The agent looks the deployment up by name on first use and caches the URL. It never deploys. If the deployment is missing, `modal.STT` raises `modal.DeploymentNotFoundError` with the exact `provision` command to run.

Whisper is a batch model (`streaming=False`): `AgentSession` wraps it in `stt.StreamAdapter`, which sends each utterance the VAD detects.

## Models

| Model | Engine | GPU | Languages |
| --- | --- | --- | --- |
| `whisper-large-v3-turbo` (default) | [faster-whisper](https://github.com/SYSTRAN/faster-whisper), float16 | L4 | 99, auto-detected |

Weights download once into the `livekit-stt-cache` Modal Volume, so later cold starts skip the download.

## Provision flags

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `whisper-large-v3-turbo` | Model to deploy |
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
| `model` | `whisper-large-v3-turbo` | Deployed model |
| `env` | your profile's | Modal environment of the deployment |
| `language` | auto-detect | Language code, e.g. `"en"` |
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

The body is 16 kHz mono 16-bit WAV; the response has `text`, `language`, `duration` and `segments`.

## Remove

```bash
modal app stop livekit-stt-whisper-large-v3-turbo --env main
```
