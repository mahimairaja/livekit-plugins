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

"""Streaming Parakeet STT server: NVIDIA NeMo on a Modal GPU, behind proxy auth.

Parakeet TDT is an offline model, so the server streams by endpointing with Silero VAD and
re-decoding the open utterance as it grows: an interim transcript every INTERIM_S of new
speech, a final one when the speaker pauses. Re-decoding a few seconds of audio takes tens
of milliseconds on the GPU.

`provision` loads this file as a standalone module and deploys `app`. Modal mounts the file
into the container and imports it there again, so it must not import the plugin package.

Protocol (all routes need the Modal-Key and Modal-Secret proxy auth headers):
    WS   /v1/stream
        client -> server: binary 16 kHz mono 16-bit PCM; text {"type": "flush"} ends the
            open utterance now; {"type": "close"} ends it and closes the socket.
        server -> client: {"type": "start_of_speech", "start"},
            {"type": "interim" | "final", "text", "start", "end"}, {"type": "end_of_speech",
            "end"}. Times are seconds of audio since the stream started. A final always
            precedes the end_of_speech of the same utterance.
    POST /v1/transcribe?language=en    body: 16 kHz mono 16-bit WAV
        -> {"text", "language", "duration", "segments": [{"start", "end", "text"}]}
    GET  /health                       {"status": "ok", "model": ...}
"""

# No `from __future__ import annotations`: FastAPI resolves route annotations against the
# module globals, and the fastapi types are imported inside `web()`, in the container only.
import asyncio
import io
import json
import os
import threading
import wave
from collections.abc import Callable
from typing import Any

import modal

CONFIG_ENV = "LIVEKIT_MODAL_SERVER_CONFIG"
CACHE_DIR = "/cache"
SAMPLE_RATE = 16000
VAD_WINDOW = 512  # samples; the only window size Silero VAD accepts at 16 kHz
PREROLL_S = 0.3  # audio kept from before the VAD fires, so first syllables are not cut
INTERIM_S = 0.5
MAX_UTTERANCE_S = 20.0

_config: dict[str, Any] = json.loads(os.environ.get(CONFIG_ENV) or "{}")
APP_NAME: str = _config.get("app_name", "livekit-stt-parakeet-tdt-0.6b-v2")
ENGINE_MODEL: str = _config.get("engine_model", "nvidia/parakeet-tdt-0.6b-v2")
GPU: str = _config.get("gpu", "L4")
REGION: str | None = _config.get("region")
MIN_CONTAINERS: int = _config.get("min_containers", 0)
SCALEDOWN_WINDOW: int = _config.get("scaledown_window", 300)
CPU: float = _config.get("cpu", 2.0)
MEMORY_MIB: int = _config.get("memory_mib", 8192)
MAX_INPUTS: int = _config.get("max_inputs", 8)
MIN_SILENCE_MS: int = _config.get("min_silence_ms", 500)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "nemo_toolkit[asr]==2.7.3",
        # torchaudio is pinned by NeMo's deps; torch must match it.
        "torch==2.11.0",
        "torchaudio==2.11.0",
        "numpy==2.3.4",
        "silero-vad==6.2.1",
        "fastapi==0.120.0",
        "huggingface-hub[hf-transfer]==0.36.0",
    )
    .env(
        {
            CONFIG_ENV: json.dumps(
                {
                    "app_name": APP_NAME,
                    "engine_model": ENGINE_MODEL,
                    "min_silence_ms": MIN_SILENCE_MS,
                }
            ),
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)

# Weights download once into the volume; later cold starts read them from there.
volume = modal.Volume.from_name("livekit-stt-cache", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    region=REGION,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={CACHE_DIR: volume},
    timeout=3600,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
class ParakeetServer:
    @modal.enter()
    def load(self) -> None:
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=ENGINE_MODEL)
        self.model.eval()
        # Streams share one model; NeMo's transcribe is not safe to call concurrently.
        self.lock = threading.Lock()
        volume.commit()
        # Compile the decoding path before the first real request.
        self.decode(_silence(1.0))

    def decode(self, audio: Any) -> str:
        with self.lock:
            out = self.model.transcribe([audio], batch_size=1, verbose=False)
        hyps = out[0] if isinstance(out, tuple) else out
        hyp = hyps[0]
        return str(getattr(hyp, "text", hyp)).strip()

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self) -> Any:
        from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

        api = FastAPI()

        @api.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "model": ENGINE_MODEL}

        @api.post("/v1/transcribe")
        async def transcribe(request: Request, language: str | None = None) -> dict[str, Any]:
            try:
                audio = _read_wav(await request.body())
            except (ValueError, EOFError, wave.Error) as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            text = await asyncio.to_thread(self.decode, audio)
            duration = len(audio) / SAMPLE_RATE
            return {
                "text": text,
                "language": "en",
                "duration": duration,
                "segments": [{"start": 0.0, "end": duration, "text": text}] if text else [],
            }

        @api.websocket("/v1/stream")
        async def stream(ws: WebSocket) -> None:
            await ws.accept()
            session = _StreamSession(self.decode, ws)
            try:
                await session.run()
            except WebSocketDisconnect:
                pass

        return api


class _StreamSession:
    """One WebSocket stream: VAD endpointing plus re-decoding of the open utterance."""

    def __init__(self, decode: Callable[[Any], str], ws: Any) -> None:
        import numpy as np
        from silero_vad import VADIterator, load_silero_vad

        self._np = np
        self._decode_blocking = decode
        self._ws = ws
        # Silero keeps its recurrent state in the model, so every stream needs its own.
        self._vad = VADIterator(
            load_silero_vad(),
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=100,
        )
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: list[Any] = []
        self._utterance: list[Any] | None = None
        self._start = 0  # sample index where the open utterance starts
        self._samples = 0  # samples fed to the VAD so far
        self._since_interim = 0
        self._last_text = ""

    async def run(self) -> None:
        while True:
            msg = await self._ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                await self._push(msg["bytes"])
                continue
            kind = json.loads(msg.get("text") or "{}").get("type")
            if kind in ("flush", "close"):
                await self._finalize()
                self._vad.reset_states()
            if kind == "close":
                await self._ws.close()
                return

    async def _push(self, pcm: bytes) -> None:
        np = self._np
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, samples])
        preroll_windows = int(PREROLL_S * SAMPLE_RATE) // VAD_WINDOW
        while len(self._pending) >= VAD_WINDOW:
            window, self._pending = self._pending[:VAD_WINDOW], self._pending[VAD_WINDOW:]
            event = self._vad(_tensor(window)) or {}
            self._samples += VAD_WINDOW
            if self._utterance is None:
                self._preroll = [*self._preroll, window][-preroll_windows:]
                if "start" in event:
                    self._utterance = list(self._preroll)
                    self._start = self._samples - len(self._utterance) * VAD_WINDOW
                    self._since_interim = 0
                    await self._send("start_of_speech", start=self._start / SAMPLE_RATE)
                continue
            self._utterance.append(window)
            self._since_interim += VAD_WINDOW
            if "end" in event:
                await self._finalize()
            elif self._samples - self._start >= MAX_UTTERANCE_S * SAMPLE_RATE:
                # The VAD is still triggered and will not report a new "start" until it sees
                # an "end", so open the next utterance here or the speech after the cut is lost.
                await self._finalize()
                self._utterance = []
                self._start = self._samples
                self._since_interim = 0
                await self._send("start_of_speech", start=self._start / SAMPLE_RATE)
        if self._utterance is not None and self._since_interim >= INTERIM_S * SAMPLE_RATE:
            self._since_interim = 0
            text = await self._decode()
            if text and text != self._last_text:
                self._last_text = text
                await self._send("interim", text=text, **self._span())

    async def _finalize(self) -> None:
        if self._utterance is None:
            return
        # An utterance opened at the length cap can be closed before it gets any audio.
        text = await self._decode() if self._utterance else ""
        span = self._span()
        if text:
            await self._send("final", text=text, **span)
        await self._send("end_of_speech", end=span["end"])
        self._utterance = None
        self._preroll = []
        self._last_text = ""

    async def _decode(self) -> str:
        assert self._utterance is not None
        audio = self._np.concatenate(self._utterance)
        return await asyncio.to_thread(self._decode_blocking, audio)

    def _span(self) -> dict[str, float]:
        assert self._utterance is not None
        end = self._start + len(self._utterance) * VAD_WINDOW
        return {"start": self._start / SAMPLE_RATE, "end": end / SAMPLE_RATE}

    async def _send(self, type: str, **fields: Any) -> None:
        await self._ws.send_text(json.dumps({"type": type, **fields}))


def _tensor(window: Any) -> Any:
    import torch

    return torch.from_numpy(window)


def _silence(seconds: float) -> Any:
    import numpy as np

    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _read_wav(wav: bytes) -> Any:
    import numpy as np

    with wave.open(io.BytesIO(wav)) as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError("expected 16 kHz mono 16-bit WAV")
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0
