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

"""Batch Whisper STT server: faster-whisper on a Modal GPU, behind proxy auth.

`provision` loads this file as a standalone module and deploys `app`. Modal mounts the file
into the container and imports it there again, so it must not import the plugin package.
Deploy settings arrive as JSON in LIVEKIT_MODAL_SERVER_CONFIG; the image carries the engine
settings so the container sees the same model.

HTTP contract (all routes need the Modal-Key and Modal-Secret proxy auth headers):
    GET  /health                       {"status": "ok", "model": ...}
    POST /v1/transcribe?language=en    body: 16 kHz mono 16-bit WAV
        -> {"text", "language", "duration", "segments": [{"start", "end", "text"}]}
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import wave
from typing import Any

import modal

CONFIG_ENV = "LIVEKIT_MODAL_SERVER_CONFIG"
CACHE_DIR = "/cache"
SAMPLE_RATE = 16000

_config: dict[str, Any] = json.loads(os.environ.get(CONFIG_ENV) or "{}")
APP_NAME: str = _config.get("app_name", "livekit-stt-whisper-large-v3-turbo")
ENGINE_MODEL: str = _config.get("engine_model", "large-v3-turbo")
GPU: str = _config.get("gpu", "L4")
REGION: str | None = _config.get("region")
MIN_CONTAINERS: int = _config.get("min_containers", 0)
SCALEDOWN_WINDOW: int = _config.get("scaledown_window", 300)
CPU: float = _config.get("cpu", 2.0)
MEMORY_MIB: int = _config.get("memory_mib", 8192)
MAX_INPUTS: int = _config.get("max_inputs", 4)

image = (
    # ctranslate2 4.6 needs CUDA 12 and cuDNN 9 at runtime.
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "faster-whisper==1.2.1",
        "ctranslate2==4.6.0",
        "numpy==2.3.4",
        "fastapi==0.120.0",
        "huggingface-hub[hf-transfer]==0.36.0",
    )
    .env(
        {
            CONFIG_ENV: json.dumps({"app_name": APP_NAME, "engine_model": ENGINE_MODEL}),
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
    timeout=600,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
class WhisperServer:
    @modal.enter()
    def load(self) -> None:
        from faster_whisper import WhisperModel

        # One worker per concurrent input, so requests decode in parallel on the GPU.
        self.model = WhisperModel(
            ENGINE_MODEL, device="cuda", compute_type="float16", num_workers=MAX_INPUTS
        )
        volume.commit()

    def transcribe(self, wav: bytes, language: str | None) -> dict[str, Any]:
        import numpy as np

        with wave.open(io.BytesIO(wav)) as f:
            if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError("expected 16 kHz mono 16-bit WAV")
            pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0

        segments, info = self.model.transcribe(
            audio, language=language, beam_size=1, vad_filter=False
        )
        parts = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments  # a generator: decoding happens here
        ]
        return {
            "text": " ".join(p["text"] for p in parts if p["text"]),
            "language": info.language,
            "duration": info.duration,
            "segments": parts,
        }

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self) -> Any:
        from fastapi import FastAPI, HTTPException, Request

        api = FastAPI()

        @api.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "model": ENGINE_MODEL}

        @api.post("/v1/transcribe")
        async def transcribe(request: Request, language: str | None = None) -> dict[str, Any]:
            body = await request.body()
            try:
                return await asyncio.to_thread(self.transcribe, body, language or None)
            except (ValueError, EOFError, wave.Error) as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        return api
