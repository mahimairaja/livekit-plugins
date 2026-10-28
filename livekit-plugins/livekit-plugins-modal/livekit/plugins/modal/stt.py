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

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    NotGivenOr,
    stt,
    utils,
)
from livekit.agents.utils import AudioBuffer, is_given

from . import provision
from .log import logger
from .models import DEFAULT_MODEL, STTModels, get_model

SAMPLE_RATE = 16000
TOKEN_ID_ENV = "MODAL_PROXY_TOKEN_ID"
TOKEN_SECRET_ENV = "MODAL_PROXY_TOKEN_SECRET"


@dataclass
class _STTOptions:
    model: str
    environment: str | None
    language: str | None


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = DEFAULT_MODEL,
        env: str | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
        proxy_token_id: NotGivenOr[str] = NOT_GIVEN,
        proxy_token_secret: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        auto_provision: bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """STT served from your own Modal deployment.

        Deploy the model once with `python -m livekit.plugins.modal provision stt`. This class
        finds the deployment by name on first use and caches its URL; it never deploys unless
        `auto_provision` is set.

        Batch models (Whisper) transcribe whole utterances, so this STT reports
        `streaming=False` and `AgentSession` runs it through `stt.StreamAdapter` with the
        session VAD.

        Args:
            model: Model to use. Must be deployed in `env`.
            env: Modal environment of the deployment. Defaults to your Modal profile's.
            language: Language code, e.g. "en". Multilingual models detect it when unset.
            proxy_token_id: Modal proxy token id (`wk-...`). Defaults to
                $MODAL_PROXY_TOKEN_ID.
            proxy_token_secret: Modal proxy token secret (`ws-...`). Defaults to
                $MODAL_PROXY_TOKEN_SECRET.
            base_url: Use this endpoint URL instead of looking the deployment up.
            auto_provision: Deploy the model (scale to zero, default GPU) if it is missing.
                Deploying takes minutes; use it in development only.
            http_session: aiohttp session to use. Defaults to the agent's shared session.
        """
        spec = get_model(model)
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False, interim_results=False, offline_recognize=True
            )
        )
        if language is NOT_GIVEN and spec.language is not None:
            language = spec.language
        self._opts = _STTOptions(
            model=model,
            environment=env,
            language=language if is_given(language) else None,
        )
        token_id = proxy_token_id if is_given(proxy_token_id) else os.environ.get(TOKEN_ID_ENV)
        token_secret = (
            proxy_token_secret if is_given(proxy_token_secret) else os.environ.get(TOKEN_SECRET_ENV)
        )
        if not token_id or not token_secret:
            raise ValueError(
                f"Modal proxy token required: set {TOKEN_ID_ENV} and {TOKEN_SECRET_ENV}, or pass "
                "proxy_token_id and proxy_token_secret. Create one with "
                "`modal workspace proxy-tokens create`."
            )
        self._headers = {"Modal-Key": token_id, "Modal-Secret": token_secret}
        self._base_url = base_url.rstrip("/") if is_given(base_url) else None
        self._pinned_url = self._base_url is not None
        self._auto_provision = auto_provision
        self._resolve_lock = asyncio.Lock()
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "modal"

    async def endpoint_url(self) -> str:
        """The deployment's URL, looked up by name once and then cached."""
        async with self._resolve_lock:
            if self._base_url is None:
                self._base_url = await self._resolve()
            return self._base_url

    async def _resolve(self) -> str:
        model, env = self._opts.model, self._opts.environment
        try:
            return await asyncio.to_thread(provision.lookup_web_url, model, env)
        except provision.DeploymentNotFoundError:
            if not self._auto_provision:
                raise
        logger.warning("auto_provision: deploying %s to Modal; this takes minutes", model)
        config = provision.ServerConfig(model=model, environment=env)
        result = await asyncio.to_thread(provision.provision_stt, config)
        return result.url

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language
        wav = _to_wav_16k_mono(buffer)
        url = await self.endpoint_url()
        params = {"language": lang} if lang else {}
        try:
            async with self._ensure_session().post(
                f"{url}/v1/transcribe",
                data=wav,
                params=params,
                headers={**self._headers, "Content-Type": "audio/wav"},
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    stale_url = resp.status == 404 and not self._pinned_url
                    if stale_url:
                        # The App was stopped or redeployed under a new URL; look it up again.
                        self._base_url = None
                    err = APIStatusError(
                        f"Modal STT request failed: {body[:200]}",
                        status_code=resp.status,
                        request_id=resp.headers.get("modal-function-call-id"),
                        body=body,
                    )
                    if stale_url:
                        # APIStatusError forces 4xx to non-retryable; this one is fixed by
                        # retrying against the re-resolved URL.
                        err.retryable = True
                    raise err
                result: dict[str, Any] = await resp.json()
        except TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e

        segments = result.get("segments") or []
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=resp.headers.get("modal-function-call-id") or utils.shortuuid(),
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(result.get("language") or lang or ""),
                    text=result.get("text", ""),
                    start_time=segments[0]["start"] if segments else 0.0,
                    end_time=segments[-1]["end"] if segments else result.get("duration", 0.0),
                )
            ],
        )


def _to_wav_16k_mono(buffer: AudioBuffer) -> bytes:
    frame = rtc.combine_audio_frames(buffer)
    if frame.num_channels > 1:
        pcm = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
        mono = pcm.mean(axis=1).astype(np.int16)
        frame = rtc.AudioFrame(mono.tobytes(), frame.sample_rate, 1, len(mono))
    if frame.sample_rate != SAMPLE_RATE:
        resampler = rtc.AudioResampler(frame.sample_rate, SAMPLE_RATE, num_channels=1)
        frame = rtc.combine_audio_frames([*resampler.push(frame), *resampler.flush()])
    return frame.to_wav_bytes()
