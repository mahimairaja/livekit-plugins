"""livekit-plugins-modal batch STT contract, against a fake endpoint on localhost (no Modal account).

The fake server speaks the deployed server's HTTP contract. The LiveKit side (STT.recognize,
retries, APIConnectOptions, audio frames) is the real SDK.
"""

from __future__ import annotations

import io
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import modal as modal_sdk
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt
from livekit.plugins import modal
from livekit.plugins.modal import provision

pytestmark = pytest.mark.unit

NO_RETRY = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
ONE_RETRY = APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=5.0)
TOKEN: dict[str, Any] = {"proxy_token_id": "wk-test", "proxy_token_secret": "ws-test"}


@dataclass
class Request:
    headers: dict[str, str]
    query: dict[str, str]
    wav: bytes


@dataclass
class FakeEndpoint:
    """The deployed server's /v1/transcribe, with scripted failures."""

    url: str = ""
    requests: list[Request] = field(default_factory=list)
    fail_with: list[int] = field(default_factory=list)
    reply: dict[str, Any] = field(
        default_factory=lambda: {
            "text": "hello world",
            "language": "en",
            "duration": 1.0,
            "segments": [{"start": 0.1, "end": 0.9, "text": "hello world"}],
        }
    )

    async def transcribe(self, request: web.Request) -> web.Response:
        self.requests.append(
            Request(dict(request.headers), dict(request.query), await request.read())
        )
        if self.fail_with:
            return web.Response(status=self.fail_with.pop(0), text="scripted failure")
        return web.json_response(self.reply)


@pytest.fixture
async def endpoint() -> AsyncIterator[FakeEndpoint]:
    fake = FakeEndpoint()
    app = web.Application()
    app.router.add_post("/v1/transcribe", fake.transcribe)
    async with TestServer(app) as server:
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake


@pytest.fixture
async def session() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as s:
        yield s


def _speech(seconds: float = 0.5, *, sample_rate: int = 48000, channels: int = 2) -> rtc.AudioFrame:
    n = int(seconds * sample_rate)
    tone = (np.sin(np.arange(n) * 2 * np.pi * 440 / sample_rate) * 8000).astype(np.int16)
    pcm = np.repeat(tone[:, None], channels, axis=1)
    return rtc.AudioFrame(pcm.tobytes(), sample_rate, channels, n)


WHISPER = "whisper-large-v3-turbo"


def _stt(session: aiohttp.ClientSession, **kwargs: Any) -> modal.STT:
    return modal.STT(**{**TOKEN, "model": WHISPER, "http_session": session, **kwargs})


async def test_recognize_sends_16k_mono_wav_with_proxy_auth(
    endpoint: FakeEndpoint, session: aiohttp.ClientSession
) -> None:
    engine = _stt(session, base_url=endpoint.url, language="en")

    event = await engine.recognize([_speech(0.5)], conn_options=NO_RETRY)

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "hello world"
    assert event.alternatives[0].language == "en"
    assert (event.alternatives[0].start_time, event.alternatives[0].end_time) == (0.1, 0.9)

    [req] = endpoint.requests
    assert req.headers["Modal-Key"] == "wk-test"
    assert req.headers["Modal-Secret"] == "ws-test"
    assert req.query == {"language": "en"}
    with wave.open(io.BytesIO(req.wav)) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (16000, 1, 2)
        assert f.getnframes() == pytest.approx(8000, abs=160)


async def test_language_is_left_to_the_server_when_unset(
    endpoint: FakeEndpoint, session: aiohttp.ClientSession
) -> None:
    endpoint.reply = {**endpoint.reply, "language": "fr", "text": "bonjour"}
    engine = _stt(session, base_url=endpoint.url)

    event = await engine.recognize([_speech()], conn_options=NO_RETRY)

    assert endpoint.requests[0].query == {}
    assert event.alternatives[0].language == "fr"


async def test_server_error_is_retried(
    endpoint: FakeEndpoint, session: aiohttp.ClientSession
) -> None:
    endpoint.fail_with = [503]
    engine = _stt(session, base_url=endpoint.url)

    event = await engine.recognize([_speech()], conn_options=ONE_RETRY)

    assert event.alternatives[0].text == "hello world"
    assert len(endpoint.requests) == 2


async def test_auth_failure_is_not_retried(
    endpoint: FakeEndpoint, session: aiohttp.ClientSession
) -> None:
    endpoint.fail_with = [401]
    engine = _stt(session, base_url=endpoint.url)

    with pytest.raises(APIStatusError) as err:
        await engine.recognize([_speech()], conn_options=ONE_RETRY)

    assert err.value.status_code == 401
    assert len(endpoint.requests) == 1


def test_proxy_token_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)

    with pytest.raises(ValueError, match="modal workspace proxy-tokens create"):
        modal.STT()


def test_proxy_token_defaults_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-env")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-env")

    engine = modal.STT()

    assert engine._headers == {"Modal-Key": "wk-env", "Modal-Secret": "ws-env"}


def test_batch_model_needs_the_stream_adapter() -> None:
    engine = modal.STT(**TOKEN, model=WHISPER)

    assert engine.capabilities.streaming is False
    assert engine.provider == "modal"
    with pytest.raises(NotImplementedError, match="StreamAdapter"):
        engine.stream()


async def test_endpoint_is_looked_up_by_name_once(
    endpoint: FakeEndpoint, session: aiohttp.ClientSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookups: list[tuple[str, str | None]] = []

    def lookup(model: str, environment: str | None = None) -> str:
        lookups.append((model, environment))
        return endpoint.url

    monkeypatch.setattr(provision, "lookup_web_url", lookup)
    engine = _stt(session, env="staging")

    await engine.recognize([_speech()], conn_options=NO_RETRY)
    await engine.recognize([_speech()], conn_options=NO_RETRY)

    assert lookups == [("whisper-large-v3-turbo", "staging")]
    assert len(endpoint.requests) == 2


async def test_not_found_drops_the_cached_url(
    endpoint: FakeEndpoint, session: aiohttp.ClientSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookups: list[str] = []

    def lookup(model: str, environment: str | None = None) -> str:
        lookups.append(model)
        return endpoint.url

    monkeypatch.setattr(provision, "lookup_web_url", lookup)
    endpoint.fail_with = [404]
    engine = _stt(session)

    with pytest.raises(APIStatusError):
        await engine.recognize([_speech()], conn_options=NO_RETRY)
    await engine.recognize([_speech()], conn_options=NO_RETRY)

    assert len(lookups) == 2


async def test_missing_deployment_names_the_provision_command(
    session: aiohttp.ClientSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modal.exception import NotFoundError

    def from_name(*args: Any, **kwargs: Any) -> Any:
        raise NotFoundError("App 'livekit-stt-whisper-large-v3-turbo' not found")

    monkeypatch.setattr(modal_sdk.Cls, "from_name", from_name)
    engine = _stt(session, env="staging")

    with pytest.raises(modal.DeploymentNotFoundError) as err:
        await engine.recognize([_speech()], conn_options=NO_RETRY)

    assert (
        "python -m livekit.plugins.modal provision stt --model whisper-large-v3-turbo "
        "--env staging" in str(err.value)
    )


async def test_auto_provision_deploys_a_missing_model(
    endpoint: FakeEndpoint, session: aiohttp.ClientSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployed: list[provision.ServerConfig] = []

    def lookup(model: str, environment: str | None = None) -> str:
        raise provision.DeploymentNotFoundError(model)

    def provision_stt(config: provision.ServerConfig, **_: Any) -> provision.ProvisionResult:
        deployed.append(config)
        return provision.ProvisionResult(
            app_name=config.app_name,
            environment=config.environment,
            url=endpoint.url,
            created=True,
            gpu="L4",
            min_containers=0,
            warm_cost_per_hour=(0.0, 0.0),
        )

    monkeypatch.setattr(provision, "lookup_web_url", lookup)
    monkeypatch.setattr(provision, "provision_stt", provision_stt)
    engine = _stt(session, env="dev", auto_provision=True)

    event = await engine.recognize([_speech()], conn_options=NO_RETRY)

    assert event.alternatives[0].text == "hello world"
    assert [(c.model, c.environment, c.min_containers) for c in deployed] == [
        ("whisper-large-v3-turbo", "dev", 0)
    ]
