"""livekit-plugins-modal streaming STT contract, against a fake /v1/stream WebSocket on localhost.

The fake server plays the deployed Parakeet server's side of the protocol from a script. The
LiveKit side (RecognizeStream, input resampling, retries, event channel) is the real SDK.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt
from livekit.plugins import modal
from livekit.plugins.modal import provision

pytestmark = pytest.mark.unit

NO_RETRY = APIConnectOptions(max_retry=0, retry_interval=0.0, timeout=5.0)
ONE_RETRY = APIConnectOptions(max_retry=1, retry_interval=0.0, timeout=5.0)

SOS = stt.SpeechEventType.START_OF_SPEECH
INTERIM = stt.SpeechEventType.INTERIM_TRANSCRIPT
FINAL = stt.SpeechEventType.FINAL_TRANSCRIPT
EOS = stt.SpeechEventType.END_OF_SPEECH
USAGE = stt.SpeechEventType.RECOGNITION_USAGE

Send = Callable[[dict[str, Any]], Any]


@dataclass
class Connection:
    headers: dict[str, str]
    audio: bytearray = field(default_factory=bytearray)
    control: list[str] = field(default_factory=list)


@dataclass
class FakeStream:
    """The server side of /v1/stream: speech starts with the first audio, an interim after
    `interim_after` bytes, and a final plus end_of_speech on flush or close."""

    url: str = ""
    connections: list[Connection] = field(default_factory=list)
    interim_after: int = 3200
    drop_first_connection_after: int | None = None
    reject_with: list[int] = field(default_factory=list)
    handshakes: int = 0

    async def handler(self, request: web.Request) -> web.StreamResponse:
        self.handshakes += 1
        if self.reject_with:
            return web.Response(status=self.reject_with.pop(0))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        conn = Connection(dict(request.headers))
        self.connections.append(conn)
        drop_after = self.drop_first_connection_after if len(self.connections) == 1 else None
        speaking = interim_sent = False

        async def send(msg: dict[str, Any]) -> None:
            await ws.send_str(json.dumps(msg))

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                conn.audio += msg.data
                if drop_after is not None and len(conn.audio) >= drop_after:
                    # what a container shutting down sends
                    await ws.close(code=aiohttp.WSCloseCode.GOING_AWAY)
                    break
                if not speaking:
                    speaking = True
                    await send({"type": "start_of_speech", "start": 0.1})
                if not interim_sent and len(conn.audio) >= self.interim_after:
                    interim_sent = True
                    await send({"type": "interim", "text": "hello", "start": 0.1, "end": 0.3})
            elif msg.type == aiohttp.WSMsgType.TEXT:
                kind = json.loads(msg.data)["type"]
                conn.control.append(kind)
                if speaking:
                    await send({"type": "final", "text": "hello world", "start": 0.1, "end": 0.6})
                    await send({"type": "end_of_speech", "end": 0.6})
                    speaking = interim_sent = False
                if kind == "close":
                    await ws.close()
        return ws


@pytest.fixture
async def server() -> AsyncIterator[FakeStream]:
    fake = FakeStream()
    app = web.Application()
    app.router.add_get("/v1/stream", fake.handler)
    async with TestServer(app) as srv:
        fake.url = str(srv.make_url("")).rstrip("/")
        yield fake


@pytest.fixture
async def session() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as s:
        yield s


def _frame(ms: int = 100, *, sample_rate: int = 16000, channels: int = 1) -> rtc.AudioFrame:
    n = sample_rate * ms // 1000
    tone = (np.sin(np.arange(n) * 2 * np.pi * 440 / sample_rate) * 8000).astype(np.int16)
    pcm = np.repeat(tone[:, None], channels, axis=1)
    return rtc.AudioFrame(pcm.tobytes(), sample_rate, channels, n)


def _stt(server: FakeStream, session: aiohttp.ClientSession) -> modal.STT:
    return modal.STT(
        base_url=server.url,
        proxy_token_id="wk-test",
        proxy_token_secret="ws-test",
        http_session=session,
    )


async def _run(
    engine: modal.STT,
    frames: list[rtc.AudioFrame],
    *,
    flush_after: int | None = None,
    conn_options: APIConnectOptions = NO_RETRY,
    offset: float = 0.0,
) -> list[stt.SpeechEvent]:
    stream = engine.stream(conn_options=conn_options)
    stream.start_time_offset = offset
    for i, frame in enumerate(frames):
        stream.push_frame(frame)
        if i == flush_after:
            stream.flush()
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()
    return events


def _types(events: list[stt.SpeechEvent]) -> list[stt.SpeechEventType]:
    return [ev.type for ev in events]


def test_parakeet_is_the_streaming_default() -> None:
    engine = modal.STT(proxy_token_id="wk", proxy_token_secret="ws")

    assert engine.model == "parakeet-tdt-0.6b-v2"
    assert engine.capabilities.streaming
    assert engine.capabilities.interim_results


async def test_stream_relays_interim_then_final_before_end_of_speech(
    server: FakeStream, session: aiohttp.ClientSession
) -> None:
    events = await _run(_stt(server, session), [_frame() for _ in range(5)], offset=10.0)

    assert _types(events) == [SOS, INTERIM, FINAL, EOS, USAGE]
    interim, final = events[1].alternatives[0], events[2].alternatives[0]
    assert (interim.text, final.text) == ("hello", "hello world")
    assert final.language == "en"
    # Server times are seconds since the stream started; the session clock is added on top.
    assert final.start_time == pytest.approx(10.1, abs=0.05)
    assert final.end_time == pytest.approx(10.6, abs=0.05)
    assert events[4].recognition_usage is not None
    assert events[4].recognition_usage.audio_duration == pytest.approx(0.5)
    assert len({ev.request_id for ev in events}) == 1


async def test_flush_ends_the_open_utterance(
    server: FakeStream, session: aiohttp.ClientSession
) -> None:
    events = await _run(_stt(server, session), [_frame() for _ in range(6)], flush_after=2)

    assert _types(events) == [SOS, INTERIM, FINAL, EOS, USAGE, SOS, INTERIM, FINAL, EOS, USAGE]
    # end_input() flushes before the stream closes.
    assert server.connections[0].control == ["flush", "flush", "close"]
    # Each utterance gets its own request id.
    assert events[0].request_id != events[5].request_id


@pytest.mark.parametrize(
    ("sample_rate", "channels"),
    [(48000, 1), (16000, 2)],  # RecognizeStream resamples mono only, so stereo stays at 16k
)
async def test_stream_sends_16k_mono_pcm_with_proxy_auth(
    server: FakeStream, session: aiohttp.ClientSession, sample_rate: int, channels: int
) -> None:
    frames = [_frame(sample_rate=sample_rate, channels=channels) for _ in range(5)]
    await _run(_stt(server, session), frames)

    [conn] = server.connections
    assert conn.headers["Modal-Key"] == "wk-test"
    assert conn.headers["Modal-Secret"] == "ws-test"
    # 0.5 s at 16 kHz mono 16-bit, give or take the resampler's latency.
    assert len(conn.audio) == pytest.approx(16000, abs=640)


async def test_dropped_connection_is_retried(
    server: FakeStream, session: aiohttp.ClientSession
) -> None:
    server.drop_first_connection_after = 3200

    await _run(_stt(server, session), [_frame() for _ in range(5)], conn_options=ONE_RETRY)

    # The stream reconnects and finishes cleanly on the second connection.
    assert len(server.connections) == 2
    assert server.connections[1].control[-1] == "close"


async def test_dropped_connection_without_retries_fails(
    server: FakeStream, session: aiohttp.ClientSession
) -> None:
    server.drop_first_connection_after = 3200

    with pytest.raises(APIConnectionError):
        await _run(_stt(server, session), [_frame() for _ in range(5)])


async def test_rejected_handshake_is_not_retried(
    server: FakeStream, session: aiohttp.ClientSession
) -> None:
    server.reject_with = [401]

    with pytest.raises(APIStatusError) as err:
        await _run(_stt(server, session), [_frame()], conn_options=ONE_RETRY)

    assert err.value.status_code == 401


async def test_not_found_handshake_reconnects_to_the_new_url(
    server: FakeStream, session: aiohttp.ClientSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookups: list[str] = []

    def lookup(model: str, environment: str | None = None) -> str:
        lookups.append(model)
        return server.url

    monkeypatch.setattr(provision, "lookup_web_url", lookup)
    server.reject_with = [404]
    engine = modal.STT(proxy_token_id="wk-test", proxy_token_secret="ws-test", http_session=session)

    events = await _run(engine, [_frame() for _ in range(5)], conn_options=ONE_RETRY)

    assert _types(events) == [SOS, INTERIM, FINAL, EOS, USAGE]
    assert len(lookups) == 2
    assert server.handshakes == 2


async def test_not_found_handshake_on_a_pinned_url_is_not_retried(
    server: FakeStream, session: aiohttp.ClientSession
) -> None:
    server.reject_with = [404]

    with pytest.raises(APIStatusError) as err:
        await _run(_stt(server, session), [_frame()], conn_options=ONE_RETRY)

    assert err.value.status_code == 404
    assert server.handshakes == 1


class TriggeredVAD:
    """Silero's VADIterator as it behaves during a long run of speech: one "start", no "end"."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._started = False

    def __call__(self, window: Any) -> dict[str, int] | None:
        if self._started:
            return None
        self._started = True
        return {"start": 0}

    def reset_states(self) -> None:
        self._started = False


@dataclass
class FakeSocket:
    """The server's side of the WebSocket: scripted client messages in, sent messages out."""

    incoming: list[dict[str, Any]]
    sent: list[dict[str, Any]] = field(default_factory=list)

    async def receive(self) -> dict[str, Any]:
        return self.incoming.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self) -> None:
        pass


async def test_server_keeps_transcribing_past_the_utterance_length_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provision.load_server_app(provision.ServerConfig(model="parakeet-tdt-0.6b-v2"))
    server_module = sys.modules["parakeet_server"]
    silero = types.ModuleType("silero_vad")
    silero.VADIterator = TriggeredVAD  # type: ignore[attr-defined]
    silero.load_silero_vad = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "silero_vad", silero)
    monkeypatch.setattr(server_module, "_tensor", lambda window: window)
    monkeypatch.setattr(server_module, "MAX_UTTERANCE_S", 1.0)
    monkeypatch.setattr(server_module, "INTERIM_S", 10.0)

    # 1.5 s of uninterrupted speech: the cap cuts it at 1 s and the VAD never reports an end.
    pcm = np.full(24000, 1000, dtype=np.int16).tobytes()
    ws = FakeSocket(
        [
            {"type": "websocket.receive", "bytes": pcm},
            {"type": "websocket.receive", "text": '{"type": "close"}'},
        ]
    )
    session = server_module._StreamSession(lambda audio: f"{len(audio)} samples", ws)

    await session.run()

    kinds = [m["type"] for m in ws.sent]
    assert kinds == ["start_of_speech", "final", "end_of_speech"] * 2
    first_final, second_start, second_final = ws.sent[1], ws.sent[3], ws.sent[4]
    assert second_start["start"] == pytest.approx(first_final["end"])
    # The speech after the cut is transcribed, up to the last whole VAD window.
    assert second_final["end"] == pytest.approx(1.5, abs=512 / 16000)
