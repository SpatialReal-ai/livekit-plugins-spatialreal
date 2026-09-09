"""Offline lifecycle tests for the SpatialReal avatar plugin.

Runs without network or a LiveKit room: the spatialreal SDK session and audio buffer are
stubbed, and the segment state machine is driven directly.

Usage: .venv/bin/python tests/test_lifecycle.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit.agents.voice import io as voice_io  # noqa: E402
from livekit.agents.voice.avatar import QueueAudioOutput  # noqa: E402
from spatialreal import PlaybackSignal  # noqa: E402

from livekit import rtc  # noqa: E402
from livekit.plugins.spatialreal.avatar import AvatarSession  # noqa: E402

pytestmark = pytest.mark.asyncio


class FakeAvatarkitSession:
    def __init__(self, req_id: str = "req-1") -> None:
        self.req_id = req_id
        self.interrupt_calls = 0
        self.sent: list[tuple[int, bool]] = []

    async def send_audio(self, audio: bytes, end: bool = False) -> str:
        self.sent.append((len(audio), end))
        return self.req_id

    async def interrupt(self) -> str:
        self.interrupt_calls += 1
        return self.req_id

    async def close(self) -> None:
        pass


class FakeBuffer:
    """Records the notify_* calls the plugin makes toward the framework."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def notify_playback_started(self) -> None:
        self.events.append("started")

    def notify_playback_finished(self, *, playback_position: float, interrupted: bool) -> None:
        self.events.append(("finished", interrupted))


def make_session() -> tuple[AvatarSession, FakeAvatarkitSession, FakeBuffer]:
    session = AvatarSession(api_key="k", app_id="a", avatar_id="av")
    fake = FakeAvatarkitSession()
    buffer = FakeBuffer()
    session._avatarkit_session = fake  # type: ignore[assignment]
    session._audio_buffer = buffer  # type: ignore[assignment]
    return session, fake, buffer


def make_frame(samples: int = 240, sample_rate: int = 24000) -> rtc.AudioFrame:
    return rtc.AudioFrame.create(sample_rate=sample_rate, num_channels=1, samples_per_channel=samples)


def provider_end_signal(req_id: str) -> PlaybackSignal:
    return PlaybackSignal(req_id=req_id, end=True)


async def test_early_provider_completion_is_preserved() -> None:
    session, fake, buffer = make_session()

    await session._send_audio_frame(make_frame())
    # provider completion arrives BEFORE the local AudioSegmentEnd
    session._on_playback_signal(provider_end_signal(fake.req_id))
    assert buffer.events == ["started"], buffer.events

    assert await session._finalize_active_segment(source="segment_end")
    assert buffer.events == ["started", ("finished", False)], buffer.events
    assert not session._segments and not session._pending_segment_ids
    assert (0, True) in fake.sent  # end=true was sent to the provider
    print("PASS early provider completion preserved (provider_end_deferred)")


async def test_duplicate_interrupts_coalesced() -> None:
    session, fake, buffer = make_session()

    await session._send_audio_frame(make_frame())
    await session._handle_interrupt()
    await session._handle_interrupt()  # duplicate clear_buffer

    assert fake.interrupt_calls == 1, fake.interrupt_calls
    # started is synthesized before finished even though playback was never observed
    assert buffer.events == ["started", ("finished", True)], buffer.events
    print("PASS duplicate interrupts coalesced, started precedes finished")


async def test_started_precedes_finished_without_signals() -> None:
    session, fake, buffer = make_session()

    await session._send_audio_frame(make_frame())
    assert await session._finalize_active_segment(source="segment_end")
    assert buffer.events == [], buffer.events  # waiting for completion

    assert session._complete_segment(req_id=fake.req_id, interrupted=False, reason="timeout_no_playback_signal")
    assert buffer.events == ["started", ("finished", False)], buffer.events
    print("PASS no-signal fallback still emits started before finished")


async def test_remote_track_marks_playback_started() -> None:
    session, fake, buffer = make_session()
    playback_events = []
    session.on("playback_started", playback_events.append)

    await session._send_audio_frame(make_frame())

    silent = make_frame()
    session._on_avatar_audio_frame(silent)
    assert buffer.events == [], buffer.events

    noise_floor = make_frame()
    noise_floor.data[0] = 40  # below AVATAR_AUDIO_ACTIVITY_THRESHOLD
    session._on_avatar_audio_frame(noise_floor)
    assert buffer.events == [], buffer.events

    speech = make_frame()
    speech.data[0] = 1200
    session._on_avatar_audio_frame(speech)
    assert buffer.events == ["started"], buffer.events
    assert len(playback_events) == 1
    assert playback_events[0].request_id == fake.req_id
    assert playback_events[0].source == "livekit_avatar_audio_track"
    assert playback_events[0].observed_at > 0

    segment = session._segments[fake.req_id]
    assert segment.playback_start_source == "livekit_avatar_audio_track"

    session._complete_segment(req_id=fake.req_id, interrupted=False, reason="test")
    print("PASS playback start detected from remote track with energy threshold")


async def test_active_speaker_secondary_signal() -> None:
    session, fake, buffer = make_session()
    playback_events = []
    session.on("playback_started", playback_events.append)
    session._represented_participant_identity = "agent-1"

    await session._send_audio_frame(make_frame())

    stranger = SimpleNamespace(identity="someone-else", attributes={})
    session._on_active_speakers_changed([stranger])
    assert buffer.events == [], buffer.events

    session._avatar_is_speaking = False
    proxy_avatar = SimpleNamespace(identity="egress-worker", attributes={"lk.publish_on_behalf": "agent-1"})
    session._on_active_speakers_changed([proxy_avatar])
    assert buffer.events == ["started"], buffer.events
    assert len(playback_events) == 1
    assert playback_events[0].source == "livekit_active_speaker"

    segment = session._segments[fake.req_id]
    assert segment.playback_start_source == "livekit_active_speaker"

    session._complete_segment(req_id=fake.req_id, interrupted=False, reason="test")
    print("PASS active-speaker signal matched via lk.publish_on_behalf")


class _Leaf(voice_io.AudioOutput):
    def __init__(self) -> None:
        super().__init__(
            label="leaf",
            next_in_chain=None,
            sample_rate=24000,
            capabilities=voice_io.AudioOutputCapabilities(pause=False),
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


class _Wrapper(voice_io.AudioOutput):
    def __init__(self, next_in_chain: voice_io.AudioOutput) -> None:
        super().__init__(
            label="wrapper",
            next_in_chain=next_in_chain,
            sample_rate=24000,
            capabilities=voice_io.AudioOutputCapabilities(pause=False),
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


async def test_audio_tail_attach_and_restore() -> None:
    session, _, _ = make_session()

    leaf = _Leaf()
    wrapper = _Wrapper(leaf)  # auto-wraps leaf behind an _AudioSinkProxy
    output = voice_io.AgentOutput(
        video_changed=lambda: None,
        audio_changed=lambda: None,
        transcription_changed=lambda: None,
    )
    output.audio = wrapper

    queue_buffer = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)
    session._audio_buffer = queue_buffer
    session._agent_session = SimpleNamespace(output=output)  # type: ignore[assignment]
    session._original_audio_output = output.audio
    session._original_audio_tail = AvatarSession._resolve_audio_tail(output.audio)
    assert session._original_audio_tail is leaf

    output.replace_audio_tail(queue_buffer)
    session._audio_output_attached = True
    assert output.audio is wrapper  # wrappers stay in place
    assert AvatarSession._resolve_audio_tail(output.audio) is queue_buffer

    session._detach_audio_output()
    assert output.audio is wrapper
    assert AvatarSession._resolve_audio_tail(output.audio) is leaf
    print("PASS tail swap keeps wrappers and restores the original tail on close")


async def test_audio_tail_restore_without_wrappers() -> None:
    session, _, _ = make_session()

    leaf = _Leaf()
    output = voice_io.AgentOutput(
        video_changed=lambda: None,
        audio_changed=lambda: None,
        transcription_changed=lambda: None,
    )
    output.audio = leaf

    queue_buffer = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)
    session._audio_buffer = queue_buffer
    session._agent_session = SimpleNamespace(output=output)  # type: ignore[assignment]
    session._original_audio_output = output.audio
    session._original_audio_tail = AvatarSession._resolve_audio_tail(output.audio)

    output.replace_audio_tail(queue_buffer)
    session._audio_output_attached = True

    session._detach_audio_output()
    assert output.audio is leaf
    print("PASS whole-chain fallback restore works when no wrappers exist")


async def test_duplicate_provider_end_ignored() -> None:
    session, fake, buffer = make_session()

    await session._send_audio_frame(make_frame())
    assert await session._finalize_active_segment(source="segment_end")
    session._on_playback_signal(provider_end_signal(fake.req_id))
    assert buffer.events == ["started", ("finished", False)], buffer.events

    # egress ALR retransmission can re-deliver end=true for the same req_id
    session._on_playback_signal(provider_end_signal(fake.req_id))
    session._on_playback_signal(provider_end_signal(fake.req_id))
    assert buffer.events == ["started", ("finished", False)], buffer.events
    assert not session._early_provider_started_ids and not session._early_provider_completed_ids
    print("PASS duplicate provider end=true events are ignored")


async def main() -> None:
    await test_early_provider_completion_is_preserved()
    await test_duplicate_provider_end_ignored()
    await test_duplicate_interrupts_coalesced()
    await test_started_precedes_finished_without_signals()
    await test_remote_track_marks_playback_started()
    await test_active_speaker_secondary_signal()
    await test_audio_tail_attach_and_restore()
    await test_audio_tail_restore_without_wrappers()
    await test_livekit_egress_token_ttl_outlives_session()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())


async def test_livekit_egress_token_ttl_outlives_session() -> None:
    """The egress LiveKit room-join token must outlive a long conversation.

    Regression for the 1h-TTL bug: after the token expired mid-session, an
    egress rejoin (triggered by a provider-WS reconnect) hit SFU 401 and the
    avatar dropped out permanently. The token TTL is decoupled from the
    SpatialReal session TTL and defaults to 24h.
    """
    import base64
    import json
    from datetime import timedelta

    from livekit.plugins.spatialreal.avatar import DEFAULT_LIVEKIT_TOKEN_TTL

    assert DEFAULT_LIVEKIT_TOKEN_TTL >= timedelta(hours=12)

    session = AvatarSession(api_key="k", app_id="a", avatar_id="av")
    assert session._livekit_token_ttl == DEFAULT_LIVEKIT_TOKEN_TTL

    # mint a token the same way start() does and decode its exp/nbf claims
    from livekit import api

    jwt = (
        api.AccessToken(api_key="devkey", api_secret="secret" * 6)
        .with_identity("spatialreal-avatar")
        .with_ttl(session._livekit_token_ttl)
        .to_jwt()
    )
    payload = jwt.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    claims = json.loads(base64.urlsafe_b64decode(payload))
    assert claims["exp"] - claims["nbf"] >= 12 * 3600, claims

    override = AvatarSession(api_key="k", app_id="a", avatar_id="av", livekit_token_ttl_seconds=7200)
    assert override._livekit_token_ttl == timedelta(seconds=7200)
    print("PASS egress LiveKit token TTL is long (>=12h default) and configurable")
