from __future__ import annotations

import asyncio
from collections import deque

import pytest
from livekit.agents.voice.avatar import AudioSegmentEnd

from livekit import rtc
from livekit.plugins.spatialreal import avatar
from livekit.plugins.spatialreal.resumable_queue_io import ResumableQueueAudioOutput


class _ProviderSpy:
    def __init__(self) -> None:
        self.generation = 1
        self.interrupt_calls = 0
        self.sent: list[tuple[str, bytes, bool]] = []

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        req_id = f"request-{self.generation}"
        self.sent.append((req_id, audio, end))
        return req_id

    async def interrupt(self) -> str:
        req_id = f"request-{self.generation}"
        self.interrupt_calls += 1
        self.generation += 1
        return req_id

    async def init(self) -> None:
        return None

    async def start(self) -> str:
        return "connection-test"

    async def close(self) -> None:
        return None


class _ReusedRequestIdProvider(_ProviderSpy):
    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        req_id = "request-1"
        self.sent.append((req_id, audio, end))
        return req_id


class _FailingInterruptProvider(_ProviderSpy):
    async def interrupt(self) -> str:
        self.interrupt_calls += 1
        raise RuntimeError("interrupt unavailable")


class _FailingResumeProvider(_ProviderSpy):
    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        raise RuntimeError("send unavailable")


class _FailFirstSendProvider(_ProviderSpy):
    def __init__(self) -> None:
        super().__init__()
        self.send_calls = 0

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        self.send_calls += 1
        if self.send_calls == 1:
            raise RuntimeError("transient send failure")
        return await super().send_audio(audio=audio, end=end)


class _BlockingInterruptProvider(_ProviderSpy):
    def __init__(self) -> None:
        super().__init__()
        self.interrupt_started = asyncio.Event()
        self.allow_interrupt = asyncio.Event()

    async def interrupt(self) -> str:
        self.interrupt_started.set()
        await self.allow_interrupt.wait()
        return await super().interrupt()


class _BlockingFailSendProvider(_ProviderSpy):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.allow_failure = asyncio.Event()
        self.fail_next_send = True

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        if self.fail_next_send:
            self.fail_next_send = False
            self.send_started.set()
            await self.allow_failure.wait()
            raise RuntimeError("send unavailable")
        return await super().send_audio(audio=audio, end=end)


class _BlockingSuccessfulSendProvider(_ProviderSpy):
    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()
        self.allow_send = asyncio.Event()
        self.send_calls = 0

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        self.send_calls += 1
        if self.send_calls == 1:
            self.send_started.set()
            await self.allow_send.wait()
        return await super().send_audio(audio=audio, end=end)


class _AudioBufferSpy:
    def __init__(self) -> None:
        self.playback_started = 0
        self.playback_finished: list[tuple[float, bool]] = []

    def notify_playback_started(self) -> None:
        self.playback_started += 1

    def notify_playback_finished(
        self,
        playback_position: float,
        interrupted: bool,
    ) -> None:
        self.playback_finished.append((playback_position, interrupted))


class _StreamingAudioBufferSpy(_AudioBufferSpy):
    def __init__(self, items: list[rtc.AudioFrame | AudioSegmentEnd]) -> None:
        super().__init__()
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item


class _AudioStreamCloseSpy:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _session() -> avatar.AvatarSession:
    session = avatar.AvatarSession(api_key="test", app_id="test", avatar_id="test")
    session._livekit_egress = object()
    session._resolved_sample_rate = 24000
    return session


def _buffered_frame(value: int, duration: float = 0.25) -> avatar._BufferedAudioFrame:
    sample_rate = 8
    samples = round(sample_rate * duration)
    return avatar._BufferedAudioFrame(
        audio=bytes([value, 0]) * samples,
        duration=duration,
        sample_rate=sample_rate,
        num_channels=1,
    )


def _rtc_frame(value: int = 9) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=bytes([value, 0]) * 4,
        sample_rate=8,
        num_channels=1,
        samples_per_channel=4,
    )


def _playing_segment() -> avatar._SegmentState:
    frames = deque(_buffered_frame(value) for value in (1, 2, 3, 4))
    return avatar._SegmentState(
        req_id="request-1",
        pushed_duration=1.0,
        playback_started=True,
        playback_started_at=100.0,
        playback_start_source="test",
        audio_frames=frames,
        buffered_duration=1.0,
        attempt_duration=1.0,
        input_finalized=True,
    )


def test_resumable_queue_advertises_pause_and_coalesces_lifecycle_events() -> None:
    async def run_test() -> None:
        output = ResumableQueueAudioOutput(sample_rate=24000, wait_playback_start=True)
        events: list[str] = []
        output.on("pause", lambda: events.append("pause"))
        output.on("resume", lambda: events.append("resume"))

        output.pause()
        output.pause()
        output.resume()
        output.resume()

        assert output.can_pause is True
        assert events == ["pause", "resume"]

    asyncio.run(run_test())


def test_false_interruption_replays_only_unplayed_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        monkeypatch.setattr(avatar.time, "time", lambda: 100.5)
        await session._handle_pause()

        assert provider.interrupt_calls == 1
        assert session._paused_segment is segment
        assert segment.playback_position_offset == pytest.approx(0.5)
        assert audio.playback_finished == []

        session._pause_requested = False
        await session._handle_resume()

        replay_audio = b"".join(audio_bytes for _, audio_bytes, end in provider.sent if not end)
        assert replay_audio == _buffered_frame(3).audio + _buffered_frame(4).audio
        assert provider.sent[-1] == ("request-2", b"", True)
        assert session._paused_segment is None
        assert list(session._pending_segment_ids) == ["request-2"]
        assert audio.playback_finished == []

    asyncio.run(run_test())


def test_confirmed_interruption_discards_paused_tail_without_second_provider_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        monkeypatch.setattr(avatar.time, "time", lambda: 100.5)
        await session._handle_pause()
        session._discard_requested = True
        session._pause_requested = False
        await session._handle_interrupt()

        assert provider.interrupt_calls == 1
        assert session._paused_segment is None
        assert session._segments == {}
        assert audio.playback_finished == [(pytest.approx(0.5), True)]

    asyncio.run(run_test())


def test_confirmed_interruption_before_provider_request_completes_framework_playout() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        audio = _AudioBufferSpy()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._pause_requested = True

        frame = rtc.AudioFrame(
            data=bytes([9, 0]) * 4,
            sample_rate=8,
            num_channels=1,
            samples_per_channel=4,
        )
        await session._send_audio_frame(frame)
        assert session._paused_segment is not None
        assert session._segments == {}

        session._discard_requested = True
        session._pause_requested = False
        await session._handle_interrupt()

        assert provider.interrupt_calls == 0
        assert session._paused_segment is None
        assert session._segments == {}
        assert audio.playback_finished == [(0.0, True)]

    asyncio.run(run_test())


def test_session_close_completes_segment_paused_before_provider_request() -> None:
    session = _session()
    audio = _AudioBufferSpy()
    segment = session._new_local_segment()
    session._audio_buffer = audio
    session._paused_segment = segment
    session._append_audio_frame(segment, _buffered_frame(1), sent_to_provider=False)

    session._complete_all_segments(interrupted=True, reason="session_close")

    assert session._paused_segment is None
    assert session._segments == {}
    assert audio.playback_finished == [(0.0, True)]


def test_partial_frame_resume_preserves_sample_alignment() -> None:
    frame = _buffered_frame(7, duration=0.5)
    segment = avatar._SegmentState(
        req_id="request-1",
        pushed_duration=0.5,
        audio_frames=deque([frame]),
        buffered_duration=0.5,
    )

    remaining = avatar.AvatarSession._remaining_audio_frames(segment, 0.25)

    assert len(remaining) == 1
    assert remaining[0].duration == pytest.approx(0.25)
    assert len(remaining[0].audio) % 2 == 0
    assert remaining[0].audio == bytes([7, 0]) * 2


def test_frames_arriving_while_paused_are_included_in_resumed_request() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        session._avatarkit_session = provider
        session._pause_requested = True
        session._paused_segment = avatar._SegmentState(req_id="paused")

        frame = rtc.AudioFrame(
            data=bytes([9, 0]) * 4,
            sample_rate=8,
            num_channels=1,
            samples_per_channel=4,
        )
        await session._send_audio_frame(frame)
        session._paused_segment.input_finalized = True

        session._pause_requested = False
        await session._handle_resume()

        assert provider.sent == [
            ("request-1", bytes([9, 0]) * 4, False),
            ("request-1", b"", True),
        ]

    asyncio.run(run_test())


def test_resume_buffer_is_bounded_to_configured_duration() -> None:
    session = _session()
    session._resume_buffer_max_seconds = 0.5
    segment = avatar._SegmentState(req_id="request-1")

    for value in (1, 2, 3, 4):
        session._append_audio_frame(
            segment,
            _buffered_frame(value),
            sent_to_provider=True,
        )

    assert segment.pushed_duration == pytest.approx(1.0)
    assert segment.buffered_duration == pytest.approx(0.5)
    assert segment.buffer_start_position == pytest.approx(0.5)
    assert [frame.audio for frame in segment.audio_frames] == [
        _buffered_frame(3).audio,
        _buffered_frame(4).audio,
    ]


def test_resume_before_pause_lock_avoids_unneeded_provider_interrupt() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        session._avatarkit_session = provider
        segment = _playing_segment()
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)

        await session._provider_io_lock.acquire()
        session._pause_requested = True
        pause_task = asyncio.create_task(session._handle_pause())
        await asyncio.sleep(0)
        session._pause_requested = False
        resume_task = asyncio.create_task(session._handle_resume())
        session._provider_io_lock.release()
        await asyncio.gather(pause_task, resume_task)

        assert provider.interrupt_calls == 0
        assert session._paused_segment is None

    asyncio.run(run_test())


def test_new_pause_supersedes_queued_resume() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._paused_segment = segment
        session._segments[segment.req_id] = segment

        await session._provider_io_lock.acquire()
        session._pause_requested = False
        resume_task = asyncio.create_task(session._handle_resume())
        await asyncio.sleep(0)
        session._pause_requested = True
        session._provider_io_lock.release()
        await resume_task

        assert provider.sent == []
        assert session._paused_segment is segment

    asyncio.run(run_test())


def test_pause_provider_failure_keeps_original_segment_authoritative() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _FailingInterruptProvider()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        await session._handle_pause()

        assert provider.interrupt_calls == 1
        assert session._pause_requested is False
        assert session._paused_segment is None
        assert session._segments[segment.req_id] is segment

    asyncio.run(run_test())


def test_resume_provider_failure_completes_speech_instead_of_stranding_it() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _FailingResumeProvider()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        segment.playback_position_offset = 0.5
        segment.playback_started = False
        segment.playback_started_at = None
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._paused_segment = segment
        session._segments[segment.req_id] = segment

        await session._handle_resume()

        assert provider.interrupt_calls == 1
        assert session._paused_segment is None
        assert session._segments == {}
        assert audio.playback_finished == [(pytest.approx(0.5), True)]

    asyncio.run(run_test())


def test_pause_before_observed_playback_does_not_skip_unheard_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = _playing_segment()
    segment.playback_started = False
    segment.playback_started_at = None
    segment.first_frame_at = 100.0
    monkeypatch.setattr(avatar.time, "time", lambda: 105.0)

    assert avatar.AvatarSession._estimate_interrupted_playback_position(segment) == 0.0


def test_reused_provider_request_id_is_reactivated_on_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ReusedRequestIdProvider()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        monkeypatch.setattr(avatar.time, "time", lambda: 100.5)
        await session._handle_pause()
        assert "request-1" in session._recently_completed_req_ids

        session._pause_requested = False
        await session._handle_resume()

        assert "request-1" not in session._recently_completed_req_ids
        assert session._segments["request-1"] is segment
        assert segment.provider_events_trusted is False

    asyncio.run(run_test())


def test_stale_completion_for_reused_request_id_cannot_finish_resumed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ReusedRequestIdProvider()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._segments[segment.req_id] = segment
        session._pending_segment_ids.append(segment.req_id)
        session._pause_requested = True

        monkeypatch.setattr(avatar.time, "time", lambda: 100.5)
        await session._handle_pause()
        session._pause_requested = False
        await session._handle_resume()
        monkeypatch.setattr(session, "_extract_req_id_from_transport_frame", lambda _: "request-1")

        session._on_transport_frame(b"stale", True)

        assert session._segments["request-1"] is segment
        assert audio.playback_finished == []

    asyncio.run(run_test())


def test_transient_provider_send_failure_does_not_kill_audio_forwarder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _FailFirstSendProvider()
        replacement = _ProviderSpy()
        monkeypatch.setattr(avatar, "new_avatar_session", lambda **_: replacement)
        audio = _StreamingAudioBufferSpy(
            [
                _rtc_frame(1),
                _rtc_frame(2),
                AudioSegmentEnd(),
                _rtc_frame(3),
            ]
        )
        session._avatarkit_session = provider
        session._audio_buffer = audio

        await session._run_main_task()

        assert provider.send_calls == 1
        assert provider.interrupt_calls == 1
        assert session._active_req_id == "request-1"
        assert audio.playback_finished == [(0.0, True)]
        assert replacement.sent == [("request-1", bytes([3, 0]) * 4, False)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_clear_after_provider_recovery_releases_next_response() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        session._avatarkit_session = provider
        session._drop_frames_until_segment_end = True

        session._on_clear_buffer()
        await session._send_audio_frame(_rtc_frame(4))

        assert session._drop_frames_until_segment_end is False
        assert provider.sent == [("request-1", bytes([4, 0]) * 4, False)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_clear_during_provider_recovery_does_not_discard_next_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _BlockingInterruptProvider()
        replacement = _ProviderSpy()
        monkeypatch.setattr(avatar, "new_avatar_session", lambda **_: replacement)
        audio = _AudioBufferSpy()
        session._avatarkit_session = provider
        session._audio_buffer = audio

        recovery = asyncio.create_task(
            session._recover_provider_forwarding_error(
                item=_rtc_frame(1),
                error=RuntimeError("send unavailable"),
            )
        )
        await provider.interrupt_started.wait()

        session._on_clear_buffer()
        provider.allow_interrupt.set()
        await recovery
        await session._send_audio_frame(_rtc_frame(2))

        assert session._provider_recovery_in_flight is False
        assert session._clear_during_provider_recovery is False
        assert session._drop_frames_until_segment_end is False
        assert provider.interrupt_calls == 1
        assert replacement.sent == [("request-1", bytes([2, 0]) * 4, False)]
        assert audio.playback_finished == [(0.0, True)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_clear_during_failing_provider_send_does_not_discard_next_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _BlockingFailSendProvider()
        replacement = _ProviderSpy()
        monkeypatch.setattr(avatar, "new_avatar_session", lambda **_: replacement)
        audio = _AudioBufferSpy()
        session._avatarkit_session = provider
        session._audio_buffer = audio

        async def send_and_recover() -> None:
            frame = _rtc_frame(1)
            try:
                await session._send_audio_frame(frame)
            except Exception as error:
                await session._recover_provider_forwarding_error(item=frame, error=error)

        failed_send = asyncio.create_task(send_and_recover())
        await provider.send_started.wait()

        session._on_clear_buffer()
        provider.allow_failure.set()
        await failed_send
        await session._send_audio_frame(_rtc_frame(2))

        assert session._provider_send_in_flight is False
        assert session._clear_during_provider_send is False
        assert session._drop_frames_until_segment_end is False
        assert provider.interrupt_calls == 1
        assert replacement.sent == [("request-1", bytes([2, 0]) * 4, False)]
        assert audio.playback_finished == [(0.0, True)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_queued_frame_from_before_clear_is_not_forwarded() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ProviderSpy()
        session._avatarkit_session = provider

        await session._provider_io_lock.acquire()
        stale_send = asyncio.create_task(session._send_audio_frame(_rtc_frame(1)))
        await asyncio.sleep(0)
        session._on_clear_buffer()
        session._provider_io_lock.release()

        await stale_send
        if session._background_tasks:
            await asyncio.gather(*list(session._background_tasks))
        await session._send_audio_frame(_rtc_frame(2))

        assert provider.sent == [("request-1", bytes([2, 0]) * 4, False)]

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_clear_during_resume_stops_before_remaining_tail_is_sent() -> None:
    async def run_test() -> None:
        session = _session()
        provider = _BlockingSuccessfulSendProvider()
        audio = _AudioBufferSpy()
        segment = _playing_segment()
        segment.playback_position_offset = 0.0
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._paused_segment = segment

        resume = asyncio.create_task(session._handle_resume())
        await provider.send_started.wait()
        session._on_clear_buffer()
        provider.allow_send.set()
        await resume

        assert provider.send_calls == 1
        assert provider.interrupt_calls == 1
        assert session._paused_segment is None
        assert session._segments == {}
        assert audio.playback_finished == [(1.0, True)]

    asyncio.run(run_test())


def test_session_close_drains_lifecycle_tasks_before_final_cleanup() -> None:
    async def run_test() -> None:
        session = _session()

        async def resurrect_paused_state_on_cancel() -> None:
            try:
                await asyncio.sleep(60)
            finally:
                session._paused_segment = session._new_local_segment()

        task = session._spawn_background_task(
            resurrect_paused_state_on_cancel(),
            name="test_resurrect_paused_state",
        )

        await session.aclose()

        assert task.done()
        assert session._paused_segment is None
        assert session._segments == {}

    asyncio.run(run_test())


def test_closing_session_rejects_new_lifecycle_tasks() -> None:
    async def run_test() -> None:
        session = _session()
        started = False

        async def late_lifecycle_work() -> None:
            nonlocal started
            started = True

        session._closing = True
        task = session._spawn_background_task(
            late_lifecycle_work(),
            name="test_late_lifecycle_work",
        )
        await asyncio.sleep(0)

        assert task is None
        assert started is False
        assert session._background_tasks == set()

    asyncio.run(run_test())


def test_session_close_awaits_avatar_audio_stream_close() -> None:
    async def run_test() -> None:
        session = _session()
        stream = _AudioStreamCloseSpy()
        session._avatar_audio_stream = stream

        await session.aclose()

        assert stream.closed is True
        assert session._avatar_audio_stream is None

    asyncio.run(run_test())


def test_stale_completion_for_reused_request_id_cannot_finish_normal_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_test() -> None:
        session = _session()
        provider = _ReusedRequestIdProvider()
        audio = _AudioBufferSpy()
        session._avatarkit_session = provider
        session._audio_buffer = audio
        session._recently_completed_req_ids.append("request-1")

        await session._send_audio_frame(_rtc_frame())
        segment = session._segments["request-1"]
        monkeypatch.setattr(
            session,
            "_extract_req_id_from_transport_frame",
            lambda _: "request-1",
        )

        session._on_transport_frame(b"stale", True)

        assert segment.provider_events_trusted is False
        assert session._segments["request-1"] is segment
        assert audio.playback_finished == []

        session._cancel_active_segment_idle_end()

    asyncio.run(run_test())


def test_segment_rollover_preserves_retained_audio_state() -> None:
    target = avatar._SegmentState(req_id="request-2")
    source = _playing_segment()
    source.buffer_start_position = 0.25
    source.playback_position_offset = 0.5

    avatar.AvatarSession._merge_segment_state(target, source)

    assert target.pushed_duration == pytest.approx(1.0)
    assert target.buffered_duration == pytest.approx(1.0)
    assert target.buffer_start_position == pytest.approx(0.25)
    assert target.playback_position_offset == pytest.approx(0.5)
    assert list(target.audio_frames) == list(source.audio_frames)
