from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit import rtc
from livekit.plugins.spatialreal import avatar


class _Provider:
    def __init__(
        self,
        *,
        on_error=None,
        on_close=None,
        start_error: Exception | None = None,
        start_gate: asyncio.Event | None = None,
    ) -> None:
        self.on_error = on_error
        self.on_close = on_close
        self.start_error = start_error
        self.start_gate = start_gate
        self.init_calls = 0
        self.start_calls = 0
        self.close_calls = 0
        self.interrupt_calls = 0
        self.send_calls = 0

    async def init(self) -> None:
        self.init_calls += 1

    async def start(self) -> str:
        self.start_calls += 1
        if self.start_gate is not None:
            await self.start_gate.wait()
        if self.start_error is not None:
            raise self.start_error
        return f"connection-{self.start_calls}"

    async def close(self) -> None:
        self.close_calls += 1
        if self.on_close is not None:
            self.on_close()

    async def interrupt(self) -> str:
        self.interrupt_calls += 1
        raise ValueError("interrupt: websocket connection is not established")

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        self.send_calls += 1
        return f"request-{self.send_calls}"


class _AudioBuffer:
    def __init__(self) -> None:
        self.started = 0
        self.finished: list[tuple[float, bool]] = []

    def notify_playback_started(self) -> None:
        self.started += 1

    def notify_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
    ) -> None:
        self.finished.append((playback_position, interrupted))


def _session() -> avatar.AvatarSession:
    session = avatar.AvatarSession(api_key="test", app_id="test", avatar_id="test")
    session._livekit_egress = SimpleNamespace()
    session._resolved_sample_rate = 24000
    session._audio_buffer = _AudioBuffer()
    return session


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame.create(
        sample_rate=24000,
        num_channels=1,
        samples_per_channel=240,
    )


@pytest.mark.asyncio
async def test_dead_websocket_reconnects_with_fresh_provider_and_confirms_playback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    original = _Provider()
    session._avatarkit_session = original
    session._provider_generation = 1
    session._provider_session_generation = 1
    states: list[str] = []
    session.on(
        "provider_connection_state_changed",
        lambda event: states.append(event.state),
    )

    replacements: list[_Provider] = []

    def factory(**kwargs):
        provider = _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
        )
        replacements.append(provider)
        return provider

    monkeypatch.setattr(avatar, "new_avatar_session", factory)

    await session._recover_provider_forwarding_error(
        item=_frame(),
        error=ValueError("WebSocket connection is not established"),
    )

    assert original.close_calls == 1
    assert len(replacements) == 1
    assert replacements[0].init_calls == 1
    assert replacements[0].start_calls == 1
    assert session._avatarkit_session is replacements[0]
    assert states == ["recovering", "connected"]

    await session._send_audio_frame(_frame())
    segment = next(iter(session._segments.values()))
    session._mark_playback_started(segment, source="livekit_avatar_audio_track")

    assert states == ["recovering", "connected", "recovered"]
    assert session._provider_terminal_failure is False


@pytest.mark.asyncio
async def test_provider_recovery_retries_with_bounded_fresh_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    original = _Provider()
    session._avatarkit_session = original
    session._provider_generation = 1
    session._provider_session_generation = 1
    created: list[_Provider] = []

    def factory(**kwargs):
        provider = _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
            start_error=(RuntimeError("first reconnect failed") if not created else None),
        )
        created.append(provider)
        return provider

    monkeypatch.setattr(avatar, "new_avatar_session", factory)
    monkeypatch.setattr(avatar, "PROVIDER_RECONNECT_DELAYS_SECONDS", (0.0, 0.0, 0.0))

    assert await session._ensure_provider_connection(reason="provider_closed") is True
    assert len(created) == 2
    assert created[0].start_calls == 1
    assert created[1].start_calls == 1
    assert session._avatarkit_session is created[1]


@pytest.mark.asyncio
async def test_provider_recovery_times_out_hung_start_and_uses_fresh_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    original = _Provider()
    session._avatarkit_session = original
    session._provider_generation = 1
    session._provider_session_generation = 1
    created: list[_Provider] = []

    def factory(**kwargs):
        provider = _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
            start_gate=(asyncio.Event() if not created else None),
        )
        created.append(provider)
        return provider

    monkeypatch.setattr(avatar, "new_avatar_session", factory)
    monkeypatch.setattr(avatar, "PROVIDER_RECONNECT_DELAYS_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(avatar, "PROVIDER_CONNECT_TIMEOUT_SECONDS", 0.01)

    assert await session._ensure_provider_connection(reason="provider_closed") is True
    assert len(created) == 2
    assert created[0].close_calls == 1
    assert created[1].start_calls == 1
    assert session._avatarkit_session is created[1]


@pytest.mark.asyncio
async def test_provider_recovery_exhaustion_emits_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    session._avatarkit_session = _Provider()
    session._provider_generation = 1
    session._provider_session_generation = 1
    states: list[str] = []
    session.on(
        "provider_connection_state_changed",
        lambda event: states.append(event.state),
    )

    def factory(**kwargs):
        return _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
            start_error=RuntimeError("provider unavailable"),
        )

    monkeypatch.setattr(avatar, "new_avatar_session", factory)
    monkeypatch.setattr(avatar, "PROVIDER_RECONNECT_DELAYS_SECONDS", (0.0, 0.0, 0.0))

    assert await session._ensure_provider_connection(reason="provider_closed") is False
    assert states == ["recovering", "failed"]
    assert session._provider_terminal_failure is True


@pytest.mark.asyncio
async def test_unexpected_provider_close_schedules_single_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session()
    session._avatarkit_session = _Provider()
    session._provider_generation = 1
    session._provider_session_generation = 1
    created: list[_Provider] = []

    def factory(**kwargs):
        provider = _Provider(
            on_error=kwargs.get("on_error"),
            on_close=kwargs.get("on_close"),
        )
        created.append(provider)
        return provider

    monkeypatch.setattr(avatar, "new_avatar_session", factory)
    monkeypatch.setattr(avatar, "PROVIDER_RECONNECT_DELAYS_SECONDS", (0.0,))

    session._on_provider_close(1)
    first_task = session._provider_reconnect_task
    session._on_provider_close(1)

    assert first_task is not None
    assert session._provider_reconnect_task is first_task
    assert await first_task is True
    assert len(created) == 1
