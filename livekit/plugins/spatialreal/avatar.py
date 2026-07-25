# Copyright 2026 SpatialReal.
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
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from avatarkit import AvatarSession as AvatarkitSession
from avatarkit import LiveKitEgressConfig, new_avatar_session
from avatarkit.proto.generated import message_pb2 as _message_pb2
from livekit.agents import (
    NOT_GIVEN,
    AgentSession,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    QueueAudioOutput,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from livekit import api, rtc

from .log import logger

message_pb2: Any = _message_pb2

__all__ = ["AvatarSession", "SpatialRealException"]

DEFAULT_AVATAR_PARTICIPANT_IDENTITY = "spatialreal-avatar"
DEFAULT_AVATAR_PARTICIPANT_NAME = "spatialreal-avatar"
DEFAULT_SAMPLE_RATE = 24000
MIN_COMPLETION_TIMEOUT_SECONDS = 3.0
COMPLETION_TIMEOUT_BUFFER_SECONDS = 2.0
ACTIVE_SEGMENT_IDLE_END_SECONDS = 1.0
DEFAULT_SESSION_TTL = timedelta(hours=1)
LIVEKIT_AVATAR_PUBLISH_SOURCES = ["camera", "microphone"]

DEFAULT_CONSOLE_ENDPOINT = "https://console.us-west.spatialwalk.cloud/v1/console"
DEFAULT_INGRESS_ENDPOINT = "wss://api.us-west.spatialwalk.cloud/v2/driveningress"


class SpatialRealException(Exception):
    """Exception raised for SpatialReal avatar integration errors."""


@dataclass
class _SegmentState:
    req_id: str
    pushed_duration: float = 0.0
    first_frame_at: float | None = None
    completion_timeout_task: asyncio.Task[None] | None = None


class AvatarSession(BaseAvatarSession):
    """A SpatialReal avatar session.

    The LiveKit agent produces speech as usual. This plugin forwards the TTS audio
    to SpatialReal, and the SpatialReal avatar worker joins the LiveKit room to publish
    synchronized avatar audio/video.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        app_id: NotGivenOr[str] = NOT_GIVEN,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        console_endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        ingress_endpoint_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout_seconds: int = 0,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        super().__init__()
        resolved_api_key = api_key if utils.is_given(api_key) else os.getenv("SPATIALREAL_API_KEY")
        if not resolved_api_key:
            raise SpatialRealException(
                "api_key must be set either by passing it to AvatarSession or "
                "by setting the SPATIALREAL_API_KEY environment variable"
            )

        resolved_app_id = app_id if utils.is_given(app_id) else os.getenv("SPATIALREAL_APP_ID")
        if not resolved_app_id:
            raise SpatialRealException(
                "app_id must be set either by passing it to AvatarSession or "
                "by setting the SPATIALREAL_APP_ID environment variable"
            )

        resolved_avatar_id = avatar_id if utils.is_given(avatar_id) else os.getenv("SPATIALREAL_AVATAR_ID")
        if not resolved_avatar_id:
            raise SpatialRealException(
                "avatar_id must be set either by passing it to AvatarSession or "
                "by setting the SPATIALREAL_AVATAR_ID environment variable"
            )

        if idle_timeout_seconds < 0:
            raise SpatialRealException("idle_timeout_seconds must be greater than or equal to 0")
        if utils.is_given(sample_rate) and sample_rate <= 0:
            raise SpatialRealException("sample_rate must be greater than 0")

        self._api_key = str(resolved_api_key)
        self._app_id = str(resolved_app_id)
        self._avatar_id = str(resolved_avatar_id)
        self._console_endpoint_url = str(
            console_endpoint_url
            if utils.is_given(console_endpoint_url)
            else os.getenv("SPATIALREAL_CONSOLE_ENDPOINT") or DEFAULT_CONSOLE_ENDPOINT
        )
        self._ingress_endpoint_url = str(
            ingress_endpoint_url
            if utils.is_given(ingress_endpoint_url)
            else os.getenv("SPATIALREAL_INGRESS_ENDPOINT") or DEFAULT_INGRESS_ENDPOINT
        )
        self._avatar_participant_identity = str(
            avatar_participant_identity
            if utils.is_given(avatar_participant_identity)
            else DEFAULT_AVATAR_PARTICIPANT_IDENTITY
        )
        self._avatar_participant_name = str(
            avatar_participant_name if utils.is_given(avatar_participant_name) else DEFAULT_AVATAR_PARTICIPANT_NAME
        )
        self._idle_timeout_seconds = idle_timeout_seconds
        self._sample_rate = sample_rate if utils.is_given(sample_rate) else None

        self._avatarkit_session: AvatarkitSession | None = None
        self._agent_session: AgentSession | None = None
        self._audio_buffer: QueueAudioOutput | None = None
        self._original_audio_output: Any | None = None
        self._audio_output_attached = False
        self._main_task: asyncio.Task | None = None
        self._initialized = False
        self._segments: dict[str, _SegmentState] = {}
        self._pending_segment_ids: deque[str] = deque()
        self._active_req_id: str | None = None
        self._active_segment_idle_end_task: asyncio.Task[None] | None = None
        self._segment_finalize_lock = asyncio.Lock()

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "spatialreal"

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Start the SpatialReal avatar session and attach it to the agent output."""
        if self._initialized:
            logger.warning("Avatar session already initialized")
            return

        await super().start(agent_session, room)

        resolved_livekit_url = livekit_url if utils.is_given(livekit_url) else os.getenv("LIVEKIT_URL")
        resolved_livekit_api_key = livekit_api_key if utils.is_given(livekit_api_key) else os.getenv("LIVEKIT_API_KEY")
        resolved_livekit_api_secret = (
            livekit_api_secret if utils.is_given(livekit_api_secret) else os.getenv("LIVEKIT_API_SECRET")
        )
        if not resolved_livekit_url or not resolved_livekit_api_key or not resolved_livekit_api_secret:
            raise SpatialRealException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set by arguments or environment variables"
            )

        room_name = room.name
        local_participant_identity = self._resolve_local_participant_identity(room)
        logger.debug(
            "starting SpatialReal avatar session",
            extra={"room": room_name},
        )

        egress_attributes = {ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity}
        livekit_token = (
            api.AccessToken(
                api_key=str(resolved_livekit_api_key),
                api_secret=str(resolved_livekit_api_secret),
            )
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_ttl(DEFAULT_SESSION_TTL)
            .with_attributes(egress_attributes)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_subscribe=False,
                    can_publish_data=False,
                    can_publish_sources=LIVEKIT_AVATAR_PUBLISH_SOURCES,
                )
            )
            .to_jwt()
        )

        livekit_egress = LiveKitEgressConfig(
            url=str(resolved_livekit_url),
            api_token=livekit_token,
            room_name=room_name,
            publisher_id=self._avatar_participant_identity,
            extra_attributes=egress_attributes,
            idle_timeout=self._idle_timeout_seconds,
        )

        resolved_sample_rate = self._sample_rate
        if resolved_sample_rate is None:
            resolved_sample_rate = agent_session.tts.sample_rate if agent_session.tts else DEFAULT_SAMPLE_RATE
        if resolved_sample_rate <= 0:
            raise SpatialRealException("sample_rate must be greater than 0")

        self._agent_session = agent_session
        self._original_audio_output = agent_session.output.audio

        try:
            self._avatarkit_session = new_avatar_session(
                api_key=self._api_key,
                app_id=self._app_id,
                avatar_id=self._avatar_id,
                console_endpoint_url=self._console_endpoint_url,
                ingress_endpoint_url=self._ingress_endpoint_url,
                expire_at=datetime.now(timezone.utc) + DEFAULT_SESSION_TTL,
                livekit_egress=livekit_egress,
                sample_rate=resolved_sample_rate,
                transport_frames=self._on_transport_frame,
            )
            await self._avatarkit_session.init()
            await self._avatarkit_session.start()

            self._audio_buffer = QueueAudioOutput(sample_rate=resolved_sample_rate)
            await self._audio_buffer.start()
            self._audio_buffer.on("clear_buffer", self._on_clear_buffer)  # type: ignore[arg-type]

            agent_session.output.audio = self._audio_buffer
            self._audio_output_attached = True
            self._main_task = asyncio.create_task(
                self._run_main_task(),
                name="spatialreal_avatar_audio_forwarder",
            )
            self._initialized = True

            # Interruption is owned entirely by the framework. The single
            # authoritative interrupt signal is ``clear_buffer`` on the
            # QueueAudioOutput (registered above): the framework emits it only
            # after its turn detector / TurnHandlingOptions.interruption gates
            # decide the user genuinely interrupted. We deliberately do NOT
            # interrupt on ``user_state_changed`` -> "speaking" — that fires on
            # raw VAD, upstream of every framework gate, so coughs, throat
            # clears and single-word fragments would truncate the avatar even
            # when turn handling would not treat them as a real interruption.
            # This matches the framework's reference receiver
            # (livekit/agents/voice/avatar/_runner.py:_on_clear_buffer); none of
            # the official avatar plugins listen to user_state_changed.

            @agent_session.on("close")
            def _on_session_close(_: Any) -> None:
                asyncio.create_task(self.aclose())

        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception as e:
            logger.debug("SpatialReal avatar session startup failed", exc_info=True)
            await self.aclose()
            raise SpatialRealException(
                self._build_start_error_message(
                    error=e,
                    room_name=room_name,
                    sample_rate=resolved_sample_rate,
                )
            ) from None

    def _build_start_error_message(
        self,
        *,
        error: Exception,
        room_name: str,
        sample_rate: int,
    ) -> str:
        return (
            "Failed to start SpatialReal avatar session. "
            "Check SpatialReal credentials, LiveKit room auth/token configuration, "
            "endpoint URLs, and outbound network access. "
            f"room={room_name}, avatar_id={self._avatar_id}, "
            f"ingress_endpoint_url={self._ingress_endpoint_url}, "
            f"sample_rate={sample_rate}. Reason: {self._format_error_reason(error)}"
        )

    @staticmethod
    def _resolve_local_participant_identity(room: rtc.Room) -> str:
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            return job_ctx.local_participant_identity
        if room.isconnected():
            return room.local_participant.identity
        raise SpatialRealException("failed to get local participant identity")

    @staticmethod
    def _format_error_reason(error: BaseException) -> str:
        root_error = error
        seen_errors: set[int] = set()

        while id(root_error) not in seen_errors:
            seen_errors.add(id(root_error))
            next_error = root_error.__cause__ or (None if root_error.__suppress_context__ else root_error.__context__)
            if next_error is None:
                break
            root_error = next_error

        message = str(root_error) or str(error)
        if message:
            return f"{type(root_error).__name__}: {message}"
        return type(root_error).__name__

    async def _run_main_task(self) -> None:
        if not self._audio_buffer or not self._avatarkit_session:
            return

        try:
            async for item in self._audio_buffer:
                if isinstance(item, rtc.AudioFrame):
                    await self._send_audio_frame(item)
                elif isinstance(item, AudioSegmentEnd):
                    if not await self._finalize_active_segment(source="segment_end"):
                        logger.debug("Avatar segment end received without an active request")
        except asyncio.CancelledError:
            logger.debug("SpatialReal avatar audio forwarder cancelled")
        except Exception as e:
            logger.error("Error in SpatialReal avatar audio forwarder", exc_info=e)

    async def _send_audio_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._avatarkit_session:
            return

        previous_req_id = self._active_req_id
        req_id = await self._avatarkit_session.send_audio(audio=bytes(frame.data), end=False)
        if previous_req_id and previous_req_id != req_id:
            logger.warning(
                "Avatar request ID changed while streaming audio",
                extra={"previous": previous_req_id, "current": req_id},
            )
            previous_segment = self._segments.get(previous_req_id)
            if previous_segment is not None:
                self._mark_segment_waiting_for_completion(previous_segment)

        segment = self._segments.get(req_id)
        if segment is None:
            segment = _SegmentState(req_id=req_id)
            self._segments[req_id] = segment

        if segment.first_frame_at is None:
            segment.first_frame_at = time.time()
            logger.debug("SpatialReal avatar first audio frame", extra={"request_id": req_id})

        segment.pushed_duration += frame.duration
        self._active_req_id = req_id
        self._schedule_active_segment_idle_end()

    def _cancel_active_segment_idle_end(self) -> None:
        if self._active_segment_idle_end_task and not self._active_segment_idle_end_task.done():
            self._active_segment_idle_end_task.cancel()
        self._active_segment_idle_end_task = None

    def _schedule_active_segment_idle_end(self) -> None:
        active_req_id = self._active_req_id
        if active_req_id is None:
            return

        self._cancel_active_segment_idle_end()
        self._active_segment_idle_end_task = asyncio.create_task(
            self._wait_for_active_segment_idle_end(active_req_id, ACTIVE_SEGMENT_IDLE_END_SECONDS),
            name=f"spatialreal_idle_segment_end_{active_req_id}",
        )

    async def _wait_for_active_segment_idle_end(self, req_id: str, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return

        if self._active_req_id != req_id:
            return
        if req_id in self._pending_segment_ids:
            return
        if req_id not in self._segments:
            return
        if await self._finalize_active_segment(source="idle_timeout"):
            logger.warning(
                "Avatar segment end marker missing; forcing finalization",
                extra={"request_id": req_id, "idle_timeout": timeout},
            )

    async def _finalize_active_segment(self, *, source: str) -> bool:
        if self._active_req_id is None or not self._avatarkit_session:
            return False

        async with self._segment_finalize_lock:
            active_req_id = self._active_req_id
            if active_req_id is None:
                return False

            self._cancel_active_segment_idle_end()
            req_id = await self._avatarkit_session.send_audio(audio=b"", end=True)
            if req_id != active_req_id:
                logger.warning(
                    "Avatar request ID changed while finalizing segment",
                    extra={"expected": active_req_id, "actual": req_id, "source": source},
                )

            self._active_req_id = None
            active_segment = self._segments.pop(active_req_id, None)
            segment = self._segments.get(req_id)

            if active_segment is None and segment is None:
                return True
            if segment is None:
                if active_segment is None:
                    return True
                active_segment.req_id = req_id
                segment = active_segment
                self._segments[req_id] = segment
            elif active_segment is not None and segment is not active_segment:
                segment.pushed_duration = max(segment.pushed_duration, active_segment.pushed_duration)
                if segment.first_frame_at is None:
                    segment.first_frame_at = active_segment.first_frame_at

            self._mark_segment_waiting_for_completion(segment)
            return True

    def _mark_segment_waiting_for_completion(self, segment: _SegmentState) -> None:
        if segment.req_id not in self._pending_segment_ids:
            self._pending_segment_ids.append(segment.req_id)

        if segment.completion_timeout_task and not segment.completion_timeout_task.done():
            segment.completion_timeout_task.cancel()

        timeout = self._compute_completion_timeout(segment)
        segment.completion_timeout_task = asyncio.create_task(
            self._wait_for_segment_completion_timeout(segment.req_id, timeout),
            name=f"spatialreal_segment_timeout_{segment.req_id}",
        )

    @staticmethod
    def _compute_completion_timeout(segment: _SegmentState) -> float:
        if segment.first_frame_at is None:
            return MIN_COMPLETION_TIMEOUT_SECONDS

        expected_playback_end = segment.first_frame_at + segment.pushed_duration
        remaining_playback = max(0.0, expected_playback_end - time.time())
        return max(
            MIN_COMPLETION_TIMEOUT_SECONDS,
            remaining_playback + COMPLETION_TIMEOUT_BUFFER_SECONDS,
        )

    async def _wait_for_segment_completion_timeout(self, req_id: str, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return

        if self._complete_segment(req_id=req_id, interrupted=False, reason="timeout"):
            logger.warning(
                "Avatar segment completion timed out, assuming playback finished",
                extra={"request_id": req_id, "timeout": timeout},
            )

    def _on_transport_frame(self, frame: bytes, is_last: bool) -> None:
        if not is_last:
            return

        req_id = self._extract_req_id_from_transport_frame(frame)
        if req_id is not None:
            if req_id not in self._pending_segment_ids:
                logger.debug(
                    "Ignoring provider completion before local segment finalization",
                    extra={"request_id": req_id},
                )
                return

            if not self._complete_segment(req_id=req_id, interrupted=False, reason="provider_end"):
                logger.debug("Completion event for unknown request", extra={"request_id": req_id})
            return

        if self._pending_segment_ids:
            fallback_req_id = self._pending_segment_ids[0]
            if self._complete_segment(
                req_id=fallback_req_id,
                interrupted=False,
                reason="provider_end_fallback",
            ):
                logger.warning(
                    "Avatar completion event missing request ID; matched oldest pending segment",
                    extra={"request_id": fallback_req_id},
                )

    @staticmethod
    def _extract_req_id_from_transport_frame(frame: bytes) -> str | None:
        try:
            envelope = message_pb2.Message()
            envelope.ParseFromString(frame)
        except Exception:
            return None

        if envelope.type != message_pb2.MESSAGE_SERVER_RESPONSE_ANIMATION:
            return None

        req_id = envelope.server_response_animation.req_id
        return req_id or None

    def _complete_segment(self, *, req_id: str, interrupted: bool, reason: str) -> bool:
        segment = self._segments.pop(req_id, None)
        if segment is None:
            return False

        self._pending_segment_ids = deque(
            pending_req_id for pending_req_id in self._pending_segment_ids if pending_req_id != req_id
        )

        if segment.completion_timeout_task and not segment.completion_timeout_task.done():
            segment.completion_timeout_task.cancel()

        if self._active_req_id == req_id:
            self._active_req_id = None
            self._cancel_active_segment_idle_end()

        playback_position = (
            self._estimate_interrupted_playback_position(segment) if interrupted else segment.pushed_duration
        )

        if self._audio_buffer:
            self._audio_buffer.notify_playback_finished(
                playback_position=playback_position,
                interrupted=interrupted,
            )

        logger.debug(
            "SpatialReal avatar segment playback completed",
            extra={
                "request_id": req_id,
                "reason": reason,
                "interrupted": interrupted,
                "playback_position": playback_position,
                "pushed_duration": segment.pushed_duration,
            },
        )
        return True

    @staticmethod
    def _estimate_interrupted_playback_position(segment: _SegmentState) -> float:
        if segment.first_frame_at is None:
            return 0.0

        elapsed = max(0.0, time.time() - segment.first_frame_at)
        return min(segment.pushed_duration, elapsed)

    def _complete_all_segments(self, *, interrupted: bool, reason: str) -> None:
        for req_id in list(self._segments.keys()):
            self._complete_segment(req_id=req_id, interrupted=interrupted, reason=reason)

        self._active_req_id = None
        self._cancel_active_segment_idle_end()
        self._pending_segment_ids.clear()

    def _on_clear_buffer(self) -> None:
        asyncio.create_task(self._handle_interrupt())

    async def _handle_interrupt(self) -> None:
        if not self._avatarkit_session:
            return

        try:
            interrupted_id = await self._avatarkit_session.interrupt()

            async with self._segment_finalize_lock:
                if (
                    not self._complete_segment(
                        req_id=interrupted_id,
                        interrupted=True,
                        reason="interrupt",
                    )
                    and self._active_req_id is not None
                ):
                    self._complete_segment(
                        req_id=self._active_req_id,
                        interrupted=True,
                        reason="interrupt_fallback",
                    )

                for req_id in list(self._segments.keys()):
                    self._complete_segment(
                        req_id=req_id,
                        interrupted=True,
                        reason="interrupt_remaining",
                    )

            logger.debug("SpatialReal avatar interrupted", extra={"request_id": interrupted_id})
        except Exception as e:
            logger.warning("Failed to interrupt SpatialReal avatar", exc_info=e)

    async def aclose(self) -> None:
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
            self._main_task = None

        self._cancel_active_segment_idle_end()
        self._complete_all_segments(interrupted=True, reason="session_close")

        if (
            self._agent_session
            and self._audio_buffer
            and self._audio_output_attached
            and self._agent_session.output.audio is self._audio_buffer
        ):
            self._agent_session.output.audio = self._original_audio_output

        self._audio_output_attached = False
        self._original_audio_output = None

        if self._audio_buffer:
            await self._audio_buffer.aclose()
            self._audio_buffer = None

        if self._avatarkit_session:
            try:
                await self._avatarkit_session.close()
                logger.debug("SpatialReal avatar session closed")
            except Exception as e:
                logger.warning("Error closing SpatialReal avatar session", exc_info=e)
            finally:
                self._avatarkit_session = None

        self._initialized = False
        self._agent_session = None
