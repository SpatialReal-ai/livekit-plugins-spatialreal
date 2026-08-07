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
from collections.abc import Coroutine
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
from livekit.agents.voice.io import AudioOutput
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from livekit import api, rtc

from .log import logger

message_pb2: Any = _message_pb2

__all__ = ["AvatarSession", "SpatialRealException"]

DEFAULT_AVATAR_PARTICIPANT_IDENTITY = "spatialreal-avatar"
DEFAULT_AVATAR_PARTICIPANT_NAME = "spatialreal-avatar"
DEFAULT_SAMPLE_RATE = 24000
MIN_COMPLETION_TIMEOUT_SECONDS = 3.0
LIVEKIT_ACTIVITY_COMPLETION_BUFFER_SECONDS = 1.0
LIVEKIT_ACTIVITY_START_WAIT_SECONDS = 8.0
LIVEKIT_ACTIVITY_MAX_OVERRUN_SECONDS = 8.0
LIVEKIT_ACTIVITY_RECHECK_SECONDS = 0.5
ACTIVE_SEGMENT_IDLE_END_SECONDS = 1.0
# int16 amplitude (~ -50 dBFS): decoded Opus "silence" is rarely exact zeros,
# so playback-start detection needs a small energy floor instead of any-nonzero
AVATAR_AUDIO_ACTIVITY_THRESHOLD = 100
# egress-mode frame retransmission (ALR) can deliver end=true more than once
# per req_id; remember recent completions so duplicates are dropped
COMPLETED_REQ_ID_HISTORY = 32
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
    playback_started: bool = False
    playback_started_at: float | None = None
    playback_start_source: str | None = None
    provider_playback_completed: bool = False
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
        self._room: rtc.Room | None = None
        self._audio_buffer: QueueAudioOutput | None = None
        self._original_audio_output: Any | None = None
        self._original_audio_tail: AudioOutput | None = None
        self._audio_output_attached = False
        self._main_task: asyncio.Task | None = None
        self._initialized = False
        self._segments: dict[str, _SegmentState] = {}
        self._pending_segment_ids: deque[str] = deque()
        self._early_provider_started_ids: set[str] = set()
        self._early_provider_completed_ids: set[str] = set()
        self._recently_completed_req_ids: deque[str] = deque(maxlen=COMPLETED_REQ_ID_HISTORY)
        self._active_req_id: str | None = None
        self._represented_participant_identity: str | None = None
        self._avatar_is_speaking = False
        self._avatar_audio_stream: rtc.AudioStream | None = None
        self._avatar_audio_monitor_task: asyncio.Task[None] | None = None
        self._active_segment_idle_end_task: asyncio.Task[None] | None = None
        self._segment_finalize_lock = asyncio.Lock()
        self._interrupt_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[Any]] = set()

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
        self._represented_participant_identity = local_participant_identity
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
        self._room = room
        self._original_audio_output = agent_session.output.audio
        self._original_audio_tail = self._resolve_audio_tail(agent_session.output.audio)

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

            # wait_playback_start defers the framework's playback_started event
            # (transcript sync, first-frame metrics) until we observe the avatar
            # actually playing in the room, instead of firing when the first TTS
            # frame is merely handed to SpatialReal.
            self._audio_buffer = QueueAudioOutput(
                sample_rate=resolved_sample_rate,
                wait_playback_start=True,
            )
            await self._audio_buffer.start()
            self._audio_buffer.on("clear_buffer", self._on_clear_buffer)  # type: ignore[arg-type]

            # Until the SpatialReal egress backend emits request-correlated
            # playback lifecycle events, playback start is observed client-side
            # from the avatar's published LiveKit audio track (primary) and the
            # room's active-speaker state (secondary).
            room.on("active_speakers_changed", self._on_active_speakers_changed)
            room.on("track_published", self._on_track_published)
            room.on("track_subscribed", self._on_track_subscribed)
            room.on("track_unsubscribed", self._on_track_unsubscribed)

            for participant in room.remote_participants.values():
                if not self._is_avatar_output_participant(participant):
                    continue
                for publication in participant.track_publications.values():
                    if publication.kind != rtc.TrackKind.KIND_AUDIO:
                        continue
                    publication.set_subscribed(True)
                    if publication.track is not None:
                        self._start_avatar_audio_monitor(publication.track)

            # keep output wrappers (TranscriptSynchronizer, recorder, ...) in
            # the chain: swap only the tail sink
            agent_session.output.replace_audio_tail(self._audio_buffer)
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
                self._spawn_background_task(self.aclose(), name="spatialreal_avatar_session_close")

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

    def _spawn_background_task(self, coro: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @staticmethod
    def _resolve_audio_tail(sink: AudioOutput | None) -> AudioOutput | None:
        while sink is not None and sink.next_in_chain is not None:
            sink = sink.next_in_chain
        return sink

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

        self._consume_early_provider_events(segment)

        if segment.first_frame_at is None:
            segment.first_frame_at = time.time()
            logger.debug("SpatialReal avatar first audio frame", extra={"request_id": req_id})

        segment.pushed_duration += frame.duration
        self._active_req_id = req_id
        self._schedule_active_segment_idle_end()

    def _consume_early_provider_events(self, segment: _SegmentState) -> None:
        if segment.req_id in self._early_provider_started_ids:
            self._early_provider_started_ids.discard(segment.req_id)
            self._mark_playback_started(segment, source="spatialreal_transport_frame")
        if segment.req_id in self._early_provider_completed_ids:
            self._early_provider_completed_ids.discard(segment.req_id)
            segment.provider_playback_completed = True

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
                segment.playback_started = segment.playback_started or active_segment.playback_started
                segment.provider_playback_completed = (
                    segment.provider_playback_completed or active_segment.provider_playback_completed
                )
                if segment.playback_started_at is None:
                    segment.playback_started_at = active_segment.playback_started_at
                    segment.playback_start_source = active_segment.playback_start_source

            self._consume_early_provider_events(segment)
            self._mark_segment_waiting_for_completion(segment)
            return True

    def _mark_segment_waiting_for_completion(self, segment: _SegmentState) -> None:
        if segment.req_id not in self._pending_segment_ids:
            self._pending_segment_ids.append(segment.req_id)

        # a provider completion that arrived before local finalization is
        # preserved (not dropped): apply it now
        if segment.provider_playback_completed:
            self._complete_segment(
                req_id=segment.req_id,
                interrupted=False,
                reason="provider_end_deferred",
            )
            return

        self._schedule_segment_completion(segment)

    def _schedule_segment_completion(self, segment: _SegmentState) -> None:
        if segment.completion_timeout_task and not segment.completion_timeout_task.done():
            segment.completion_timeout_task.cancel()

        timeout = self._compute_completion_timeout(segment)
        segment.completion_timeout_task = asyncio.create_task(
            self._wait_for_segment_completion_timeout(segment.req_id, timeout),
            name=f"spatialreal_segment_timeout_{segment.req_id}",
        )

    @staticmethod
    def _compute_completion_timeout(segment: _SegmentState) -> float:
        if segment.playback_started_at is not None:
            expected_playback_end = segment.playback_started_at + segment.pushed_duration
            remaining_playback = max(0.0, expected_playback_end - time.time())
            return max(
                MIN_COMPLETION_TIMEOUT_SECONDS,
                remaining_playback + LIVEKIT_ACTIVITY_COMPLETION_BUFFER_SECONDS,
            )

        if segment.first_frame_at is None:
            return MIN_COMPLETION_TIMEOUT_SECONDS

        expected_playback_end = segment.first_frame_at + segment.pushed_duration
        remaining_playback = max(0.0, expected_playback_end - time.time())
        return max(
            MIN_COMPLETION_TIMEOUT_SECONDS,
            remaining_playback + LIVEKIT_ACTIVITY_START_WAIT_SECONDS,
        )

    async def _wait_for_segment_completion_timeout(self, req_id: str, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return

        segment = self._segments.get(req_id)
        if segment is None:
            return

        # if the avatar is still audibly speaking, allow a bounded overrun
        # before declaring the segment finished
        if segment.playback_started_at is not None and self._avatar_is_speaking:
            expected_end = segment.playback_started_at + segment.pushed_duration
            if time.time() < expected_end + LIVEKIT_ACTIVITY_MAX_OVERRUN_SECONDS:
                segment.completion_timeout_task = asyncio.create_task(
                    self._wait_for_segment_completion_timeout(req_id, LIVEKIT_ACTIVITY_RECHECK_SECONDS),
                    name=f"spatialreal_segment_activity_wait_{req_id}",
                )
                return

        playback_observed = segment.playback_started_at is not None
        reason = "livekit_activity_duration" if playback_observed else "timeout_no_playback_signal"
        if self._complete_segment(req_id=req_id, interrupted=False, reason=reason):
            if playback_observed:
                logger.debug(
                    "SpatialReal avatar completion derived from observed LiveKit playback",
                    extra={
                        "request_id": req_id,
                        "playback_start_source": segment.playback_start_source,
                        "pushed_duration": segment.pushed_duration,
                    },
                )
            else:
                logger.warning(
                    "Avatar segment received no provider or LiveKit playback signal; using duration fallback",
                    extra={"request_id": req_id, "timeout": timeout},
                )

    def _on_transport_frame(self, frame: bytes, is_last: bool) -> None:
        req_id = self._extract_req_id_from_transport_frame(frame)
        if req_id is not None and req_id in self._recently_completed_req_ids:
            logger.debug(
                "Ignoring duplicate provider event for completed request",
                extra={"request_id": req_id, "is_last": is_last},
            )
            return
        if req_id is not None:
            segment = self._segments.get(req_id)
            if segment is None:
                self._early_provider_started_ids.add(req_id)
            else:
                self._mark_playback_started(segment, source="spatialreal_transport_frame")

            if not is_last:
                return

            if segment is None:
                self._early_provider_completed_ids.add(req_id)
                logger.debug(
                    "Preserving provider completion received before local segment creation",
                    extra={"request_id": req_id},
                )
                return

            segment.provider_playback_completed = True
            if req_id not in self._pending_segment_ids:
                logger.debug(
                    "Deferring provider completion until local segment finalization",
                    extra={"request_id": req_id},
                )
                return

            if not self._complete_segment(req_id=req_id, interrupted=False, reason="provider_end"):
                logger.debug("Completion event for unknown request", extra={"request_id": req_id})
            return

        if not is_last:
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
            return

        if self._active_req_id is not None:
            active_segment = self._segments.get(self._active_req_id)
            if active_segment is not None:
                self._mark_playback_started(active_segment, source="spatialreal_transport_frame")
                active_segment.provider_playback_completed = True
                logger.warning(
                    "Avatar completion event missing request ID; deferred against active segment",
                    extra={"request_id": self._active_req_id},
                )

    def _is_avatar_output_participant(self, participant: rtc.Participant) -> bool:
        if participant.identity == self._avatar_participant_identity:
            return True

        represented_identity = self._represented_participant_identity
        if represented_identity is None:
            return False

        attributes = getattr(participant, "attributes", {}) or {}
        return attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF) == represented_identity

    def _on_active_speakers_changed(self, speakers: list[rtc.Participant]) -> None:
        avatar_is_speaking = any(self._is_avatar_output_participant(speaker) for speaker in speakers)
        if avatar_is_speaking == self._avatar_is_speaking:
            return

        self._avatar_is_speaking = avatar_is_speaking
        logger.debug(
            "SpatialReal avatar LiveKit speaking state changed",
            extra={
                "is_speaking": avatar_is_speaking,
                "active_speaker_identities": [speaker.identity for speaker in speakers],
            },
        )
        if not avatar_is_speaking:
            return

        segment = self._current_observable_segment()
        if segment is not None:
            self._mark_playback_started(segment, source="livekit_active_speaker")

    def _on_track_published(
        self,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        # rooms configured without auto-subscribe never fire track_subscribed
        # unless we opt in explicitly
        if publication.kind == rtc.TrackKind.KIND_AUDIO and self._is_avatar_output_participant(participant):
            publication.set_subscribed(True)

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO or not self._is_avatar_output_participant(participant):
            return

        self._start_avatar_audio_monitor(track)
        logger.debug(
            "SpatialReal avatar LiveKit audio track subscribed",
            extra={
                "participant_identity": participant.identity,
                "track_sid": publication.sid,
            },
        )

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        _: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO and self._is_avatar_output_participant(participant):
            self._stop_avatar_audio_monitor()

    def _start_avatar_audio_monitor(self, track: rtc.Track) -> None:
        if self._avatar_audio_monitor_task and not self._avatar_audio_monitor_task.done():
            return

        self._avatar_audio_stream = rtc.AudioStream(track, capacity=10)
        self._avatar_audio_monitor_task = asyncio.create_task(
            self._monitor_avatar_audio(),
            name="spatialreal_avatar_audio_monitor",
        )

    def _stop_avatar_audio_monitor(self) -> None:
        if self._avatar_audio_monitor_task and not self._avatar_audio_monitor_task.done():
            self._avatar_audio_monitor_task.cancel()
        self._avatar_audio_monitor_task = None

        if self._avatar_audio_stream is not None:
            self._spawn_background_task(
                self._avatar_audio_stream.aclose(),
                name="spatialreal_avatar_audio_stream_close",
            )
        self._avatar_audio_stream = None

    async def _monitor_avatar_audio(self) -> None:
        stream = self._avatar_audio_stream
        if stream is None:
            return

        try:
            async for event in stream:
                self._on_avatar_audio_frame(event.frame)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning("SpatialReal avatar audio monitor failed", exc_info=True)

    def _current_observable_segment(self) -> _SegmentState | None:
        if self._pending_segment_ids:
            segment = self._segments.get(self._pending_segment_ids[0])
            if segment is not None:
                return segment
        if self._active_req_id is not None:
            return self._segments.get(self._active_req_id)
        return None

    def _on_avatar_audio_frame(self, frame: rtc.AudioFrame) -> None:
        segment = self._current_observable_segment()
        if segment is None or segment.playback_started:
            return

        if not self._frame_has_audible_audio(frame):
            return

        self._mark_playback_started(segment, source="livekit_avatar_audio_track")

    @staticmethod
    def _frame_has_audible_audio(frame: rtc.AudioFrame) -> bool:
        threshold = AVATAR_AUDIO_ACTIVITY_THRESHOLD
        for sample in frame.data:
            if sample >= threshold or sample <= -threshold:
                return True
        return False

    def _mark_playback_started(self, segment: _SegmentState, *, source: str) -> None:
        if segment.playback_started:
            return

        segment.playback_started = True
        segment.playback_started_at = time.time()
        segment.playback_start_source = source
        if self._audio_buffer:
            self._audio_buffer.notify_playback_started()
        if segment.req_id in self._pending_segment_ids:
            self._schedule_segment_completion(segment)
        logger.debug(
            "SpatialReal avatar playback started",
            extra={"request_id": segment.req_id, "source": source},
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

        self._early_provider_started_ids.discard(req_id)
        self._early_provider_completed_ids.discard(req_id)
        self._recently_completed_req_ids.append(req_id)

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
            # with wait_playback_start the framework only sees playback_started
            # when we notify it; guarantee started precedes finished even if no
            # playback signal was ever observed
            if not segment.playback_started:
                self._audio_buffer.notify_playback_started()
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
                "playback_start_source": segment.playback_start_source,
                "provider_playback_completed": segment.provider_playback_completed,
            },
        )
        return True

    @staticmethod
    def _estimate_interrupted_playback_position(segment: _SegmentState) -> float:
        anchor = segment.playback_started_at if segment.playback_started_at is not None else segment.first_frame_at
        if anchor is None:
            return 0.0

        elapsed = max(0.0, time.time() - anchor)
        return min(segment.pushed_duration, elapsed)

    def _complete_all_segments(self, *, interrupted: bool, reason: str) -> None:
        for req_id in list(self._segments.keys()):
            self._complete_segment(req_id=req_id, interrupted=interrupted, reason=reason)

        self._active_req_id = None
        self._cancel_active_segment_idle_end()
        self._pending_segment_ids.clear()
        self._early_provider_started_ids.clear()
        self._early_provider_completed_ids.clear()
        self._avatar_is_speaking = False

    def _on_clear_buffer(self) -> None:
        self._spawn_background_task(self._handle_interrupt(), name="spatialreal_avatar_interrupt")

    async def _handle_interrupt(self) -> None:
        if not self._avatarkit_session:
            return

        async with self._interrupt_lock:
            # coalesce duplicate clear_buffer callbacks: once the segments for
            # the interrupted turn are completed, later duplicates are no-ops
            if not self._segments and self._active_req_id is None:
                logger.debug("Ignoring duplicate SpatialReal avatar interrupt")
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

    def _detach_audio_output(self) -> None:
        if not self._agent_session or not self._audio_output_attached or self._audio_buffer is None:
            return

        output = self._agent_session.output
        if output.audio is self._audio_buffer:
            # replace_audio_tail fell back to a whole-chain assignment (no
            # wrappers were present), so restore the whole chain
            output.audio = self._original_audio_output
        elif self._resolve_audio_tail(output.audio) is self._audio_buffer:
            # we were swapped in as the tail under wrapper(s); swap the
            # original tail back so the closed buffer doesn't linger in the
            # chain
            if self._original_audio_tail is not None:
                output.replace_audio_tail(self._original_audio_tail)

        self._audio_output_attached = False

    async def aclose(self) -> None:
        if self._room is not None:
            self._room.off("active_speakers_changed", self._on_active_speakers_changed)
            self._room.off("track_published", self._on_track_published)
            self._room.off("track_subscribed", self._on_track_subscribed)
            self._room.off("track_unsubscribed", self._on_track_unsubscribed)
            self._room = None

        self._stop_avatar_audio_monitor()

        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
            self._main_task = None

        self._cancel_active_segment_idle_end()
        self._complete_all_segments(interrupted=True, reason="session_close")

        self._detach_audio_output()
        self._original_audio_output = None
        self._original_audio_tail = None
        self._represented_participant_identity = None

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
