"""
SpatialReal Avatar integration for LiveKit Agents.

This module provides AvatarSession which hooks into an AgentSession
to route TTS audio to the SpatialReal avatar service.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from avatarkit import (
    AvatarSession as AvatarkitSession,
)
from avatarkit import (
    LiveKitEgressConfig,
    new_avatar_session,
)
from livekit.agents import AgentSession, UserStateChangedEvent
from livekit.agents.voice.avatar import AudioSegmentEnd, QueueAudioOutput

from livekit import rtc

from .log import logger

__all__ = ["AvatarSession", "SpatialRealException"]

DEFAULT_AVATAR_PARTICIPANT_IDENTITY = "spatialreal-avatar"
DEFAULT_SAMPLE_RATE = 24000

DEFAULT_CONSOLE_ENDPOINT = "https://console.us-west.spatialwalk.cloud/v1/console"
DEFAULT_INGRESS_ENDPOINT = "wss://api.us-west.spatialwalk.cloud/v2/driveningress"


class SpatialRealException(Exception):
    """Exception raised for SpatialReal-related errors."""

    pass


class AvatarSession:
    """
    This connects to SpatialReal's avatar service and routes TTS audio
    from the agent to the avatar for lip-synced rendering. The avatar
    service joins the LiveKit room and publishes synchronized video + audio.

    Args:
        api_key: SpatialReal API key. Falls back to SPATIALREAL_API_KEY env var.
        app_id: SpatialReal application ID. Falls back to SPATIALREAL_APP_ID env var.
        avatar_id: Avatar ID to use. Falls back to SPATIALREAL_AVATAR_ID env var.
        console_endpoint_url: Console endpoint URL. Falls back to
            SPATIALREAL_CONSOLE_ENDPOINT env var or default.
        ingress_endpoint_url: Ingress endpoint URL. Falls back to
            SPATIALREAL_INGRESS_ENDPOINT env var or default.
        avatar_participant_identity: LiveKit identity for the avatar participant.
        idle_timeout_seconds: Idle timeout in seconds for the egress connection.
            A value of 0 uses server defaults.
        sample_rate: Optional audio sample rate override for avatar audio.
            Falls back to agent_session.tts.sample_rate or a default value.

    Usage:
        avatar = AvatarSession()
        await avatar.start(session, room=ctx.room)
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        app_id: str | None = None,
        avatar_id: str | None = None,
        console_endpoint_url: str | None = None,
        ingress_endpoint_url: str | None = None,
        avatar_participant_identity: str | None = None,
        idle_timeout_seconds: int = 0,
        sample_rate: int | None = None,
    ) -> None:
        # Resolve API key
        self._api_key = api_key or os.getenv("SPATIALREAL_API_KEY")
        if not self._api_key:
            raise SpatialRealException(
                "api_key must be provided or SPATIALREAL_API_KEY environment variable must be set"
            )

        # Resolve app ID
        self._app_id = app_id or os.getenv("SPATIALREAL_APP_ID")
        if not self._app_id:
            raise SpatialRealException("app_id must be provided or SPATIALREAL_APP_ID environment variable must be set")

        # Resolve avatar ID
        self._avatar_id = avatar_id or os.getenv("SPATIALREAL_AVATAR_ID")
        if not self._avatar_id:
            raise SpatialRealException(
                "avatar_id must be provided or SPATIALREAL_AVATAR_ID environment variable must be set"
            )

        # Resolve endpoints
        self._console_endpoint_url = (
            console_endpoint_url or os.getenv("SPATIALREAL_CONSOLE_ENDPOINT") or DEFAULT_CONSOLE_ENDPOINT
        )
        self._ingress_endpoint_url = (
            ingress_endpoint_url or os.getenv("SPATIALREAL_INGRESS_ENDPOINT") or DEFAULT_INGRESS_ENDPOINT
        )

        # Avatar participant configuration
        self._avatar_participant_identity = avatar_participant_identity or DEFAULT_AVATAR_PARTICIPANT_IDENTITY

        if idle_timeout_seconds < 0:
            raise SpatialRealException("idle_timeout_seconds must be greater than or equal to 0")
        self._idle_timeout_seconds = idle_timeout_seconds

        if sample_rate is not None and sample_rate <= 0:
            raise SpatialRealException("sample_rate must be greater than 0")
        self._sample_rate = sample_rate

        # Internal state
        self._avatarkit_session: AvatarkitSession | None = None
        self._agent_session: AgentSession | None = None
        self._audio_buffer: QueueAudioOutput | None = None
        self._original_audio_output: Any | None = None
        self._audio_output_attached = False
        self._main_task: asyncio.Task | None = None
        self._initialized = False

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: str | None = None,
        livekit_api_key: str | None = None,
        livekit_api_secret: str | None = None,
    ) -> None:
        """
        Start the avatar session and hook into the agent session.

        Args:
            agent_session: The AgentSession to hook into for TTS audio.
            room: The LiveKit room for egress configuration.
            livekit_url: LiveKit server URL. Falls back to LIVEKIT_URL env var.
            livekit_api_key: LiveKit API key. Falls back to LIVEKIT_API_KEY env var.
            livekit_api_secret: LiveKit API secret. Falls back to LIVEKIT_API_SECRET env var.
        """
        if self._initialized:
            logger.warning("Avatar session already initialized")
            return

        # Resolve LiveKit credentials
        lk_url = livekit_url or os.getenv("LIVEKIT_URL")
        lk_api_key = livekit_api_key or os.getenv("LIVEKIT_API_KEY")
        lk_api_secret = livekit_api_secret or os.getenv("LIVEKIT_API_SECRET")

        if not lk_url or not lk_api_key or not lk_api_secret:
            raise SpatialRealException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be provided "
                "or LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET environment variables must be set"
            )

        room_name = room.name
        agent_participant_identity = room.local_participant.identity
        logger.info(f"Initializing SpatialReal avatar session for room: {room_name}")
        logger.debug(f"Console endpoint: {self._console_endpoint_url}")
        logger.debug(f"Ingress endpoint: {self._ingress_endpoint_url}")

        egress_attributes = {"lk.publish_on_behalf": agent_participant_identity}

        # Create LiveKit egress configuration for the avatar to join the room
        livekit_egress_kwargs: dict[str, Any] = {
            "url": lk_url,
            "api_key": lk_api_key,
            "api_secret": lk_api_secret,
            "room_name": room_name,
            "publisher_id": self._avatar_participant_identity,
            "extra_attributes": egress_attributes,
            "idle_timeout": self._idle_timeout_seconds,
        }
        livekit_egress = LiveKitEgressConfig(**livekit_egress_kwargs)

        resolved_sample_rate = self._sample_rate
        if resolved_sample_rate is None:
            resolved_sample_rate = agent_session.tts.sample_rate if agent_session.tts else DEFAULT_SAMPLE_RATE
        if resolved_sample_rate <= 0:
            raise SpatialRealException("sample_rate must be greater than 0")

        self._agent_session = agent_session
        self._original_audio_output = agent_session.output.audio

        try:
            # Create avatar session with LiveKit egress mode
            self._avatarkit_session = new_avatar_session(
                api_key=self._api_key,
                app_id=self._app_id,
                avatar_id=self._avatar_id,
                console_endpoint_url=self._console_endpoint_url,
                ingress_endpoint_url=self._ingress_endpoint_url,
                expire_at=datetime.now(timezone.utc) + timedelta(hours=1),
                livekit_egress=livekit_egress,
                sample_rate=resolved_sample_rate,
            )

            # Initialize and start the avatar session
            await self._avatarkit_session.init()
            await self._avatarkit_session.start()
            logger.info("SpatialReal avatar session connected")

            # Create audio buffer using livekit-agents' QueueAudioOutput
            self._audio_buffer = QueueAudioOutput(sample_rate=resolved_sample_rate)

            # Hook into agent session's audio output
            agent_session.output.audio = self._audio_buffer
            self._audio_output_attached = True

            # Start the audio buffer
            await self._audio_buffer.start()

            # Register for clear_buffer events (interruptions)
            @self._audio_buffer.on("clear_buffer")
            def on_clear_buffer() -> None:
                asyncio.create_task(self._handle_interrupt())

            # Register for user_state_changed events (interrupt on user speaking)
            @agent_session.on("user_state_changed")
            def on_user_state_changed(ev: UserStateChangedEvent) -> None:
                if ev.new_state == "speaking":
                    asyncio.create_task(self._handle_interrupt())

            # Start the main task that forwards audio to avatar
            self._main_task = asyncio.create_task(self._run_main_task())

            self._initialized = True
            logger.info("Avatar audio output attached to agent session")

            # Register cleanup on session close
            @agent_session.on("close")
            def on_session_close() -> None:
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
        reason = self._format_error_reason(error)
        return (
            "Failed to start SpatialReal avatar session. "
            "Check SpatialReal credentials, endpoint URLs, and outbound network access. "
            f"room={room_name}, avatar_id={self._avatar_id}, ingress_endpoint_url={self._ingress_endpoint_url}, "
            f"sample_rate={sample_rate}. Reason: {reason}"
        )

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
        """Main task that forwards audio from the buffer to the avatar service."""
        if not self._audio_buffer or not self._avatarkit_session:
            return

        try:
            frame_count = 0
            async for item in self._audio_buffer:
                if isinstance(item, rtc.AudioFrame):
                    # Convert AudioFrame to bytes and send to avatar
                    audio_bytes = bytes(item.data)
                    frame_count += 1

                    if frame_count == 1:
                        logger.debug("Avatar: First audio frame received")

                    await self._avatarkit_session.send_audio(
                        audio=audio_bytes,
                        end=False,
                    )

                elif isinstance(item, AudioSegmentEnd):
                    # End of audio segment - signal completion to avatar
                    logger.debug(f"Avatar: Segment end, sent {frame_count} frames")
                    await self._avatarkit_session.send_audio(
                        audio=b"",
                        end=True,
                    )

                    # Notify the buffer that playback is finished
                    self._audio_buffer.notify_playback_finished(
                        playback_position=0.0,
                        interrupted=False,
                    )
                    frame_count = 0

        except asyncio.CancelledError:
            logger.debug("Avatar main task cancelled")
        except Exception as e:
            logger.error(f"Error in avatar main task: {e}")

    async def _handle_interrupt(self) -> None:
        """Handle interruption - stop avatar's current audio processing."""
        if not self._avatarkit_session:
            return

        try:
            interrupted_id = await self._avatarkit_session.interrupt()
            logger.debug(f"Avatar interrupted, request_id={interrupted_id}")
        except Exception as e:
            logger.warning(f"Failed to interrupt avatar: {e}")

    async def aclose(self) -> None:
        """Clean up avatar session resources."""
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
            self._main_task = None

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
                logger.info("Avatar session closed")
            except Exception as e:
                logger.warning(f"Error closing avatar session: {e}")
            finally:
                self._avatarkit_session = None

        self._initialized = False
        self._agent_session = None
        self._audio_output_attached = False
        self._original_audio_output = None
