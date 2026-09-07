# LiveKit Agents Plugin for SpatialReal Avatar

This plugin provides integration with [SpatialReal](https://spatialreal.com)'s avatar service for lip-synced avatar rendering in LiveKit voice agents.

## Installation

```bash
pip install livekit-plugins-spatialreal
```

### Versioning

Starting with 1.7.1, the plugin's major and minor version track the `livekit-agents`
release it was validated against (e.g. plugin 1.8.x targets `livekit-agents` 1.8.x). The
patch number is the plugin's own and may advance independently.

The current release, 1.8.0, is validated against `livekit-agents` 1.8.0.

Or install from source:

```bash
pip install -e .
```

## Configuration

Set the following environment variables:

```bash
# Required
SPATIALREAL_API_KEY=your-api-key
SPATIALREAL_APP_ID=your-app-id
SPATIALREAL_AVATAR_ID=your-avatar-id

# Optional
SPATIALREAL_CONSOLE_ENDPOINT=
SPATIALREAL_INGRESS_ENDPOINT=
# Max seconds of per-response audio retained for false-interruption resume (default 180, min 10)
SPATIALREAL_RESUME_BUFFER_MAX_SECONDS=

# LiveKit credentials
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

## Usage

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import spatialreal


class VoiceAssistant(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant.")


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Configure your pipeline components (VAD, STT, LLM, TTS)
    session = AgentSession(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )

    # Initialize and start the avatar session
    avatar = spatialreal.AvatarSession()
    await avatar.start(session, room=ctx.room)

    # Start the agent session
    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

For production agents, catch `SpatialRealException` so you can decide whether to fail the job or continue without avatar output:

```python
try:
    await avatar.start(session, room=ctx.room)
except spatialreal.SpatialRealException as err:
    logger.error("Avatar startup failed: %s", err)
    raise
```

## API Reference

### `AvatarSession`

Main class for integrating SpatialReal avatars with LiveKit agents.

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | `str` | SpatialReal API key (or set `SPATIALREAL_API_KEY`) |
| `app_id` | `str` | SpatialReal application ID (or set `SPATIALREAL_APP_ID`) |
| `avatar_id` | `str` | Avatar ID to use (or set `SPATIALREAL_AVATAR_ID`) |
| `console_endpoint_url` | `str` | Custom console endpoint URL |
| `ingress_endpoint_url` | `str` | Custom ingress endpoint URL |
| `avatar_participant_identity` | `str` | LiveKit identity for avatar participant |
| `avatar_participant_name` | `str` | LiveKit display name for avatar participant |
| `idle_timeout_seconds` | `int` | LiveKit egress idle timeout in seconds (`0` uses server defaults) |
| `sample_rate` | `int \| None` | Optional avatar audio sample rate override |

#### Methods

- `start(agent_session, room, *, livekit_url, livekit_api_key, livekit_api_secret)`: Start the avatar session and hook into the agent's audio output. Raises `SpatialRealException` with actionable context if startup fails.
- `aclose()`: Clean up avatar session resources.

When starting, the plugin automatically sets `lk.publish_on_behalf` to the
agent participant identity for avatar worker association in LiveKit frontends.

#### Events

- `playback_started` — emitted with an `AvatarPlaybackStartedEvent` when audible avatar
  playback is first observed for a request.

```python
@avatar.on("playback_started")
def _on_playback_started(ev: spatialreal.AvatarPlaybackStartedEvent):
    logger.info("avatar playing request %s (via %s)", ev.request_id, ev.source)
```

### `AvatarPlaybackStartedEvent`

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | SpatialReal request the playback belongs to |
| `source` | `str` | Signal that observed the start (e.g. `livekit_avatar_audio_track`) |
| `observed_at` | `float` | Unix timestamp of the observation |

### `SpatialRealException`

Exception raised for SpatialReal-related errors.

## How It Works

1. The plugin intercepts TTS audio output from the agent session
2. Audio frames are forwarded to SpatialReal's avatar service
3. SpatialReal generates lip-synced video and audio
4. The avatar joins the LiveKit room and publishes the synchronized streams

### Interruptions and false-interruption recovery

LiveKit Agents owns turn-taking and interruption decisions; the plugin never adds its own VAD
or interruption policy. The audio output the plugin installs advertises pause support, so the
framework's `resume_false_interruption` behaviour (on by default) works with the avatar:

- When the framework detects a possible interruption it calls `pause()`. The plugin stops the
  current SpatialReal request and retains the unplayed PCM tail locally.
- If the framework later rejects the interruption (a laugh, a backchannel, a cough), it calls
  `resume()`. The plugin re-sends only the estimated unplayed remainder as a new request.
- If the interruption is confirmed, `clear_buffer()` discards the retained audio.

The retained tail is bounded by `SPATIALREAL_RESUME_BUFFER_MAX_SECONDS`.

## License

Apache-2.0
