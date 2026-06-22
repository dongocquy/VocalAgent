# WASAPI Loopback Audio Capture — Design

**Date:** 2026-06-17
**Status:** approved
**Replaces:** Chrome extension tabCapture audio source

## Motivation

Current audio capture relies on Chrome extension `tabCapture.getMediaStreamId` + content script `getUserMedia` with `chromeMediaSource: 'tab'`. Problems:
- Chrome autoplay policy suspends `AudioContext` (requires manual `resume()`)
- `ScriptProcessorNode` deprecated, unreliable
- `getUserMedia` with `mandatory` constraint syntax deprecated
- Only works with Chrome browser Teams; native Teams desktop app unsupported
- Extension has ~200 lines of audio capture code mixing concerns

## Goal

Replace extension-side audio capture with Python `sounddevice` reading WASAPI loopback device on Windows. Extension becomes a thin subtitle overlay client.

## Architecture

```
┌──────────────────┐     WASAPI loopback     ┌─────────────────────────┐
│  Windows Audio   │────────────────────────▶│  Python Server (Win)    │
│  (Teams output)  │   float32 16kHz mono    │                         │
└──────────────────┘                         │  sounddevice InputStream│
                                             │  ↓                      │
                                             │  VAD (Silero)           │
                                             │  ↓                      │
                                             │  Whisper (faster-w)     │
                                             │  ↓                      │
                                             │  Translator (DeepSeek)  │
                                             │  ↓                      │
                                             │  WebSocket broadcast    │
                                             └────────┬────────────────┘
                                                      │ subtitle events
                                      ┌───────────────┼───────────────┐
                                      ▼               │               ▼
                              ┌──────────┐    ┌──────────────┐  ┌──────────┐
                              │Extension │    │ Web UI       │  │ (future  │
                              │(overlay) │    │ /            │  │ clients) │
                              └──────────┘    └──────────────┘  └──────────┘
```

## Components

### 1. Python Server (`server/server.py`)

**Changes from current:**

| Aspect | Current | New |
|--------|---------|-----|
| Audio source | WebSocket `/ws` receives bytes from extension | `sounddevice.InputStream` on WASAPI loopback |
| WebSocket role | Input + output (bi-directional) | Output only (broadcast subtitle events) |
| `/ws` endpoint | One connection from extension | Multiple connections (extension + web UI + future) |
| Event loop | uvicorn owns | Shared: sounddevice callback posts to asyncio queue |

**Key implementation details:**

- `sounddevice.query_devices()` to find WASAPI loopback device (`"Stereo Mix"` or `"Speakers (Loopback)"`)
- Device selection: auto-detect, configurable via `config.json`
- `InputStream` with `callback` that pushes `(audio_chunk,)` to `asyncio.Queue`
- Consumers: one per WebSocket connection reads from broadcast queue
- Pipeline runs in main asyncio task: reads audio queue → `pipeline.process_chunk()` → fans out to all connected WebSocket clients
- Server must run on Windows host (not WSL2) for WASAPI access

**New dependency:** `sounddevice` (pip)

### 2. Chrome Extension

**Remove:**
- `startAudioCapture()` (getUserMedia, AudioContext, ScriptProcessor)
- `stopCapture()` (cleanup of audio objects)
- `capture-start` / `capture-stop` message handlers
- `onaudioprocess` audio processing
- `capture-start`/`capture-stop` message forwarding in `background.js`

**Keep:**
- `setupDOM()` — overlay creation
- `showSubtitle()`, `showStatus()` — UI helpers
- Ping/pong for content script injection check

**Add:**
- Passive WebSocket client: connect to `ws://127.0.0.1:{port}/ws`, receive `{type: "sentence", text: ...}` events
- `startCapture()` / `stopCapture()` simplified to WebSocket connect/disconnect

**`background.js` changes:**
- Remove `tabCapture.getMediaStreamId` call
- Remove `injectContentScript`, `capture-start`/`capture-stop` messaging
- New flow: Start = inject content script + tell it to connect WebSocket; Stop = tell it to disconnect
- `isCapturing` state still persisted for popup UI

**`popup.js`:** Minimal changes. "Start" still triggers capture flow (now just connects WebSocket). "Stop" disconnects. Status indicators unchanged.

### 3. Web UI (`server/web/` or embedded in server)

- HTML page served at `/` or `/subtitle`
- Dark theme, minimal
- WebSocket auto-connect to `/ws`
- Shows last 3-5 subtitles with fade animation
- Language selector (source/target) — future iteration

### 4. Pipeline (`server/pipeline.py`)

**No changes.** VAD → Whisper → Translate chain unchanged. Input is `np.ndarray float32 16kHz` regardless of source.

### 5. Config (`config.json`)

**Add:**
```json
{
  "audio": {
    "device": null,       // null = auto-detect WASAPI loopback
    "sample_rate": 16000,
    "chunk_size": 4096,
    "channels": 1
  }
}
```

## Data Flow

```
1. sounddevice callback fires every CHUNK_SIZE samples
2. Audio chunk float32[4096] pushed to asyncio.Queue
3. Main asyncio loop dequeues, calls pipeline.process_chunk(audio)
4. Pipeline yields events: {type:"status", status:"listening"} or {type:"sentence", text:"..."} or {type:"error", message:"..."}
5. Events broadcast to all connected WebSocket clients
6. Extension recvs → showSubtitle(text)
7. Web UI recvs → append subtitle element
```

## Error Handling

| Failure | Behavior |
|---------|----------|
| WASAPI device not found | Log available devices, exit with helpful message |
| Device disconnected mid-capture | Callback receives error status, attempt reconnect, notify clients |
| Pipeline error (Whisper/API) | Catch, yield error event, continue processing |
| WebSocket client disconnects | Remove from broadcast set, server continues |
| No clients connected | Pipeline still runs (future: pause when no clients to save API costs) |

## Testing

- Unit: VAD, Whisper mock, Translator mock — already covered in `test_pipeline.py`
- Integration: `sounddevice.InputStream` with sine wave or WAV file → pipeline → verify events
- Manual: Start Teams meeting with audio, run server, open extension + web UI, verify subtitles appear

## Limitations

- Captures ALL system audio, not just Teams. VAD + Whisper filter non-speech. Music or other voices during meeting waste API calls.
- Requires Python server on Windows host (WSL2 has no audio device access).
- Single WASAPI loopback device — cannot choose per-app audio source.
