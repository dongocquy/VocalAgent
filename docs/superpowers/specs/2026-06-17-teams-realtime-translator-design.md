# Teams Realtime Translator — Design Spec

**Date:** 2026-06-17
**Status:** Approved

## Overview

Chrome extension that captures Microsoft Teams (web) meeting audio, transcribes English speech locally via Whisper on RTX 5080 GPU, translates to Vietnamese via DeepSeek v4-flash API, and injects subtitle overlays into the Teams DOM.

**Goal:** ≤1.5s end-to-end latency from speaker utterance to translated subtitle display.

---

## Architecture

```
┌──────────────────────────────────┐
│  Chrome Extension (Manifest V3)  │
│                                  │
│  Popup ── Content Script ── BG   │
│  (settings)  (Teams DOM)   Worker│
│                              │   │
└──────────────────────────────┼───┘
                               │ WebSocket (ws://localhost:8765)
┌──────────────────────────────┼───┐
│  Python FastAPI Server       │   │
│                              │   │
│  /admin ── /ws               │   │
│ (config)   (pipeline)        │   │
│                              │   │
│  VAD ⟶ Whisper ⟶ LLM API    │   │
└──────────────────────────────────┘
```

### Components

| Component | File(s) | Role |
|-----------|---------|------|
| Popup UI | `popup.html`, `popup.js` | Language selection, server status, start/stop |
| Background Worker | `background.js` | `chrome.tabCapture` audio, WebSocket client, routing |
| Content Script | `content.js`, `overlay.css` | Inject subtitle overlay into Teams DOM |
| HTTP Server | `server.py` | FastAPI app, WS endpoint, admin page |
| Pipeline | `pipeline.py` | VAD → Whisper → Translate orchestrator |
| Config Manager | `config.py` | Read/write `config.json` |
| Admin UI | `templates/admin.html` | Provider selection, API keys, settings form |

---

## Data Flow

1. User opens Teams meeting tab → clicks extension popup → selects EN→VI → clicks Start
2. Popup sends `{action: "start"}` to background service worker
3. Background worker: `chrome.tabCapture.getMediaStreamId()` → obtains stream ID
4. Background worker sends stream ID to content script → content script creates `AudioContext` from stream
5. Audio resampled to 16kHz mono PCM, chunked into 4096-sample frames
6. Binary frames sent via WebSocket to `ws://127.0.0.1:8765/ws`
7. Server pipeline:
   - **VAD** (Silero): detects speech start/end, accumulates speech segments
   - On silence ≥500ms: flush segment → Whisper transcription
   - **Whisper** (faster-whisper medium, fp16, CUDA): transcribes English text
   - **DeepSeek v4-flash**: translates English → Vietnamese
   - Result pushed back via WebSocket as JSON
8. Background worker receives translation → `chrome.tabs.sendMessage` to content script
9. Content script renders subtitle in overlay

### WebSocket Protocol

**Client → Server:** Binary frames (PCM 16-bit, 16kHz, mono)
**Server → Client:** JSON messages

```json
{"type": "status",   "status": "listening"}
{"type": "sentence", "text": "Tôi nghĩ rằng chúng ta cần tập trung vào dự án này."}
{"type": "error",   "message": "Translation timeout"}
```

- `status` — "listening" when VAD detects active speech. Shown as dim indicator.
- `sentence` — completed translated sentence. Rendered white, final, stays 5s.
- `error` — pipeline error, displayed briefly then hidden.

---

## Component Details

### 1. Popup UI

- Source language dropdown (default: English)
- Target language dropdown (default: Tiếng Việt)
- Server connection indicator (green dot = connected)
- Start/Stop button (toggle)
- Settings: server port (default 8765)
- No framework — vanilla HTML/JS/CSS

### 2. Background Service Worker

**State machine:** `IDLE → CAPTURING → IDLE`

- Handles popup messages (start/stop)
- Manages `chrome.tabCapture` stream
- WebSocket client with auto-reconnect (exponential backoff: 1s → 2s → 4s → max 8s)
- Routes translated text to content script via `chrome.tabs.sendMessage`
- Listens for tab close → cleanup stream + WS

**Manifest V3 permissions:**
- `tabCapture` — capture tab audio
- `activeTab` — access Teams tab
- `storage` — persist settings
- `scripting` — inject content script

### 3. Content Script & Subtitle Overlay

- Injects single `<div id="vocaltranslator-overlay">` into Teams DOM
- All styles scoped via unique ID prefix, no CSS leak
- Option: Shadow DOM for strongest isolation
- Position: `fixed`, bottom-center, `z-index: 999999`
- Each sentence: fade in (200ms), stay 5s, fade out + shift up
- Max 3 sentences visible simultaneously
- Semi-transparent dark background (`rgba(0,0,0,0.75)`), white text
- Draggable handle to reposition overlay
- [X] button to temporarily hide
- Padding: 12px 16px, border-radius: 8px, font: 15px system-ui
- Partial text rendered in `rgba(255,255,255,0.5)`, final in solid white

### 4. Backend Server (`server.py`)

FastAPI application with two endpoints:

**`GET /admin`** — Configuration page (HTML form):
- Provider selector: DeepSeek | OpenAI (dropdown)
- API key input (password-masked)
- Model name input
- Whisper model size: tiny | base | small | medium (dropdown)
- VAD threshold slider (0.0–1.0)
- Status display: GPU memory used, active WS connections, uptime

**`POST /admin`** — Save configuration:
- Accepts form data, validates API key format
- Writes `config.json`
- Returns redirect to `GET /admin` with success/error flash message
- Active pipelines continue with old config; new connections use updated config

**`WS /ws`** — Audio pipeline WebSocket:
- Accepts binary audio frames
- Returns JSON translation results
- One pipeline instance per connection
- Handles disconnect cleanup

Run: `python server.py` → `http://127.0.0.1:8765`

### 5. Pipeline (`pipeline.py`)

Three-stage processing:

**Stage 1 — VAD (Silero VAD):**
```
Audio frames → Silero VAD iterator → speech segments
```
- Model: `silero-vad` (ONNX, runs on CPU, negligible overhead)
- Threshold: 0.5 (configurable)
- Min speech duration: 250ms
- Min silence duration: 500ms (triggers segment flush)
- Max speech duration: 15s (force flush, prevent OOM on long utterances)
- Yields `partial` events for interim text display

**Stage 2 — Whisper STT (faster-whisper):**
```
Speech segment (float32 array) → faster-whisper → English text
```
- Model: `medium` (recommended for RTX 5080 16GB), fp16, CUDA
- Fallback: `small` if VRAM constrained
- Language: `en` (faster, no auto-detection needed)
- Beam size: 1 (fastest), no timestamps
- Expected latency: 150–300ms per segment

**Stage 3 — Translation (DeepSeek API):**
```
English text + system prompt → deepseek-v4-flash → Vietnamese text
```
- SDK: `openai` Python client (compatible with DeepSeek API)
- Base URL: `https://api.deepseek.com`
- System prompt: `"Translate from English to Vietnamese. Output only the translation, no explanation. Keep proper nouns, numbers, and acronyms unchanged. Use natural conversational Vietnamese."`
- `max_tokens: 256`, `temperature: 0.0`
- Expected latency: 300–500ms (network + inference)

### 6. Translator Abstraction

```python
class Translator(ABC):
    @abstractmethod
    async def translate(self, text: str, source: str, target: str) -> str: ...

class DeepSeekTranslator(Translator): ...
class OpenAITranslator(Translator): ...
```

Provider selected via `config.json`, instantiated at pipeline startup.

### 7. Config (`config.json`)

```json
{
  "provider": "deepseek",
  "deepseek": {
    "api_key": "",
    "model": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com"
  },
  "openai": {
    "api_key": "",
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1"
  },
  "whisper": {
    "model": "medium",
    "device": "cuda",
    "compute_type": "float16",
    "language": "en"
  },
  "vad": {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 500
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8765
  }
}
```

Auto-created with defaults on first run if missing.

---

## Error Handling

| Error | Detection | Recovery |
|-------|-----------|----------|
| Server offline | WS connection refused | Popup shows "Server not running. Run `python server.py`" |
| API key invalid | DeepSeek 401 | Admin page red badge, logged |
| API rate limit | DeepSeek 429 | Queue sentence, retry after `Retry-After` header |
| Tab audio stopped | `MediaStream.onended` | Auto-reconnect attempt, notify "Audio lost — reload tab" |
| GPU OOM | CUDA error in faster-whisper | Fallback to smaller model, notify in admin |
| No speech detected | >5s silence | Periodic status message, check audio source |
| WebSocket disconnect | Connection close | Exponential backoff reconnection (1s→2s→4s→max 8s) |

---

## Latency Budget

End-to-end latency = **speech duration** + **processing pipeline**. Speech duration varies (2–15s per sentence), processing pipeline is fixed.

**Processing latency** (after speaker stops talking):

| Stage | Target | Notes |
|-------|--------|-------|
| VAD silence detection | 500ms | Wait for silence gap to confirm sentence end |
| Whisper transcription | 150–300ms | `medium` model, fp16, RTX 5080, segment ≤15s |
| DeepSeek API | 300–500ms | Network + inference, `max_tokens: 256` |
| **Processing total** | **~1–1.3s** | Within ≤1.5s target |

**End-to-end example:** Speaker talks 5s → 500ms silence → 300ms Whisper → 400ms API = ~6.2s từ lúc bắt đầu nói đến khi có phụ đề. Processing latency (1.2s) là phần extension kiểm soát được.

---

## Edge Cases

- **Multiple speakers simultaneously:** VAD segments treated independently. Translation not speaker-attributed (acceptable for v1).
- **Background noise:** Silero VAD has strong noise filtering. Threshold configurable in admin.
- **Proper nouns/numbers:** System prompt instructs model to preserve them unchanged.
- **Long utterances (>15s):** Force flush at 15s max speech duration to prevent infinite accumulation.
- **No audio detected:** Server sends periodic heartbeat; client shows "Listening..." status.

---

## Dependencies

**Python:**
- `fastapi` + `uvicorn` — HTTP/WS server
- `faster-whisper` — local STT (CTranslate2 backend, CUDA)
- `silero-vad` — voice activity detection (ONNX)
- `openai` — DeepSeek API client (OpenAI-compatible SDK)
- `numpy` — audio buffer processing
- `jinja2` — admin page templating

**Chrome Extension:**
- Manifest V3, vanilla JS, no framework
- Permissions: `tabCapture`, `activeTab`, `storage`, `scripting`

---

## Testing Strategy

```
test/
├── test_vad.py          # Silero VAD: segment detection with sample audio
├── test_whisper.py      # Whisper: transcription accuracy
├── test_translator.py   # Mock API, verify translate flow + prompt
├── test_ws.py           # WebSocket: connect, binary send, JSON receive
├── test_pipeline.py     # End-to-end: audio file → full pipeline → VI text
├── test_content.js      # Overlay: inject, render, animate, remove
└── fixtures/
    ├── sample_en.wav    # "I think we should focus on this project"
    └── silence.wav      # 5 seconds of silence
```

- Python: `pytest` + `pytest-asyncio`
- Extension: manual verification on live Teams meeting + chrome.debugger
- Pipeline: run `pipeline.py` standalone with WAV file input → verify output

---

## Project Structure

```
VocalAgent/
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   ├── popup.css
│   ├── background.js
│   ├── content.js
│   └── overlay.css
├── server/
│   ├── server.py
│   ├── pipeline.py
│   ├── config.py
│   └── templates/
│       └── admin.html
├── test/
│   ├── test_vad.py
│   ├── test_whisper.py
│   ├── test_translator.py
│   ├── test_ws.py
│   ├── test_pipeline.py
│   ├── test_content.js
│   └── fixtures/
│       ├── sample_en.wav
│       └── silence.wav
├── requirements.txt
└── README.md
```
