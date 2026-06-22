# WASAPI Loopback Audio Capture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Chrome extension `tabCapture` audio source with Python `sounddevice` WASAPI loopback capture. Extension becomes thin subtitle overlay client. Web UI for viewing subtitles outside Teams.

**Architecture:** Python server opens WASAPI loopback `InputStream` via `sounddevice`, pushing float32 audio chunks to `asyncio.Queue` via thread-safe callback. Main asyncio loop dequeues chunks → `pipeline.process_chunk()` → broadcasts events to all WebSocket clients. Extension and Web UI connect to `/ws` as passive receivers. No audio capture logic remains in the extension.

**Tech Stack:** Python 3.12, FastAPI, sounddevice, faster-whisper, silero-vad, openai SDK, Chrome Manifest V3 (vanilla JS).

---

## File Structure

```
VocalAgent/
├── server/
│   ├── config.py              # MODIFY: add get_audio_config()
│   ├── pipeline.py            # NO CHANGE
│   ├── server.py              # MAJOR REWRITE: WASAPI capture + broadcast + Web UI
├── extension/
│   ├── manifest.json          # MODIFY: remove tabCapture permission
│   ├── popup.html             # NO CHANGE
│   ├── popup.js               # SLIGHT MODIFY: simplified start/stop flow
│   ├── popup.css              # NO CHANGE
│   ├── background.js          # MAJOR SIMPLIFY: remove tabCapture, inject+connect only
│   ├── content.js             # MAJOR SIMPLIFY: remove audio capture, WS client only
│   └── overlay.css            # NO CHANGE
├── config.json                # MODIFY: add "audio" section
├── requirements.txt           # MODIFY: add sounddevice
└── test/
    └── test_ws.py             # MODIFY: new WS broadcast tests
```

---

### Task 1: Add `sounddevice` dependency and audio config

**Files:**
- Modify: `requirements.txt`
- Modify: `config.json`
- Modify: `server/config.py:24-28`

- [ ] **Step 1: Add sounddevice to requirements.txt**

```bash
echo "sounddevice>=0.5.0" >> /mnt/d/Projects/VocalAgent/requirements.txt
```

- [ ] **Step 2: Install sounddevice**

```bash
cd /mnt/d/Projects/VocalAgent && pip install sounddevice
```

Expected: `sounddevice` and PortAudio bindings installed.

- [ ] **Step 3: Add audio section to config.json**

Edit `/mnt/d/Projects/VocalAgent/config.json` — add after `"server"` block:

```json
  "audio": {
    "device": null,
    "sample_rate": 16000,
    "chunk_size": 4096,
    "channels": 1
  }
```

Full config.json after edit:

```json
{
  "provider": "openai",
  "deepseek": {
    "api_key": "key-placeholder",
    "model": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com"
  },
  "openai": {
    "api_key": "key-placeholder",
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1"
  },
  "whisper": {
    "model": "small",
    "device": "cuda",
    "compute_type": "float16",
    "language": "en"
  },
  "vad": {
    "threshold": 0.6,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 500
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8765
  },
  "audio": {
    "device": null,
    "sample_rate": 16000,
    "chunk_size": 4096,
    "channels": 1
  }
}
```

- [ ] **Step 4: Add get_audio_config() to ConfigManager**

Edit `/mnt/d/Projects/VocalAgent/server/config.py` — add after `get_server_config` (line 93):

```python
    def get_audio_config(self) -> Dict[str, Any]:
        return dict(self._config.get("audio", {
            "device": None,
            "sample_rate": 16000,
            "chunk_size": 4096,
            "channels": 1,
        }))
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add requirements.txt config.json server/config.py && git commit -m "feat: add sounddevice dependency and audio config section

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Rewrite server.py — WASAPI loopback + WebSocket broadcast + Web UI

**Files:**
- Modify: `server/server.py`

- [ ] **Step 1: Write new server.py**

Overwrite `/mnt/d/Projects/VocalAgent/server/server.py`:

```python
"""FastAPI server: WASAPI loopback audio → pipeline → WebSocket broadcast.

Audio flows:
  WASAPI loopback → asyncio.Queue → Pipeline.process_chunk() → broadcast to WS clients

WebSocket:
  Clients connect to /ws to receive events. Multiple clients supported.
  Events: {type:"sentence", text:...} | {type:"status", status:...} | {type:"error", message:...}

HTTP:
  GET /admin  → config page
  POST /admin → save config
  GET /       → subtitle web UI
"""

import asyncio
import json
import os
import queue
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config import ConfigManager
from pipeline import DeepSeekTranslator, OpenAITranslator, Pipeline

CONFIG_PATH = os.environ.get(
    "VOCALAGENT_CONFIG",
    os.path.join(Path(__file__).parent.parent, "config.json"),
)
config_manager = ConfigManager(CONFIG_PATH)

# ── Globals ──────────────────────────────────────────────────

_pipeline: Pipeline = None
_clients: set[WebSocket] = set()
_audio_queue: queue.Queue = queue.Queue()
_stream: sd.InputStream = None
_running = False


def _create_pipeline():
    provider = config_manager.get_provider()
    provider_cfg = config_manager.get_provider_config()
    api_key = provider_cfg.get("api_key", "")
    model = provider_cfg.get("model", "deepseek-v4-flash")

    if provider == "openai":
        translator = OpenAITranslator(api_key=api_key, model=model)
    else:
        translator = DeepSeekTranslator(api_key=api_key, model=model)

    return Pipeline(
        translator=translator,
        whisper_config=config_manager.get_whisper_config(),
        vad_config=config_manager.get_vad_config(),
    )


# ── Audio Capture ────────────────────────────────────────────


def _find_loopback_device():
    """Auto-detect WASAPI loopback device. Returns device index or None."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if dev["max_input_channels"] > 0 and (
            "loopback" in name
            or "stereo mix" in name
            or "wasapi" in name
            or "speakers" in name
        ):
            print(f"[Audio] Found loopback device [{i}]: {dev['name']}")
            return i
    # Fallback: use default input device
    default_input = sd.default.device[0]
    if default_input is not None:
        dev = sd.query_devices(default_input)
        print(f"[Audio] Using default input device [{default_input}]: {dev['name']}")
        return default_input
    return None


def _audio_callback(indata, frames, time_info, status):
    """Called from PortAudio thread. Push audio chunk to thread-safe queue."""
    if status:
        print(f"[Audio] Status: {status}")
    chunk = indata.copy().flatten()
    _audio_queue.put(chunk)


def start_audio_capture():
    """Start WASAPI loopback capture stream."""
    global _stream, _running
    audio_cfg = config_manager.get_audio_config()
    device = audio_cfg.get("device") or _find_loopback_device()

    if device is None:
        print("[Audio] ERROR: No loopback device found. Available devices:")
        print(sd.query_devices())
        raise RuntimeError("No loopback audio device available")

    sample_rate = audio_cfg.get("sample_rate", 16000)
    chunk_size = audio_cfg.get("chunk_size", 4096)
    channels = audio_cfg.get("channels", 1)

    _stream = sd.InputStream(
        device=device,
        channels=channels,
        samplerate=sample_rate,
        blocksize=chunk_size,
        callback=_audio_callback,
        dtype=np.float32,
    )
    _stream.start()
    _running = True
    print(f"[Audio] Capture started: device={device}, sr={sample_rate}, "
          f"chunk={chunk_size}, channels={channels}")


def stop_audio_capture():
    """Stop WASAPI capture stream."""
    global _stream, _running
    _running = False
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None
    print("[Audio] Capture stopped")


# ── Broadcast ─────────────────────────────────────────────────


async def broadcast_event(event: dict):
    """Send event to all connected WebSocket clients."""
    disconnected = set()
    for ws in _clients:
        try:
            await ws.send_json(event)
        except Exception:
            disconnected.add(ws)
    _clients.difference_update(disconnected)


async def audio_processing_loop():
    """Main loop: read audio chunks from queue, run pipeline, broadcast events."""
    global _pipeline
    print("[Pipeline] Processing loop started")
    while _running:
        try:
            chunk = _audio_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        try:
            async for event in _pipeline.process_chunk(chunk):
                await broadcast_event(event)
        except Exception as e:
            print(f"[Pipeline] Error: {e}")
            await broadcast_event({"type": "error", "message": str(e)})
    print("[Pipeline] Processing loop stopped")


# ── App ───────────────────────────────────────────────────────

app = FastAPI(title="VocalAgent Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Admin Page ────────────────────────────────────────────────

ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VocalAgent — Admin</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: system-ui, sans-serif; background: #0f1117; color: #e1e4e8; max-width: 640px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ font-size: 24px; margin-bottom: 24px; }}
        label {{ display: block; font-size: 13px; font-weight: 600; margin: 16px 0 6px; color: #8b949e; }}
        input, select {{ width: 100%; padding: 8px 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #e1e4e8; font-size: 14px; }}
        input:focus, select:focus {{ border-color: #58a6ff; outline: none; }}
        fieldset {{ border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 16px 0; }}
        legend {{ font-weight: 600; padding: 0 8px; }}
        button {{ margin-top: 20px; padding: 10px 24px; background: #238636; color: #fff; border: none; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; }}
        button:hover {{ background: #2ea043; }}
        .flash {{ padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; }}
        .flash-success {{ background: #1a3a2a; border: 1px solid #238636; }}
        .flash-error {{ background: #3a1a1a; border: 1px solid #da3633; }}
        hr {{ border: none; border-top: 1px solid #30363d; margin: 20px 0; }}
        .status {{ font-size: 13px; color: #8b949e; }}
        .status span {{ color: #56d364; }}
    </style>
</head>
<body>
    <h1>VocalAgent Admin</h1>
    {flash}
    <form method="POST" action="/admin">
        <fieldset>
            <legend>LLM Provider</legend>
            <label for="provider">Provider</label>
            <select name="provider" id="provider">
                <option value="deepseek" {ds_sel}>DeepSeek</option>
                <option value="openai" {oa_sel}>OpenAI</option>
            </select>
            <label for="deepseek_api_key">DeepSeek API Key</label>
            <input type="password" name="deepseek_api_key" id="deepseek_api_key" value="{ds_key}" placeholder="key-...">
            <label for="deepseek_model">DeepSeek Model</label>
            <input type="text" name="deepseek_model" id="deepseek_model" value="{ds_model}">
            <label for="openai_api_key">OpenAI API Key</label>
            <input type="password" name="openai_api_key" id="openai_api_key" value="{oa_key}" placeholder="key-...">
            <label for="openai_model">OpenAI Model</label>
            <input type="text" name="openai_model" id="openai_model" value="{oa_model}">
        </fieldset>
        <fieldset>
            <legend>Whisper</legend>
            <label for="whisper_model">Model Size</label>
            <select name="whisper_model" id="whisper_model">
                {whisper_opts}
            </select>
        </fieldset>
        <fieldset>
            <legend>VAD</legend>
            <label for="vad_threshold">Threshold ({vad_threshold})</label>
            <input type="range" name="vad_threshold" id="vad_threshold" min="0.1" max="0.9" step="0.05" value="{vad_threshold}">
        </fieldset>
        <fieldset>
            <legend>Audio</legend>
            <label for="audio_device">Device Index (empty = auto-detect)</label>
            <input type="number" name="audio_device" id="audio_device" value="{audio_device}" placeholder="Auto">
        </fieldset>
        <button type="submit">Save Configuration</button>
    </form>
    <hr>
    <p class="status">Config: <span>{config_path}</span></p>
</body>
</html>"""


def _render_admin(flash: str = "") -> str:
    cfg = config_manager.config
    whisper_cfg = cfg.get("whisper", {})
    vad_cfg = cfg.get("vad", {})
    ds_cfg = cfg.get("deepseek", {})
    oa_cfg = cfg.get("openai", {})
    audio_cfg = cfg.get("audio", {})

    whisper_models = ["tiny", "base", "small", "medium"]
    current_whisper = whisper_cfg.get("model", "medium")
    whisper_opts = "".join(
        f'<option value="{m}" {"selected" if m == current_whisper else ""}>{m}</option>'
        for m in whisper_models
    )

    return ADMIN_HTML.format(
        flash=flash,
        ds_sel="selected" if cfg.get("provider") == "deepseek" else "",
        oa_sel="selected" if cfg.get("provider") == "openai" else "",
        ds_key=ds_cfg.get("api_key", ""),
        ds_model=ds_cfg.get("model", "deepseek-v4-flash"),
        oa_key=oa_cfg.get("api_key", ""),
        oa_model=oa_cfg.get("model", "gpt-4o-mini"),
        whisper_opts=whisper_opts,
        vad_threshold=vad_cfg.get("threshold", 0.5),
        audio_device=audio_cfg.get("device") or "",
        config_path=CONFIG_PATH,
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_get():
    return HTMLResponse(content=_render_admin())


@app.post("/admin", response_class=HTMLResponse)
async def admin_post(
    provider: str = Form("deepseek"),
    deepseek_api_key: str = Form(""),
    deepseek_model: str = Form("deepseek-v4-flash"),
    openai_api_key: str = Form(""),
    openai_model: str = Form("gpt-4o-mini"),
    whisper_model: str = Form("medium"),
    vad_threshold: str = Form("0.5"),
    audio_device: str = Form(""),
):
    try:
        update_data = {
            "provider": provider,
            "deepseek": {
                "api_key": deepseek_api_key,
                "model": deepseek_model,
            },
            "openai": {
                "api_key": openai_api_key,
                "model": openai_model,
            },
            "whisper": {"model": whisper_model},
            "vad": {"threshold": float(vad_threshold)},
            "audio": {
                "device": int(audio_device) if audio_device.strip() else None,
            },
        }
        config_manager.update(update_data)
        return HTMLResponse(
            content=_render_admin(
                '<div class="flash flash-success">&#10003; Configuration saved. Restart server for audio changes.</div>'
            )
        )
    except Exception as e:
        return HTMLResponse(
            content=_render_admin(
                f'<div class="flash flash-error">Error: {str(e)}</div>'
            ),
            status_code=400,
        )


# ── Subtitle Web UI ──────────────────────────────────────────

SUBTITLE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VocalAgent — Subtitles</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            background: #0f1117;
            color: #e1e4e8;
            height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-end;
            padding-bottom: 80px;
            overflow: hidden;
        }}
        #subtitles {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            max-width: 700px;
            width: 90%;
        }}
        .subtitle {{
            background: rgba(0, 0, 0, 0.78);
            color: rgba(255, 255, 255, 0.95);
            padding: 12px 22px;
            border-radius: 12px;
            font-size: 17px;
            line-height: 1.5;
            text-align: center;
            animation: fadeIn 200ms ease-out;
            transition: opacity 400ms ease-out, transform 300ms ease-out;
            max-width: 100%;
            word-wrap: break-word;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
        }}
        .subtitle.fading {{
            opacity: 0;
            transform: translateY(-12px);
        }}
        #status {{
            font-size: 12px;
            color: #8b949e;
            margin-bottom: 12px;
            min-height: 20px;
        }}
        #status.connected {{ color: #56d364; }}
        #status.disconnected {{ color: #da3633; }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <div id="status">Connecting...</div>
    <div id="subtitles"></div>
    <script>
        const statusEl = document.getElementById('status');
        const subsEl = document.getElementById('subtitles');
        const MAX_VISIBLE = 3;
        const STAY_MS = 5000;
        let ws;

        function connect() {{
            const port = {server_port};
            ws = new WebSocket(`ws://127.0.0.1:${{port}}/ws`);
            ws.onopen = () => {{
                statusEl.textContent = 'Connected';
                statusEl.className = 'connected';
            }};
            ws.onclose = () => {{
                statusEl.textContent = 'Disconnected — reconnecting...';
                statusEl.className = 'disconnected';
                setTimeout(connect, 2000);
            }};
            ws.onerror = () => {{
                statusEl.textContent = 'Connection error';
                statusEl.className = 'disconnected';
            }};
            ws.onmessage = (event) => {{
                try {{
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'sentence') {{
                        addSubtitle(msg.text);
                    }} else if (msg.type === 'status' && msg.status === 'listening') {{
                        statusEl.textContent = 'Listening...';
                        statusEl.className = 'connected';
                    }} else if (msg.type === 'error') {{
                        statusEl.textContent = msg.message;
                        setTimeout(() => {{
                            statusEl.textContent = 'Connected';
                            statusEl.className = 'connected';
                        }}, 3000);
                    }}
                }} catch (e) {{ /* ignore */ }}
            }};
        }}

        function addSubtitle(text) {{
            const el = document.createElement('div');
            el.className = 'subtitle';
            el.textContent = text;
            subsEl.appendChild(el);

            const subtitles = subsEl.querySelectorAll('.subtitle');
            if (subtitles.length > MAX_VISIBLE) {{
                subtitles[0].classList.add('fading');
                setTimeout(() => subtitles[0].remove(), 400);
            }}

            setTimeout(() => {{
                if (el.parentNode) {{
                    el.classList.add('fading');
                    setTimeout(() => {{ if (el.parentNode) el.remove(); }}, 400);
                }}
            }}, STAY_MS);
        }}

        connect();
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def subtitle_page():
    server_cfg = config_manager.get_server_config()
    return HTMLResponse(
        content=SUBTITLE_PAGE.format(server_port=server_cfg.get("port", 8765))
    )


# ── WebSocket Broadcast ──────────────────────────────────────


@app.websocket("/ws")
async def websocket_subscriber(ws: WebSocket):
    """Clients connect to receive subtitle/status events. No audio input expected."""
    await ws.accept()
    _clients.add(ws)
    print(f"[WS] Client connected ({len(_clients)} total)")
    try:
        # Keep connection alive; discard any messages from client
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _clients.discard(ws)
        print(f"[WS] Client disconnected ({len(_clients)} remaining)")


# ── Startup / Shutdown ───────────────────────────────────────


@app.on_event("startup")
async def on_startup():
    global _pipeline
    _pipeline = _create_pipeline()
    try:
        start_audio_capture()
    except Exception as e:
        print(f"[Startup] Audio capture failed: {e}")
        print("[Startup] Server running without audio — WS broadcast only")
    asyncio.create_task(audio_processing_loop())


@app.on_event("shutdown")
async def on_shutdown():
    stop_audio_capture()


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    server_cfg = config_manager.get_server_config()
    uvicorn.run(
        "server:app",
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 8765),
        reload=False,
    )
```

- [ ] **Step 2: Verify syntax**

```bash
cd /mnt/d/Projects/VocalAgent && python -c "import ast; ast.parse(open('server/server.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add server/server.py && git commit -m "feat: rewrite server for WASAPI loopback capture + WS broadcast + web UI

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Simplify extension — remove audio capture, keep overlay + WS client

**Files:**
- Modify: `extension/manifest.json`
- Modify: `extension/background.js`
- Modify: `extension/content.js`
- Modify: `extension/popup.js`

- [ ] **Step 1: Update manifest.json — remove tabCapture permission**

Edit `/mnt/d/Projects/VocalAgent/extension/manifest.json`:

```json
{
  "manifest_version": 3,
  "name": "VocalAgent — Teams Translator",
  "version": "1.0.0",
  "description": "Realtime English→Vietnamese translation for Microsoft Teams meetings.",
  "permissions": [
    "activeTab",
    "storage",
    "scripting"
  ],
  "host_permissions": [
    "https://teams.microsoft.com/*",
    "https://teams.live.com/*",
    "http://127.0.0.1:*"
  ],
  "action": {
    "default_popup": "popup.html",
    "default_title": "VocalAgent Translator"
  },
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [
    {
      "matches": [
        "https://teams.microsoft.com/*",
        "https://teams.live.com/*"
      ],
      "js": ["content.js"],
      "css": ["overlay.css"]
    }
  ]
}
```

- [ ] **Step 2: Rewrite background.js — inject only, no tabCapture**

Overwrite `/mnt/d/Projects/VocalAgent/extension/background.js`:

```javascript
// State
let activeTabId = null;
let port = 8765;
let isCapturing = false;

// Persist capture state so popup shows correct status on reopen
function setCaptureState(active) {
  isCapturing = active;
  chrome.storage.local.set({ isCapturing: active, captureTabId: activeTabId });
}

// Handle messages from popup
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === 'checkServer') {
    checkServerConnection(msg.port || 8765)
      .then(ok => sendResponse({ ok }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
  if (msg.action === 'getState') {
    sendResponse({ isCapturing, activeTabId, port });
    return true;
  }
  if (msg.action === 'start') {
    activeTabId = msg.tabId;
    port = msg.port || 8765;
    startSubtitleOverlay(msg.tabId);
    sendResponse({ ok: true });
  } else if (msg.action === 'stop') {
    stopSubtitleOverlay();
    sendResponse({ ok: true });
  }
  return true;
});

async function checkServerConnection(serverPort) {
  try {
    const resp = await fetch(`http://127.0.0.1:${serverPort}/admin`, {
      signal: AbortSignal.timeout(3000),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

async function startSubtitleOverlay(tabId) {
  // Guard: stop any existing session
  if (isCapturing) {
    stopSubtitleOverlay();
    await new Promise(r => setTimeout(r, 300));
  }

  // Inject content script (if not already injected)
  try {
    await injectContentScript(tabId);
  } catch (err) {
    console.error('startSubtitleOverlay: inject failed:', err.message);
    setCaptureState(false);
    updatePopupStatus('stopped');
    return;
  }

  // Retry sendMessage (Teams SPA may lose context during navigation/focus)
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      await chrome.tabs.sendMessage(tabId, {
        type: 'subtitle-start',
        port: port,
      });
      console.log('startSubtitleOverlay: sendMessage succeeded on attempt', attempt);
      break;
    } catch (sendErr) {
      console.log('startSubtitleOverlay: sendMessage attempt', attempt, 'failed:', sendErr.message);
      if (attempt === 3) throw sendErr;
      await new Promise(r => setTimeout(r, 200));
    }
  }

  setCaptureState(true);
  updatePopupStatus('capturing');
}

async function injectContentScript(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: 'ping' });
    return; // Already loaded
  } catch {
    // Not loaded — inject now
  }
  await chrome.scripting.executeScript({
    target: { tabId: tabId },
    files: ['content.js'],
  });
  await chrome.scripting.insertCSS({
    target: { tabId: tabId },
    files: ['overlay.css'],
  });
}

function stopSubtitleOverlay() {
  setCaptureState(false);
  if (activeTabId) {
    chrome.tabs.sendMessage(activeTabId, { type: 'subtitle-stop' }).catch(() => {});
    activeTabId = null;
  }
  updatePopupStatus('stopped');
}

function updatePopupStatus(status) {
  chrome.runtime.sendMessage({
    type: 'status',
    status: status,
  }).catch(() => {}); // Popup may not be open
}

// Clean up when Teams tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === activeTabId) {
    stopSubtitleOverlay();
  }
});
```

- [ ] **Step 3: Rewrite content.js — WS client + overlay only, no audio**

Overwrite `/mnt/d/Projects/VocalAgent/extension/content.js`:

```javascript
(function () {
  const alreadyInjected = document.getElementById('vocaltranslator-overlay');

  let ws = null;
  let overlay = null;
  let statusEl = null;

  // ── One-time DOM setup ──────────────────────────────────
  function setupDOM() {
    if (overlay) return;
    overlay = document.createElement('div');
    overlay.id = 'vocaltranslator-overlay';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'vt-close';
    closeBtn.textContent = '×';
    closeBtn.title = 'Hide subtitles';
    overlay.appendChild(closeBtn);

    const dragHandle = document.createElement('div');
    dragHandle.className = 'vt-drag-handle';
    overlay.appendChild(dragHandle);

    statusEl = document.createElement('div');
    statusEl.className = 'vt-status';
    statusEl.style.display = 'none';
    overlay.appendChild(statusEl);

    document.body.appendChild(overlay);

    // Drag
    let dragging = false, dragStartY = 0, overlayStartBottom = 80;
    dragHandle.addEventListener('mousedown', (e) => {
      dragging = true;
      dragStartY = e.clientY;
      overlayStartBottom = parseInt(overlay.style.bottom) || 80;
      e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      overlay.style.bottom = Math.max(0, overlayStartBottom + dragStartY - e.clientY) + 'px';
    });
    document.addEventListener('mouseup', () => { dragging = false; });

    closeBtn.addEventListener('click', () => {
      overlay.style.display = overlay.style.display === 'none' ? '' : 'none';
    });
  }

  if (!alreadyInjected) setupDOM();

  // ── Shared helpers ─────────────────────────────────────
  function getOverlay() {
    if (!overlay) overlay = document.getElementById('vocaltranslator-overlay');
    return overlay;
  }
  function getStatusEl() {
    if (!statusEl) statusEl = getOverlay() && getOverlay().querySelector('.vt-status');
    return statusEl;
  }

  const MAX_VISIBLE = 3;
  const STAY_MS = 5000;

  function showSubtitle(text) {
    const ov = getOverlay();
    const st = getStatusEl();
    if (!ov || !st) return;
    const el = document.createElement('div');
    el.className = 'vt-subtitle';
    el.textContent = text;
    ov.insertBefore(el, st);
    const subtitles = ov.querySelectorAll('.vt-subtitle');
    if (subtitles.length > MAX_VISIBLE) {
      const oldest = subtitles[0];
      oldest.classList.add('fading');
      setTimeout(() => oldest.remove(), 400);
    }
    setTimeout(() => {
      if (el.parentNode) {
        el.classList.add('fading');
        setTimeout(() => { if (el.parentNode) el.remove(); }, 400);
      }
    }, STAY_MS);
  }

  function showStatus(text) {
    const st = getStatusEl();
    if (!st) return;
    st.style.display = text ? 'block' : 'none';
    st.textContent = text;
    if (text) setTimeout(() => { st.style.display = 'none'; }, 3000);
  }

  // ── WebSocket client (receive only) ────────────────────
  function connectWebSocket(serverPort) {
    if (ws) {
      ws.close();
      ws = null;
    }

    ws = new WebSocket(`ws://127.0.0.1:${serverPort}/ws`);

    ws.onopen = () => { showStatus('Connected'); };
    ws.onclose = () => { showStatus('Disconnected'); };
    ws.onerror = () => { showStatus('Connection error'); };
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'sentence') showSubtitle(msg.text);
        else if (msg.type === 'status' && msg.status === 'listening') showStatus('Listening...');
        else if (msg.type === 'error') showStatus(msg.message);
      } catch (e) { /* ignore malformed */ }
    };
  }

  function disconnectWebSocket() {
    if (ws) { ws.close(); ws = null; }
    showStatus('Stopped');
  }

  // ── Message listener ───────────────────────────────────
  let _listenerRegistered = false;
  if (!_listenerRegistered) {
    _listenerRegistered = true;
    chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
      if (msg.type === 'ping') {
        sendResponse({ ok: true });
        return true;
      }
      if (msg.type === 'subtitle-start') {
        connectWebSocket(msg.port || 8765);
        sendResponse({ ok: true });
        return true;
      }
      if (msg.type === 'subtitle-stop') {
        disconnectWebSocket();
        sendResponse({ ok: true });
        return true;
      }
      return false;
    });
  }
})();
```

- [ ] **Step 4: Update popup.js — simplified flow, no audio capture state**

Overwrite `/mnt/d/Projects/VocalAgent/extension/popup.js`:

```javascript
const PORT = document.getElementById('server-port');
const STATUS_DOT = document.querySelector('.dot');
const STATUS_TEXT = document.getElementById('server-text');
const TOGGLE_BTN = document.getElementById('toggle-btn');
const SOURCE_LANG = document.getElementById('source-lang');
const TARGET_LANG = document.getElementById('target-lang');

let isRunning = false;

// On popup open: load settings + check server + check if already capturing
chrome.storage.local.get(
  ['port', 'sourceLang', 'targetLang', 'isCapturing'],
  (items) => {
    if (items.port) PORT.value = items.port;
    if (items.sourceLang) SOURCE_LANG.value = items.sourceLang;
    if (items.targetLang) TARGET_LANG.value = items.targetLang;

    // Restore capture state if background is already active
    if (items.isCapturing) {
      isRunning = true;
      TOGGLE_BTN.textContent = 'Stop';
      TOGGLE_BTN.className = 'btn-stop';
      TOGGLE_BTN.disabled = false;
      STATUS_DOT.className = 'dot connected';
      STATUS_TEXT.textContent = 'Capturing...';
    } else {
      checkServer();
    }
  }
);

PORT.addEventListener('change', () => {
  chrome.storage.local.set({ port: PORT.value });
  if (!isRunning) checkServer();
});

SOURCE_LANG.addEventListener('change', () => {
  chrome.storage.local.set({ sourceLang: SOURCE_LANG.value });
});

TARGET_LANG.addEventListener('change', () => {
  chrome.storage.local.set({ targetLang: TARGET_LANG.value });
});

function checkServer() {
  STATUS_DOT.className = 'dot checking';
  STATUS_TEXT.textContent = 'Checking...';
  TOGGLE_BTN.disabled = true;

  chrome.runtime.sendMessage(
    { action: 'checkServer', port: parseInt(PORT.value) },
    (response) => {
      if (chrome.runtime.lastError || !response || !response.ok) {
        STATUS_DOT.className = 'dot disconnected';
        STATUS_TEXT.textContent = 'Server not running';
        TOGGLE_BTN.disabled = true;
        return;
      }
      STATUS_DOT.className = 'dot connected';
      STATUS_TEXT.textContent = 'Connected';
      TOGGLE_BTN.disabled = false;
    }
  );
}

TOGGLE_BTN.addEventListener('click', async () => {
  if (isRunning) {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.runtime.sendMessage({ action: 'stop', tabId: tab.id });
    isRunning = false;
    TOGGLE_BTN.textContent = 'Start';
    TOGGLE_BTN.className = 'btn-start';
    STATUS_TEXT.textContent = 'Connected';
    return;
  }

  // Start subtitle overlay
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (
    !tab.url.includes('teams.microsoft.com') &&
    !tab.url.includes('teams.live.com')
  ) {
    STATUS_TEXT.textContent = 'Open Teams meeting first';
    return;
  }

  chrome.runtime.sendMessage({
    action: 'start',
    tabId: tab.id,
    port: parseInt(PORT.value),
    sourceLang: SOURCE_LANG.value,
    targetLang: TARGET_LANG.value,
  });

  // Popup will close when background focuses the Teams tab.
  // State is persisted in chrome.storage, restored on next open.
  isRunning = true;
  TOGGLE_BTN.textContent = 'Stop';
  TOGGLE_BTN.className = 'btn-stop';
  STATUS_TEXT.textContent = 'Starting...';
});

// Listen for status updates from background
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'status') {
    if (msg.status === 'capturing') {
      isRunning = true;
      TOGGLE_BTN.textContent = 'Stop';
      TOGGLE_BTN.className = 'btn-stop';
    } else if (msg.status === 'stopped') {
      isRunning = false;
      TOGGLE_BTN.textContent = 'Start';
      TOGGLE_BTN.className = 'btn-start';
      if (STATUS_TEXT.textContent !== 'Server not running') {
        STATUS_TEXT.textContent = 'Connected';
      }
    }
  }
});
```

- [ ] **Step 5: Verify JS syntax**

```bash
node --check /mnt/d/Projects/VocalAgent/extension/background.js && echo "OK"
node --check /mnt/d/Projects/VocalAgent/extension/content.js && echo "OK"
node --check /mnt/d/Projects/VocalAgent/extension/popup.js && echo "OK"
```

Expected: three `OK`

- [ ] **Step 6: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add extension/ && git commit -m "feat: simplify extension to subtitle overlay client, remove audio capture

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Update WebSocket tests for new broadcast-only behavior

**Files:**
- Modify: `test/test_ws.py`

- [ ] **Step 1: Rewrite test_ws.py**

Overwrite `/mnt/d/Projects/VocalAgent/test/test_ws.py`:

```python
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from fastapi.testclient import TestClient
from server import app


def test_admin_page_returns_html():
    client = TestClient(app)
    response = client.get("/admin")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_admin_post_updates_config():
    client = TestClient(app)
    response = client.post(
        "/admin",
        data={
            "provider": "deepseek",
            "deepseek_api_key": "placeholder",
            "deepseek_model": "deepseek-v4-flash",
            "whisper_model": "small",
            "vad_threshold": "0.6",
        },
        follow_redirects=False,
    )
    assert response.status_code in (200, 302, 303)


def test_subtitle_page_returns_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Subtitles" in response.text or "subtitle" in response.text.lower()


def test_websocket_route_exists():
    routes = [r.path for r in app.routes]
    assert "/ws" in routes


def test_admin_route_exists():
    routes = [r.path for r in app.routes]
    assert "/admin" in routes


def test_root_route_exists():
    routes = [r.path for r in app.routes]
    assert "/" in routes
```

- [ ] **Step 2: Run tests**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_ws.py -v
```

Expected: 6 PASS

- [ ] **Step 3: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add test/test_ws.py && git commit -m "test: update WS tests for broadcast-only endpoint and web UI route

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Full test suite verification

**Files:**
- None new

- [ ] **Step 1: Run all Python tests**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/ -v
```

Expected: all tests pass (config, translator, VAD, whisper, pipeline, WS)

- [ ] **Step 2: Verify audio device detection (Windows only)**

```bash
cd /mnt/d/Projects/VocalAgent && python -c "
import sounddevice as sd
print('=== Audio Devices ===')
print(sd.query_devices())
print()
# Check for loopback
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(f'Input device [{i}]: {dev[\"name\"]} (channels={dev[\"max_input_channels\"]})')
"
```

Expected on Windows: list includes loopback device (e.g., "Speakers (Loopback)" or "Stereo Mix").

Expected on WSL2/Linux: may show no devices or dummy devices. Audio capture works only on Windows host.

- [ ] **Step 3: Commit any remaining changes**

```bash
cd /mnt/d/Projects/VocalAgent && git status
```

---

## Summary

| Task | Component | Files Changed |
|------|-----------|---------------|
| 1 | Dependencies + Config | `requirements.txt`, `config.json`, `server/config.py` |
| 2 | Server rewrite | `server/server.py` |
| 3 | Extension simplify | `manifest.json`, `background.js`, `content.js`, `popup.js` |
| 4 | Test update | `test/test_ws.py` |
| 5 | Verification | Run all tests, check audio devices |

**Total: 5 tasks. 0 new files. 9 files modified.**

**Key behavioral changes:**
- Server starts audio capture at boot, runs continuously with or without clients
- Multiple WebSocket clients can receive events simultaneously
- Extension no longer handles audio — pure display client
- Web UI at `/` for standalone subtitle viewing
- Admin page at `/admin` now includes audio device config
