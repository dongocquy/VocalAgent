"""FastAPI server: WASAPI loopback audio → pipeline → WebSocket broadcast."""

import asyncio
import os
import queue
import sys
from pathlib import Path

# Force UTF-8 everywhere on Windows (before any other imports)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from config import ConfigManager
from pipeline import DeepSeekTranslator, OpenAITranslator, Pipeline

CONFIG_PATH = os.environ.get(
    "VOCALAGENT_CONFIG",
    os.path.join(Path(__file__).parent.parent, "config.db"),
)
config_manager = ConfigManager(CONFIG_PATH)

# ── Globals ──────────────────────────────────────────────────

_pipeline: Pipeline = None
_clients: set[WebSocket] = set()
_audio_queue: queue.Queue = queue.Queue()
_stream: sd.InputStream = None
_running = False
_processing_task: asyncio.Task = None


def _create_pipeline():
    provider = config_manager.get_provider()
    provider_cfg = config_manager.get_provider_config()
    api_key = config_manager.get_api_key()
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
    """Auto-detect WASAPI loopback device. Returns device index or None.

    Prioritizes WASAPI host API (required for loopback on Windows).
    Falls back to any input device with 'stereo mix' or 'loopback' in name.
    """
    devices = sd.query_devices()
    host_apis = sd.query_hostapis()

    # Find WASAPI host API index
    wasapi_api = None
    for i, api in enumerate(host_apis):
        if "wasapi" in api["name"].lower():
            wasapi_api = i
            break

    # Priority 1: WASAPI loopback devices
    if wasapi_api is not None:
        for i, dev in enumerate(devices):
            if dev["hostapi"] == wasapi_api and dev["max_input_channels"] > 0:
                name = dev["name"].lower()
                if "loopback" in name:
                    print(f"[Audio] Found WASAPI loopback device [{i}]: {dev['name']}")
                    return i

    # Priority 2: Any host API device with stereo mix / loopback / wasapi in name
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if dev["max_input_channels"] > 0 and (
            "loopback" in name
            or "stereo mix" in name
            or "wasapi" in name
        ):
            print(f"[Audio] Found loopback device [{i}]: {dev['name']} (hostapi={dev['hostapi']})")
            return i

    # Priority 3: WASAPI input devices (may support loopback)
    if wasapi_api is not None:
        for i, dev in enumerate(devices):
            if dev["hostapi"] == wasapi_api and dev["max_input_channels"] > 0:
                print(f"[Audio] Trying WASAPI input device [{i}]: {dev['name']}")
                return i

    # Fallback: default input device
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

    # Diagnostic: log first chunk and periodic counts
    _audio_callback.count = getattr(_audio_callback, 'count', 0) + 1
    if _audio_callback.count == 1:
        peak = abs(chunk).max()
        print(f"[Audio] First chunk received: {len(chunk)} samples, peak={peak:.4f}")
    if _audio_callback.count % 100 == 0:
        qsize = _audio_queue.qsize()
        print(f"[Audio] Chunks: {_audio_callback.count}, queue size: {qsize}")

    _audio_queue.put(chunk)


def start_audio_capture():
    """Start WASAPI loopback capture stream."""
    global _stream, _running
    audio_cfg = config_manager.get_audio_config()
    device = audio_cfg.get("device")
    if device is None:
        device = _find_loopback_device()

    if device is None:
        print("[Audio] ERROR: No loopback device found. Available devices:")
        print(sd.query_devices())
        print()
        print("[Audio] HINT: On Windows, try these steps:")
        print("  1. Right-click speaker icon → Sounds → Recording tab")
        print("  2. Right-click empty area → 'Show Disabled Devices'")
        print("  3. Enable 'Stereo Mix' if present")
        print("  4. Or install VB-Cable (https://vb-audio.com/Cable/)")
        raise RuntimeError("No loopback audio device available")

    sample_rate = audio_cfg.get("sample_rate", 16000)
    chunk_size = audio_cfg.get("chunk_size", 4096)
    channels = audio_cfg.get("channels", 1)

    # Reset diagnostic counter
    _audio_callback.count = 0

    _stream = sd.InputStream(
        device=device,
        channels=channels,
        samplerate=sample_rate,
        blocksize=chunk_size,
        callback=_audio_callback,
        dtype=np.float32,
    )
    _running = True
    _stream.start()
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
    for ws in list(_clients):
        try:
            await ws.send_json(event)
        except Exception:
            disconnected.add(ws)
    _clients.difference_update(disconnected)


async def audio_processing_loop():
    """Main loop: read audio chunks from queue, run pipeline, broadcast events."""
    global _pipeline
    print("[Pipeline] Processing loop started")
    chunk_count = 0
    event_count = 0
    while _running:
        # Drain all available chunks in batch
        try:
            chunk = _audio_queue.get(timeout=0.05)
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        chunk_count += 1
        # Apply gain from config
        audio_cfg = config_manager.get_audio_config()
        gain = audio_cfg.get("gain", 1.0)
        if gain != 1.0:
            chunk = np.clip(chunk * gain, -1.0, 1.0)

        if chunk_count == 1:
            print(f"[Pipeline] First chunk dequeued, peak={abs(chunk).max():.4f} (gain={gain}x)")
        if chunk_count % 100 == 0:
            qsize = _audio_queue.qsize()
            print(f"[Pipeline] Chunks processed: {chunk_count}, queue: {qsize}, events: {event_count}")

        if _pipeline is None:
            continue

        try:
            async for event in _pipeline.process_chunk(chunk):
                event_count += 1
                if event.get("type") == "sentence":
                    print(f"[Pipeline] Sentence: {event['text'][:80]}")
                elif event.get("type") == "sentence_start":
                    print(f"[Stream] ▶ [{event['id']}]")
                elif event.get("type") == "sentence_end":
                    text_full = event.get("text", "")
                    text_en = event.get("text_en", "")
                    print(f"[Stream] ✓ [{event['id']}]: {text_full[:80]}")
                    # Persist translation to DB
                    if text_full and text_en:
                        try:
                            config_manager.save_translation(text_en, text_full)
                        except Exception as e:
                            print(f"[DB] Save translation error: {e}")
                await broadcast_event(event)
        except Exception as e:
            print(f"[Pipeline] Error: {e}")
    print(f"[Pipeline] Processing loop stopped ({chunk_count} chunks, {event_count} events)")


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
            <label for="audio_gain">Gain ({audio_gain}x — boost quiet audio)</label>
            <input type="range" name="audio_gain" id="audio_gain" min="0.5" max="20.0" step="0.5" value="{audio_gain}">
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
        audio_gain=audio_cfg.get("gain", 1.0),
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
    audio_gain: str = Form("1.0"),
):
    try:
        # Build update data — only include api_key if non-empty (password fields
        # return empty when user didn't type anything, don't overwrite stored keys)
        deepseek_update = {"model": deepseek_model}
        if deepseek_api_key.strip():
            deepseek_update["api_key"] = deepseek_api_key

        openai_update = {"model": openai_model}
        if openai_api_key.strip():
            openai_update["api_key"] = openai_api_key

        update_data = {
            "provider": provider,
            "deepseek": deepseek_update,
            "openai": openai_update,
            "whisper": {"model": whisper_model},
            "vad": {"threshold": float(vad_threshold)},
            "audio": {
                "device": int(audio_device) if audio_device.strip() else None,
                "gain": float(audio_gain),
            },
        }
        config_manager.update(update_data)
        # Reload pipeline with new config (background task)
        asyncio.create_task(reload_pipeline())
        return HTMLResponse(
            content=_render_admin(
                '<div class="flash flash-success">&#10003; Configuration saved — pipeline reloaded.</div>'
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


@app.get("/api/translations")
async def api_translations(limit: int = 50):
    """Return recent translations as JSON."""
    rows = config_manager.get_translations(limit)
    return rows


HISTORY_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Translation History</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    background: #0f0f1a; color: #eee;
    max-width: 960px; margin: 0 auto; padding: 24px;
  }
  h1 { font-size: 22px; margin-bottom: 4px; color: #e94560; }
  #stats { font-size: 12px; color: #666; margin-bottom: 20px; }
  .entry {
    background: #1a1a2e; border-radius: 10px; padding: 14px 18px;
    margin-bottom: 10px;
  }
  .entry .meta {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px; font-size: 11px; color: #555;
  }
  .entry .meta .id { color: #e94560; font-weight: 600; }
  .row {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  }
  .badge {
    border-radius: 8px; padding: 10px 14px; font-size: 14px;
    line-height: 1.5; word-break: break-word;
  }
  .badge.en {
    background: #0d2137; border: 1px solid #1e4d7a; color: #7ec8f8;
  }
  .badge.vi {
    background: #2d1320; border: 1px solid #6e2543; color: #f5a8c8;
  }
  .badge .tag {
    display: inline-block; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px;
    padding: 2px 8px; border-radius: 4px; margin-bottom: 6px;
  }
  .badge.en .tag { background: #1e4d7a; color: #a0d8ff; }
  .badge.vi .tag { background: #6e2543; color: #f8c8d8; }
</style>
</head>
<body>
<h1>Translation History</h1>
<div id="stats"></div>
<div id="list"></div>
<script>
  const API = '/api/translations?limit=';
  const LIMIT = 50;

  function timeAgo(ts) {
    const iso = ts.replace(' ', 'T') + 'Z';
    const diff = (Date.now() - Date.parse(iso)) / 1000;
    if (diff < 60) return Math.floor(diff) + 's ago';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    return Math.floor(diff / 3600) + 'h ago';
  }

  async function refresh() {
    try {
      const res = await fetch(API + LIMIT);
      const rows = await res.json();
      document.getElementById('stats').textContent = rows.length + ' translations';
      document.getElementById('list').innerHTML = rows.map(r => `
        <div class="entry">
          <div class="meta">
            <span>${timeAgo(r.timestamp)}</span>
            <span class="id">#${r.id}</span>
          </div>
          <div class="row">
            <div class="badge en">
              <span class="tag">EN</span>
              <div>${r.text_en}</div>
            </div>
            <div class="badge vi">
              <span class="tag">VI</span>
              <div>${r.text_vi}</div>
            </div>
          </div>
        </div>
      `).join('');
    } catch(e) { console.error(e); }
  }

  refresh();
  setInterval(refresh, 3000);
</script>
</body>
</html>"""


@app.get("/history", response_class=HTMLResponse)
async def history_page():
    return HTMLResponse(content=HISTORY_PAGE)


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


def _print_config():
    """Print current configuration to terminal."""
    cfg = config_manager.config
    audio_cfg = cfg.get("audio", {})
    vad_cfg = cfg.get("vad", {})
    whisper_cfg = cfg.get("whisper", {})
    provider = cfg.get("provider", "deepseek")
    provider_cfg = cfg.get(provider, {})
    print("─" * 50)
    print(f"  Provider : {provider} ({provider_cfg.get('model', '?')})")
    print(f"  Whisper  : {whisper_cfg.get('model', '?')} on {whisper_cfg.get('device', '?')}")
    print(f"  VAD      : threshold={vad_cfg.get('threshold', 0.5)}")
    print(f"  Audio    : device={audio_cfg.get('device') or 'auto'}, "
          f"sr={audio_cfg.get('sample_rate', 16000)}, "
          f"chunk={audio_cfg.get('chunk_size', 4096)}, "
          f"gain={audio_cfg.get('gain', 1.0)}x")
    print("─" * 50)


async def reload_pipeline():
    """Stop capture, recreate pipeline with current config, restart capture."""
    global _pipeline
    print("[Reload] Recreating pipeline with current config...")
    old_pipeline = _pipeline
    stop_audio_capture()
    # Wait for stream to fully close
    await asyncio.sleep(0.5)
    # Clear audio queue
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
        except queue.Empty:
            break
    try:
        _pipeline = _create_pipeline()
    except Exception as e:
        print(f"[Reload] Failed to create pipeline: {e}")
        print("[Reload] Keeping old pipeline, restarting capture...")
        _pipeline = old_pipeline
    try:
        start_audio_capture()
        _print_config()
    except Exception as e:
        print(f"[Reload] Audio capture failed: {e}")
        print("[Reload] Server running without audio — WS broadcast only")
    # Restart processing task (old one exited when _running was set to False)
    global _processing_task
    _processing_task = asyncio.create_task(audio_processing_loop())


@app.on_event("startup")
async def on_startup():
    global _pipeline
    _print_config()
    try:
        _pipeline = _create_pipeline()
    except Exception as e:
        print(f"[Startup] Failed to create pipeline: {e}")
        print("[Startup] Translation disabled — check API key in admin page")
        _pipeline = None
    try:
        start_audio_capture()
    except Exception as e:
        print(f"[Startup] Audio capture failed: {e}")
        print("[Startup] Server running without audio — WS broadcast only")
    global _processing_task
    _processing_task = asyncio.create_task(audio_processing_loop())


@app.on_event("shutdown")
async def on_shutdown():
    global _processing_task
    stop_audio_capture()
    if _processing_task is not None:
        _processing_task.cancel()
        try:
            await asyncio.wait_for(_processing_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    print("[Shutdown] Complete")


# ── Entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    server_cfg = config_manager.get_server_config()
    uvicorn.run(
        "server:app",
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 8765),
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
    )
