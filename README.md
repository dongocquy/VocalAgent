# VocalAgent — Teams Realtime Translator

Chrome extension + Python server for realtime English→Vietnamese meeting translation on Microsoft Teams (web).

Hỗ trợ **OpenAI GPT-4o-mini** và **DeepSeek v4-flash**.

## Architecture

- Chrome Extension captures Teams tab audio via `chrome.tabCapture`
- Local Python server runs Whisper STT on RTX 5080 GPU
- OpenAI GPT-4o-mini / DeepSeek v4-flash API handles translation
- Subtitles injected as DOM overlay

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Start server: `python server/server.py`
3. Configure API key at `http://127.0.0.1:8765/admin`
4. Load extension: `chrome://extensions` → "Load unpacked" → select `extension/`
5. Open Teams meeting → click extension icon → Start

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 12+ (RTX 5080 recommended)
- Chrome 120+
- OpenAI API key or DeepSeek API key

## Development

### Project Structure
```
VocalAgent/
├── server/           # Python FastAPI backend
│   ├── config.py     # ConfigManager with defaults
│   ├── pipeline.py   # VAD, WhisperSTT, Translator, Pipeline
│   └── server.py     # FastAPI app (/admin, /ws)
├── extension/        # Chrome extension (Manifest V3)
│   ├── manifest.json
│   ├── popup.html/js/css
│   ├── background.js # Service worker
│   ├── content.js    # Audio capture + subtitle overlay
│   └── overlay.css
├── test/             # Python tests
│   ├── test_config.py
│   ├── test_translator.py
│   ├── test_vad.py
│   ├── test_whisper.py
│   ├── test_ws.py
│   ├── test_pipeline.py
│   └── fixtures/
└── requirements.txt
```

### Running Tests
```bash
pip install -r requirements.txt
python -m pytest test/ -v
```

### Loading Extension
1. Go to `chrome://extensions`
2. Enable "Developer mode"
3. Click "Load unpacked" → select `extension/` folder
4. Open Teams meeting → click extension icon → Start

### Configuration

**Cách 1 — Environment variables (ưu tiên cao nhất):**
Tạo file `.env` (xem `.env.example`):
```
VOCALAGENT_PROVIDER=openai
VOCALAGENT_OPENAI_API_KEY=...
# hoặc VOCALAGENT_DEEPSEEK_API_KEY=...
```

**Cách 2 — Admin UI:**
`http://127.0.0.1:8765/admin` — chọn provider + nhập key.

**Cách 3 — Config file:**
`config.json` (auto-created, auto-migrated to SQLite).

### API Providers
| Provider | Default Model | Env Var |
|----------|--------------|---------|
| OpenAI   | `gpt-4o-mini` | `VOCALAGENT_OPENAI_API_KEY` |
| DeepSeek | `deepseek-v4-flash` | `VOCALAGENT_DEEPSEEK_API_KEY` |
