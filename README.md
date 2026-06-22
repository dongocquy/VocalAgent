# VocalAgent — Teams Realtime Translator

Chrome extension for realtime English→Vietnamese meeting translation on Microsoft Teams (web).

## Architecture

- Chrome Extension captures Teams tab audio via `chrome.tabCapture`
- Local Python server runs Whisper STT on RTX 5080 GPU
- DeepSeek v4-flash API handles translation
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
- DeepSeek API key

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
- Admin UI: `http://127.0.0.1:8765/admin`
- Config file: `config.json` (auto-created in project root)
- Environment variable: `VOCALAGENT_CONFIG` overrides config file path

### API Providers
- DeepSeek: Set API key at `/admin` or in `config.json`
- OpenAI: Switch provider at `/admin`, set API key
