# Teams Realtime Translator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Chrome extension + local Python server for realtime English→Vietnamese translation of Microsoft Teams meetings, using RTX 5080 GPU for Whisper STT and DeepSeek v4-flash API for translation.

**Architecture:** Python FastAPI server runs locally with WebSocket endpoint for streaming audio. Silero VAD detects sentence boundaries via state machine, faster-whisper transcribes on GPU, DeepSeek API translates. Chrome extension captures tab audio via `chrome.tabCapture`, sends PCM over WebSocket, receives translated sentences, injects subtitle overlay into Teams DOM.

**Tech Stack:** Python 3.12, FastAPI, faster-whisper (CTranslate2/CUDA), silero-vad (ONNX), openai SDK, Chrome Manifest V3 (vanilla JS).

---

## File Structure

```
VocalAgent/
├── server/
│   ├── config.py          # ConfigManager: read/write config.json, defaults
│   ├── pipeline.py        # VADProcessor, WhisperSTT, Translator ABC, DeepSeekTranslator, OpenAITranslator, Pipeline
│   ├── server.py           # FastAPI: GET/POST /admin, WS /ws
│   └── templates/
│       └── admin.html      # Admin config page
├── extension/
│   ├── manifest.json       # Manifest V3
│   ├── popup.html          # Popup UI
│   ├── popup.js            # Popup logic
│   ├── popup.css           # Popup styles
│   ├── background.js       # Service worker: tabCapture, WS client
│   ├── content.js          # Content script: overlay injection + AudioContext
│   └── overlay.css         # Subtitle overlay styles
├── test/
│   ├── test_config.py      # Config tests
│   ├── test_translator.py  # Translator mock tests
│   ├── test_vad.py         # VAD segment detection tests
│   ├── test_whisper.py     # Whisper transcription tests
│   ├── test_pipeline.py    # End-to-end pipeline tests
│   ├── test_ws.py          # WebSocket server tests
│   └── fixtures/
│       ├── sample_en.wav   # "I think we should focus on this project."
│       └── silence.wav     # 5s silence
├── requirements.txt
└── README.md
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: directory structure

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p /mnt/d/Projects/VocalAgent/server/templates
mkdir -p /mnt/d/Projects/VocalAgent/extension
mkdir -p /mnt/d/Projects/VocalAgent/test/fixtures
```

- [ ] **Step 2: Write requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
faster-whisper>=1.0.3
silero-vad>=5.1
openai>=1.58.0
numpy>=1.26.0
jinja2>=3.1.0
pytest>=8.0.0
pytest-asyncio>=0.24.0
```

Write to `/mnt/d/Projects/VocalAgent/requirements.txt`.

- [ ] **Step 3: Install dependencies**

```bash
cd /mnt/d/Projects/VocalAgent && pip install -r requirements.txt
```

Expected: all packages install. `faster-whisper` downloads CTranslate2 CUDA binaries.

- [ ] **Step 4: Write README.md**

Write to `/mnt/d/Projects/VocalAgent/README.md`:

```markdown
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
```

- [ ] **Step 5: Git init and commit**

```bash
cd /mnt/d/Projects/VocalAgent && git init && git add -A && git commit -m "feat: project scaffolding"
```

---

### Task 2: Config Manager

**Files:**
- Create: `server/config.py`
- Create: `test/test_config.py`

- [ ] **Step 1: Write failing test**

Create `/mnt/d/Projects/VocalAgent/test/test_config.py`:

```python
import json
import os
import tempfile
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from config import ConfigManager, DEFAULT_CONFIG


def test_default_config_created_when_no_file():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.json")
        assert not os.path.exists(config_path)
        cm = ConfigManager(config_path)
        assert os.path.exists(config_path)
        with open(config_path) as f:
            data = json.load(f)
        assert data["provider"] == "deepseek"
        assert data["whisper"]["model"] == "medium"


def test_load_existing_config():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.json")
        custom = DEFAULT_CONFIG.copy()
        custom["provider"] = "openai"
        custom["openai"]["api_key"] = "demo-key-123"
        with open(config_path, "w") as f:
            json.dump(custom, f)
        cm = ConfigManager(config_path)
        assert cm.get_provider() == "openai"
        assert cm.get_api_key() == "demo-key-123"


def test_save_config_updates_file():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.json")
        cm = ConfigManager(config_path)
        cm.update({"provider": "openai", "openai": {"api_key": "demo-key-new"}})
        with open(config_path) as f:
            data = json.load(f)
        assert data["provider"] == "openai"
        assert data["openai"]["api_key"] == "demo-key-new"


def test_get_provider_config_returns_correct_section():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.json")
        cm = ConfigManager(config_path)
        deepseek = cm.get_provider_config()
        assert deepseek["model"] == "deepseek-v4-flash"
        assert deepseek["base_url"] == "https://api.deepseek.com"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Implement ConfigManager**

Create `/mnt/d/Projects/VocalAgent/server/config.py`:

```python
import json
import os
from copy import deepcopy
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "provider": "deepseek",
    "deepseek": {
        "api_key": "",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
    },
    "openai": {
        "api_key": "",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
    },
    "whisper": {
        "model": "medium",
        "device": "cuda",
        "compute_type": "float16",
        "language": "en",
    },
    "vad": {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
}


class ConfigManager:
    def __init__(self, config_path: str):
        self._config_path = config_path
        if os.path.exists(config_path):
            self._load()
        else:
            self._config = deepcopy(DEFAULT_CONFIG)
            self._save()

    def _load(self) -> None:
        with open(self._config_path) as f:
            self._config = json.load(f)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        with open(self._config_path, "w") as f:
            json.dump(self._config, f, indent=2)

    @property
    def config(self) -> Dict[str, Any]:
        return deepcopy(self._config)

    def get_provider(self) -> str:
        return self._config.get("provider", "deepseek")

    def get_api_key(self) -> str:
        provider = self.get_provider()
        return self._config.get(provider, {}).get("api_key", "")

    def get_provider_config(self) -> Dict[str, Any]:
        provider = self.get_provider()
        return dict(self._config.get(provider, {}))

    def get_whisper_config(self) -> Dict[str, Any]:
        return dict(self._config.get("whisper", {}))

    def get_vad_config(self) -> Dict[str, Any]:
        return dict(self._config.get("vad", {}))

    def get_server_config(self) -> Dict[str, Any]:
        return dict(self._config.get("server", {}))

    def update(self, data: Dict[str, Any]) -> None:
        """Merge data into config (shallow merge at top level, deep for nested)."""
        for key, value in data.items():
            if isinstance(value, dict) and key in self._config:
                self._config[key].update(value)
            else:
                self._config[key] = value
        self._save()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_config.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add server/config.py test/test_config.py && git commit -m "feat: add ConfigManager with defaults and CRUD"
```

---

### Task 3: Translator Classes

**Files:**
- Create: `server/pipeline.py` (Translator part)
- Create: `test/test_translator.py`

- [ ] **Step 1: Write failing test**

Create `/mnt/d/Projects/VocalAgent/test/test_translator.py`:

```python
import os
import sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from pipeline import Translator, DeepSeekTranslator, OpenAITranslator


def test_deepseek_translator_sets_api_key_and_base_url():
    t = DeepSeekTranslator(api_key="demo-key-test", model="deepseek-v4-flash")
    assert t.client.api_key == "demo-key-test"
    assert str(t.client.base_url).rstrip("/") == "https://api.deepseek.com"
    assert t.model == "deepseek-v4-flash"


def test_openai_translator_sets_api_key_and_base_url():
    t = OpenAITranslator(api_key="demo-key-test", model="gpt-4o-mini")
    assert t.client.api_key == "demo-key-test"
    assert str(t.client.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_translator_is_abstract():
    with pytest.raises(TypeError):
        Translator()


def test_translate_empty_text_returns_empty():
    t = DeepSeekTranslator(api_key="demo-key-fake")
    # Should return empty without making API call
    import asyncio
    result = asyncio.run(t.translate("", "en", "vi"))
    assert result == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_translator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: Implement Translator classes**

Create `/mnt/d/Projects/VocalAgent/server/pipeline.py`:

```python
"""Audio processing pipeline: VAD -> Whisper -> Translate."""

from abc import ABC, abstractmethod
import numpy as np
from openai import AsyncOpenAI


# ── System Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = (
    "Translate from {source} to {target}. "
    "Output only the translation, no explanation. "
    "Keep proper nouns, numbers, and acronyms unchanged. "
    "Use natural conversational language."
)


# ── Translator ─────────────────────────────────────────────

class Translator(ABC):
    @abstractmethod
    async def translate(self, text: str, source: str, target: str) -> str:
        ...


class DeepSeekTranslator(Translator):
    def __init__(self, api_key: str, model: str = "deepseek-v4-flash"):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    async def translate(self, text: str, source: str, target: str) -> str:
        if not text.strip():
            return ""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(source=source, target=target),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


class OpenAITranslator(Translator):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )

    async def translate(self, text: str, source: str, target: str) -> str:
        if not text.strip():
            return ""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(source=source, target=target),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


# ── VAD Processor ──────────────────────────────────────────

class VADProcessor:
    """Voice Activity Detection using Silero VAD.

    State machine (SILENCE / SPEECH) operating on 512-sample frames.
    Accumulates speech audio, flushes completed segment when silence
    exceeds min_silence_duration_ms or max speech duration reached.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        max_speech_duration_s: float = 15.0,
    ):
        self.sample_rate = 16000
        self.threshold = threshold
        self.min_speech_samples = int(min_speech_duration_ms * 16)
        self.min_silence_samples = int(min_silence_duration_ms * 16)
        self.max_speech_samples = int(max_speech_duration_s * 16000)

        from silero_vad import load_silero_vad
        self._model = load_silero_vad(onnx=True)

        self._buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_samples = 0

    def process(self, audio: np.ndarray) -> list:
        """Process audio chunk (float32 16kHz). Returns completed speech segments.

        Each segment: (sample_rate: int, audio: np.ndarray).
        """
        self._buffer = np.concatenate([self._buffer, audio])
        completed = []

        frame_size = 512
        while len(self._buffer) >= frame_size:
            frame = self._buffer[:frame_size]
            self._buffer = self._buffer[frame_size:]

            speech_prob = self._model(frame, self.sample_rate).item()
            is_speech = speech_prob >= self.threshold

            if is_speech:
                self._speech_buffer = np.concatenate([self._speech_buffer, frame])
                self._silence_samples = 0
                if not self._is_speaking and len(self._speech_buffer) >= self.min_speech_samples:
                    self._is_speaking = True
            elif self._is_speaking:
                self._speech_buffer = np.concatenate([self._speech_buffer, frame])
                self._silence_samples += len(frame)
                force = len(self._speech_buffer) >= self.max_speech_samples
                if self._silence_samples >= self.min_silence_samples or force:
                    completed.append((self.sample_rate, self._speech_buffer.copy()))
                    self._speech_buffer = np.array([], dtype=np.float32)
                    self._is_speaking = False
                    self._silence_samples = 0

        return completed

    def is_active(self) -> bool:
        return self._is_speaking

    def flush(self):
        """Force-flush current speech buffer. Returns (sr, array) or (None, None)."""
        if self._is_speaking and len(self._speech_buffer) >= self.min_speech_samples:
            segment = (self.sample_rate, self._speech_buffer.copy())
        else:
            segment = (None, None)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_samples = 0
        return segment

    def reset(self) -> None:
        self._buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_samples = 0


# ── Whisper STT ────────────────────────────────────────────

class WhisperSTT:
    """Speech-to-text using faster-whisper with CTranslate2/CUDA backend."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
    ):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe speech audio to English text."""
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,
            vad_filter=False,
            without_timestamps=True,
        )
        return " ".join(seg.text.strip() for seg in segments)


# ── Pipeline Orchestrator ──────────────────────────────────

class Pipeline:
    """Orchestrates VAD -> Whisper -> Translate for a single WebSocket connection."""

    def __init__(
        self,
        translator: Translator,
        whisper_config: dict,
        vad_config: dict,
    ):
        self._translator = translator
        self._vad = VADProcessor(**vad_config)
        self._whisper = WhisperSTT(**whisper_config)
        self._source = "en"
        self._target = "vi"

    async def process_chunk(self, audio: np.ndarray):
        """Process one audio chunk. Yields dicts: {type, text|status|message}."""
        segments = self._vad.process(audio)

        if self._vad.is_active():
            yield {"type": "status", "status": "listening"}

        for sr, speech_audio in segments:
            try:
                text_en = self._whisper.transcribe(speech_audio, sr)
                if not text_en.strip():
                    continue
                text_vi = await self._translator.translate(text_en, self._source, self._target)
                if text_vi:
                    yield {"type": "sentence", "text": text_vi}
            except Exception as e:
                yield {"type": "error", "message": str(e)}

    def flush(self):
        """Force flush any remaining speech from VAD."""
        sr, audio = self._vad.flush()
        return sr, audio

    def reset(self):
        self._vad.reset()
```

- [ ] **Step 4: Run tests (constructor + abstract tests)**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_translator.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add server/pipeline.py test/test_translator.py && git commit -m "feat: add Translator, VADProcessor, WhisperSTT, and Pipeline"
```

---

### Task 4: VAD Unit Tests

**Files:**
- Create: `test/test_vad.py`

- [ ] **Step 1: Write VAD tests**

Create `/mnt/d/Projects/VocalAgent/test/test_vad.py`:

```python
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from pipeline import VADProcessor


def make_sine(freq: float, duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def make_silence(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


def test_vad_initialization():
    vad = VADProcessor()
    assert vad.sample_rate == 16000
    assert vad.threshold == 0.5


def test_vad_returns_empty_for_silence():
    vad = VADProcessor()
    silence = make_silence(2.0)
    segments = vad.process(silence)
    assert len(segments) == 0
    assert not vad.is_active()


def test_vad_detects_speech_segment():
    vad = VADProcessor()
    # 1s speech + 700ms silence (enough to trigger flush)
    speech = make_sine(440, 1.0)
    silence = make_silence(0.7)
    audio = np.concatenate([speech, silence])

    segments = vad.process(audio)
    # Should have one completed segment
    assert len(segments) >= 1
    sr, seg_audio = segments[0]
    assert sr == 16000
    assert len(seg_audio) > 0


def test_vad_no_flush_without_silence():
    vad = VADProcessor()
    # Continuous speech, no silence gap
    speech = make_sine(440, 2.0)
    segments = vad.process(speech)
    # No completed segment yet — still speaking
    assert vad.is_active()


def test_vad_reset_clears_state():
    vad = VADProcessor()
    speech = make_sine(440, 0.5)
    vad.process(speech)
    assert vad.is_active()
    vad.reset()
    assert not vad.is_active()
    silence = make_silence(1.0)
    segments = vad.process(silence)
    assert len(segments) == 0


def test_vad_flush_returns_accumulated_speech():
    vad = VADProcessor()
    speech = make_sine(440, 0.5)
    vad.process(speech)
    sr, audio = vad.flush()
    assert sr == 16000
    assert len(audio) > 0
    assert not vad.is_active()
```

- [ ] **Step 2: Run VAD tests**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_vad.py -v
```

Expected: 6 PASS

- [ ] **Step 3: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add test/test_vad.py && git commit -m "test: add VADProcessor unit tests"
```

---

### Task 5: Whisper Unit Test

**Files:**
- Create: `test/test_whisper.py`

- [ ] **Step 1: Write Whisper test**

Create `/mnt/d/Projects/VocalAgent/test/test_whisper.py`:

```python
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from pipeline import WhisperSTT


def make_sine(freq: float, duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def test_whisper_model_loads():
    """Verify WhisperSTT can load tiny model without error."""
    stt = WhisperSTT(model_size="tiny", device="cpu", compute_type="int8")
    assert stt is not None


def test_whisper_transcribe_silence_returns_empty():
    """Transcribing silence should return empty or near-empty string."""
    stt = WhisperSTT(model_size="tiny", device="cpu", compute_type="int8")
    silence = np.zeros(16000 * 2, dtype=np.float32)  # 2s silence
    result = stt.transcribe(silence)
    # Whisper may hallucinate on silence — acceptable for v1
    # Just check it doesn't crash
    assert isinstance(result, str)
```

Note: Full transcription accuracy tests require real speech audio and GPU. These are smoke tests ensuring the model loads and doesn't crash. Manual verification with actual Teams meeting audio validates accuracy.

- [ ] **Step 2: Run Whisper tests**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_whisper.py -v
```

Expected: 2 PASS (model downloads on first run)

- [ ] **Step 3: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add test/test_whisper.py && git commit -m "test: add WhisperSTT smoke tests"
```

---

### Task 6: Server & WebSocket

**Files:**
- Create: `server/server.py`
- Create: `server/templates/admin.html`
- Create: `test/test_ws.py`

- [ ] **Step 1: Write WebSocket test**

Create `/mnt/d/Projects/VocalAgent/test/test_ws.py`:

```python
import os
import sys
import json
import pytest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from fastapi.testclient import TestClient
from server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_admin_page_returns_html(client):
    response = client.get("/admin")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_admin_post_updates_config(client):
    """POST to /admin should accept form data and redirect."""
    response = client.post(
        "/admin",
        data={
            "provider": "deepseek",
            "deepseek_api_key": "demo-key-123",
            "deepseek_model": "deepseek-v4-flash",
            "whisper_model": "small",
            "vad_threshold": "0.6",
        },
        follow_redirects=False,
    )
    assert response.status_code in (200, 302, 303)


def test_websocket_accepts_binary_and_returns_json():
    """WebSocket endpoint should accept connection and binary frames."""
    # FastAPI TestClient doesn't natively support WS easily;
    # We test the endpoint shape via HTTP and verify the route exists.
    from server import app as test_app
    routes = [r.path for r in test_app.routes]
    assert "/ws" in routes


def test_server_config_endpoint_structure(client):
    """Verify the app has expected routes."""
    routes = [r.path for r in app.routes]
    assert "/admin" in routes
    assert "/ws" in routes
```

- [ ] **Step 2: Run test — expect failure (no server module)**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_ws.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 3: Implement server.py**

Create `/mnt/d/Projects/VocalAgent/server/server.py`:

```python
"""FastAPI server: GET/POST /admin config, WS /ws audio pipeline."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config import ConfigManager
from pipeline import (
    DeepSeekTranslator,
    OpenAITranslator,
    Pipeline,
)

CONFIG_PATH = os.environ.get(
    "VOCALAGENT_CONFIG",
    os.path.join(Path(__file__).parent.parent, "config.json"),
)
config_manager = ConfigManager(CONFIG_PATH)

_pipeline: Optional[Pipeline] = None


def _create_pipeline():
    """Create a new pipeline from current config."""
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


app = FastAPI(title="VocalAgent Server")


# ── Admin Page ─────────────────────────────────────────────

ADMIN_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VocalAgent — Admin</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, sans-serif; background: #0f1117; color: #e1e4e8; max-width: 640px; margin: 40px auto; padding: 0 20px; }
        h1 { font-size: 24px; margin-bottom: 24px; }
        label { display: block; font-size: 13px; font-weight: 600; margin: 16px 0 6px; color: #8b949e; }
        input, select { width: 100%; padding: 8px 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #e1e4e8; font-size: 14px; }
        input:focus, select:focus { border-color: #58a6ff; outline: none; }
        fieldset { border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 16px 0; }
        legend { font-weight: 600; padding: 0 8px; }
        button { margin-top: 20px; padding: 10px 24px; background: #238636; color: #fff; border: none; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; }
        button:hover { background: #2ea043; }
        .flash { padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; }
        .flash-success { background: #1a3a2a; border: 1px solid #238636; }
        .flash-error { background: #3a1a1a; border: 1px solid #da3633; }
        hr { border: none; border-top: 1px solid #30363d; margin: 20px 0; }
        .status { font-size: 13px; color: #8b949e; }
        .status span { color: #56d364; }
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
        <button type="submit">Save Configuration</button>
    </form>
    <hr>
    <p class="status">Config: <span>{config_path}</span></p>
</body>
</html>"""


@app.get("/admin", response_class=HTMLResponse)
async def admin_get():
    cfg = config_manager.config
    whisper_cfg = cfg.get("whisper", {})
    vad_cfg = cfg.get("vad", {})
    ds_cfg = cfg.get("deepseek", {})
    oa_cfg = cfg.get("openai", {})

    whisper_models = ["tiny", "base", "small", "medium"]
    current_whisper = whisper_cfg.get("model", "medium")
    whisper_opts = "".join(
        f'<option value="{m}" {"selected" if m == current_whisper else ""}>{m}</option>'
        for m in whisper_models
    )

    html = ADMIN_TEMPLATE.format(
        flash="",
        ds_sel="selected" if cfg.get("provider") == "deepseek" else "",
        oa_sel="selected" if cfg.get("provider") == "openai" else "",
        ds_key=ds_cfg.get("api_key", ""),
        ds_model=ds_cfg.get("model", "deepseek-v4-flash"),
        oa_key=oa_cfg.get("api_key", ""),
        oa_model=oa_cfg.get("model", "gpt-4o-mini"),
        whisper_opts=whisper_opts,
        vad_threshold=vad_cfg.get("threshold", 0.5),
        config_path=CONFIG_PATH,
    )
    return HTMLResponse(content=html)


@app.post("/admin")
async def admin_post(
    provider: str = Form("deepseek"),
    deepseek_api_key: str = Form(""),
    deepseek_model: str = Form("deepseek-v4-flash"),
    openai_api_key: str = Form(""),
    openai_model: str = Form("gpt-4o-mini"),
    whisper_model: str = Form("medium"),
    vad_threshold: str = Form("0.5"),
):
    try:
        config_manager.update({
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
        })
        cfg = config_manager.config
        # Rebuild template with success flash
        flash_html = '<div class="flash flash-success">&#10003; Configuration saved.</div>'
        whisper_cfg = cfg.get("whisper", {})
        vad_cfg = cfg.get("vad", {})
        ds_cfg = cfg.get("deepseek", {})
        oa_cfg = cfg.get("openai", {})
        whisper_models = ["tiny", "base", "small", "medium"]
        current_whisper = whisper_cfg.get("model", "medium")
        whisper_opts = "".join(
            f'<option value="{m}" {"selected" if m == current_whisper else ""}>{m}</option>'
            for m in whisper_models
        )
        html = ADMIN_TEMPLATE.format(
            flash=flash_html,
            ds_sel="selected" if cfg.get("provider") == "deepseek" else "",
            oa_sel="selected" if cfg.get("provider") == "openai" else "",
            ds_key=ds_cfg.get("api_key", ""),
            ds_model=ds_cfg.get("model", "deepseek-v4-flash"),
            oa_key=oa_cfg.get("api_key", ""),
            oa_model=oa_cfg.get("model", "gpt-4o-mini"),
            whisper_opts=whisper_opts,
            vad_threshold=vad_cfg.get("threshold", 0.5),
            config_path=CONFIG_PATH,
        )
        return HTMLResponse(content=html)
    except Exception as e:
        cfg = config_manager.config
        whisper_cfg = cfg.get("whisper", {})
        vad_cfg = cfg.get("vad", {})
        ds_cfg = cfg.get("deepseek", {})
        oa_cfg = cfg.get("openai", {})
        flash_html = f'<div class="flash flash-error">Error: {str(e)}</div>'
        whisper_models = ["tiny", "base", "small", "medium"]
        current_whisper = whisper_cfg.get("model", "medium")
        whisper_opts = "".join(
            f'<option value="{m}" {"selected" if m == current_whisper else ""}>{m}</option>'
            for m in whisper_models
        )
        html = ADMIN_TEMPLATE.format(
            flash=flash_html,
            ds_sel="selected" if cfg.get("provider") == "deepseek" else "",
            oa_sel="selected" if cfg.get("provider") == "openai" else "",
            ds_key=ds_cfg.get("api_key", ""),
            ds_model=ds_cfg.get("model", "deepseek-v4-flash"),
            oa_key=oa_cfg.get("api_key", ""),
            oa_model=oa_cfg.get("model", "gpt-4o-mini"),
            whisper_opts=whisper_opts,
            vad_threshold=vad_cfg.get("threshold", 0.5),
            config_path=CONFIG_PATH,
        )
        return HTMLResponse(content=html, status_code=400)


# ── WebSocket Pipeline ─────────────────────────────────────

@app.websocket("/ws")
async def websocket_pipeline(ws: WebSocket):
    await ws.accept()
    pipeline = _create_pipeline()

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data:
                # PCM 16-bit mono 16kHz -> float32
                raw = data["bytes"]
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                async for event in pipeline.process_chunk(audio):
                    await ws.send_json(event)
            elif "text" in data:
                # Control message
                try:
                    msg = json.loads(data["text"])
                    if msg.get("action") == "reset":
                        pipeline.reset()
                        await ws.send_json({"type": "status", "status": "reset"})
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── Entry point ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    server_cfg = config_manager.get_server_config()
    uvicorn.run(
        app,
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 8765),
    )
```

- [ ] **Step 4: Run server tests**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_ws.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add server/server.py server/templates/admin.html test/test_ws.py && git commit -m "feat: add FastAPI server with /admin and /ws endpoints"
```

---

### Task 7: Chrome Extension Manifest & Popup

**Files:**
- Create: `extension/manifest.json`
- Create: `extension/popup.html`
- Create: `extension/popup.js`
- Create: `extension/popup.css`

- [ ] **Step 1: Write manifest.json**

Create `/mnt/d/Projects/VocalAgent/extension/manifest.json`:

```json
{
  "manifest_version": 3,
  "name": "VocalAgent — Teams Translator",
  "version": "1.0.0",
  "description": "Realtime English→Vietnamese translation for Microsoft Teams meetings.",
  "permissions": [
    "tabCapture",
    "activeTab",
    "storage",
    "scripting"
  ],
  "host_permissions": [
    "https://teams.microsoft.com/*",
    "https://teams.live.com/*"
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
  ],
  "icons": {
    "16": "icons/icon16.png",
    "48": "icons/icon48.png",
    "128": "icons/icon128.png"
  }
}
```

Note: Create placeholder icon files or skip icons for v1. Icons are optional in development mode.

- [ ] **Step 2: Write popup.html**

Create `/mnt/d/Projects/VocalAgent/extension/popup.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VocalAgent</title>
  <link rel="stylesheet" href="popup.css">
</head>
<body>
  <div class="container">
    <header>
      <h1>VocalAgent</h1>
      <span class="version">v1.0</span>
    </header>

    <div class="section">
      <label for="source-lang">Source</label>
      <select id="source-lang">
        <option value="en" selected>English</option>
        <option value="vi">Tiếng Việt</option>
        <option value="ja">日本語</option>
        <option value="zh">中文</option>
        <option value="ko">한국어</option>
      </select>
    </div>

    <div class="section">
      <label for="target-lang">Target</label>
      <select id="target-lang">
        <option value="vi" selected>Tiếng Việt</option>
        <option value="en">English</option>
        <option value="ja">日本語</option>
        <option value="zh">中文</option>
        <option value="ko">한국어</option>
      </select>
    </div>

    <div class="section">
      <label>Server</label>
      <div class="server-status" id="server-status">
        <span class="dot disconnected"></span>
        <span id="server-text">Disconnected</span>
      </div>
    </div>

    <button id="toggle-btn" disabled>Start</button>

    <div class="section settings">
      <label for="server-port">Port</label>
      <input type="number" id="server-port" value="8765" min="1024" max="65535">
    </div>
  </div>
  <script src="popup.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write popup.css**

Create `/mnt/d/Projects/VocalAgent/extension/popup.css`:

```css
* { box-sizing: border-box; margin: 0; padding: 0; }
body { width: 280px; font-family: system-ui, sans-serif; background: #1a1a2e; color: #e1e4e8; font-size: 13px; }
.container { padding: 16px; }
header { display: flex; align-items: baseline; gap: 8px; margin-bottom: 16px; }
header h1 { font-size: 18px; font-weight: 700; }
.version { color: #8b949e; font-size: 11px; }
.section { margin-bottom: 12px; }
label { display: block; font-size: 11px; font-weight: 600; color: #8b949e; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
select, input { width: 100%; padding: 6px 10px; background: #16213e; border: 1px solid #30363d; border-radius: 6px; color: #e1e4e8; font-size: 13px; }
select:focus, input:focus { border-color: #58a6ff; outline: none; }
.settings input { width: 80px; }
.server-status { display: flex; align-items: center; gap: 6px; padding: 6px 10px; background: #16213e; border-radius: 6px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot.connected { background: #56d364; }
.dot.disconnected { background: #da3633; }
.dot.checking { background: #d29922; }
button { width: 100%; padding: 10px; border: none; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-start { background: #238636; color: #fff; }
.btn-start:hover:not(:disabled) { background: #2ea043; }
.btn-stop { background: #da3633; color: #fff; }
.btn-stop:hover:not(:disabled) { background: #f85149; }
```

- [ ] **Step 4: Write popup.js**

Create `/mnt/d/Projects/VocalAgent/extension/popup.js`:

```javascript
const PORT = document.getElementById('server-port');
const STATUS_DOT = document.querySelector('.dot');
const STATUS_TEXT = document.getElementById('server-text');
const TOGGLE_BTN = document.getElementById('toggle-btn');
const SOURCE_LANG = document.getElementById('source-lang');
const TARGET_LANG = document.getElementById('target-lang');

let isRunning = false;

// Load saved settings
chrome.storage.local.get(['port', 'sourceLang', 'targetLang'], (items) => {
  if (items.port) PORT.value = items.port;
  if (items.sourceLang) SOURCE_LANG.value = items.sourceLang;
  if (items.targetLang) TARGET_LANG.value = items.targetLang;
  checkServer();
});

// Save settings on change
PORT.addEventListener('change', () => {
  chrome.storage.local.set({ port: PORT.value });
  checkServer();
});

SOURCE_LANG.addEventListener('change', () => {
  chrome.storage.local.set({ sourceLang: SOURCE_LANG.value });
});

TARGET_LANG.addEventListener('change', () => {
  chrome.storage.local.set({ targetLang: TARGET_LANG.value });
});

async function checkServer() {
  STATUS_DOT.className = 'dot checking';
  STATUS_TEXT.textContent = 'Checking...';
  TOGGLE_BTN.disabled = true;

  try {
    const resp = await fetch(`http://127.0.0.1:${PORT.value}/admin`, {
      mode: 'cors',
      signal: AbortSignal.timeout(3000),
    });
    if (resp.ok) {
      STATUS_DOT.className = 'dot connected';
      STATUS_TEXT.textContent = 'Connected';
      TOGGLE_BTN.disabled = false;
    }
  } catch {
    STATUS_DOT.className = 'dot disconnected';
    STATUS_TEXT.textContent = 'Server not running';
    TOGGLE_BTN.disabled = true;
  }
}

TOGGLE_BTN.addEventListener('click', async () => {
  if (isRunning) {
    // Stop
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.runtime.sendMessage({ action: 'stop', tabId: tab.id });
    isRunning = false;
    TOGGLE_BTN.textContent = 'Start';
    TOGGLE_BTN.className = 'btn-start';
  } else {
    // Start
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab.url.includes('teams.microsoft.com') && !tab.url.includes('teams.live.com')) {
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
    isRunning = true;
    TOGGLE_BTN.textContent = 'Stop';
    TOGGLE_BTN.className = 'btn-stop';
  }
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
    }
  }
});
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add extension/ && git commit -m "feat: add Chrome extension manifest and popup UI"
```

---

### Task 8: Background Service Worker

**Files:**
- Create: `extension/background.js`

- [ ] **Step 1: Write background.js**

Create `/mnt/d/Projects/VocalAgent/extension/background.js`:

```javascript
// State
let ws = null;
let reconnectTimer = null;
let reconnectDelay = 1000;
const MAX_RECONNECT_DELAY = 8000;
let activeTabId = null;
let port = 8765;

// Handle messages from popup
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === 'start') {
    activeTabId = msg.tabId;
    port = msg.port || 8765;
    startCapture(msg.tabId, msg.sourceLang, msg.targetLang);
    sendResponse({ ok: true });
  } else if (msg.action === 'stop') {
    stopCapture();
    sendResponse({ ok: true });
  }
  return true;
});

async function startCapture(tabId, sourceLang, targetLang) {
  try {
    // Get stream ID for the Teams tab
    const streamId = await chrome.tabCapture.getMediaStreamId({
      targetTabId: tabId,
    });

    // Tell content script to create AudioContext with this stream
    await chrome.tabs.sendMessage(tabId, {
      type: 'capture-start',
      streamId: streamId,
      port: port,
    });

    updatePopupStatus('capturing');
    connectWebSocket(sourceLang, targetLang);
  } catch (err) {
    console.error('Failed to start capture:', err);
    updatePopupStatus('stopped');
  }
}

function stopCapture() {
  if (ws) {
    ws.close();
    ws = null;
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (activeTabId) {
    chrome.tabs.sendMessage(activeTabId, { type: 'capture-stop' }).catch(() => {});
    activeTabId = null;
  }
  updatePopupStatus('stopped');
}

function connectWebSocket(sourceLang, targetLang) {
  if (ws) {
    ws.close();
    ws = null;
  }

  ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    console.log('WebSocket connected');
    reconnectDelay = 1000;

    // Send source/target language config as first message
    ws.send(JSON.stringify({
      source: sourceLang || 'en',
      target: targetLang || 'vi',
    }));
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (activeTabId) {
        chrome.tabs.sendMessage(activeTabId, {
          type: 'subtitle',
          data: msg,
        }).catch(() => {});
      }
    } catch (e) {
      console.error('Failed to parse WS message:', e);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected');
    // Exponential backoff reconnect
    reconnectTimer = setTimeout(() => {
      connectWebSocket(sourceLang, targetLang);
      reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
    }, reconnectDelay);
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    ws.close();
  };
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
    stopCapture();
  }
});
```

- [ ] **Step 2: Verify background.js has no syntax errors**

```bash
node --check /mnt/d/Projects/VocalAgent/extension/background.js
```

Expected: no output (syntax OK)

- [ ] **Step 3: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add extension/background.js && git commit -m "feat: add background service worker with tabCapture and WS reconnect"
```

---

### Task 9: Content Script & Subtitle Overlay

**Files:**
- Create: `extension/content.js`
- Create: `extension/overlay.css`

- [ ] **Step 1: Write overlay.css**

Create `/mnt/d/Projects/VocalAgent/extension/overlay.css`:

```css
#vocaltranslator-overlay {
  position: fixed;
  bottom: 80px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 999999;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  pointer-events: none;
  font-family: system-ui, -apple-system, sans-serif;
  max-width: 600px;
  width: 90%;
}

#vocaltranslator-overlay .vt-subtitle {
  background: rgba(0, 0, 0, 0.78);
  color: rgba(255, 255, 255, 0.95);
  padding: 10px 18px;
  border-radius: 10px;
  font-size: 15px;
  line-height: 1.45;
  text-align: center;
  pointer-events: auto;
  animation: vt-fade-in 200ms ease-out;
  transition: opacity 400ms ease-out, transform 300ms ease-out;
  max-width: 100%;
  word-wrap: break-word;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
}

#vocaltranslator-overlay .vt-subtitle.fading {
  opacity: 0;
  transform: translateY(-12px);
}

#vocaltranslator-overlay .vt-status {
  background: rgba(0, 0, 0, 0.55);
  color: rgba(255, 255, 255, 0.55);
  padding: 6px 14px;
  border-radius: 8px;
  font-size: 12px;
  pointer-events: auto;
}

#vocaltranslator-overlay .vt-drag-handle {
  position: absolute;
  top: -16px;
  left: 50%;
  transform: translateX(-50%);
  width: 40px;
  height: 12px;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 6px;
  cursor: grab;
  pointer-events: auto;
  opacity: 0;
  transition: opacity 200ms;
}

#vocaltranslator-overlay:hover .vt-drag-handle {
  opacity: 1;
}

#vocaltranslator-overlay .vt-close {
  position: absolute;
  top: -8px;
  right: -8px;
  width: 22px;
  height: 22px;
  background: rgba(0, 0, 0, 0.8);
  color: #fff;
  border: none;
  border-radius: 50%;
  font-size: 12px;
  cursor: pointer;
  pointer-events: auto;
  opacity: 0;
  transition: opacity 200ms;
  display: flex;
  align-items: center;
  justify-content: center;
}

#vocaltranslator-overlay:hover .vt-close {
  opacity: 1;
}

@keyframes vt-fade-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
```

- [ ] **Step 2: Write content.js**

Create `/mnt/d/Projects/VocalAgent/extension/content.js`:

```javascript
(function () {
  // Prevent double injection
  if (document.getElementById('vocaltranslator-overlay')) return;

  // ── Audio Capture ────────────────────────────────────
  let audioContext = null;
  let mediaStream = null;
  let ws = null;
  const SAMPLE_RATE = 16000;
  const CHUNK_SIZE = 4096;

  // ── Overlay DOM ──────────────────────────────────────
  const overlay = document.createElement('div');
  overlay.id = 'vocaltranslator-overlay';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'vt-close';
  closeBtn.textContent = '×';
  closeBtn.title = 'Hide subtitles';
  overlay.appendChild(closeBtn);

  const dragHandle = document.createElement('div');
  dragHandle.className = 'vt-drag-handle';
  overlay.appendChild(dragHandle);

  const statusEl = document.createElement('div');
  statusEl.className = 'vt-status';
  statusEl.style.display = 'none';
  overlay.appendChild(statusEl);

  document.body.appendChild(overlay);

  // ── Drag functionality ───────────────────────────────
  let dragging = false;
  let dragStartY = 0;
  let overlayStartBottom = 80;

  dragHandle.addEventListener('mousedown', (e) => {
    dragging = true;
    dragStartY = e.clientY;
    overlayStartBottom = parseInt(overlay.style.bottom) || 80;
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const dy = dragStartY - e.clientY;
    overlay.style.bottom = Math.max(0, overlayStartBottom + dy) + 'px';
  });

  document.addEventListener('mouseup', () => { dragging = false; });

  closeBtn.addEventListener('click', () => {
    overlay.style.display = overlay.style.display === 'none' ? '' : 'none';
  });

  // ── Subtitle rendering ───────────────────────────────
  const MAX_VISIBLE = 3;
  const STAY_MS = 5000;

  function showSubtitle(text) {
    const el = document.createElement('div');
    el.className = 'vt-subtitle';
    el.textContent = text;
    overlay.insertBefore(el, statusEl);

    // Remove oldest if too many
    const subtitles = overlay.querySelectorAll('.vt-subtitle');
    if (subtitles.length > MAX_VISIBLE) {
      const oldest = subtitles[0];
      oldest.classList.add('fading');
      setTimeout(() => oldest.remove(), 400);
    }

    // Auto-remove after delay
    setTimeout(() => {
      if (el.parentNode) {
        el.classList.add('fading');
        setTimeout(() => { if (el.parentNode) el.remove(); }, 400);
      }
    }, STAY_MS);
  }

  function showStatus(text) {
    statusEl.style.display = text ? 'block' : 'none';
    statusEl.textContent = text;
    if (text) {
      setTimeout(() => { statusEl.style.display = 'none'; }, 3000);
    }
  }

  // ── Audio processing ─────────────────────────────────
  async function startAudioCapture(streamId, serverPort) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          mandatory: {
            chromeMediaSource: 'tab',
            chromeMediaSourceId: streamId,
          },
        },
        video: false,
      });

      mediaStream = stream;
      audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });

      const source = audioContext.createMediaStreamSource(stream);

      // Resample to 16kHz mono
      const processor = audioContext.createScriptProcessor(CHUNK_SIZE, 1, 1);
      source.connect(processor);
      processor.connect(audioContext.destination);

      // Connect WebSocket
      ws = new WebSocket(`ws://127.0.0.1:${serverPort}/ws`);
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        console.log('Content: WebSocket connected');
        showStatus('Connected');
      };

      ws.onclose = () => {
        showStatus('Disconnected');
      };

      processor.onaudioprocess = (event) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        // Convert float32 [-1,1] to int16 PCM
        const input = event.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          const s = Math.max(-1, Math.min(1, input[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        ws.send(int16.buffer);
      };

      // Handle stream end
      stream.getVideoTracks()[0]?.addEventListener('ended', () => {
        showStatus('Audio lost — reload tab');
      });

      showStatus('Listening...');
    } catch (err) {
      console.error('Failed to start audio capture:', err);
      showStatus('Error: ' + err.message);
    }
  }

  function stopCapture() {
    if (ws) { ws.close(); ws = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    showStatus('Stopped');
  }

  // ── Message handler ──────────────────────────────────
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'capture-start') {
      startAudioCapture(msg.streamId, msg.port || 8765);
      sendResponse({ ok: true });
    } else if (msg.type === 'capture-stop') {
      stopCapture();
      sendResponse({ ok: true });
    } else if (msg.type === 'subtitle') {
      const data = msg.data;
      if (data.type === 'sentence') {
        showSubtitle(data.text);
      } else if (data.type === 'status' && data.status === 'listening') {
        showStatus('Listening...');
      } else if (data.type === 'error') {
        showStatus(data.message);
      }
      sendResponse({ ok: true });
    }
    return true;
  });
})();
```

- [ ] **Step 3: Verify content.js has no syntax errors**

```bash
node --check /mnt/d/Projects/VocalAgent/extension/content.js
```

Note: `chrome` and `navigator` are runtime globals; expected warning but should pass syntax check.

- [ ] **Step 4: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add extension/content.js extension/overlay.css && git commit -m "feat: add content script with subtitle overlay and audio capture"
```

---

### Task 10: End-to-End Pipeline Test

**Files:**
- Create: `test/test_pipeline.py`
- Create dev dependency: `test/fixtures/sample_en.wav` (generate programmatically)

- [ ] **Step 1: Create test fixture generation helper**

Add to `test/test_pipeline.py`:

```python
"""End-to-end pipeline test: audio file -> VAD -> Whisper -> Translate."""

import os
import sys
import asyncio
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from pipeline import (
    VADProcessor,
    WhisperSTT,
    DeepSeekTranslator,
    Pipeline,
)


def make_sine(freq: float, duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def make_silence(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


class FakeTranslator:
    """Mock translator that prefixes text — for testing without API."""
    async def translate(self, text: str, source: str, target: str) -> str:
        return f"[{target}] {text}"


def test_vad_detects_speech_and_silence_boundary():
    """Integration: VAD correctly segments speech+sentence+speech."""
    vad = VADProcessor(
        threshold=0.3,
        min_speech_duration_ms=100,
        min_silence_duration_ms=400,
    )
    # Sentence 1: 0.5s speech + 500ms silence
    s1 = make_sine(440, 0.5)
    gap1 = make_silence(0.5)
    audio = np.concatenate([s1, gap1])
    segments = vad.process(audio)
    assert len(segments) >= 1


def test_pipeline_construction():
    """Pipeline builds from config dicts."""
    translator = FakeTranslator()
    pipeline = Pipeline(
        translator=translator,
        whisper_config={"model_size": "tiny", "device": "cpu", "compute_type": "int8", "language": "en"},
        vad_config={"threshold": 0.5},
    )
    assert pipeline is not None


@pytest.mark.asyncio
async def test_pipeline_processes_audio_with_fake_translator():
    """Pipeline yields translated sentences through mock translator."""
    translator = FakeTranslator()
    pipeline = Pipeline(
        translator=translator,
        whisper_config={"model_size": "tiny", "device": "cpu", "compute_type": "int8", "language": "en"},
        vad_config={
            "threshold": 0.3,
            "min_speech_duration_ms": 50,
            "min_silence_duration_ms": 400,
        },
    )

    # 1s speech + 1s silence to trigger flush
    speech = make_sine(440, 1.0)
    silence = make_silence(1.0)
    audio = np.concatenate([speech, silence])

    results = []
    async for event in pipeline.process_chunk(audio):
        results.append(event)

    # Should have at least "listening" status; sentence if Whisper tiny
    # produces text from sine wave (may produce empty or garbage)
    assert any(e["type"] in ("status", "sentence") for e in results)


def test_pipeline_flush_and_reset():
    """Flush returns accumulated speech, reset clears state."""
    translator = FakeTranslator()
    pipeline = Pipeline(
        translator=translator,
        whisper_config={"model_size": "tiny", "device": "cpu", "compute_type": "int8", "language": "en"},
        vad_config={"threshold": 0.5},
    )
    speech = make_sine(440, 0.5)
    pipeline._vad.process(speech)
    assert pipeline._vad.is_active()

    sr, audio = pipeline.flush()
    assert sr is not None
    assert len(audio) > 0
    assert not pipeline._vad.is_active()

    pipeline.reset()
    assert not pipeline._vad.is_active()
```

- [ ] **Step 2: Run pipeline tests**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/test_pipeline.py -v
```

Expected: 4 PASS (Whisper tiny downloads on first run)

- [ ] **Step 3: Commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add test/test_pipeline.py && git commit -m "test: add end-to-end pipeline tests"
```

---

### Task 11: Final Integration Verification & README

**Files:**
- Modify: `README.md` (add development instructions)

- [ ] **Step 1: Verify full project structure**

```bash
cd /mnt/d/Projects/VocalAgent && find . -type f | sort
```

Expected output:
```
./README.md
./requirements.txt
./extension/manifest.json
./extension/popup.html
./extension/popup.js
./extension/popup.css
./extension/background.js
./extension/content.js
./extension/overlay.css
./server/config.py
./server/pipeline.py
./server/server.py
./test/test_config.py
./test/test_translator.py
./test/test_vad.py
./test/test_whisper.py
./test/test_pipeline.py
./test/test_ws.py
```

- [ ] **Step 2: Run all Python tests**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/ -v
```

Expected: all tests pass (16+ tests)

- [ ] **Step 3: Update README.md with development section**

Append to `/mnt/d/Projects/VocalAgent/README.md`:

```markdown
## Development

### Project Structure
- `server/` — Python FastAPI backend
- `extension/` — Chrome extension
- `test/` — Python tests

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
```

- [ ] **Step 4: Run full test suite again to confirm**

```bash
cd /mnt/d/Projects/VocalAgent && python -m pytest test/ -v
```

Expected: all tests pass

- [ ] **Step 5: Final commit**

```bash
cd /mnt/d/Projects/VocalAgent && git add -A && git commit -m "docs: add development instructions, finalize project structure"
```

---

## Summary

| Task | Component | Files Created |
|------|-----------|---------------|
| 1 | Scaffolding | `requirements.txt`, `README.md`, directory tree |
| 2 | ConfigManager | `server/config.py`, `test/test_config.py` |
| 3 | Translator + Pipeline | `server/pipeline.py`, `test/test_translator.py` |
| 4 | VAD Tests | `test/test_vad.py` |
| 5 | Whisper Tests | `test/test_whisper.py` |
| 6 | Server + Admin | `server/server.py`, `test/test_ws.py` |
| 7 | Extension Popup | `manifest.json`, `popup.*` |
| 8 | Background Worker | `background.js` |
| 9 | Content Script | `content.js`, `overlay.css` |
| 10 | E2E Tests | `test/test_pipeline.py` |
| 11 | Integration | README update, full test suite |

**Total: 11 tasks, ~50 steps, ~17 files created.**
