"""Audio processing pipeline: VAD -> Whisper -> Translate."""

import asyncio
import logging
import os
import sys
from abc import ABC, abstractmethod
import numpy as np
import torch
import re
import uuid
from collections import deque

from openai import AsyncOpenAI

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Suppress httpx/httpcore internal logging which can raise UnicodeEncodeError
# on non-ASCII API responses (e.g., Vietnamese translations)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── System Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = (
    "Translate from {source} to {target}. "
    "Output only the translation, no explanation. "
    "Keep proper nouns, numbers, and acronyms unchanged. "
    "Use natural conversational language."
)


def _split_sentences(text: str, max_chars: int = 180) -> list[str]:
    """Split English text into sentences. Falls back to word-boundary
    chunks for long segments without punctuation (common in Whisper output)."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in parts if s.strip()]
    if not sentences:
        return []

    # Fallback: split long segments at word boundaries so no single
    # API call exceeds token limits (Whisper often outputs run-on text).
    result = []
    for s in sentences:
        if len(s) <= max_chars:
            result.append(s)
        else:
            words = s.split()
            chunk, char_count = [], 0
            for w in words:
                chunk.append(w)
                char_count += len(w) + 1  # +1 for space
                if char_count >= max_chars:
                    result.append(" ".join(chunk))
                    chunk, char_count = [], 0
            if chunk:
                result.append(" ".join(chunk))
    return result


# ── Translator ─────────────────────────────────────────────

class Translator(ABC):
    @abstractmethod
    async def translate(self, text: str, source: str, target: str) -> str:
        ...


class OpenAICompatibleTranslator(Translator):
    """Translator for any OpenAI-compatible API (DeepSeek, OpenAI, etc.)."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 30.0):
        self.model = model
        # Validate API key is ASCII before creating client — non-ASCII keys
        # cause cryptic UnicodeEncodeError inside httpx when building HTTP headers.
        if api_key and not api_key.isascii():
            raise ValueError(
                f"API key contains non-ASCII characters ({len(api_key)} chars). "
                f"HTTP headers must be ASCII. Check your config.json or re-enter "
                f"the API key in the admin page."
            )
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        # Rolling context buffer for streaming translation (max 3 previous sentences)
        self._context: deque[str] = deque(maxlen=3)

    async def translate(self, text: str, source: str, target: str) -> str:
        if not text.strip():
            return ""
        response = None
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(source=source, target=target),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            result = response.choices[0].message.content.strip()
            return result
        except UnicodeEncodeError as e:
            # Print full traceback to identify exact source of encoding error
            import traceback
            print(f"[TL] UnicodeEncodeError caught, full traceback:")
            traceback.print_exc()
            if response is not None:
                try:
                    result = response.choices[0].message.content.strip()
                    return result
                except Exception as ex:
                    raise RuntimeError(
                        f"Translation returned but cannot access content: {ex}"
                    ) from ex
            raise RuntimeError(
                f"Translation failed — UnicodeEncodeError before response assigned: {e}"
            ) from e
        except Exception as e:
            err_msg = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"[TL] API error: {err_msg}")
            raise

    async def translate_stream(self, text: str, source: str, target: str):
        """Async generator. Yields translation tokens one at a time via LLM streaming.

        Injects previous sentences from context buffer for translation quality.
        After stream completes, stores the full translation in the context buffer.
        """
        if not text.strip():
            return

        # Build context-aware system prompt
        system_content = SYSTEM_PROMPT.format(source=source, target=target)
        if self._context:
            ctx_lines = "\n".join(f"- {c}" for c in self._context)
            system_content += (
                f"\n\nPrevious sentences for context "
                f"(do NOT translate these, use only for reference):\n{ctx_lines}"
            )

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text},
                ],
                max_tokens=512,
                temperature=0.0,
                stream=True,
            )
            full_text: list[str] = []
            async for chunk in stream:
                delta = chunk.choices[0].delta
                token = delta.content
                if token:
                    full_text.append(token)
                    yield token
            # Store completed translation in context buffer
            completed = "".join(full_text).strip()
            if completed:
                self._context.append(completed)
        except UnicodeEncodeError:
            import traceback
            print("[TL-Stream] UnicodeEncodeError:")
            traceback.print_exc()
            raise
        except Exception as e:
            err_msg = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"[TL-Stream] API error: {err_msg}")
            raise


def DeepSeekTranslator(api_key: str, model: str = "deepseek-v4-flash", timeout: float = 30.0) -> Translator:
    """Factory for DeepSeek translator."""
    return OpenAICompatibleTranslator(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model=model,
        timeout=timeout,
    )


def OpenAITranslator(api_key: str, model: str = "gpt-4o-mini", timeout: float = 30.0) -> Translator:
    """Factory for OpenAI translator."""
    return OpenAICompatibleTranslator(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
        model=model,
        timeout=timeout,
    )


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

            frame_tensor = torch.from_numpy(frame)
            speech_prob = self._model(frame_tensor, self.sample_rate).item()
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
        model: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
    ):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
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

    def reset_context(self):
        """Clear the translator's context buffer (e.g., on new conversation)."""
        if hasattr(self._translator, '_context'):
            self._translator._context.clear()

    async def process_chunk(self, audio: np.ndarray):
        """Process one audio chunk. Yields dicts: {type, text|status|message}."""
        segments = self._vad.process(audio)

        if self._vad.is_active() and not getattr(self, '_was_active', False):
            print("[VAD]  ▶ Speech started")
            self._was_active = True
        elif not self._vad.is_active() and getattr(self, '_was_active', False):
            print("[VAD]  ■ Speech ended")
            self._was_active = False

        for sr, speech_audio in segments:
            dur = len(speech_audio) / sr
            print(f"[VAD]  → Segment: {dur:.1f}s ({len(speech_audio)} samples)")
            sent_id = None
            try:
                text_en = await asyncio.to_thread(
                    self._whisper.transcribe, speech_audio, sr
                )
                if not text_en.strip():
                    print("[STT]  → (silence / no text)")
                    continue
                print(f"[STT]  → \"{text_en[:100]}\"")

                # Split into sentences, stream all into ONE subtitle
                sentences = _split_sentences(text_en)
                sent_id = uuid.uuid4().hex[:8]
                yield {"type": "sentence_start", "id": sent_id}

                all_tokens: list[str] = []
                for i, sent in enumerate(sentences):
                    async for token in self._translator.translate_stream(
                        sent, self._source, self._target
                    ):
                        all_tokens.append(token)
                        yield {"type": "token", "id": sent_id, "text": token}
                    # Add newline between sentences (but not after last)
                    if i < len(sentences) - 1:
                        all_tokens.append("\n")
                        yield {"type": "token", "id": sent_id, "text": "\n"}

                completed = "".join(all_tokens).strip()
                if completed:
                    try:
                        print(f"[TL]   → \"{completed[:100]}\"")
                    except UnicodeEncodeError:
                        print(f"[TL]   → (Vietnamese text, {len(completed)} chars)")
                yield {
                    "type": "sentence_end",
                    "id": sent_id,
                    "text": completed,
                    "text_en": text_en,
                }

            except Exception as e:
                print(f"[Pipeline] Error: {e}")
                # Clean up streaming state if sentence_start was already sent
                if sent_id:
                    yield {"type": "sentence_end", "id": sent_id, "text": ""}
                yield {"type": "error", "message": str(e)}

    def flush(self):
        """Force flush any remaining speech from VAD."""
        sr, audio = self._vad.flush()
        return sr, audio

    def reset(self):
        self._vad.reset()
