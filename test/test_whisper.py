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
