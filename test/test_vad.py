"""Unit tests for VADProcessor state machine.

The Silero VAD model requires real speech audio and does not respond
to synthetic sine waves.  We mock the model's __call__ to return
controlled probabilities that exercise the VADProcessor state
machine: speech (>=threshold) vs silence (<threshold).
"""

import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from pipeline import VADProcessor


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_probs(
    signal_frames: int,
    silence_frames: int,
    threshold: float = 0.5,
    speech_prob: float = 0.9,
    silence_prob: float = 0.1,
) -> list:
    """Return an ordered list of speech probabilities for the mock
    model, one entry per 512-sample frame.  Each entry is a tensor-like
    object with a ``.item()`` method returning the probability."""
    probs = [speech_prob] * signal_frames + [silence_prob] * silence_frames

    class _Prob:
        def __init__(self, val: float):
            self.val = val
        def item(self) -> float:
            return self.val

    return [_Prob(p) for p in probs]


def _make_audio(probs: list, sr: int = 16000) -> np.ndarray:
    """Build a flat float32 audio array whose *length* equals the number
    of 512-sample frames implied by ``probs``.  The actual sample
    values don't matter (the mock ignores them)."""
    n_samples = len(probs) * 512
    # Use a non-zero signal to distinguish from silence at the numpy level
    arr = np.full(n_samples, 0.1, dtype=np.float32)
    return arr


# A side-effect-free model that yields pre-defined probabilities.
class _MockModel:
    def __init__(self, probs: list):
        self._probs = iter(probs)

    def __call__(self, _frame, _sr):
        return next(self._probs)

    def reset_states(self):
        pass


# ── tests ────────────────────────────────────────────────────────────────────


def test_vad_initialization():
    vad = VADProcessor()
    assert vad.sample_rate == 16000
    assert vad.threshold == 0.5


def test_vad_returns_empty_for_silence():
    """All-zero audio → every frame is silence → no segment."""
    probs = _make_probs(signal_frames=0, silence_frames=63)          # ~2 s
    vad = VADProcessor()
    vad._model = _MockModel(probs)

    audio = _make_audio(probs)
    segments = vad.process(audio)
    assert len(segments) == 0
    assert not vad.is_active()


def test_vad_detects_speech_segment():
    """1 s speech + 700 ms silence → one complete segment."""
    probs = _make_probs(signal_frames=31, silence_frames=22)        # ~1 s + ~0.7 s
    vad = VADProcessor()
    vad._model = _MockModel(probs)

    audio = _make_audio(probs)
    segments = vad.process(audio)

    assert len(segments) >= 1
    sr, seg_audio = segments[0]
    assert sr == 16000
    assert len(seg_audio) > 0


def test_vad_no_flush_without_silence():
    """Continuous speech → still active, no completed segment."""
    probs = _make_probs(signal_frames=63, silence_frames=0)          # ~2 s
    vad = VADProcessor()
    vad._model = _MockModel(probs)

    audio = _make_audio(probs)
    segments = vad.process(audio)
    assert len(segments) == 0
    assert vad.is_active()


def test_vad_reset_clears_state():
    """Reset mid-speech should clear buffers and return to silence."""
    probs_active = _make_probs(signal_frames=16, silence_frames=0)   # ~0.5 s
    vad = VADProcessor()
    vad._model = _MockModel(probs_active)

    audio = _make_audio(probs_active)
    vad.process(audio)
    assert vad.is_active()

    vad.reset()
    assert not vad.is_active()

    # Subsequent silence should not produce a segment
    probs_silence = _make_probs(signal_frames=0, silence_frames=32)  # ~1 s
    vad._model = _MockModel(probs_silence)
    segments = vad.process(_make_audio(probs_silence))
    assert len(segments) == 0


def test_vad_flush_returns_accumulated_speech():
    """flush() returns accumulated audio and deactivates."""
    probs = _make_probs(signal_frames=16, silence_frames=0)          # ~0.5 s
    vad = VADProcessor()
    vad._model = _MockModel(probs)

    vad.process(_make_audio(probs))
    sr, audio = vad.flush()
    assert sr == 16000
    assert len(audio) > 0
    assert not vad.is_active()
