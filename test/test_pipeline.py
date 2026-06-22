"""Unit tests for pipeline helpers."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from pipeline import _split_sentences


def test_split_single_sentence():
    assert _split_sentences("Hello world.") == ["Hello world."]


def test_split_multiple_sentences():
    result = _split_sentences("Hello. How are you? I'm fine!")
    assert result == ["Hello.", "How are you?", "I'm fine!"]


def test_split_empty_string():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


def test_split_no_punctuation():
    assert _split_sentences("hello world") == ["hello world"]


def test_split_handles_abbreviations():
    """Known: simple regex splits on 'Dr.' — acceptable trade-off."""
    result = _split_sentences("Dr. Smith came. He said hi.")
    # "Dr." triggers split — we accept this, real speech rarely has "Dr."
    assert len(result) == 3
    assert result[-1] == "He said hi."


def test_split_fallback_long_no_punctuation():
    """Long text without punctuation falls back to word-boundary chunks."""
    words = "the quick brown fox " * 40  # ~760 chars, no punctuation
    result = _split_sentences(words, max_chars=180)
    assert len(result) >= 4  # Should split into 4+ chunks
    # Each chunk should be <= max_chars (plus some slack for last word)
    for chunk in result:
        assert len(chunk) <= 210  # max_chars 180 + ~30 chars for last word


def test_split_short_text_not_split():
    """Short text without punctuation stays as one sentence."""
    result = _split_sentences("hello world", max_chars=180)
    assert result == ["hello world"]
