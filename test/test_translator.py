import os
import sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))

from pipeline import Translator, DeepSeekTranslator, OpenAITranslator


def test_deepseek_translator_sets_api_key_and_base_url():
    t = DeepSeekTranslator(api_key="test-key-01", model="deepseek-v4-flash")
    assert t.client.api_key == "test-key-01"
    assert str(t.client.base_url).rstrip("/") == "https://api.deepseek.com"
    assert t.model == "deepseek-v4-flash"


def test_openai_translator_sets_api_key_and_base_url():
    t = OpenAITranslator(api_key="test-key-02", model="gpt-4o-mini")
    assert t.client.api_key == "test-key-02"
    assert str(t.client.base_url).rstrip("/") == "https://api.openai.com/v1"


def test_translator_is_abstract():
    with pytest.raises(TypeError):
        Translator()


def test_translate_empty_text_returns_empty():
    t = DeepSeekTranslator(api_key="test-key-fake")
    import asyncio
    result = asyncio.run(t.translate("", "en", "vi"))
    assert result == ""
