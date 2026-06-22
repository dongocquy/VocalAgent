import json
import os
import sqlite3
import sys
import tempfile
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from config import ConfigManager, DEFAULT_CONFIG


def _db_has_row(db_path: str, section: str, key: str) -> bool:
    """Check that a specific (section, key) exists in the SQLite DB."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT value FROM config WHERE section = ? AND key = ?",
        (section, key),
    ).fetchone()
    conn.close()
    return row is not None


def _db_value(db_path: str, section: str, key: str) -> object:
    """Read a single value from the SQLite DB."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT value FROM config WHERE section = ? AND key = ?",
        (section, key),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return json.loads(row[0])


# ── Basic DB tests ──────────────────────────────────────────


def test_default_config_created_when_no_file():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "config.db")
        assert not os.path.exists(db_path)
        cm = ConfigManager(db_path)
        try:
            assert os.path.exists(db_path)
            assert _db_has_row(db_path, "__root__", "provider")
            assert _db_value(db_path, "__root__", "provider") == "deepseek"
            assert _db_value(db_path, "whisper", "model") == "medium"
        finally:
            cm.close()


def test_load_existing_config():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "config.db")
        cm1 = ConfigManager(db_path)
        try:
            cm1.update({"provider": "openai", "openai": {"api_key": "TEST-KEY-DEMO"}})
        finally:
            cm1.close()
        # Now load with a fresh instance
        cm2 = ConfigManager(db_path)
        try:
            assert cm2.get_provider() == "openai"
            assert cm2.get_api_key() == "TEST-KEY-DEMO"
        finally:
            cm2.close()


def test_save_config_persists_to_db():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "config.db")
        cm = ConfigManager(db_path)
        try:
            cm.update({"provider": "openai", "openai": {"api_key": "TEST-KEY-NEW"}})
            assert _db_value(db_path, "__root__", "provider") == "openai"
            assert _db_value(db_path, "openai", "api_key") == "TEST-KEY-NEW"
        finally:
            cm.close()


def test_get_provider_config_returns_correct_section():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "config.db")
        cm = ConfigManager(db_path)
        try:
            deepseek = cm.get_provider_config()
            assert deepseek["model"] == "deepseek-v4-flash"
            assert deepseek["base_url"] == "https://api.deepseek.com"
        finally:
            cm.close()


# ── API key ASCII validation ────────────────────────────────


def test_non_ascii_api_key_is_auto_reset():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "config.db")
        cm = ConfigManager(db_path)
        try:
            cm.update({"provider": "openai", "openai": {"api_key": "khóa-giả-unicode"}})
            key = cm.get_api_key()
            assert key == ""
            assert _db_value(db_path, "openai", "api_key") == ""
        finally:
            cm.close()


# ── JSON auto-migration ─────────────────────────────────────


def test_migrate_from_json_file():
    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "config.json")
        db_path = os.path.join(tmp, "config.db")
        custom = deepcopy(DEFAULT_CONFIG)
        custom["provider"] = "openai"
        custom["openai"]["api_key"] = "TEST-KEY-MIGRATED"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(custom, f)
        cm = ConfigManager(db_path)
        try:
            assert cm.get_provider() == "openai"
            assert cm.get_api_key() == "TEST-KEY-MIGRATED"
            assert _db_value(db_path, "whisper", "model") == custom["whisper"]["model"]
        finally:
            cm.close()


# ── Edge cases ──────────────────────────────────────────────


def test_empty_db_seeds_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "config.db")
        # Create empty DB manually
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS config ("
            "  section TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,"
            "  PRIMARY KEY (section, key))"
        )
        conn.commit()
        conn.close()
        cm = ConfigManager(db_path)
        try:
            assert cm.get_provider() == "deepseek"
            assert cm.get_whisper_config()["model"] == "medium"
        finally:
            cm.close()


def test_update_preserves_unrelated_sections():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "config.db")
        cm = ConfigManager(db_path)
        try:
            cm.update({"vad": {"threshold": 0.9}})
            whisper = cm.get_whisper_config()
            assert whisper["model"] == "medium"
            assert cm.get_vad_config()["threshold"] == 0.9
        finally:
            cm.close()


def test_config_property_returns_full_dict():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "config.db")
        cm = ConfigManager(db_path)
        try:
            cfg = cm.config
            assert isinstance(cfg, dict)
            assert cfg["provider"] == "deepseek"
            assert isinstance(cfg["deepseek"], dict)
            assert cfg["deepseek"]["model"] == "deepseek-v4-flash"
            cfg["provider"] = "changed"
            assert cm.get_provider() == "deepseek"
        finally:
            cm.close()
