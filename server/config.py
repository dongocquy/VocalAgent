"""Configuration manager backed by SQLite.

Stores settings in a SQLite database to prevent file corruption that can
occur with plain JSON files (e.g., process output accidentally written into
the config file by a misdirected stdout/stderr).

Schema:
    config(section TEXT, key TEXT, value TEXT, PRIMARY KEY(section, key))

    - section = top-level config key ("deepseek", "openai", "whisper", …)
    - key     = sub-key within the section ("api_key", "model", …)
    - value   = JSON-serialized value (string, number, bool, null)
    - Special section "__root__" holds top-level scalar values (e.g., "provider")

Auto-migration: if the DB is empty and a config.json exists next to it,
data is imported automatically.  Otherwise defaults are seeded.
"""

import json
import os
import sqlite3
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

# Load .env if present (python-dotenv optional — works without it)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
        "max_speech_duration_s": 10.0,
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "audio": {
        "device": None,
        "sample_rate": 16000,
        "chunk_size": 4096,
        "channels": 1,
        "gain": 5.0,
    },
}

# Sections that contain nested dicts (vs top-level scalars).
_SECTIONS = [
    "deepseek", "openai", "whisper", "vad", "server", "audio",
]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


class ConfigManager:
    """SQLite-backed configuration manager.

    Usage::

        cm = ConfigManager("/path/to/config.db")
        api_key = cm.get_api_key()
        cm.update({"provider": "openai", "openai": {"api_key": "key-here"}})

    The public API is identical to the previous JSON-file implementation.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()
        if self._is_empty():
            self._migrate_or_seed()

    # ── schema / migration ──────────────────────────────────

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                section TEXT NOT NULL,
                key     TEXT NOT NULL,
                value   TEXT NOT NULL,
                PRIMARY KEY (section, key)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                text_en TEXT NOT NULL,
                text_vi TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _is_empty(self) -> bool:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM config"
        ).fetchone()
        return row["cnt"] == 0

    def _migrate_or_seed(self) -> None:
        """Try to import from an old config.json; otherwise seed defaults."""
        json_path = self._guess_json_path()
        if json_path and os.path.isfile(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._write_nested(data)
                print(f"[Config] Migrated {json_path} → {self._db_path}")
                return
            except (json.JSONDecodeError, IOError, OSError) as exc:
                print(f"[Config] JSON migration failed: {exc}", file=sys.stderr)
        # No JSON to migrate — seed with factory defaults
        self._write_nested(DEFAULT_CONFIG)
        self._conn.commit()

    def _guess_json_path(self) -> str | None:
        """Guess the location of an old config.json file."""
        db_dir = os.path.dirname(self._db_path) or "."
        # 1) config.json right next to the .db file
        candidate = os.path.join(db_dir, "config.json")
        if os.path.isfile(candidate):
            return candidate
        # 2) same path but replace .db → .json
        if self._db_path.endswith(".db"):
            candidate = self._db_path[:-3] + ".json"
            if os.path.isfile(candidate):
                return candidate
        return None

    # ── low-level read / write ──────────────────────────────

    def _write_nested(self, data: Dict[str, Any]) -> None:
        """Persist a full nested config dict to the DB."""
        with self._conn:
            for section, section_data in data.items():
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        self._upsert(section, key, value)
                else:
                    self._upsert("__root__", section, section_data)

    def _upsert(self, section: str, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO config (section, key, value) "
            "VALUES (?, ?, ?)",
            (section, key, json.dumps(value)),
        )

    def _read_nested(self) -> Dict[str, Any]:
        """Read the DB, merge with defaults, return full config dict."""
        result: Dict[str, Any] = {s: {} for s in _SECTIONS}
        rows = self._conn.execute(
            "SELECT section, key, value FROM config"
        ).fetchall()
        for row in rows:
            section, key, raw = row["section"], row["key"], row["value"]
            parsed = json.loads(raw)
            if section == "__root__":
                result[key] = parsed
            else:
                if section not in result:
                    result[section] = {}
                result[section][key] = parsed
        # Merge with defaults so missing keys are always filled in
        merged = deepcopy(DEFAULT_CONFIG)
        _deep_merge(merged, result)
        return merged

    # ── public API (identical to JSON version) ──────────────

    @property
    def config(self) -> Dict[str, Any]:
        """Return a deep copy of the full configuration."""
        return self._read_nested()

    def get_provider(self) -> str:
        # Allow env var override (highest priority)
        env_provider = os.environ.get("VOCALAGENT_PROVIDER")
        if env_provider and env_provider in ("deepseek", "openai"):
            return env_provider
        row = self._conn.execute(
            "SELECT value FROM config WHERE section = '__root__' AND key = 'provider'"
        ).fetchone()
        if row is None:
            return DEFAULT_CONFIG["provider"]
        return json.loads(row["value"])

    def get_api_key(self) -> str:
        provider = self.get_provider()

        # Check env var first (highest priority)
        env_var = f"VOCALAGENT_{provider.upper()}_API_KEY"
        env_key = os.environ.get(env_var)
        if env_key:
            return env_key

        # Fallback to DB
        row = self._conn.execute(
            "SELECT value FROM config WHERE section = ? AND key = 'api_key'",
            (provider,),
        ).fetchone()
        if row is None:
            return DEFAULT_CONFIG.get(provider, {}).get("api_key", "")
        key = json.loads(row["value"])
        # Guard against non-ASCII garbage (e.g. terminal output pasted into form)
        if key and not key.isascii():
            print(
                f"[Config] WARNING: {provider} API key contains non-ASCII "
                f"characters ({len(key)} chars). HTTP headers require ASCII. "
                f"Resetting to empty — re-enter your API key in the admin page.",
                file=sys.stderr,
            )
            self._upsert(provider, "api_key", "")
            self._conn.commit()
            return ""
        return key

    def get_provider_config(self) -> Dict[str, Any]:
        provider = self.get_provider()
        return self._read_section(provider)

    def get_whisper_config(self) -> Dict[str, Any]:
        return self._read_section("whisper")

    def get_vad_config(self) -> Dict[str, Any]:
        return self._read_section("vad")

    def get_server_config(self) -> Dict[str, Any]:
        return self._read_section("server")

    def get_audio_config(self) -> Dict[str, Any]:
        return self._read_section("audio")

    def _read_section(self, section: str) -> Dict[str, Any]:
        rows = self._conn.execute(
            "SELECT key, value FROM config WHERE section = ?", (section,)
        ).fetchall()
        result = {}
        for row in rows:
            result[row["key"]] = json.loads(row["value"])
        # Merge with defaults for this section
        default_section = DEFAULT_CONFIG.get(section, {})
        if isinstance(default_section, dict):
            merged = dict(default_section)
            merged.update(result)
            return merged
        return result if result else default_section

    def save_translation(self, text_en: str, text_vi: str) -> None:
        """Persist a translation pair to the translations table."""
        self._conn.execute(
            "INSERT INTO translations (text_en, text_vi) VALUES (?, ?)",
            (text_en, text_vi),
        )
        self._conn.commit()

    def get_translations(self, limit: int = 50) -> list[dict]:
        """Return recent translations, newest first."""
        rows = self._conn.execute(
            "SELECT id, timestamp, text_en, text_vi "
            "FROM translations ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r["id"], "timestamp": r["timestamp"],
                 "text_en": r["text_en"], "text_vi": r["text_vi"]}
                for r in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def update(self, data: Dict[str, Any]) -> None:
        """Merge *data* into config and persist.

        Shallow merge at top level, deep (dict.update) for nested sections.
        """
        with self._conn:
            for key, value in data.items():
                if isinstance(value, dict) and key in _SECTIONS:
                    # Deep-merge: load current section, update, write back
                    current = self._read_section(key)
                    current.update(value)
                    for subkey, subvalue in current.items():
                        self._upsert(key, subkey, subvalue)
                else:
                    self._upsert("__root__", key, value)
        self._conn.commit()
