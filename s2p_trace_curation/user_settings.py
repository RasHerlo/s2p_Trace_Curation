"""Persist UI preferences between sessions (JSON in the user config dir)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def settings_path() -> Path:
    return Path.home() / ".s2p_trace_curation" / "settings.json"


def load_settings() -> dict[str, Any]:
    path = settings_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(updates: dict[str, Any]) -> None:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_settings()
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def last_open_start_dir(settings: dict[str, Any] | None = None) -> str:
    """Directory to start the Open dialog in (last suite2p folder if it still exists)."""
    if settings is None:
        settings = load_settings()
    raw = settings.get("last_suite2p_dir")
    if not raw:
        return ""
    path = Path(raw)
    if path.is_dir():
        return str(path)
    parent = path.parent
    if parent.is_dir():
        return str(parent)
    return ""
