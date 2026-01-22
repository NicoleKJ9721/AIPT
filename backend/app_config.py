from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def app_home_dir() -> Path:
    """
    Returns a per-user, out-of-repo home directory for AIPT runtime files.
    Can be overridden via env var `AIPT_HOME_DIR` (or legacy `AIPT_CONFIG_DIR`).
    """
    override = os.getenv("AIPT_HOME_DIR") or os.getenv("AIPT_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()

    if sys.platform.startswith("win"):
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home())
        return (Path(base) / "AIPT").resolve()

    return (Path.home() / ".aipt").resolve()


def config_path() -> Path:
    return app_home_dir() / "config.json"


def _default_projects_root() -> Path:
    # Keep backward-compat: AIPT_STORAGE_DIR historically controlled storage root.
    env = os.getenv("AIPT_STORAGE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (app_home_dir() / "storage").resolve()


def _default_resources_root() -> Path:
    env = os.getenv("AIPT_RESOURCES_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (app_home_dir() / "resources").resolve()


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        key = (v or "").strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def load_settings() -> dict[str, Any]:
    """
    Load persisted settings from disk (JSON) and merge with defaults.
    """
    defaults: dict[str, Any] = {
        "projects_root_dir": str(_default_projects_root()),
        "resources_root_dir": str(_default_resources_root()),
        "recent_projects_root_dirs": [],
        "recent_resources_root_dirs": [],
        "default_model_resource_id": None,
    }

    path = config_path()
    if not path.exists():
        # First run: seed recents with defaults.
        defaults["recent_projects_root_dirs"] = [defaults["projects_root_dir"]]
        defaults["recent_resources_root_dirs"] = [defaults["resources_root_dir"]]
        return defaults

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

    merged = {**defaults, **data}

    merged["projects_root_dir"] = str(Path(str(merged["projects_root_dir"])).expanduser().resolve())
    merged["resources_root_dir"] = str(Path(str(merged["resources_root_dir"])).expanduser().resolve())

    merged["recent_projects_root_dirs"] = _dedupe_keep_order(
        [merged["projects_root_dir"], *list(merged.get("recent_projects_root_dirs") or [])]
    )[:10]
    merged["recent_resources_root_dirs"] = _dedupe_keep_order(
        [merged["resources_root_dir"], *list(merged.get("recent_resources_root_dirs") or [])]
    )[:10]

    return merged


def save_settings(settings: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def update_settings(patch: dict[str, Any]) -> dict[str, Any]:
    """
    Update persisted settings, ensuring paths are absolute and recents updated.
    """
    current = load_settings()
    next_settings = {**current, **{k: v for k, v in patch.items() if v is not None}}

    if "projects_root_dir" in patch and patch.get("projects_root_dir"):
        next_settings["projects_root_dir"] = str(Path(str(patch["projects_root_dir"])).expanduser().resolve())

    if "resources_root_dir" in patch and patch.get("resources_root_dir"):
        next_settings["resources_root_dir"] = str(Path(str(patch["resources_root_dir"])).expanduser().resolve())

    next_settings["recent_projects_root_dirs"] = _dedupe_keep_order(
        [str(next_settings["projects_root_dir"]), *list(next_settings.get("recent_projects_root_dirs") or [])]
    )[:10]
    next_settings["recent_resources_root_dirs"] = _dedupe_keep_order(
        [str(next_settings["resources_root_dir"]), *list(next_settings.get("recent_resources_root_dirs") or [])]
    )[:10]

    # Ensure directories exist (local-only mode).
    Path(str(next_settings["projects_root_dir"])).mkdir(parents=True, exist_ok=True)
    Path(str(next_settings["resources_root_dir"])).mkdir(parents=True, exist_ok=True)

    save_settings(next_settings)
    return next_settings

