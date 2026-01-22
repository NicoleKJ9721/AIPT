import json
import os
import sys
from pathlib import Path

import pytest


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app_config  # noqa: E402


def test_app_home_dir_non_windows_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AIPT_HOME_DIR", raising=False)
    monkeypatch.delenv("AIPT_CONFIG_DIR", raising=False)

    # Force the non-Windows branch for coverage; Path.home is also patched for determinism.
    monkeypatch.setattr(app_config.sys, "platform", "linux")
    monkeypatch.setattr(app_config.Path, "home", classmethod(lambda cls: tmp_path / "user"))

    assert app_config.app_home_dir() == (tmp_path / "user" / ".aipt").resolve()


def test_load_settings_merges_and_dedupes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "aipt_home"
    monkeypatch.setenv("AIPT_HOME_DIR", str(home))

    config_path = home / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    projects_root = home / "projects_root"
    resources_root = home / "resources_root"

    config_path.write_text(
        json.dumps(
            {
                "projects_root_dir": str(projects_root),
                "resources_root_dir": str(resources_root),
                "recent_projects_root_dirs": ["", str(projects_root), str(home / "projects_other"), str(home / "projects_other")],
                "recent_resources_root_dirs": [str(resources_root), str(home / "res_other"), str(home / "res_other")],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    settings = app_config.load_settings()
    assert Path(settings["projects_root_dir"]).resolve() == projects_root.resolve()
    assert Path(settings["resources_root_dir"]).resolve() == resources_root.resolve()

    # Recents are deduped and always start with current roots.
    assert settings["recent_projects_root_dirs"][0] == settings["projects_root_dir"]
    assert settings["recent_resources_root_dirs"][0] == settings["resources_root_dir"]
    assert len(settings["recent_projects_root_dirs"]) == 2
    assert len(settings["recent_resources_root_dirs"]) == 2


def test_load_settings_invalid_json_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "aipt_home"
    monkeypatch.setenv("AIPT_HOME_DIR", str(home))

    config_path = home / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{not json", encoding="utf-8")

    settings = app_config.load_settings()
    assert settings["projects_root_dir"]
    assert settings["resources_root_dir"]
    assert settings["recent_projects_root_dirs"][0] == settings["projects_root_dir"]
    assert settings["recent_resources_root_dirs"][0] == settings["resources_root_dir"]


def test_load_settings_non_dict_json_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "aipt_home"
    monkeypatch.setenv("AIPT_HOME_DIR", str(home))

    config_path = home / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(["not", "a", "dict"], ensure_ascii=False), encoding="utf-8")

    settings = app_config.load_settings()
    assert settings["projects_root_dir"]
    assert settings["resources_root_dir"]


def test_default_resources_root_can_be_overridden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "aipt_home"
    monkeypatch.setenv("AIPT_HOME_DIR", str(home))

    resources_override = tmp_path / "resources_override"
    monkeypatch.setenv("AIPT_RESOURCES_DIR", str(resources_override))

    settings = app_config.load_settings()
    assert Path(settings["resources_root_dir"]).resolve() == resources_override.resolve()
