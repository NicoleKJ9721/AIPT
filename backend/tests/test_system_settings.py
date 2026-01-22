import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


def _api_headers(api_key: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {"X-User": "tester"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def test_system_settings_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "aipt_home"
    monkeypatch.setenv("AIPT_HOME_DIR", str(home))
    monkeypatch.delenv("AIPT_STORAGE_DIR", raising=False)
    monkeypatch.delenv("AIPT_RESOURCES_DIR", raising=False)
    monkeypatch.delenv("AIPT_API_KEY", raising=False)

    with TestClient(app) as client:
        resp = client.get("/system/settings")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["code"] == 200
        data = body["data"]
        assert Path(data["projects_root_dir"]).resolve() == (home / "storage").resolve()
        assert Path(data["resources_root_dir"]).resolve() == (home / "resources").resolve()
        assert data["recent_projects_root_dirs"][0] == data["projects_root_dir"]
        assert data["recent_resources_root_dirs"][0] == data["resources_root_dir"]


def test_put_system_settings_requires_key_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "aipt_home"
    monkeypatch.setenv("AIPT_HOME_DIR", str(home))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    with TestClient(app) as client:
        resp = client.put("/system/settings", json={"projects_root_dir": str(tmp_path / "x")})
        assert resp.status_code == 401, resp.text
        body = resp.json()
        assert body["code"] == 401


def test_put_system_settings_persists_and_creates_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "aipt_home"
    monkeypatch.setenv("AIPT_HOME_DIR", str(home))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    projects_root = tmp_path / "projects_root"
    resources_root = tmp_path / "resources_root"

    with TestClient(app) as client:
        resp = client.put(
            "/system/settings",
            headers=_api_headers(api_key="test-key"),
            json={"projects_root_dir": str(projects_root), "resources_root_dir": str(resources_root)},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["code"] == 200
        data = body["data"]
        assert Path(data["projects_root_dir"]).resolve() == projects_root.resolve()
        assert Path(data["resources_root_dir"]).resolve() == resources_root.resolve()

    assert projects_root.exists()
    assert resources_root.exists()

    config_path = home / "config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert Path(config["projects_root_dir"]).resolve() == projects_root.resolve()
    assert Path(config["resources_root_dir"]).resolve() == resources_root.resolve()
    assert config["recent_projects_root_dirs"][0] == config["projects_root_dir"]
    assert config["recent_resources_root_dirs"][0] == config["resources_root_dir"]


def test_select_directory_dialog_returns_501_when_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "aipt_home"
    monkeypatch.setenv("AIPT_HOME_DIR", str(home))

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tkinter" or name.startswith("tkinter."):
            raise ImportError("no tkinter")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with TestClient(app) as client:
        resp = client.post("/system/dialogs/select-directory", json={"title": "x"})
        assert resp.status_code == 501, resp.text
        body = resp.json()
        assert body["code"] == 501
        assert "Directory picker is not available" in body["message"]

