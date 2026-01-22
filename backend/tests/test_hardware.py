import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


def test_hardware_endpoint_returns_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage"))
    with TestClient(app) as client:
        resp = client.get("/hardware")
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["code"] == 200
        devices = payload["data"]
        assert isinstance(devices, list)
        assert any(d.get("type") == "CPU" for d in devices)

