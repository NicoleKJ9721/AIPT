import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


def test_dashboard_summary_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.delenv("AIPT_API_KEY", raising=False)

    with TestClient(app) as client:
        resp = client.post("/projects", json={"name": "P1", "type": "目标检测"})
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            headers={"X-User": "alice"},
            json={"project_id": project_id, "name": "DS", "version": "v1"},
        )
        assert resp.status_code == 201, resp.text

        resp = client.post(
            f"/projects/{project_id}/images",
            json={"filename": "img.png", "width": 100, "height": 100},
        )
        assert resp.status_code == 201, resp.text
        image_id = resp.json()["id"]

        resp = client.put(
            f"/images/{image_id}/annotations",
            json=[
                {
                    "type": "rect",
                    "label": "hd",
                    "color": "#ef4444",
                    "visible": True,
                    "x": 10,
                    "y": 10,
                    "width": 20,
                    "height": 20,
                }
            ],
        )
        assert resp.status_code == 200, resp.text

        resp = client.get("/dashboard/summary")
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["code"] == 200
        data = payload["data"]
        assert data["projects_total"] == 1
        assert data["datasets_total"] == 1
        assert data["images_total"] == 1
        assert data["images_annotated_total"] == 1
        assert data["images_pending_total"] == 0

