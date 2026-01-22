import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


def test_project_image_annotation_crud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage"))
    with TestClient(app) as client:
        # Create project
        resp = client.post("/projects", json={"name": "Test Project", "type": "目标检测"})
        assert resp.status_code == 201, resp.text
        project = resp.json()
        project_id = project["id"]

        # List projects
        resp = client.get("/projects")
        assert resp.status_code == 200
        assert any(p["id"] == project_id for p in resp.json())

        # Update project
        resp = client.patch(
            f"/projects/{project_id}",
            json={"status": "已完成", "latest_commit": "abc123"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "已完成"
        assert resp.json()["latest_commit"] == "abc123"

        # Create image
        resp = client.post(
            f"/projects/{project_id}/images",
            json={"filename": "img_001.jpg", "width": 640, "height": 480},
        )
        assert resp.status_code == 201, resp.text
        image_id = resp.json()["id"]

        # Replace annotations
        resp = client.put(
            f"/images/{image_id}/annotations",
            json=[
                {
                    "type": "rect",
                    "label": "person",
                    "color": "#ef4444",
                    "visible": True,
                    "x": 10,
                    "y": 20,
                    "width": 30,
                    "height": 40,
                }
            ],
        )
        assert resp.status_code == 200, resp.text
        anns = resp.json()
        assert len(anns) == 1
        ann_id = anns[0]["id"]

        # List annotations
        resp = client.get(f"/images/{image_id}/annotations")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        # Update annotation
        resp = client.patch(f"/annotations/{ann_id}", json={"label": "car"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["label"] == "car"

        # Delete annotation
        resp = client.delete(f"/annotations/{ann_id}")
        assert resp.status_code == 204, resp.text

        resp = client.get(f"/images/{image_id}/annotations")
        assert resp.status_code == 200
        assert resp.json() == []


def test_project_create_with_custom_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage"))
    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"id": "fixed-id", "name": "Custom ID Project", "type": "目标检测"},
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["id"] == "fixed-id"
