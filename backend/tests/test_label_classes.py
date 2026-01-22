import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


def test_label_class_crud_and_propagation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage"))
    with TestClient(app) as client:
        # Create project
        resp = client.post("/projects", json={"name": "Label Project", "type": "目标检测"})
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        # Create label class
        resp = client.post(
            f"/projects/{project_id}/labels",
            json={"name": "hd", "color": "#ef4444", "shortcut": "1"},
        )
        assert resp.status_code == 201, resp.text
        created = resp.json()["data"]
        label_id = created["id"]
        assert created["name"] == "hd"
        assert created["color"] == "#ef4444"

        # List label classes
        resp = client.get(f"/projects/{project_id}/labels")
        assert resp.status_code == 200, resp.text
        labels = resp.json()["data"]
        assert any(label_row["id"] == label_id for label_row in labels)

        # Create image + annotation using this label
        resp = client.post(
            f"/projects/{project_id}/images",
            json={"filename": "img.jpg", "width": 640, "height": 480},
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
                    "x": 1,
                    "y": 2,
                    "width": 3,
                    "height": 4,
                }
            ],
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()[0]["label"] == "hd"

        # Update label class (rename + recolor) and ensure annotations are updated
        resp = client.put(
            f"/projects/{project_id}/labels/{label_id}",
            json={"name": "crack", "color": "#3b82f6"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["label"]["name"] == "crack"
        assert data["label"]["color"] == "#3b82f6"
        assert data["updated"] == 1

        resp = client.get(f"/images/{image_id}/annotations")
        assert resp.status_code == 200, resp.text
        anns = resp.json()
        assert anns[0]["label"] == "crack"
        assert anns[0]["color"] == "#3b82f6"

        # Deleting an in-use label should be blocked
        resp = client.delete(f"/projects/{project_id}/labels/{label_id}")
        assert resp.status_code == 409, resp.text

        # Create an unused label and delete it
        resp = client.post(
            f"/projects/{project_id}/labels",
            json={"name": "scratch", "color": "#22c55e", "shortcut": "2"},
        )
        assert resp.status_code == 201, resp.text
        scratch_id = resp.json()["data"]["id"]

        resp = client.delete(f"/projects/{project_id}/labels/{scratch_id}")
        assert resp.status_code == 200, resp.text

        resp = client.get(f"/projects/{project_id}/labels")
        assert resp.status_code == 200, resp.text
        assert all(label_row["id"] != scratch_id for label_row in resp.json()["data"])
