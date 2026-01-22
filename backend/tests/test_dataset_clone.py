import io
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


def test_dataset_clone_copies_images_and_annotations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Clone Project", "type": "目标检测", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            headers={"X-User": "alice", "X-API-Key": "test-key"},
            json={"name": f"ds-{uuid4().hex[:8]}", "project_id": project_id},
        )
        assert resp.status_code == 201, resp.text
        src_dataset_id = resp.json()["data"]["id"]

        img = PILImage.new("RGB", (32, 24), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        resp = client.post(
            f"/datasets/{src_dataset_id}/files",
            headers={"X-User": "alice", "X-API-Key": "test-key"},
            files=[("files", ("a.png", img_bytes, "image/png"))],
        )
        assert resp.status_code == 201, resp.text

        resp = client.get(f"/projects/{project_id}/images")
        assert resp.status_code == 200, resp.text
        image_id = resp.json()[0]["id"]

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
                    "width": 10,
                    "height": 8,
                    "points": None,
                }
            ],
        )
        assert resp.status_code == 200, resp.text

        resp = client.post(
            f"/datasets/{src_dataset_id}/clone",
            headers={"X-User": "alice", "X-API-Key": "test-key"},
            json={"name": "snapshot", "version": "v2"},
        )
        assert resp.status_code == 201, resp.text
        dst_dataset_id = resp.json()["data"]["id"]

        resp = client.get(f"/datasets/{dst_dataset_id}/files", headers={"X-User": "alice"})
        assert resp.status_code == 200, resp.text
        files = resp.json()["data"]
        assert len(files) == 1

        resp = client.get(f"/projects/{project_id}/images", params={"dataset_id": dst_dataset_id})
        assert resp.status_code == 200, resp.text
        dst_images = resp.json()
        assert len(dst_images) == 1
        dst_image_id = dst_images[0]["id"]

        resp = client.get(f"/images/{dst_image_id}/annotations")
        assert resp.status_code == 200, resp.text
        anns = resp.json()
        assert len(anns) == 1
        assert anns[0]["label"] == "hd"
