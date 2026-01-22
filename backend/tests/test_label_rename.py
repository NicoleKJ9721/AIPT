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


def _headers(user: str = "alice", api_key: str | None = None) -> dict[str, str]:
    headers = {"X-User": user}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _png_bytes() -> bytes:
    img = PILImage.new("RGB", (8, 6), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_project_label_rename_scoped_to_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    with TestClient(app) as client:
        # Project
        resp = client.post("/projects", json={"name": "Rename Project", "type": "目标检测"})
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        # Two datasets under the same project
        ds1 = client.post(
            "/datasets",
            headers=_headers(api_key="test-key"),
            json={"name": f"ds1-{uuid4().hex[:6]}", "project_id": project_id},
        )
        assert ds1.status_code == 201, ds1.text
        dataset1_id = ds1.json()["data"]["id"]

        ds2 = client.post(
            "/datasets",
            headers=_headers(api_key="test-key"),
            json={"name": f"ds2-{uuid4().hex[:6]}", "project_id": project_id},
        )
        assert ds2.status_code == 201, ds2.text
        dataset2_id = ds2.json()["data"]["id"]

        # Upload one image to each dataset (auto-creates Image records)
        img_bytes = _png_bytes()
        resp = client.post(
            f"/datasets/{dataset1_id}/files",
            headers=_headers(api_key="test-key"),
            files=[("files", ("a.png", img_bytes, "image/png"))],
        )
        assert resp.status_code == 201, resp.text

        resp = client.post(
            f"/datasets/{dataset2_id}/files",
            headers=_headers(api_key="test-key"),
            files=[("files", ("b.png", img_bytes, "image/png"))],
        )
        assert resp.status_code == 201, resp.text

        # Resolve image ids
        resp = client.get(f"/projects/{project_id}/images", params={"dataset_id": dataset1_id})
        assert resp.status_code == 200, resp.text
        img1_id = resp.json()[0]["id"]

        resp = client.get(f"/projects/{project_id}/images", params={"dataset_id": dataset2_id})
        assert resp.status_code == 200, resp.text
        img2_id = resp.json()[0]["id"]

        # Add one annotation to each image with the same label
        payload1 = [
            {"id": "ann-1", "type": "rect", "label": "hd", "color": "#ef4444", "visible": True, "x": 1, "y": 2, "width": 3, "height": 4}
        ]
        payload2 = [
            {"id": "ann-2", "type": "rect", "label": "hd", "color": "#ef4444", "visible": True, "x": 1, "y": 2, "width": 3, "height": 4}
        ]
        resp = client.put(f"/images/{img1_id}/annotations", json=payload1)
        assert resp.status_code == 200, resp.text
        resp = client.put(f"/images/{img2_id}/annotations", json=payload2)
        assert resp.status_code == 200, resp.text

        # Rename scoped to dataset1 only
        resp = client.post(
            f"/projects/{project_id}/labels/rename",
            headers=_headers(api_key="test-key"),
            json={"from_label": "hd", "to_label": "crack", "dataset_id": dataset1_id},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["updated"] == 1

        # Dataset1 image updated
        resp = client.get(f"/images/{img1_id}/annotations")
        assert resp.status_code == 200
        assert resp.json()[0]["label"] == "crack"

        # Dataset2 image unchanged
        resp = client.get(f"/images/{img2_id}/annotations")
        assert resp.status_code == 200
        assert resp.json()[0]["label"] == "hd"
