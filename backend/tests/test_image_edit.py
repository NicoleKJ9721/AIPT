import io
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image as PILImage
import pytest


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


def _headers(user: str = "alice", api_key: str | None = None) -> dict[str, str]:
    headers = {"X-User": user}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def test_image_edit_rotate_and_crop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage_env"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_root = tmp_path / "storage_project"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "P1", "type": "目标检测", "storage_root": str(project_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            json={"name": "ds1", "project_id": project_id},
            headers=_headers(api_key="test-key"),
        )
        assert resp.status_code == 201, resp.text
        dataset_id = resp.json()["data"]["id"]

        img = PILImage.new("RGB", (8, 6), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers=_headers(api_key="test-key"),
            files=[("files", ("a.png", img_bytes, "image/png"))],
        )
        assert resp.status_code == 201, resp.text
        file_id = resp.json()["data"][0]["id"]

        resp = client.get(f"/projects/{project_id}/images", headers=_headers())
        assert resp.status_code == 200, resp.text
        images = resp.json()
        assert len(images) == 1
        image_id = images[0]["id"]
        assert images[0]["width"] == 8
        assert images[0]["height"] == 6

        resp = client.get(f"/datasets/{dataset_id}/files", headers=_headers())
        assert resp.status_code == 200, resp.text
        before = next(f for f in resp.json()["data"] if f["id"] == file_id)
        sha_before = before["sha256"]

        # Requires API key when AIPT_API_KEY is set.
        resp = client.post(f"/images/{image_id}/edit", json={"rotate": 90}, headers=_headers(api_key=None))
        assert resp.status_code == 401, resp.text

        resp = client.post(f"/images/{image_id}/edit", json={"rotate": 90}, headers=_headers(api_key="test-key"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["code"] == 200
        assert resp.json()["data"]["width"] == 6
        assert resp.json()["data"]["height"] == 8

        resp = client.post(
            f"/images/{image_id}/edit",
            json={"crop": {"x": 0, "y": 0, "width": 3, "height": 4}},
            headers=_headers(api_key="test-key"),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["width"] == 3
        assert resp.json()["data"]["height"] == 4

        resp = client.get(f"/datasets/{dataset_id}/files", headers=_headers())
        assert resp.status_code == 200, resp.text
        after = next(f for f in resp.json()["data"] if f["id"] == file_id)
        assert after["sha256"] != sha_before

