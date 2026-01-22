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


def _api_headers(user: str = "alice", api_key: str | None = None) -> dict[str, str]:
    headers = {"X-User": user}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _png_bytes(w: int = 8, h: int = 6) -> bytes:
    img = PILImage.new("RGB", (w, h), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_delete_image_removes_dataset_file_when_no_annotations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage_env"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_storage_root = tmp_path / "storage_project"
    dataset_name = f"ds-{uuid4().hex[:8]}"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Test Project", "type": "커깃쇱꿎", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            json={"name": dataset_name, "project_id": project_id},
            headers=_api_headers(api_key="test-key"),
        )
        assert resp.status_code == 201, resp.text
        dataset_id = resp.json()["data"]["id"]

        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers=_api_headers(api_key="test-key"),
            files=[("files", ("a.png", _png_bytes(), "image/png"))],
        )
        assert resp.status_code == 201, resp.text

        resp = client.get(f"/projects/{project_id}/images", headers=_api_headers())
        assert resp.status_code == 200
        images = resp.json()
        assert len(images) == 1
        image_id = images[0]["id"]

        resp = client.delete(f"/images/{image_id}", headers=_api_headers(api_key="test-key"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["code"] == 200

        resp = client.get(f"/projects/{project_id}/images", headers=_api_headers())
        assert resp.status_code == 200
        assert resp.json() == []

        resp = client.get(f"/datasets/{dataset_id}/files", headers=_api_headers())
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"] == []

    ds_dir = project_storage_root / "projects" / project_id / "datasets" / dataset_id
    if ds_dir.exists():
        assert list(ds_dir.glob("*")) == []


def test_delete_image_rejects_when_annotations_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage_env"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_storage_root = tmp_path / "storage_project"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Test Project", "type": "커깃쇱꿎", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            json={"name": f"ds-{uuid4().hex[:8]}", "project_id": project_id},
            headers=_api_headers(api_key="test-key"),
        )
        assert resp.status_code == 201, resp.text
        dataset_id = resp.json()["data"]["id"]

        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers=_api_headers(api_key="test-key"),
            files=[("files", ("a.png", _png_bytes(), "image/png"))],
        )
        assert resp.status_code == 201, resp.text

        resp = client.get(f"/projects/{project_id}/images", headers=_api_headers())
        assert resp.status_code == 200
        image_id = resp.json()[0]["id"]

        payload = [
            {
                "type": "rect",
                "label": "hd",
                "color": "#ef4444",
                "visible": True,
                "x": 1,
                "y": 1,
                "width": 2,
                "height": 2,
                "points": None,
            }
        ]
        resp = client.put(f"/images/{image_id}/annotations", json=payload)
        assert resp.status_code == 200, resp.text
        assert len(resp.json()) == 1

        resp = client.delete(f"/images/{image_id}", headers=_api_headers(api_key="test-key"))
        assert resp.status_code == 409, resp.text
        assert resp.json()["code"] == 409

