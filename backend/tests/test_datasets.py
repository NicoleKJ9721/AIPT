import io
import os
import sys
import zipfile
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


def test_dataset_crud_files_and_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Use an env storage dir that differs from the per-project storage_root to
    # ensure endpoints consistently respect the project's storage_root.
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage_env"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_storage_root = tmp_path / "storage_project"
    dataset_name = f"ds-{uuid4().hex[:8]}"

    with TestClient(app) as client:
        # Create a project
        resp = client.post(
            "/projects",
            json={"name": "Test Project", "type": "目标检测", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        # Write operations must include API key when AIPT_API_KEY is set.
        resp = client.post(
            "/datasets",
            json={"name": dataset_name, "project_id": project_id},
            headers=_api_headers(api_key=None),
        )
        assert resp.status_code == 401, resp.text
        assert resp.json()["code"] == 401

        # Create dataset
        resp = client.post(
            "/datasets",
            json={"name": dataset_name, "project_id": project_id, "description": "test dataset"},
            headers=_api_headers(api_key="test-key"),
        )
        assert resp.status_code == 201, resp.text
        created = resp.json()["data"]
        dataset_id = created["id"]
        assert created["name"] == dataset_name
        assert created["owner"] == "alice"
        assert created["project_id"] == project_id

        # Conflict (same name + version + project_id)
        resp = client.post(
            "/datasets",
            json={"name": dataset_name, "project_id": project_id, "version": created["version"]},
            headers=_api_headers(api_key="test-key"),
        )
        assert resp.status_code == 409, resp.text
        assert resp.json()["code"] == 409

        # List datasets
        resp = client.get("/datasets", headers=_api_headers())
        assert resp.status_code == 200, resp.text
        items = resp.json()["data"]
        assert any(d["id"] == dataset_id for d in items)

        # project_id is required
        resp = client.post(
            "/datasets",
            json={"name": "no-project"},
            headers=_api_headers(api_key="test-key"),
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == 400

        # Get dataset stats
        resp = client.get(f"/datasets/{dataset_id}", headers=_api_headers())
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["file_count"] == 0

        # Upload an image + a text file
        img = PILImage.new("RGB", (8, 6), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers=_api_headers(api_key="test-key"),
            files=[
                ("files", ("a.png", img_bytes, "image/png")),
                ("files", ("note.txt", b"hello", "text/plain")),
            ],
        )
        assert resp.status_code == 201, resp.text
        uploaded_files = resp.json()["data"]
        assert len(uploaded_files) == 2
        file_id = uploaded_files[0]["id"]

        # List files
        resp = client.get(f"/datasets/{dataset_id}/files", headers=_api_headers())
        assert resp.status_code == 200, resp.text
        files = resp.json()["data"]
        assert len(files) == 2

        # Image record should be created for image files
        resp = client.get(f"/projects/{project_id}/images", headers=_api_headers())
        assert resp.status_code == 200, resp.text
        images = resp.json()
        assert len(images) == 1
        assert images[0]["dataset_id"] == dataset_id
        assert images[0]["dataset_file_id"] is not None
        assert images[0]["width"] == 8
        assert images[0]["height"] == 6

        # Upload a zip with images (auto-extract)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("nested/x.txt", "x")
            zf.writestr("img1.png", img_bytes)
            zf.writestr("nested/img2.png", img_bytes)
        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers=_api_headers(api_key="test-key"),
            files=[("files", ("pack.zip", zip_buf.getvalue(), "application/zip"))],
        )
        assert resp.status_code == 201, resp.text
        assert len(resp.json()["data"]) >= 3

        resp = client.get(f"/projects/{project_id}/images", headers=_api_headers())
        assert resp.status_code == 200
        images = resp.json()
        assert len(images) == 3

        # Download one file (binary response, not wrapped)
        resp = client.get(f"/datasets/{dataset_id}/files/{file_id}/download", headers=_api_headers())
        assert resp.status_code == 200
        assert resp.content in (img_bytes, b"hello")

        # Download dataset zip (binary response, not wrapped)
        resp = client.get(f"/datasets/{dataset_id}/download", headers=_api_headers())
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("application/zip")
        with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
            names = zf.namelist()
            assert any(n.endswith("/a.png") for n in names)
            assert any(n.endswith("/note.txt") for n in names)

        # Update metadata
        resp = client.patch(
            f"/datasets/{dataset_id}",
            headers=_api_headers(api_key="test-key"),
            json={
                "description": "updated",
                "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["description"] == "updated"
        assert resp.json()["data"]["splits"]["train"] == 0.8

        # Permission: other user cannot read private dataset
        resp = client.get(f"/datasets/{dataset_id}", headers=_api_headers(user="bob"))
        assert resp.status_code == 403, resp.text
        assert resp.json()["code"] == 403

        # Make public
        resp = client.patch(
            f"/datasets/{dataset_id}",
            headers=_api_headers(api_key="test-key"),
            json={"is_public": True},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["is_public"] is True

        # Now bob can read
        resp = client.get(f"/datasets/{dataset_id}", headers=_api_headers(user="bob"))
        assert resp.status_code == 200, resp.text

        # Other user cannot write
        resp = client.patch(
            f"/datasets/{dataset_id}",
            headers=_api_headers(user="bob", api_key="test-key"),
            json={"description": "hacked"},
        )
        assert resp.status_code == 403, resp.text
        assert resp.json()["code"] == 403

        # Delete one file
        resp = client.delete(f"/datasets/{dataset_id}/files/{file_id}", headers=_api_headers(api_key="test-key"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["code"] == 200

        resp = client.get(f"/projects/{project_id}/images", headers=_api_headers())
        assert resp.status_code == 200
        images = resp.json()
        assert len(images) == 2

        resp = client.get(f"/datasets/{dataset_id}/files", headers=_api_headers())
        assert resp.status_code == 200
        assert len(resp.json()["data"]) >= 1

        # Delete dataset
        resp = client.delete(f"/datasets/{dataset_id}", headers=_api_headers(api_key="test-key"))
        assert resp.status_code == 200, resp.text
        assert resp.json()["code"] == 200

        resp = client.get(f"/datasets/{dataset_id}", headers=_api_headers())
        assert resp.status_code == 404, resp.text

        resp = client.get(f"/projects/{project_id}/images", headers=_api_headers())
        assert resp.status_code == 200
        assert resp.json() == []

    # Stored files should be removed (directory may remain)
    ds_dir = (project_storage_root / "projects" / project_id / "datasets" / dataset_id)
    if ds_dir.exists():
        assert list(ds_dir.glob("*")) == []


def test_dataset_validation_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    with TestClient(app) as client:
        resp = client.post("/projects", json={"name": "Test Project", "type": "目标检测"})
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            headers=_api_headers(api_key="test-key"),
            json={
                "name": f"ds-{uuid4().hex[:8]}",
                "project_id": project_id,
                "splits": {"train": 0.5, "val": 0.5, "test": 0.5},
            },
        )
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["code"] == 422
        assert body["message"] == "Validation error"
        assert isinstance(body["data"], list)


def test_project_dataset_annotation_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    with TestClient(app) as client:
        resp = client.post("/projects", json={"name": "Flow Project", "type": "目标检测"})
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            json={"name": f"flow-{uuid4().hex[:8]}", "project_id": project_id},
            headers=_api_headers(api_key="test-key"),
        )
        assert resp.status_code == 201, resp.text
        dataset_id = resp.json()["data"]["id"]

        img = PILImage.new("RGB", (16, 12), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers=_api_headers(api_key="test-key"),
            files=[("files", ("img.png", img_bytes, "image/png"))],
        )
        assert resp.status_code == 201, resp.text

        resp = client.get(f"/datasets/{dataset_id}/stats", headers=_api_headers())
        assert resp.status_code == 200, resp.text
        stats = resp.json()["data"]
        assert stats["image_count"] == 1
        assert stats["avg_width"] == 16
        assert stats["avg_height"] == 12
        assert stats["total_pixels"] == 16 * 12

        resp = client.get(f"/projects/{project_id}/images", params={"dataset_id": dataset_id})
        assert resp.status_code == 200, resp.text
        images = resp.json()
        assert len(images) == 1
        image_id = images[0]["id"]
        assert images[0]["annotations_count"] == 0

        payload = [
            {
                "id": "ann-rect",
                "type": "rect",
                "label": "defect",
                "color": "#ef4444",
                "visible": True,
                "x": 1.0,
                "y": 2.0,
                "width": 3.0,
                "height": 4.0,
            },
            {
                "id": "ann-poly",
                "type": "polygon",
                "label": "defect",
                "color": "#3b82f6",
                "visible": True,
                "points": [0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0],
            },
        ]

        resp = client.put(f"/images/{image_id}/annotations", json=payload)
        assert resp.status_code == 200, resp.text
        anns = resp.json()
        assert {a["id"] for a in anns} == {"ann-rect", "ann-poly"}

        resp = client.get(f"/images/{image_id}/annotations")
        assert resp.status_code == 200, resp.text
        assert len(resp.json()) == 2

        resp = client.get(f"/projects/{project_id}/images", params={"dataset_id": dataset_id})
        assert resp.status_code == 200, resp.text
        assert resp.json()[0]["annotations_count"] == 2

        resp = client.put(f"/images/{image_id}/annotations", json=[])
        assert resp.status_code == 200, resp.text
        assert resp.json() == []

        resp = client.get(f"/projects/{project_id}/images", params={"dataset_id": dataset_id})
        assert resp.status_code == 200, resp.text
        assert resp.json()[0]["annotations_count"] == 0
