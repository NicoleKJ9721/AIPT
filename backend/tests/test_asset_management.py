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

from db import SessionLocal  # noqa: E402
from db_models import Dataset, Pipeline, TrainedModel  # noqa: E402
from main import app  # noqa: E402


def _png_bytes(w: int = 8, h: int = 6) -> bytes:
    img = PILImage.new("RGB", (w, h), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_delete_project_removes_storage_and_related_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Asset Project", "type": "detection", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            headers={"X-User": "alice", "X-API-Key": "test-key"},
            json={"name": f"ds-{uuid4().hex[:8]}", "project_id": project_id},
        )
        assert resp.status_code == 201, resp.text
        dataset_id = resp.json()["data"]["id"]

        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers={"X-User": "alice", "X-API-Key": "test-key"},
            files=[("files", ("a.png", _png_bytes(), "image/png"))],
        )
        assert resp.status_code == 201, resp.text

        resp = client.post(
            "/pipelines",
            json={
                "project_id": project_id,
                "name": "pipe-a",
                "description": "",
                "steps": [
                    {
                        "id": "s1",
                        "title": "detect",
                        "model_id": "m1",
                        "conf": 0.25,
                        "iou": 0.7,
                        "max_det": 50,
                    }
                ],
            },
        )
        assert resp.status_code == 201, resp.text
        pipeline_id = resp.json()["data"]["id"]

        model_id = uuid4().hex
        model_dir = project_storage_root / "projects" / project_id / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "best.pt").write_bytes(b"dummy-weights")

        db = SessionLocal()
        try:
            db.add(
                TrainedModel(
                    id=model_id,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    name="asset-model",
                    base_model="yolo26m",
                    weights_path=f"projects/{project_id}/models/{model_id}/best.pt",
                    results_path=None,
                    metrics=None,
                )
            )
            db.commit()
        finally:
            db.close()

        deploy_cache_dir = (tmp_path / "aipt_home" / "resources" / "deploy_cache" / project_id / model_id / "tensorrt" / "end2end_0").resolve()
        deploy_cache_dir.mkdir(parents=True, exist_ok=True)
        (deploy_cache_dir / "model.engine").write_bytes(b"engine")

        resp = client.delete(f"/projects/{project_id}")
        assert resp.status_code == 204, resp.text

        assert not (project_storage_root / "projects" / project_id).exists()
        assert not (tmp_path / "aipt_home" / "resources" / "deploy_cache" / project_id).exists()

        resp = client.get(f"/projects/{project_id}")
        assert resp.status_code == 404, resp.text

        resp = client.get("/datasets", headers={"X-User": "alice"})
        assert resp.status_code == 200, resp.text
        dataset_ids = {d["id"] for d in (resp.json()["data"] or [])}
        assert dataset_id not in dataset_ids

        db = SessionLocal()
        try:
            assert db.get(Dataset, dataset_id) is None
            assert db.get(Pipeline, pipeline_id) is None
            assert db.get(TrainedModel, model_id) is None
        finally:
            db.close()
