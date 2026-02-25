import io
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


def test_train_diagnostics_and_job_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    with TestClient(app) as client:
        resp = client.get("/train/diagnostics")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["code"] == 200
        assert body["data"]["python_executable"]

        resp = client.post("/train", json={"data": "coco128.yaml", "epochs": 2})
        assert resp.status_code == 200, resp.text
        job_id = resp.json().get("job_id")
        assert job_id

        # Poll until the job leaves "queued".
        status = None
        for _ in range(40):
            resp = client.get(f"/train/jobs/{job_id}")
            assert resp.status_code == 200, resp.text
            status = resp.json()["data"]["status"]
            if status in ("running", "completed", "failed"):
                break
            time.sleep(0.05)  # pragma: no cover

        assert status in ("running", "completed", "failed")

        # Logs should be readable; allow a short delay for the writer thread.
        offset = 0
        text = ""
        for _ in range(40):
            resp = client.get(f"/train/jobs/{job_id}/logs", params={"offset": offset})
            assert resp.status_code == 200, resp.text
            chunk = resp.json()["data"]
            offset = chunk["offset"]
            text += chunk["text"]
            if text.strip():
                break
            time.sleep(0.05)  # pragma: no cover

        assert "[train]" in text

        # Job should complete quickly in test mode (mock run when ultralytics is missing).
        for _ in range(80):
            resp = client.get(f"/train/jobs/{job_id}")
            status = resp.json()["data"]["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert status in ("completed", "failed")

        resp = client.get(f"/train/jobs/{job_id}/metrics")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["code"] == 200
        assert "epochs" in body["data"]
        assert "series" in body["data"]

        # Create a lightweight results.csv to exercise the CSV metrics parser (ultralytics-style).
        job_dir = tmp_path / "aipt_home" / "resources" / "training" / job_id / "runs" / "train"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "results.csv").write_text(
            "\n".join(
                [
                    "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B),metrics/mAP50-95(B)",
                    "0,0.9,0.8,0.7,0.1,0.05",
                    "1,0.8,0.7,0.6,0.2,0.10",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        resp = client.get(f"/train/jobs/{job_id}/metrics")
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["epochs"] == [0, 1]
        assert body["series"]["box_loss"] == [0.9, 0.8]
        assert body["series"]["map50"] == [0.1, 0.2]


def test_train_dataset_id_exports_yolo_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Train Project", "type": "目标检测", "storage_root": str(project_storage_root)},
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

        # Upload an image so the dataset has at least one sample.
        img = PILImage.new("RGB", (32, 24), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers={"X-User": "alice", "X-API-Key": "test-key"},
            files=[("files", ("a.png", img_bytes, "image/png"))],
        )
        assert resp.status_code == 201, resp.text

        resp = client.get(f"/projects/{project_id}/images", headers={"X-User": "alice"})
        assert resp.status_code == 200, resp.text
        image_id = resp.json()[0]["id"]

        # Add a simple bbox so labels/train gets written.
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
            "/train",
            json={"data": f"dataset:{dataset_id}", "epochs": 1, "project_id": project_id, "dataset_id": dataset_id},
        )
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

        # Wait for the job to complete; in tests it uses a fast mock run.
        status = None
        for _ in range(80):
            resp = client.get(f"/train/jobs/{job_id}")
            assert resp.status_code == 200, resp.text
            status = resp.json()["data"]["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(0.05)
        assert status in ("completed", "failed")

    # Validate dataset artifacts are materialized under resources_root_dir/training/<job_id>/dataset
    job_dir = tmp_path / "aipt_home" / "resources" / "training" / job_id
    data_yaml = job_dir / "dataset" / "data.yaml"
    assert data_yaml.exists()
    content = data_yaml.read_text(encoding="utf-8")
    assert "train:" in content
    assert "names:" in content


def test_train_incremental_requires_base_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    with TestClient(app) as client:
        resp = client.post("/train", json={"data": "coco128.yaml", "epochs": 1, "mode": "incremental", "project_id": "p"})
        assert resp.status_code == 400, resp.text
        assert "base_model_id" in (resp.json().get("message") or "")
