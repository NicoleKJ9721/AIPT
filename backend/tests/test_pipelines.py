import json
import os
import sys
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


def test_pipelines_crud_and_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    class DummyScalar:
        def __init__(self, value: float):
            self._value = float(value)

        def item(self):
            return self._value

    class DummyArray(list):
        def tolist(self):
            return list(self)

    class DummyBox:
        def __init__(self):
            self.xyxy = [DummyArray([1.0, 2.0, 10.0, 20.0])]
            self.conf = [DummyScalar(0.9)]
            self.cls = [DummyScalar(0.0)]

    class DummyResult:
        def __init__(self):
            self.boxes = [DummyBox()]
            self.names = {0: "ok"}

    class DummyYOLO:
        def __init__(self, _weights: str):
            self.names = {0: "ok"}

        def predict(self, _source, **_kwargs):
            return [DummyResult()]

    dummy_mod = ModuleType("ultralytics")
    dummy_mod.YOLO = DummyYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_mod)

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Pipe Project", "type": "目标检测", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        model_id = uuid4().hex
        model_dir = project_storage_root / "projects" / project_id / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "best.pt").write_bytes(b"dummy-weights")

        from db import SessionLocal  # noqa: E402
        from db_models import TrainedModel  # noqa: E402

        db = SessionLocal()
        try:
            rec = TrainedModel(
                id=model_id,
                project_id=project_id,
                dataset_id=None,
                name="yolo26m-test",
                base_model="yolo26m",
                weights_path=f"projects/{project_id}/models/{model_id}/best.pt",
                results_path=None,
                metrics=None,
            )
            db.add(rec)
            db.commit()
        finally:
            db.close()

        pipeline_payload = {
            "project_id": project_id,
            "name": "p1",
            "steps": [
                {
                    "id": "s1",
                    "title": "step1",
                    "model_id": model_id,
                    "conf": 0.25,
                    "iou": 0.7,
                    "max_det": 50,
                    "classes": None,
                    "crop": False,
                    "crop_padding": 0.0,
                    "crop_max_regions": None,
                }
            ],
        }

        resp = client.post("/pipelines", json=pipeline_payload)
        assert resp.status_code == 201, resp.text
        pipeline_id = resp.json()["data"]["id"]

        resp = client.get(f"/projects/{project_id}/pipelines")
        assert resp.status_code == 200, resp.text
        ids = [p["id"] for p in resp.json()["data"]]
        assert pipeline_id in ids

        img = tmp_path / "test.jpg"
        Image.new("RGB", (64, 64), color="white").save(img, format="JPEG")

        with img.open("rb") as f:
            resp = client.post(
                f"/pipelines/{pipeline_id}/run",
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["pipeline_id"] == pipeline_id
        assert len(data["steps"]) == 1
        assert len(data["final_detections"]) == 1
        assert len(data["merged_detections"]) == 1

        with img.open("rb") as f:
            resp = client.post(
                "/pipelines/run",
                data={"payload": json.dumps({"project_id": project_id, "steps": pipeline_payload["steps"]})},
                files={"file": ("test.jpg", f, "image/jpeg")},
            )
        assert resp.status_code == 200, resp.text
        data2 = resp.json()["data"]
        assert data2["pipeline_id"] is None
        assert len(data2["steps"]) == 1
        assert len(data2["final_detections"]) == 1


def test_pipeline_fixed_input_roi_crops_and_maps_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    class DummyScalar:
        def __init__(self, value: float):
            self._value = float(value)

        def item(self):
            return self._value

    class DummyArray(list):
        def tolist(self):
            return list(self)

    class DummyBox:
        def __init__(self):
            self.xyxy = [DummyArray([1.0, 2.0, 10.0, 20.0])]
            self.conf = [DummyScalar(0.9)]
            self.cls = [DummyScalar(0.0)]

    class DummyResult:
        def __init__(self):
            self.boxes = [DummyBox()]
            self.names = {0: "ok"}

    class DummyYOLO:
        seen_sizes: list[tuple[int, int]] = []

        def __init__(self, _weights: str):
            self.names = {0: "ok"}

        def predict(self, source, **_kwargs):
            DummyYOLO.seen_sizes.append(tuple(source.size))
            return [DummyResult()]

    dummy_mod = ModuleType("ultralytics")
    dummy_mod.YOLO = DummyYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_mod)

    project_storage_root = tmp_path / "project_storage"
    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Fixed ROI", "type": "目标检测", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        model_id = uuid4().hex
        model_dir = project_storage_root / "projects" / project_id / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "best.pt").write_bytes(b"dummy-weights")

        from db import SessionLocal  # noqa: E402
        from db_models import TrainedModel  # noqa: E402

        db = SessionLocal()
        try:
            db.add(
                TrainedModel(
                    id=model_id,
                    project_id=project_id,
                    dataset_id=None,
                    name="fixed-roi-test",
                    base_model="yolov8s",
                    weights_path=f"projects/{project_id}/models/{model_id}/best.pt",
                    results_path=None,
                    metrics=None,
                )
            )
            db.commit()
        finally:
            db.close()

        img = tmp_path / "input.png"
        Image.new("RGB", (64, 64), color="white").save(img, format="PNG")
        payload = {
            "project_id": project_id,
            "steps": [
                {
                    "id": "s1",
                    "title": "fixed roi",
                    "model_id": model_id,
                    "input_roi": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5},
                }
            ],
        }

        with img.open("rb") as f:
            resp = client.post(
                "/pipelines/run",
                data={"payload": json.dumps(payload)},
                files={"file": ("input.png", f, "image/png")},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert DummyYOLO.seen_sizes == [(32, 32)]
        assert data["final_detections"][0]["bbox"] == [17.0, 18.0, 26.0, 36.0]
        assert "Fixed input ROI applied" in (data["steps"][0]["note"] or "")
