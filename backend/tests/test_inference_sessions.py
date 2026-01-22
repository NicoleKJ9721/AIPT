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


def test_inference_session_is_cached_and_predict_reuses_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    init_calls: list[str] = []
    predict_calls: list[str] = []

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
        def __init__(self, weights: str):
            init_calls.append(str(weights))
            self.weights = weights

        def export(self, format: str = "engine", save_dir: str | None = None, project: str | None = None, name: str | None = None, **_):
            base = Path(save_dir or project or Path(self.weights).parent).resolve()
            out_dir = base / (name or "export")
            out_dir.mkdir(parents=True, exist_ok=True)
            if format == "engine":
                p = (out_dir / "model.engine").resolve()
                p.write_bytes(b"dummy-engine")
                return str(p)
            if format == "openvino":
                # Return an xml file path.
                (out_dir / "model.bin").write_bytes(b"bin")
                p = (out_dir / "model.xml").resolve()
                p.write_text("<xml/>", encoding="utf-8")
                return str(p)
            raise ValueError("unsupported format")

        def predict(self, _source: str, **_kwargs):
            predict_calls.append(self.weights)
            return [DummyResult()]

    dummy_mod = ModuleType("ultralytics")
    dummy_mod.YOLO = DummyYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_mod)

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Infer Project", "type": "目标检测", "storage_root": str(project_storage_root)},
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
                name="yolov8m-test",
                base_model="yolov8m",
                weights_path=f"projects/{project_id}/models/{model_id}/best.pt",
                results_path=None,
                metrics=None,
            )
            db.add(rec)
            db.commit()
        finally:
            db.close()

        payload = {"project_id": project_id, "model_id": model_id, "format": "tensorrt", "device": "0"}

        resp = client.post("/inference/sessions", json=payload)
        assert resp.status_code in (200, 201), resp.text
        session_id = resp.json()["data"]["id"]
        assert session_id
        assert len(init_calls) == 2  # 1) load .pt for export, 2) load exported engine

        # Second create call should reuse the existing session without reloading.
        resp2 = client.post("/inference/sessions", json=payload)
        assert resp2.status_code == 200, resp2.text
        assert resp2.json()["data"]["id"] == session_id
        assert len(init_calls) == 2

        # Predict should run on the already loaded engine model.
        img = tmp_path / "test.jpg"
        Image.new("RGB", (8, 8), color="white").save(img, format="JPEG")
        with img.open("rb") as f:
            resp3 = client.post(
                f"/inference/sessions/{session_id}/predict",
                files={"file": ("test.jpg", f, "image/jpeg")},
                params={"conf": 0.25, "iou": 0.7, "max_det": 50, "imgsz": 640},
            )
        assert resp3.status_code == 200, resp3.text
        data = resp3.json()["data"]
        assert data["session_id"] == session_id
        assert isinstance(data["detections"], list)
        assert len(data["detections"]) == 1
        assert len(init_calls) == 2
        assert len(predict_calls) >= 1

        resp4 = client.delete(f"/inference/sessions/{session_id}")
        assert resp4.status_code == 200, resp4.text
