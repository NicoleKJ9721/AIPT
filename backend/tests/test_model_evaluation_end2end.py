import io
import os
import sys
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


def test_model_evaluation_forwards_end2end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    predict_kwargs_seen: list[dict] = []

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
            pass

        def predict(self, _source: str, **kwargs):
            predict_kwargs_seen.append(dict(kwargs))
            return [DummyResult()]

    dummy_mod = ModuleType("ultralytics")
    dummy_mod.YOLO = DummyYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_mod)

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Eval Project", "type": "目标检测", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            headers={"X-User": "alice", "X-API-Key": "test-key"},
            json={
                "name": f"ds-{uuid4().hex[:8]}",
                "project_id": project_id,
                "splits": {"train": 0.0, "val": 0.0, "test": 1.0},
            },
        )
        assert resp.status_code == 201, resp.text
        dataset_id = resp.json()["data"]["id"]

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

        # Create a trained model bound to the dataset.
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
                    dataset_id=dataset_id,
                    name="eval-model",
                    base_model="yolo26m",
                    weights_path=f"projects/{project_id}/models/{model_id}/best.pt",
                    results_path=None,
                    metrics=None,
                )
            )
            db.commit()
        finally:
            db.close()

        resp = client.get(f"/models/{model_id}/evaluation", params={"split": "test", "limit": 1, "end2end": False})
        assert resp.status_code == 200, resp.text
        assert predict_kwargs_seen, "predict() was not called"
        assert predict_kwargs_seen[-1].get("end2end") is False

        resp = client.get(f"/models/{model_id}/evaluation", params={"split": "test", "limit": 1})
        assert resp.status_code == 200, resp.text
        assert predict_kwargs_seen, "predict() was not called"
        assert predict_kwargs_seen[-1].get("end2end") is False
