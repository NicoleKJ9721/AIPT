import os
import sys
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


def test_models_list_delete_and_export_pt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Model Project", "type": "目标检测", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        model_id = uuid4().hex
        model_dir = project_storage_root / "projects" / project_id / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "best.pt").write_bytes(b"dummy-weights")
        model_cache_dir = (tmp_path / "aipt_home" / "resources" / "deploy_cache" / project_id / model_id / "tensorrt" / "end2end_0").resolve()
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        (model_cache_dir / "model.engine").write_bytes(b"engine")

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
                metrics={"map50": 0.1234, "map": 0.0567},
            )
            db.add(rec)
            db.commit()
        finally:
            db.close()

        resp = client.get(f"/projects/{project_id}/models")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["code"] == 200
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == model_id

        resp = client.get(f"/models/{model_id}/export", params={"format": "pt"})
        assert resp.status_code == 200, resp.text
        assert resp.content == b"dummy-weights"

        resp = client.delete(f"/models/{model_id}", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200, resp.text
        assert not model_dir.exists()
        assert not (tmp_path / "aipt_home" / "resources" / "deploy_cache" / project_id / model_id).exists()


def test_models_export_onnx_with_stub_ultralytics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    class DummyYOLO:
        def __init__(self, weights: str):
            self.weights = weights

        def export(
            self,
            format: str = "onnx",
            project: str | None = None,
            name: str | None = None,
            exist_ok: bool = True,
            save_dir: str | None = None,
            device: str | None = None,
            half: bool = False,
            int8: bool = False,
            opset: int = 12,
            simplify: bool = True,
            dynamic: bool = False,
            workspace: int | None = None,
            batch: int | None = None,
            imgsz: int | None = None,
        ) -> str:
            base = Path(save_dir or project or Path(self.weights).parent).resolve()
            out = base / (name or "export")
            out.mkdir(parents=True, exist_ok=True)
            path = out / "model.onnx"
            path.write_bytes(b"dummy-onnx")
            return str(path)

    dummy_mod = ModuleType("ultralytics")
    dummy_mod.YOLO = DummyYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_mod)

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Export Project", "type": "目标检测", "storage_root": str(project_storage_root)},
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
                    name="yolo26m-export",
                    base_model="yolo26m",
                    weights_path=f"projects/{project_id}/models/{model_id}/best.pt",
                    results_path=None,
                    metrics=None,
                )
            )
            db.commit()
        finally:
            db.close()

        resp = client.get(f"/models/{model_id}/export", params={"format": "onnx", "opset": 12, "simplify": True, "dynamic": False})
        assert resp.status_code == 200, resp.text
        assert resp.content == b"dummy-onnx"


def test_models_export_openvino_zips_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    class DummyYOLO:
        def __init__(self, weights: str):
            self.weights = weights

        def export(self, format: str = "openvino", project: str | None = None, name: str | None = None, exist_ok: bool = True) -> str:
            base = Path(project or Path(self.weights).parent).resolve()
            out = base / (name or "export")
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.xml").write_text("<xml />", encoding="utf-8")
            (out / "model.bin").write_bytes(b"bin")
            return str(out)

    dummy_mod = ModuleType("ultralytics")
    dummy_mod.YOLO = DummyYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_mod)

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Export Project 2", "type": "目标检测", "storage_root": str(project_storage_root)},
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
                    name="yolo26m-export-ov",
                    base_model="yolo26m",
                    weights_path=f"projects/{project_id}/models/{model_id}/best.pt",
                    results_path=None,
                    metrics=None,
                )
            )
            db.commit()
        finally:
            db.close()

        resp = client.get(f"/models/{model_id}/export", params={"format": "openvino"})
        assert resp.status_code == 200, resp.text
        assert resp.content[:2] == b"PK"
