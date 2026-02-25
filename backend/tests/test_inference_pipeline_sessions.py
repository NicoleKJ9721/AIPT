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


class _DummyScalar:
    def __init__(self, value: float):
        self._value = float(value)

    def item(self):
        return self._value


class _DummyArray(list):
    def tolist(self):
        return list(self)


class _DummyBox:
    def __init__(self, bbox: tuple[float, float, float, float], conf: float = 0.9, cls_id: int = 0):
        self.xyxy = [_DummyArray([float(v) for v in bbox])]
        self.conf = [_DummyScalar(float(conf))]
        self.cls = [_DummyScalar(float(cls_id))]


class _DummyMasks:
    def __init__(self, polygons: list[list[list[float]]]):
        self.xy = polygons


class _DummyResult:
    def __init__(
        self,
        *,
        boxes: list[_DummyBox] | None = None,
        masks: list[list[list[float]]] | None = None,
        names: dict[int, str] | None = None,
    ):
        self.boxes = boxes if boxes is not None else []
        self.names = names or {0: "obj"}
        if masks is not None:
            self.masks = _DummyMasks(masks)


def _install_dummy_ultralytics(monkeypatch: pytest.MonkeyPatch, call_counts: dict[str, int]) -> None:
    class DummyYOLO:
        def __init__(self, weights: str):
            self.weights = str(weights)
            self.basename = Path(self.weights).name.lower()
            self.names = {0: "obj"}

        def export(
            self,
            format: str = "engine",
            save_dir: str | None = None,
            project: str | None = None,
            name: str | None = None,
            **_,
        ):
            base = Path(save_dir or project or Path(self.weights).parent).resolve()
            out_dir = base / (name or "export")
            out_dir.mkdir(parents=True, exist_ok=True)
            if format == "engine":
                p = (out_dir / f"{Path(self.weights).stem}.engine").resolve()
                p.write_bytes(b"dummy-engine")
                return str(p)
            if format == "openvino":
                (out_dir / "model.bin").write_bytes(b"bin")
                p = (out_dir / "model.xml").resolve()
                p.write_text("<xml/>", encoding="utf-8")
                return str(p)
            raise ValueError(f"unsupported format: {format}")

        def predict(self, source, **_kwargs):
            key = self.basename
            call_counts[key] = int(call_counts.get(key, 0)) + 1

            if isinstance(source, Image.Image):
                w, h = source.size
            else:
                try:
                    with Image.open(str(source)) as img:
                        w, h = img.size
                except Exception:
                    w, h = 64, 64

            if key.endswith("seg.pt"):
                # Segmentation-only output: mask polygon, no boxes.
                poly = [[20.0, 20.0], [44.0, 20.0], [44.0, 44.0], [20.0, 44.0]]
                return [_DummyResult(boxes=[], masks=[poly], names={0: "segment"})]

            if key.endswith("empty.pt"):
                return [_DummyResult(boxes=[], names={0: "empty"})]

            if key.endswith("roi_det.pt"):
                # Emit detections only when fed ROI crops (smaller than full image).
                if w <= 40 and h <= 40:
                    return [_DummyResult(boxes=[_DummyBox((2.0, 2.0, 18.0, 18.0), conf=0.95, cls_id=0)], names={0: "roi"})]
                return [_DummyResult(boxes=[], names={0: "roi"})]

            # Default detector used by legacy model session test.
            return [_DummyResult(boxes=[_DummyBox((1.0, 2.0, 10.0, 20.0), conf=0.91, cls_id=0)], names={0: "ok"})]

    dummy_mod = ModuleType("ultralytics")
    dummy_mod.YOLO = DummyYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_mod)


def _create_project(client: TestClient, storage_root: Path) -> str:
    resp = client.post(
        "/projects",
        json={"name": f"infer-{uuid4().hex[:8]}", "type": "detection", "storage_root": str(storage_root)},
    )
    assert resp.status_code == 201, resp.text
    return str(resp.json()["id"])


def _insert_model(*, project_storage_root: Path, project_id: str, model_id: str, filename: str, name: str) -> str:
    model_dir = project_storage_root / "projects" / project_id / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / filename
    weights_path.write_bytes(b"dummy-weights")

    from db import SessionLocal  # noqa: E402
    from db_models import TrainedModel  # noqa: E402

    db = SessionLocal()
    try:
        rec = TrainedModel(
            id=model_id,
            project_id=project_id,
            dataset_id=None,
            name=name,
            base_model="yolo26m",
            weights_path=f"projects/{project_id}/models/{model_id}/{filename}",
            results_path=None,
            metrics=None,
        )
        db.add(rec)
        db.commit()
    finally:
        db.close()
    return model_id


def _create_pipeline(client: TestClient, project_id: str, name: str, steps: list[dict]) -> str:
    resp = client.post("/pipelines", json={"project_id": project_id, "name": name, "steps": steps})
    assert resp.status_code == 201, resp.text
    return str(resp.json()["data"]["id"])


def _write_test_image(path: Path) -> None:
    Image.new("RGB", (64, 64), color="white").save(path, format="JPEG")


def _close_all_sessions(client: TestClient) -> None:
    status = client.get("/inference/status")
    if status.status_code != 200:
        return
    for sess in list(status.json().get("data", {}).get("sessions", [])):
        client.delete(f"/inference/sessions/{sess['id']}")


def test_inference_session_legacy_model_id_compatibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    call_counts: dict[str, int] = {}
    _install_dummy_ultralytics(monkeypatch, call_counts)

    project_storage_root = tmp_path / "project_storage"
    with TestClient(app) as client:
        _close_all_sessions(client)
        project_id = _create_project(client, project_storage_root)
        model_id = _insert_model(
            project_storage_root=project_storage_root,
            project_id=project_id,
            model_id=uuid4().hex,
            filename="base.pt",
            name="legacy-model",
        )

        resp = client.post("/inference/sessions", json={"project_id": project_id, "model_id": model_id})
        assert resp.status_code in (200, 201), resp.text
        data = resp.json()["data"]
        assert data["kind"] == "model"
        assert data["target_id"] == model_id
        assert data["model_id"] == model_id
        assert data["format"] == "tensorrt"

        status = client.get("/inference/status")
        assert status.status_code == 200, status.text
        sessions = status.json()["data"]["sessions"]
        assert any(s["id"] == data["id"] and s["kind"] == "model" and s["target_id"] == model_id for s in sessions)

        closed = client.delete(f"/inference/sessions/{data['id']}")
        assert closed.status_code == 200, closed.text


def test_pipeline_session_seg_connector_verbose_and_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    call_counts: dict[str, int] = {}
    _install_dummy_ultralytics(monkeypatch, call_counts)

    project_storage_root = tmp_path / "project_storage"
    with TestClient(app) as client:
        _close_all_sessions(client)
        project_id = _create_project(client, project_storage_root)
        seg_model_id = _insert_model(
            project_storage_root=project_storage_root,
            project_id=project_id,
            model_id=uuid4().hex,
            filename="seg.pt",
            name="seg-model",
        )
        det_model_id = _insert_model(
            project_storage_root=project_storage_root,
            project_id=project_id,
            model_id=uuid4().hex,
            filename="roi_det.pt",
            name="roi-detector",
        )

        pipeline_id = _create_pipeline(
            client,
            project_id,
            "seg-connector",
            steps=[
                {
                    "id": "s1",
                    "title": "seg",
                    "model_id": seg_model_id,
                    "conf": 0.25,
                    "iou": 0.7,
                    "max_det": 50,
                    "connector": {
                        "source": "prev_segments",
                        "min_conf": 0.0,
                        "padding": 0.0,
                        "max_regions": 1,
                        "on_empty": "stop",
                    },
                },
                {
                    "id": "s2",
                    "title": "det",
                    "model_id": det_model_id,
                    "conf": 0.25,
                    "iou": 0.7,
                    "max_det": 50,
                },
            ],
        )

        payload = {"project_id": project_id, "kind": "pipeline", "target_id": pipeline_id, "device": "cpu"}
        create = client.post("/inference/sessions", json=payload)
        assert create.status_code in (200, 201), create.text
        session = create.json()["data"]
        assert session["kind"] == "pipeline"
        assert session["target_id"] == pipeline_id
        assert session["target_name"] == "seg-connector"
        session_id = session["id"]

        # Reuse existing pipeline session.
        create2 = client.post("/inference/sessions", json=payload)
        assert create2.status_code == 200, create2.text
        assert create2.json()["data"]["id"] == session_id

        img = tmp_path / "predict.jpg"
        _write_test_image(img)

        with img.open("rb") as f:
            pred = client.post(
                f"/inference/sessions/{session_id}/predict",
                files={"file": ("predict.jpg", f, "image/jpeg")},
            )
        assert pred.status_code == 200, pred.text
        pred_data = pred.json()["data"]
        assert len(pred_data["detections"]) == 1
        assert len(pred_data["merged_detections"]) >= 2
        assert pred_data.get("steps") in (None, [])

        with img.open("rb") as f:
            pred_verbose = client.post(
                f"/inference/sessions/{session_id}/predict",
                files={"file": ("predict.jpg", f, "image/jpeg")},
                params={"verbose": 1},
            )
        assert pred_verbose.status_code == 200, pred_verbose.text
        verbose_data = pred_verbose.json()["data"]
        assert isinstance(verbose_data.get("steps"), list)
        assert len(verbose_data["steps"]) == 2

        status = client.get("/inference/status")
        assert status.status_code == 200, status.text
        sessions = status.json()["data"]["sessions"]
        assert any(s["id"] == session_id and s["kind"] == "pipeline" and s["target_name"] == "seg-connector" for s in sessions)

        closed = client.delete(f"/inference/sessions/{session_id}")
        assert closed.status_code == 200, closed.text


def test_pipeline_session_snapshot_is_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    call_counts: dict[str, int] = {}
    _install_dummy_ultralytics(monkeypatch, call_counts)

    project_storage_root = tmp_path / "project_storage"
    with TestClient(app) as client:
        _close_all_sessions(client)
        project_id = _create_project(client, project_storage_root)
        seg_model_id = _insert_model(
            project_storage_root=project_storage_root,
            project_id=project_id,
            model_id=uuid4().hex,
            filename="seg.pt",
            name="seg-model",
        )
        empty_model_id = _insert_model(
            project_storage_root=project_storage_root,
            project_id=project_id,
            model_id=uuid4().hex,
            filename="empty.pt",
            name="empty-model",
        )

        pipeline_id = _create_pipeline(
            client,
            project_id,
            "snapshot-pipeline",
            steps=[
                {
                    "id": "s1",
                    "title": "seg",
                    "model_id": seg_model_id,
                    "conf": 0.25,
                    "iou": 0.7,
                    "max_det": 50,
                }
            ],
        )

        created = client.post(
            "/inference/sessions",
            json={"project_id": project_id, "kind": "pipeline", "target_id": pipeline_id},
        )
        assert created.status_code in (200, 201), created.text
        old_session_id = created.json()["data"]["id"]

        # Modify saved pipeline after session creation.
        update = client.put(
            f"/pipelines/{pipeline_id}",
            json={
                "name": "snapshot-pipeline",
                "steps": [
                    {
                        "id": "s1",
                        "title": "empty",
                        "model_id": empty_model_id,
                        "conf": 0.25,
                        "iou": 0.7,
                        "max_det": 50,
                    }
                ],
            },
        )
        assert update.status_code == 200, update.text

        img = tmp_path / "predict.jpg"
        _write_test_image(img)

        with img.open("rb") as f:
            old_pred = client.post(
                f"/inference/sessions/{old_session_id}/predict",
                files={"file": ("predict.jpg", f, "image/jpeg")},
            )
        assert old_pred.status_code == 200, old_pred.text
        assert len(old_pred.json()["data"]["detections"]) >= 1

        # New session should use updated pipeline definition.
        created_new = client.post(
            "/inference/sessions",
            json={"project_id": project_id, "kind": "pipeline", "target_id": pipeline_id},
        )
        assert created_new.status_code in (200, 201), created_new.text
        new_session_id = created_new.json()["data"]["id"]

        with img.open("rb") as f:
            new_pred = client.post(
                f"/inference/sessions/{new_session_id}/predict",
                files={"file": ("predict.jpg", f, "image/jpeg")},
            )
        assert new_pred.status_code == 200, new_pred.text
        assert len(new_pred.json()["data"]["detections"]) == 0

        client.delete(f"/inference/sessions/{old_session_id}")
        client.delete(f"/inference/sessions/{new_session_id}")


def test_pipeline_connector_empty_defaults_to_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    call_counts: dict[str, int] = {}
    _install_dummy_ultralytics(monkeypatch, call_counts)

    project_storage_root = tmp_path / "project_storage"
    with TestClient(app) as client:
        _close_all_sessions(client)
        project_id = _create_project(client, project_storage_root)
        empty_model_id = _insert_model(
            project_storage_root=project_storage_root,
            project_id=project_id,
            model_id=uuid4().hex,
            filename="empty.pt",
            name="empty-model",
        )
        det_model_id = _insert_model(
            project_storage_root=project_storage_root,
            project_id=project_id,
            model_id=uuid4().hex,
            filename="roi_det.pt",
            name="det-model",
        )

        pipeline_id = _create_pipeline(
            client,
            project_id,
            "empty-stop",
            steps=[
                {
                    "id": "s1",
                    "title": "empty",
                    "model_id": empty_model_id,
                    "conf": 0.25,
                    "iou": 0.7,
                    "max_det": 50,
                    "connector": {
                        "source": "prev_detections",
                        "min_conf": 0.0,
                        "padding": 0.0,
                        "max_regions": 1,
                    },
                },
                {
                    "id": "s2",
                    "title": "next",
                    "model_id": det_model_id,
                    "conf": 0.25,
                    "iou": 0.7,
                    "max_det": 50,
                },
            ],
        )

        create = client.post(
            "/inference/sessions",
            json={"project_id": project_id, "kind": "pipeline", "target_id": pipeline_id},
        )
        assert create.status_code in (200, 201), create.text
        session_id = create.json()["data"]["id"]

        img = tmp_path / "predict.jpg"
        _write_test_image(img)
        with img.open("rb") as f:
            pred = client.post(
                f"/inference/sessions/{session_id}/predict",
                files={"file": ("predict.jpg", f, "image/jpeg")},
                params={"verbose": 1},
            )
        assert pred.status_code == 200, pred.text
        data = pred.json()["data"]
        assert data["detections"] == []
        assert isinstance(data.get("note"), str)
        assert "stopped early" in data["note"]
        assert isinstance(data.get("steps"), list)
        assert len(data["steps"]) == 1
        assert int(call_counts.get("roi_det.pt", 0)) == 0

        client.delete(f"/inference/sessions/{session_id}")


def test_pipeline_session_create_rejects_model_only_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    call_counts: dict[str, int] = {}
    _install_dummy_ultralytics(monkeypatch, call_counts)

    project_storage_root = tmp_path / "project_storage"
    with TestClient(app) as client:
        _close_all_sessions(client)
        project_id = _create_project(client, project_storage_root)
        model_id = _insert_model(
            project_storage_root=project_storage_root,
            project_id=project_id,
            model_id=uuid4().hex,
            filename="seg.pt",
            name="seg-model",
        )
        pipeline_id = _create_pipeline(
            client,
            project_id,
            "reject-fields",
            steps=[
                {
                    "id": "s1",
                    "title": "seg",
                    "model_id": model_id,
                    "conf": 0.25,
                    "iou": 0.7,
                    "max_det": 50,
                }
            ],
        )

        resp = client.post(
            "/inference/sessions",
            json={
                "project_id": project_id,
                "kind": "pipeline",
                "target_id": pipeline_id,
                "format": "tensorrt",
                "end2end": True,
            },
        )
        assert resp.status_code == 422, resp.text
