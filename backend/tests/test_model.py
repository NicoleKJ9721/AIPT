import os
import sys
from types import ModuleType

import numpy as np
from PIL import Image


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model  # noqa: E402


class _DummyBox:
    def __init__(self):
        self.xyxy = np.array([[1.0, 2.0, 3.0, 4.0]])
        self.conf = np.array([0.9])
        self.cls = np.array([1.0])


class _DummyResult:
    def __init__(self):
        self.names = {0: "ok", 1: "defect"}
        self.boxes = [_DummyBox()]


class _DummyYOLO:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._train_calls: list[tuple[str, int]] = []

    def __call__(self, _image: Image.Image):
        return [_DummyResult()]

    def train(self, data: str, epochs: int = 1):
        self._train_calls.append((data, epochs))
        return {"status": "trained", "data": data, "epochs": epochs}


def test_object_detector_predict_and_train(monkeypatch):
    dummy_mod = ModuleType("ultralytics")
    dummy_mod.YOLO = _DummyYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_mod)

    det = model.ObjectDetector("dummy.pt")
    img = Image.new("RGB", (32, 32), color="white")

    detections = det.predict(img)
    assert isinstance(detections, list)
    assert len(detections) == 1
    assert detections[0]["class"] == "defect"
    assert detections[0]["class_id"] == 1
    assert detections[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]

    train_out = det.train("dummy.yaml", epochs=2)
    assert train_out["status"] == "trained"
    assert train_out["epochs"] == 2
