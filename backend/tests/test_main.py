from fastapi.testclient import TestClient
import sys
import os
import io
import time
from PIL import Image
import pytest
from pathlib import Path

# Use an in-memory SQLite DB for tests.
os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
import main as backend_main

def test_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_predict_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    class DummyDetector:
        def predict(self, _image):
            return [{"bbox": [0.0, 0.0, 1.0, 1.0], "confidence": 0.5, "class": "ok", "class_id": 0}]

    # Create a dummy image
    img = Image.new('RGB', (640, 640), color = 'white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    with TestClient(app) as client:
        backend_main.detector = DummyDetector()
        response = client.post(
            "/predict", 
            files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert isinstance(data["detections"], list)

def test_train_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))

    class DummyDetector:
        pass

    with TestClient(app) as client:
        backend_main.detector = DummyDetector()
        response = client.post(
            "/train",
            json={"data": "coco128.yaml", "epochs": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["config"]["epochs"] == 10
        assert data["config"]["imgsz"] == 640  # default value

        job_id = data.get("job_id")
        assert job_id
        # Ensure the background training thread finishes so later tests can start a new job.
        for _ in range(200):
            status_resp = client.get(f"/train/jobs/{job_id}")
            assert status_resp.status_code == 200
            status = status_resp.json()["data"]["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(0.02)
