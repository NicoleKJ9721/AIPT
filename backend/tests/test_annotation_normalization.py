import io
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_models import Annotation  # noqa: E402
from main import _annotation_to_yolo_lines, app  # noqa: E402


def test_annotation_to_yolo_lines_accepts_negative_rect_sizes():
    ann = Annotation(
        image_id="img",
        type="rect",
        label="defect",
        color="#ef4444",
        visible=True,
        x=10.0,
        y=12.0,
        width=-4.0,
        height=-6.0,
    )
    lines = _annotation_to_yolo_lines(ann, {"defect": 0}, img_w=100, img_h=100)
    assert len(lines) == 1
    parts = lines[0].split()
    assert len(parts) == 5
    assert parts[0] == "0"
    assert float(parts[3]) == pytest.approx(0.04, abs=1e-6)
    assert float(parts[4]) == pytest.approx(0.06, abs=1e-6)


def test_annotation_api_normalizes_negative_rect_sizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_storage_root = tmp_path / "project_storage"

    with TestClient(app) as client:
        resp = client.post(
            "/projects",
            json={"name": "Anno Norm", "type": "detection", "storage_root": str(project_storage_root)},
        )
        assert resp.status_code == 201, resp.text
        project_id = resp.json()["id"]

        resp = client.post(
            "/datasets",
            headers={"X-User": "alice", "X-API-Key": "test-key"},
            json={"name": "norm-ds", "project_id": project_id},
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

        resp = client.get(f"/projects/{project_id}/images", headers={"X-User": "alice"})
        assert resp.status_code == 200, resp.text
        image_id = resp.json()[0]["id"]

        put_payload = [
            {
                "id": "neg-rect",
                "type": "rect",
                "label": "defect",
                "color": "#ef4444",
                "visible": True,
                "x": 12.0,
                "y": 8.0,
                "width": -4.0,
                "height": -3.0,
            }
        ]
        resp = client.put(f"/images/{image_id}/annotations", json=put_payload)
        assert resp.status_code == 200, resp.text
        ann = resp.json()[0]
        assert ann["x"] == pytest.approx(8.0, abs=1e-6)
        assert ann["y"] == pytest.approx(5.0, abs=1e-6)
        assert ann["width"] == pytest.approx(4.0, abs=1e-6)
        assert ann["height"] == pytest.approx(3.0, abs=1e-6)

        ann_id = ann["id"]
        resp = client.patch(f"/annotations/{ann_id}", json={"x": 20.0, "width": -6.0})
        assert resp.status_code == 200, resp.text
        patched = resp.json()
        assert patched["x"] == pytest.approx(14.0, abs=1e-6)
        assert patched["width"] == pytest.approx(6.0, abs=1e-6)

        cleanup_ds = client.delete(f"/datasets/{dataset_id}", headers={"X-User": "alice", "X-API-Key": "test-key"})
        assert cleanup_ds.status_code == 200, cleanup_ds.text
        cleanup_project = client.delete(f"/projects/{project_id}")
        assert cleanup_project.status_code == 204, cleanup_project.text

