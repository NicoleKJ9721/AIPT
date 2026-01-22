import io
import os
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


def _api_headers(user: str = "alice", api_key: str | None = None) -> dict[str, str]:
    headers = {"X-User": user}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _png_bytes(arr: np.ndarray) -> bytes:
    img = PILImage.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_project_and_dataset(client: TestClient, storage_root: Path) -> tuple[str, str]:
    resp = client.post(
        "/projects",
        json={"name": f"p-{uuid4().hex[:8]}", "type": "detection", "storage_root": str(storage_root)},
    )
    assert resp.status_code == 201, resp.text
    project_id = resp.json()["id"]

    resp = client.post(
        "/datasets",
        json={"name": f"ds-{uuid4().hex[:8]}", "project_id": project_id},
        headers=_api_headers(api_key="test-key"),
    )
    assert resp.status_code == 201, resp.text
    dataset_id = resp.json()["data"]["id"]
    return project_id, dataset_id


def test_smart_detect_creates_rects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage_env"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")

    project_root = tmp_path / "storage_project"

    rng = np.random.default_rng(0)
    patch = rng.integers(0, 255, size=(12, 12), dtype=np.uint8)

    placements = [(10, 10), (34, 18), (18, 36)]
    images: list[tuple[str, bytes]] = []
    for idx, (x, y) in enumerate(placements, start=1):
        arr = np.full((64, 64), 40, dtype=np.uint8)
        arr[y : y + patch.shape[0], x : x + patch.shape[1]] = patch
        images.append((f"img{idx}.png", _png_bytes(arr)))

    with TestClient(app) as client:
        project_id, dataset_id = _create_project_and_dataset(client, project_root)

        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers=_api_headers(api_key="test-key"),
            files=[("files", (name, data, "image/png")) for name, data in images],
        )
        assert resp.status_code == 201, resp.text

        resp = client.get(f"/projects/{project_id}/images", params={"dataset_id": dataset_id}, headers=_api_headers())
        assert resp.status_code == 200, resp.text
        project_images = resp.json()
        assert len(project_images) == 3
        reference = project_images[0]

        # Reference box corresponds to the patch placement in the first image.
        x0, y0 = placements[0]
        box = [float(x0), float(y0), float(x0 + patch.shape[1]), float(y0 + patch.shape[0])]

        resp = client.post(
            "/smart-annotation/detect",
            headers=_api_headers(),
            json={
                "dataset_id": dataset_id,
                "reference_image_id": reference["id"],
                "label": "obj",
                "color": "#22c55e",
                "box": box,
                "scope": "dataset",
                "max_images": 3,
                "threshold": 0.9,
                "max_det_per_image": 1,
                "min_distance": 6,
                "dedup_iou": 0.9,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()["data"]
        assert payload["processed_images"] == 3
        assert payload["created_annotations"] == 3

        for img in project_images:
            resp = client.get(f"/images/{img['id']}/annotations", headers=_api_headers())
            assert resp.status_code == 200, resp.text
            anns = resp.json()
            rects = [a for a in anns if a.get("type") == "rect" and a.get("label") == "obj"]
            assert len(rects) == 1


def test_smart_segment_returns_polygon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_STORAGE_DIR", str(tmp_path / "storage_env"))
    monkeypatch.setenv("AIPT_API_KEY", "test-key")
    monkeypatch.setenv("AIPT_SMART_SEG_ENGINE", "flood")

    project_root = tmp_path / "storage_project"

    h = w = 64
    y0 = x0 = 32
    r = 12
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= r**2
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[mask] = 255

    with TestClient(app) as client:
        project_id, dataset_id = _create_project_and_dataset(client, project_root)

        resp = client.post(
            f"/datasets/{dataset_id}/files",
            headers=_api_headers(api_key="test-key"),
            files=[("files", ("circle.png", _png_bytes(arr), "image/png"))],
        )
        assert resp.status_code == 201, resp.text

        resp = client.get(f"/projects/{project_id}/images", params={"dataset_id": dataset_id}, headers=_api_headers())
        assert resp.status_code == 200, resp.text
        project_images = resp.json()
        assert len(project_images) == 1
        image_id = project_images[0]["id"]

        resp = client.post(
            "/smart-annotation/segment",
            headers=_api_headers(),
            json={"image_id": image_id, "point": [x0, y0], "tolerance": 0.2, "simplify": 1.0},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert isinstance(data["points"], list)
        assert len(data["points"]) >= 6
        assert data["area"] > 200
