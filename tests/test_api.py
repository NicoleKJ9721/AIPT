import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_datasets_list() -> None:
    response = client.get("/datasets/")
    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert payload[0]["id"] == "demo-dataset"


def test_training_jobs_list() -> None:
    response = client.get("/training/jobs")
    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert payload[0]["id"] == "job-1"
