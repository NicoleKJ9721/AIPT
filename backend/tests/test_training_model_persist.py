import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import SessionLocal  # noqa: E402
from db import init_db  # noqa: E402
from db_models import Dataset, Project, TrainedModel  # noqa: E402
from main import TrainConfig, _persist_trained_model  # noqa: E402


def test_persist_trained_model_copies_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    init_db()

    project_root = tmp_path / "project_storage"
    project_id = uuid4().hex

    db = SessionLocal()
    try:
        db.add(
            Project(
                id=project_id,
                name="Persist Project",
                type="目标检测",
                status="进行中",
                latest_commit="",
                storage_root=str(project_root),
            )
        )
        db.commit()
        db.add(
            Dataset(
                id="dummy",
                project_id=project_id,
                name="Dummy Dataset",
            )
        )
        db.commit()
    finally:
        db.close()

    job_dir = tmp_path / "job"
    weights_src = job_dir / "runs" / "train" / "weights" / "best.pt"
    weights_src.parent.mkdir(parents=True, exist_ok=True)
    weights_src.write_bytes(b"weights")

    (job_dir / "runs" / "train").mkdir(parents=True, exist_ok=True)
    (job_dir / "runs" / "train" / "results.csv").write_text(
        "\n".join(
            [
                "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B),metrics/mAP50-95(B)",
                "0,0.9,0.8,0.7,0.1,0.05",
                "1,0.8,0.7,0.6,0.2,0.10",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = TrainConfig(
        data="dataset:dummy",
        epochs=2,
        imgsz=640,
        batch=4,
        lr0=0.01,
        model="yolo26m",
        project_id=project_id,
        dataset_id="dummy",
    )
    _persist_trained_model(job_id="job-id", cfg=cfg, job_dir=job_dir)

    db = SessionLocal()
    try:
        rows = db.query(TrainedModel).filter(TrainedModel.project_id == project_id).all()
        assert len(rows) == 1
        rec = rows[0]
        assert rec.weights_path.endswith("/best.pt")
        assert rec.results_path and rec.results_path.endswith("/results.csv")
        assert rec.metrics and "map50" in rec.metrics and "map" in rec.metrics

        weights_dst = project_root / rec.weights_path
        results_dst = project_root / (rec.results_path or "")
        assert weights_dst.exists()
        assert results_dst.exists()
    finally:
        db.close()


def test_persist_trained_model_records_incremental_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPT_HOME_DIR", str(tmp_path / "aipt_home"))
    init_db()

    project_root = tmp_path / "project_storage"
    project_id = uuid4().hex
    dataset_id = uuid4().hex
    parent_id = uuid4().hex

    db = SessionLocal()
    try:
        db.add(
            Project(
                id=project_id,
                name="Incremental Persist Project",
                type="目标检测",
                status="进行中",
                latest_commit="",
                storage_root=str(project_root),
            )
        )
        db.commit()
        db.add(
            Dataset(
                id=dataset_id,
                project_id=project_id,
                name="Dummy Dataset",
            )
        )
        db.commit()

        # Seed a parent model record (for lineage linkage).
        parent_dir = project_root / "projects" / project_id / "models" / parent_id
        parent_dir.mkdir(parents=True, exist_ok=True)
        (parent_dir / "best.pt").write_bytes(b"parent-weights")
        db.add(
            TrainedModel(
                id=parent_id,
                project_id=project_id,
                dataset_id=dataset_id,
                name="base-model",
                base_model="yolo26m",
                weights_path=f"projects/{project_id}/models/{parent_id}/best.pt",
                results_path=None,
                metrics=None,
            )
        )
        db.commit()
    finally:
        db.close()

    job_dir = tmp_path / "job_inc"
    weights_src = job_dir / "runs" / "train" / "weights" / "best.pt"
    weights_src.parent.mkdir(parents=True, exist_ok=True)
    weights_src.write_bytes(b"weights")

    (job_dir / "runs" / "train").mkdir(parents=True, exist_ok=True)
    (job_dir / "runs" / "train" / "results.csv").write_text(
        "\n".join(
            [
                "epoch,metrics/mAP50(B),metrics/mAP50-95(B)",
                "0,0.1,0.05",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = TrainConfig(
        data=f"dataset:{dataset_id}",
        epochs=1,
        imgsz=640,
        batch=4,
        lr0=0.01,
        model=None,
        mode="incremental",
        output_name="base-model-inc",
        base_model_id=parent_id,
        project_id=project_id,
        dataset_id=dataset_id,
    )
    _persist_trained_model(job_id="job-id-inc", cfg=cfg, job_dir=job_dir)

    db = SessionLocal()
    try:
        rows = db.query(TrainedModel).filter(TrainedModel.project_id == project_id).order_by(TrainedModel.created_at.asc()).all()
        assert len(rows) == 2
        inc = [r for r in rows if r.parent_model_id == parent_id]
        assert len(inc) == 1
        rec = inc[0]
        assert rec.train_mode == "incremental"
        assert rec.name == "base-model-inc"
        assert rec.base_model == "yolo26m"
        assert rec.train_config and rec.train_config.get("mode") == "incremental"
        assert rec.train_config.get("base_model_id") == parent_id
    finally:
        db.close()
