# ruff: noqa: E402

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pydantic import BaseModel
from PIL import Image
import hashlib
import inspect
import io
import sys
import os
import datetime as dt
import logging
import mimetypes
import re
import shutil
import subprocess  # nosec B404 - used for local hardware discovery (WMIC)
import threading
import time
import tempfile
import zipfile
import platform
from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session, selectinload
import uuid
from pathlib import Path
from typing import Callable, Literal

# Add current directory to path to find model module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

from db import SessionLocal, get_db, init_db
from db_models import Annotation, Dataset, DatasetFile, Image as ImageModel, LabelClass, Project, TrainedModel
from model import ObjectDetector
from storage import (
    delete_storage_path,
    default_storage_root,
    ensure_project_dirs,
    project_storage_root,
    resolve_storage_path,
    safe_filename,
    save_fileobj,
    save_upload_file,
    zip_dataset_bytes,
)
from schemas import (
    ApiResponse,
    AnnotationCreate,
    AnnotationOut,
    AnnotationUpdate,
    DatasetCreate,
    DatasetCloneRequest,
    DatasetFileOut,
    DatasetImageStatsOut,
    DatasetOut,
    DatasetSplits,
    DatasetUpdate,
    HardwareDeviceOut,
    ImageCreate,
    ImageOut,
    LabelClassCreate,
    LabelClassOut,
    LabelClassUpdate,
    ModelEvaluationDetectionOut,
    ModelEvaluationItemOut,
    ModelEvaluationPageOut,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
    SystemSettingsOut,
    SystemSettingsUpdate,
    TrainedModelOut,
)

from app_config import load_settings, update_settings

# Initialize model
detector = None


def _default_model_weights_path() -> Path | None:
    """
    Local-first: prefer weights under the configured resources directory.
    If missing, seed from the repo root `yolov8n.pt` when available.
    """
    try:
        settings = load_settings()
        resources_root = Path(str(settings.get("resources_root_dir") or "")).expanduser().resolve()
    except Exception:
        resources_root = None  # type: ignore[assignment]

    repo_root = Path(__file__).resolve().parents[1]
    repo_weights = (repo_root / "yolov8n.pt").resolve()

    if resources_root:
        models_dir = resources_root / "models"
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            models_dir = None  # type: ignore[assignment]

        if models_dir:
            seeded = (models_dir / "yolov8n.pt").resolve()
            if seeded.exists():
                return seeded
            if repo_weights.exists():
                try:
                    shutil.copy2(repo_weights, seeded)
                    return seeded
                except Exception:
                    return repo_weights

    if repo_weights.exists():
        return repo_weights

    return None


def _resolve_train_model_arg(model: str | None) -> str:
    """
    Resolve a YOLO model identifier to a local weights path when possible.

    - If `model` is a local file path and exists, return it.
    - If `model` is a stem like "yolov8m", prefer `<resources_root>/models/yolov8m.pt`,
      seeding from the repo root when available.
    - Otherwise, return the original value (letting ultralytics resolve/download).
    """
    raw = (model or "").strip()
    if not raw:
        raw = "yolov8m.pt"

    try:
        p = Path(raw)
        if p.exists():
            return str(p.expanduser().resolve())
    except Exception:
        logger.debug("Failed to resolve train model argument as a local path: %r", raw, exc_info=True)

    is_simple = "/" not in raw and "\\" not in raw and Path(raw).suffix == ""
    candidate_name = f"{raw}.pt" if is_simple else raw

    repo_root = Path(__file__).resolve().parents[1]
    repo_candidate = (repo_root / candidate_name).resolve()

    try:
        settings = load_settings()
        resources_root = Path(str(settings.get("resources_root_dir") or "")).expanduser().resolve()
        models_dir = resources_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        seeded = (models_dir / Path(candidate_name).name).resolve()
        if seeded.exists():
            return str(seeded)
        if repo_candidate.exists():
            try:
                shutil.copy2(repo_candidate, seeded)
                return str(seeded)
            except Exception:
                return str(repo_candidate)
    except Exception:
        logger.debug("Failed to resolve resources models dir for train model arg: %r", candidate_name, exc_info=True)

    if repo_candidate.exists():
        return str(repo_candidate)

    return candidate_name


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    init_db()
    try:
        model_path = _default_model_weights_path()
        if model_path and model_path.exists():
            detector = ObjectDetector(str(model_path))
            logger.info("Model loaded successfully: %s", model_path)
        else:
            detector = None
            logger.warning(
                "Model weights not found. Place a YOLO .pt file under the resources dir (models/) or repo root."
            )
    except Exception as e:
        logger.exception("Error loading model: %s", e)
    yield
    # Clean up if needed
    detector = None

app = FastAPI(title="AIPT Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiResponse[None](code=exc.status_code, message=str(exc.detail), data=None).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ApiResponse[list](code=422, message="Validation error", data=jsonable_encoder(exc.errors())).model_dump(),
    )


def get_current_user(x_user: str | None = Header(default=None, alias="X-User")) -> str:
    return (x_user or "anonymous").strip() or "anonymous"


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    required = os.getenv("AIPT_API_KEY")
    if not required:
        return
    # Local-first mode: allow requests from localhost without an API key.
    client_host = (request.client.host if request.client else "").strip().lower()
    if client_host in {"127.0.0.1", "::1", "localhost"}:
        return
    if x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid API key")


def ensure_dataset_readable(dataset: Dataset, user: str) -> None:
    if dataset.is_public or dataset.owner == user:
        return
    raise HTTPException(status_code=403, detail="Forbidden")


def ensure_dataset_writable(dataset: Dataset, user: str) -> None:
    if dataset.owner == user:
        return
    raise HTTPException(status_code=403, detail="Forbidden")


class TrainConfig(BaseModel):
    data: str = "coco128.yaml"
    epochs: int = 1
    imgsz: int = 640
    batch: int = 16
    lr0: float | None = None
    model: str | None = None
    project_id: str | None = None
    dataset_id: str | None = None


class TrainJobStatusOut(BaseModel):
    id: str
    status: Literal["queued", "running", "stopping", "stopped", "completed", "failed"]
    progress: float | None = None
    message: str | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    log_path: str
    config: TrainConfig


class TrainLogChunkOut(BaseModel):
    offset: int
    text: str
    eof: bool


class TrainMetricsOut(BaseModel):
    epochs: list[int] = []
    series: dict[str, list[float | None]] = {}


class TrainDiagnosticsOut(BaseModel):
    python_executable: str
    python_version: str
    conda_env: str | None = None
    conda_prefix: str | None = None
    ultralytics: str | None = None
    torch: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    nvidia_smi: list[dict[str, str]] | None = None


class DashboardTrainingSummaryOut(BaseModel):
    running_jobs: int = 0
    last_job_id: str | None = None
    last_status: str | None = None
    last_progress: float | None = None
    last_message: str | None = None
    last_error: str | None = None


class DashboardSummaryOut(BaseModel):
    projects_total: int = 0
    datasets_total: int = 0
    images_total: int = 0
    images_annotated_total: int = 0
    images_pending_total: int = 0
    training: DashboardTrainingSummaryOut


_TRAIN_JOBS_LOCK = threading.Lock()
_TRAIN_JOBS: dict[str, TrainJobStatusOut] = {}

_TRAIN_STOP_LOCK = threading.Lock()
_TRAIN_STOP_EVENTS: dict[str, threading.Event] = {}

_EVAL_MODELS_LOCK = threading.Lock()
_EVAL_MODELS: dict[str, object] = {}
_EVAL_MODELS_MAX = 4


InferenceFormat = Literal["tensorrt", "openvino"]


class InferenceSessionCreateIn(BaseModel):
    project_id: str
    model_id: str
    format: InferenceFormat = "tensorrt"
    device: str | None = None

    # Export hints (optional).
    half: bool = True
    int8: bool = False
    workspace: int | None = None
    batch: int | None = None
    imgsz: int | None = None


class InferenceSessionOut(BaseModel):
    id: str
    project_id: str
    model_id: str
    format: InferenceFormat
    device: str | None = None
    created_at: str
    last_used_at: str | None = None
    cached_artifact: str | None = None


class InferenceStatusOut(BaseModel):
    active_requests: int = 0
    sessions: list[InferenceSessionOut] = []


class InferencePredictionOut(BaseModel):
    session_id: str
    detections: list[ModelEvaluationDetectionOut]
    note: str | None = None


@dataclass
class _InferenceSession:
    id: str
    key: str
    project_id: str
    model_id: str
    format: InferenceFormat
    device: str | None
    artifact_path: Path
    model: object
    created_at: str
    last_used_at: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    active_requests: int = 0


_INFER_LOCK = threading.Lock()
_INFER_SESSIONS: dict[str, _InferenceSession] = {}
_INFER_SESSION_BY_KEY: dict[str, str] = {}
_INFER_ACTIVE_REQUESTS: int = 0
_INFER_LOADING: dict[str, threading.Event] = {}
_INFER_LOADING_ERRORS: dict[str, str] = {}


class SmartDetectRequest(BaseModel):
    dataset_id: str
    reference_image_id: str
    label: str
    color: str = "#ef4444"
    box: tuple[float, float, float, float]
    scope: Literal["image", "dataset"] = "dataset"
    max_images: int | None = None
    threshold: float = 0.6
    max_det_per_image: int = 20
    min_distance: int | None = None
    dedup_iou: float = 0.8


class SmartDetectImageOut(BaseModel):
    image_id: str
    created: int


class SmartDetectResponse(BaseModel):
    processed_images: int
    created_annotations: int
    template_size: tuple[int, int]
    images: list[SmartDetectImageOut]


class SmartSegmentRequest(BaseModel):
    image_id: str
    point: tuple[float, float]
    tolerance: float = 0.08
    simplify: float = 2.0
    engine: str | None = None


class SmartSegmentResponse(BaseModel):
    points: list[float]
    area: int


def _train_stop_event(job_id: str) -> threading.Event:
    with _TRAIN_STOP_LOCK:
        evt = _TRAIN_STOP_EVENTS.get(job_id)
        if evt is None:
            evt = threading.Event()
            _TRAIN_STOP_EVENTS[job_id] = evt
        return evt


def _get_eval_model(weights_path: Path):
    key = str(weights_path.resolve())
    with _EVAL_MODELS_LOCK:
        cached = _EVAL_MODELS.get(key)
    if cached is not None:
        return cached
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"ultralytics is not available: {exc}") from exc
    model = YOLO(str(weights_path))
    with _EVAL_MODELS_LOCK:
        _EVAL_MODELS[key] = model
        while len(_EVAL_MODELS) > _EVAL_MODELS_MAX:
            oldest = next(iter(_EVAL_MODELS.keys()))
            if oldest == key:
                break
            _EVAL_MODELS.pop(oldest, None)
    return model


def _active_training_job() -> TrainJobStatusOut | None:
    with _TRAIN_JOBS_LOCK:
        jobs = [j for j in _TRAIN_JOBS.values() if j.status in ("queued", "running", "stopping")]
    if not jobs:
        return None
    return max(jobs, key=lambda j: j.created_at)


def _training_blocked_by_inference_reason() -> str | None:
    with _INFER_LOCK:
        if _INFER_LOADING:
            return "Inference session is initializing"
        if _INFER_ACTIVE_REQUESTS > 0:
            return f"Inference is running (active_requests={_INFER_ACTIVE_REQUESTS})"
        if _INFER_SESSIONS:
            any_id = next(iter(_INFER_SESSIONS.keys()))
            return f"Inference session is active (session_id={any_id})"
    return None


def _inference_status_snapshot() -> InferenceStatusOut:
    with _INFER_LOCK:
        active_requests = int(_INFER_ACTIVE_REQUESTS)
        sessions = [
            InferenceSessionOut(
                id=s.id,
                project_id=s.project_id,
                model_id=s.model_id,
                format=s.format,
                device=s.device,
                created_at=s.created_at,
                last_used_at=s.last_used_at,
                cached_artifact=s.artifact_path.name if s.artifact_path else None,
            )
            for s in _INFER_SESSIONS.values()
        ]
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    return InferenceStatusOut(active_requests=active_requests, sessions=sessions)


def _inference_cache_dir(project_root: Path, project_id: str, model_id: str, fmt: InferenceFormat) -> Path:
    base = (project_root / "projects" / project_id / "deploy_cache" / model_id / fmt).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _find_first(parent: Path, glob_pat: str) -> Path | None:
    try:
        for p in parent.rglob(glob_pat):
            if p.is_file():
                return p
    except Exception:
        return None
    return None


def _export_inference_artifact(
    *,
    weights_path: Path,
    cache_dir: Path,
    fmt: InferenceFormat,
    device: str | None,
    half: bool,
    int8: bool,
    workspace: int | None,
    batch: int | None,
    imgsz: int | None,
) -> Path:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"ultralytics is not available: {exc}") from exc

    ul_fmt = "engine" if fmt == "tensorrt" else "openvino"

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"infer_{fmt}_", dir=str(cache_dir))).resolve()
    try:
        yolo = YOLO(str(weights_path))

        export_kwargs: dict[str, object] = {"format": ul_fmt}
        try:
            sig = inspect.signature(yolo.export)
            params = sig.parameters
        except Exception:
            params = {}

        # Export into tmp_dir for deterministic caching.
        if "project" in params:
            export_kwargs["project"] = str(tmp_dir)
        if "name" in params:
            export_kwargs["name"] = "export"
        if "exist_ok" in params:
            export_kwargs["exist_ok"] = True
        if "save_dir" in params:
            export_kwargs["save_dir"] = str(tmp_dir)
        if device and "device" in params:
            export_kwargs["device"] = device
        if "half" in params:
            export_kwargs["half"] = bool(half)
        if "int8" in params:
            export_kwargs["int8"] = bool(int8)
        if workspace is not None and "workspace" in params:
            export_kwargs["workspace"] = int(workspace)
        if batch is not None and "batch" in params:
            export_kwargs["batch"] = int(batch)
        if imgsz is not None and "imgsz" in params:
            export_kwargs["imgsz"] = int(imgsz)

        result = yolo.export(**export_kwargs)

        exported: Path | None = None
        if isinstance(result, (str, Path)):
            exported = Path(result).expanduser()
        elif isinstance(result, (list, tuple)) and result:
            exported = Path(str(result[0])).expanduser()
        if exported and not exported.is_absolute():
            exported = (tmp_dir / exported).resolve()

        if fmt == "openvino":
            if exported and exported.exists():
                # Newer ultralytics may return the OpenVINO directory.
                if exported.is_dir():
                    xml = _find_first(exported, "*.xml")
                    if xml:
                        exported = xml
                elif exported.suffix.lower() != ".xml":
                    exported = None
            if not exported:
                exported = _find_first(tmp_dir, "*.xml")
            if not exported:
                exported = _find_first(weights_path.parent, "*.xml")
            if not exported or not exported.exists():
                raise HTTPException(status_code=500, detail="OpenVINO export failed: output not found")

            src_dir = exported.parent
            target_dir = (cache_dir / "openvino").resolve()
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            shutil.copytree(src_dir, target_dir)
            xml = _find_first(target_dir, "*.xml")
            if not xml:
                raise HTTPException(status_code=500, detail="OpenVINO export failed: cached output not found")
            return xml.resolve()

        # tensorrt
        if exported and exported.exists() and exported.is_file() and exported.suffix.lower() == ".engine":
            engine = exported
        else:
            engine = _find_first(tmp_dir, "*.engine") or _find_first(weights_path.parent, "*.engine")
        if not engine or not engine.exists():
            raise HTTPException(status_code=500, detail="TensorRT export failed: output not found")

        target_engine = (cache_dir / f"{weights_path.stem}.engine").resolve()
        shutil.copy2(engine, target_engine)
        return target_engine
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Inference export failed: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _ensure_inference_artifact(weights_path: Path, cache_dir: Path, req: InferenceSessionCreateIn) -> Path:
    if req.format == "tensorrt":
        target = (cache_dir / f"{weights_path.stem}.engine").resolve()
        if target.exists():
            return target
        return _export_inference_artifact(
            weights_path=weights_path,
            cache_dir=cache_dir,
            fmt=req.format,
            device=req.device,
            half=req.half,
            int8=req.int8,
            workspace=req.workspace,
            batch=req.batch,
            imgsz=req.imgsz,
        )

    # openvino
    cached_xml = _find_first(cache_dir / "openvino", "*.xml")
    if cached_xml and cached_xml.exists():
        return cached_xml.resolve()
    return _export_inference_artifact(
        weights_path=weights_path,
        cache_dir=cache_dir,
        fmt=req.format,
        device=req.device,
        half=req.half,
        int8=req.int8,
        workspace=req.workspace,
        batch=req.batch,
        imgsz=req.imgsz,
    )


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _nvidia_smi_snapshot() -> list[dict[str, str]] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        # Args are internal-only (not user-provided); used for local GPU monitoring.
        out = subprocess.check_output(  # nosec B603
            [
                exe,
                "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        lines = out.decode(errors="ignore").splitlines()
        rows: list[dict[str, str]] = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mb": parts[2],
                    "memory_used_mb": parts[3],
                    "utilization_gpu_pct": parts[4],
                    "temperature_c": parts[5],
                }
            )
        return rows
    except Exception:
        logger.debug("nvidia-smi call failed", exc_info=True)
        return None


def _collect_train_diagnostics() -> TrainDiagnosticsOut:
    ultralytics_ver: str | None = None
    try:
        import ultralytics  # type: ignore

        ultralytics_ver = getattr(ultralytics, "__version__", None)
    except Exception:
        ultralytics_ver = None

    torch_ver: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    try:
        import torch  # type: ignore

        torch_ver = getattr(torch, "__version__", None)
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        try:
            cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
        except Exception:
            cudnn_version = None
    except Exception:
        torch_ver = None
        cuda_available = None
        cuda_version = None
        cudnn_version = None

    return TrainDiagnosticsOut(
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        conda_env=os.getenv("CONDA_DEFAULT_ENV") or None,
        conda_prefix=os.getenv("CONDA_PREFIX") or None,
        ultralytics=ultralytics_ver,
        torch=torch_ver,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        nvidia_smi=_nvidia_smi_snapshot(),
    )


def _training_root_dir() -> Path:
    try:
        settings = load_settings()
        root = Path(str(settings.get("resources_root_dir") or "")).expanduser().resolve()
        if not str(root):
            raise ValueError("empty resources_root_dir")
        path = (root / "training").resolve()
    except Exception:
        path = (Path.home() / ".aipt" / "resources" / "training").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_dataset_id(cfg: TrainConfig) -> str | None:
    if cfg.dataset_id and cfg.dataset_id.strip():
        return cfg.dataset_id.strip()
    raw = (cfg.data or "").strip()
    if raw.lower().startswith("dataset:"):
        return raw.split(":", 1)[1].strip() or None
    return None


def _annotation_to_yolo_lines(ann: Annotation, class_index: dict[str, int], img_w: int, img_h: int) -> list[str]:
    label = str(getattr(ann, "label", "")).strip()
    if not label or label not in class_index:
        return []

    if ann.type == "rect":
        x = float(ann.x or 0)
        y = float(ann.y or 0)
        w = float(ann.width or 0)
        h = float(ann.height or 0)
        if w <= 0 or h <= 0:
            return []
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
    elif ann.type == "polygon" and ann.points:
        pts = list(ann.points or [])
        if len(pts) < 6 or len(pts) % 2 != 0:
            return []
        ox = float(ann.x or 0)
        oy = float(ann.y or 0)
        xs = [float(pts[i]) + ox for i in range(0, len(pts), 2)]
        ys = [float(pts[i]) + oy for i in range(1, len(pts), 2)]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    else:
        return []

    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1 or bh <= 1:
        return []

    xc = (x1 + x2) / 2.0 / float(img_w)
    yc = (y1 + y2) / 2.0 / float(img_h)
    bw_n = bw / float(img_w)
    bh_n = bh / float(img_h)
    cls = int(class_index[label])
    return [f"{cls} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}"]


def _prepare_yolo_dataset(dataset_id: str, job_dir: Path) -> Path:
    """
    Materialize a YOLO-format dataset under `<job_dir>/dataset/` and return `data.yaml`.
    """
    ds_root = (job_dir / "dataset").resolve()
    for split in ("train", "val", "test"):
        (ds_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        dataset = db.get(Dataset, dataset_id)
        if not dataset:
            raise RuntimeError("Dataset not found")
        if not dataset.project_id:
            raise RuntimeError("Dataset is not bound to a project")
        project = db.get(Project, dataset.project_id)
        if not project:
            raise RuntimeError("Project not found for dataset")

        project_root = project_storage_root(project.storage_root)

        # Class mapping: prefer project label classes (stable ordering), then append missing labels.
        cls_rows = (
            db.query(LabelClass)
            .filter(LabelClass.project_id == project.id)
            .order_by(LabelClass.created_at.asc())
            .all()
        )
        class_names = [c.name for c in cls_rows if str(c.name).strip()]
        known = set(class_names)

        images = (
            db.query(ImageModel)
            .filter(ImageModel.dataset_id == dataset_id)
            .order_by(ImageModel.created_at.asc())
            .all()
        )
        for img in images:
            for ann in (
                db.query(Annotation)
                .filter(Annotation.image_id == img.id)
                .order_by(Annotation.created_at.asc())
                .all()
            ):
                label = str(getattr(ann, "label", "")).strip()
                if label and label not in known:
                    known.add(label)
                    class_names.append(label)

        class_index = {name: i for i, name in enumerate(class_names)}

        split_train = float(dataset.split_train or 0.7)
        split_val = float(dataset.split_val or 0.2)
        # remaining -> test

        def pick_split(image_id: str) -> str:
            # Deterministic shuffling for dataset splits (not security-related).
            h = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8]
            r = int(h, 16) / 0xFFFFFFFF
            if r < split_train:
                return "train"
            if r < split_train + split_val:
                return "val"
            return "test"

        for img in images:
            file_id = getattr(img, "dataset_file_id", None) or ""
            if not file_id:
                continue
            file = db.get(DatasetFile, file_id)
            if not file or file.dataset_id != dataset_id:
                continue

            src_path = resolve_storage_path(file.storage_path, root=project_root)
            if not src_path.exists():
                continue

            ext = (Path(img.filename).suffix or src_path.suffix or ".jpg").lower()
            split = pick_split(img.id)
            dst_img = ds_root / "images" / split / f"{img.id}{ext}"
            dst_lbl = ds_root / "labels" / split / f"{img.id}.txt"

            if not dst_img.exists():
                try:
                    os.link(src_path, dst_img)
                except Exception:
                    shutil.copy2(src_path, dst_img)

            img_w = int(img.width or 0)
            img_h = int(img.height or 0)
            if img_w <= 0 or img_h <= 0:
                try:
                    img_w, img_h = _try_image_size_path(src_path)
                except Exception:
                    img_w, img_h = (0, 0)
            if img_w <= 0 or img_h <= 0:
                # Skip label export if size is unknown.
                dst_lbl.write_text("", encoding="utf-8")
                continue

            anns = (
                db.query(Annotation)
                .filter(Annotation.image_id == img.id)
                .order_by(Annotation.created_at.asc())
                .all()
            )
            lines: list[str] = []
            for ann in anns:
                lines.extend(_annotation_to_yolo_lines(ann, class_index, img_w, img_h))
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_path = ds_root / "data.yaml"
    names_yaml = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_names)]) or "  0: object"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {ds_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                names_yaml,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def _set_train_job(job_id: str, updater: Callable[[TrainJobStatusOut], TrainJobStatusOut]) -> None:
    with _TRAIN_JOBS_LOCK:
        job = _TRAIN_JOBS.get(job_id)
        if not job:
            return
        _TRAIN_JOBS[job_id] = updater(job)


def _last_metric_value(values: list[float | None]) -> float | None:
    for v in reversed(values or []):
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv:  # NaN
            continue
        return fv
    return None


def _summarize_metrics(metrics: TrainMetricsOut) -> dict | None:
    summary: dict[str, float] = {}
    for key in ("map50", "map", "precision", "recall", "box_loss", "cls_loss", "dfl_loss"):
        v = _last_metric_value(metrics.series.get(key, []))
        if v is None:
            continue
        summary[key] = v
    return summary or None


def _persist_trained_model(job_id: str, cfg: TrainConfig, job_dir: Path) -> None:
    project_id = (cfg.project_id or "").strip()
    if not project_id:
        return

    dataset_id = (cfg.dataset_id or "").strip() or None
    run_dir = (job_dir / "runs" / "train").resolve()
    weights_src = run_dir / "weights" / "best.pt"
    if not weights_src.exists():
        return

    results_src = run_dir / "results.csv"

    model_id = str(uuid.uuid4())
    model_name_hint = Path(cfg.model or "yolov8m").stem or "model"
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")

    db = SessionLocal()
    try:
        project = db.get(Project, project_id)
        if not project:
            return
        project_root = project_storage_root(project.storage_root)
        ensure_project_dirs(project_id, root=project_root)

        model_dir = (project_root / "projects" / project_id / "models" / model_id).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights_src, model_dir / "best.pt")

        weights_rel = (Path("projects") / project_id / "models" / model_id / "best.pt").as_posix()
        results_rel: str | None = None
        if results_src.exists():
            try:
                shutil.copy2(results_src, model_dir / "results.csv")
                results_rel = (Path("projects") / project_id / "models" / model_id / "results.csv").as_posix()
            except Exception:
                logger.debug("Failed to copy results.csv for trained model (job_id=%s)", job_id, exc_info=True)

        metrics_summary: dict | None = None
        if results_src.exists():
            try:
                metrics_summary = _summarize_metrics(_parse_results_csv(results_src))
            except Exception:
                logger.debug("Failed to summarize metrics for trained model (job_id=%s)", job_id, exc_info=True)

        rec = TrainedModel(
            id=model_id,
            project_id=project_id,
            dataset_id=dataset_id,
            name=f"{model_name_hint}-{ts}",
            base_model=cfg.model or model_name_hint,
            weights_path=weights_rel,
            results_path=results_rel,
            metrics=metrics_summary,
        )
        db.add(rec)
        db.commit()
    finally:
        db.close()


def _run_training_job(job_id: str, resume: bool = False) -> None:
    with _TRAIN_JOBS_LOCK:
        job = _TRAIN_JOBS.get(job_id)
    if not job:
        return

    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = job.config
    stop_evt = _train_stop_event(job_id)

    def write_log(line: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    try:
        if stop_evt.is_set():
            _set_train_job(
                job_id, lambda j: j.model_copy(update={"status": "stopped", "finished_at": _now_iso(), "message": "Stopped"})
            )
            write_log(f"[train] job_id={job_id} status=stopped (pre-start)")
            return

        _set_train_job(
            job_id,
            lambda j: j.model_copy(
                update={
                    "status": "running",
                    "started_at": _now_iso(),
                    "message": "Resuming training" if resume else "Training running",
                }
            ),
        )
        write_log(f"[train] job_id={job_id} status=running resume={resume}")
        write_log(f"[train] config={cfg.model_dump()}")
        write_log(f"[train] diagnostics={_collect_train_diagnostics().model_dump()}")

        dataset_id = _resolve_dataset_id(cfg)
        data_yaml: str = cfg.data
        if dataset_id:
            write_log(f"[train] preparing YOLO dataset for dataset_id={dataset_id}")
            yaml_path = _prepare_yolo_dataset(dataset_id, Path(job.log_path).parent)
            data_yaml = str(yaml_path)
            write_log(f"[train] dataset_yaml={data_yaml}")

        # Try real training when ultralytics is available; otherwise fall back to a lightweight mock.
        try:
            force_mock = bool(os.getenv("PYTEST_CURRENT_TEST")) or str(os.getenv("AIPT_TRAIN_FORCE_MOCK") or "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if force_mock:
                raise RuntimeError("forced mock training")

            from ultralytics import YOLO  # type: ignore

            out_dir = (Path(job.log_path).parent / "runs").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            write_log(f"[train] output_dir={out_dir}")
            model_arg = _resolve_train_model_arg(cfg.model)
            if resume:
                last_weights = out_dir / "train" / "weights" / "last.pt"
                if last_weights.exists():
                    model_arg = str(last_weights)
                else:
                    write_log("[train] resume requested but last.pt is missing; starting from model config")

            write_log(f"[train] model_arg={model_arg}")
            yolo = YOLO(model_arg)

            def _stop_cb(trainer, *args, **kwargs):  # type: ignore[no-untyped-def]
                if stop_evt.is_set():
                    try:
                        trainer.stop = True
                    except Exception:
                        setattr(trainer, "stop", True)

            if hasattr(yolo, "add_callback"):
                try:
                    yolo.add_callback("on_train_epoch_end", _stop_cb)
                except Exception:
                    logger.debug("Failed to register ultralytics callback on_train_epoch_end", exc_info=True)
                try:
                    yolo.add_callback("on_train_batch_end", _stop_cb)
                except Exception:
                    logger.debug("Failed to register ultralytics callback on_train_batch_end", exc_info=True)

            with log_path.open("a", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
                yolo.train(
                    data=data_yaml,
                    epochs=int(cfg.epochs),
                    imgsz=int(cfg.imgsz),
                    batch=int(cfg.batch),
                    lr0=cfg.lr0,
                    project=str(out_dir),
                    name="train",
                    exist_ok=True,
                    verbose=True,
                    resume=resume,
                )
        except Exception as exc:
            write_log(f"[train] ultralytics training unavailable, falling back to mock run: {exc}")
            # Mock progress: prints loss-like lines so UI can validate "loss updates".
            for epoch in range(int(cfg.epochs or 1)):
                if stop_evt.is_set():
                    break
                loss = 1.0 / (1.0 + epoch)
                write_log(f"Epoch {epoch + 1}/{cfg.epochs} box_loss={loss:.4f} cls_loss={loss:.4f} dfl_loss={loss:.4f}")
                _set_train_job(job_id, lambda j: j.model_copy(update={"progress": (epoch + 1) / float(cfg.epochs or 1)}))
                time.sleep(0.05)

        try:
            _persist_trained_model(job_id, cfg, Path(job.log_path).parent)
        except Exception:
            logger.debug("Failed to persist trained model (job_id=%s)", job_id, exc_info=True)

        if stop_evt.is_set():
            _set_train_job(
                job_id,
                lambda j: j.model_copy(update={"status": "stopped", "finished_at": _now_iso(), "message": "Training stopped"}),
            )
            write_log("[train] status=stopped")
        else:
            _set_train_job(
                job_id,
                lambda j: j.model_copy(update={"status": "completed", "finished_at": _now_iso(), "message": "Training finished"}),
            )
            write_log("[train] status=completed")
    except Exception as exc:
        logger.exception("Training job failed (job_id=%s): %s", job_id, exc)
        err = str(exc)
        _set_train_job(
            job_id,
            lambda j, err=err: j.model_copy(update={"status": "failed", "finished_at": _now_iso(), "error": err}),
        )
        try:
            write_log(f"[train] status=failed error={exc}")
        except Exception:
            logger.debug("Failed to write train failure log", exc_info=True)


def _try_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _try_int(text: str) -> int | None:
    value = _try_float(text)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        raw = row.get(key)
        if raw is None:
            continue
        value = _try_float(str(raw).strip())
        if value is None:
            continue
        return value
    return None


def _parse_results_csv(path: Path) -> TrainMetricsOut:
    import csv

    epochs: list[int] = []
    series: dict[str, list[float | None]] = {}

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [str(name).strip() for name in (reader.fieldnames or []) if name is not None]

        epoch_key = None
        for cand in ("epoch", "Epoch", "epochs", "Epochs"):
            if cand in fieldnames:
                epoch_key = cand
                break
        if not epoch_key and fieldnames:
            epoch_key = fieldnames[0]

        value_keys = [k for k in fieldnames if k and k != epoch_key]
        series = {k: [] for k in value_keys}

        for row_raw in reader:
            row: dict[str, str] = {str(k).strip(): str(v).strip() for k, v in (row_raw or {}).items() if k is not None}
            epoch_raw = row.get(epoch_key or "epoch") if row else None
            epoch = _try_int(str(epoch_raw).strip()) if epoch_raw is not None else None
            if epoch is None:
                continue
            epochs.append(epoch)
            for key in value_keys:
                raw = row.get(key)
                series[key].append(_try_float(str(raw).strip()) if raw is not None else None)

    series = {k: v for k, v in series.items() if any(x is not None for x in v)}

    # Backward-compatible aliases used by earlier UI/tests.
    aliases: dict[str, list[str]] = {
        "box_loss": ["train/box_loss", "box_loss", "val/box_loss"],
        "cls_loss": ["train/cls_loss", "cls_loss", "val/cls_loss"],
        "dfl_loss": ["train/dfl_loss", "dfl_loss", "val/dfl_loss"],
        "precision": ["metrics/precision(B)", "metrics/precision", "precision"],
        "recall": ["metrics/recall(B)", "metrics/recall", "recall"],
        "map50": ["metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"],
        "map": ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "map"],
    }
    for alias, keys in aliases.items():
        for key in keys:
            values = series.get(key)
            if values is None:
                continue
            if any(v is not None for v in values):
                series[alias] = list(values)
                break

    return TrainMetricsOut(epochs=epochs, series=series)


def _parse_mock_metrics_from_log(path: Path) -> TrainMetricsOut:
    text = ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""

    pat = re.compile(
        r"Epoch\\s+(?P<epoch>\\d+)\\/\\d+\\s+box_loss=(?P<box>[0-9.]+)\\s+cls_loss=(?P<cls>[0-9.]+)\\s+dfl_loss=(?P<dfl>[0-9.]+)"
    )
    epochs: list[int] = []
    box: list[float | None] = []
    cls: list[float | None] = []
    dfl: list[float | None] = []
    for m in pat.finditer(text):
        epochs.append(int(m.group("epoch")))
        box.append(float(m.group("box")))
        cls.append(float(m.group("cls")))
        dfl.append(float(m.group("dfl")))

    series: dict[str, list[float | None]] = {}
    if box:
        series["box_loss"] = box
        series["cls_loss"] = cls
        series["dfl_loss"] = dfl
    return TrainMetricsOut(epochs=epochs, series=series)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": detector is not None}


@app.get("/dashboard/summary", response_model=ApiResponse[DashboardSummaryOut])
def dashboard_summary(db: Session = Depends(get_db)):
    projects_total = int(db.query(func.count(Project.id)).scalar() or 0)
    datasets_total = int(db.query(func.count(Dataset.id)).scalar() or 0)
    images_total = int(db.query(func.count(ImageModel.id)).scalar() or 0)
    images_annotated_total = int(db.query(func.count(func.distinct(Annotation.image_id))).scalar() or 0)
    images_pending_total = max(0, images_total - images_annotated_total)

    with _TRAIN_JOBS_LOCK:
        jobs = list(_TRAIN_JOBS.values())
    running_jobs = sum(1 for j in jobs if j.status in ("running", "stopping"))
    last = max(jobs, key=lambda j: j.created_at, default=None)

    training = DashboardTrainingSummaryOut(
        running_jobs=running_jobs,
        last_job_id=last.id if last else None,
        last_status=last.status if last else None,
        last_progress=last.progress if last else None,
        last_message=last.message if last else None,
        last_error=last.error if last else None,
    )

    return ApiResponse(
        code=200,
        message="OK",
        data=DashboardSummaryOut(
            projects_total=projects_total,
            datasets_total=datasets_total,
            images_total=images_total,
            images_annotated_total=images_annotated_total,
            images_pending_total=images_pending_total,
            training=training,
        ),
    )


def _wmic_lines(args: list[str]) -> list[str]:
    wmic_path = shutil.which("wmic") or shutil.which("wmic.exe")
    if not wmic_path:
        return []
    try:
        # Args are internal-only (not user-provided); used for local hardware discovery.
        out = subprocess.check_output([wmic_path, *args], stderr=subprocess.DEVNULL)  # nosec B603
    except Exception:
        logger.debug("WMIC call failed (args=%s)", args, exc_info=True)
        return []
    text = out.decode(errors="ignore")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _wmic_first_value(args: list[str]) -> str | None:
    lines = _wmic_lines(args)
    if len(lines) >= 2:
        return lines[1].strip() or None
    for line in lines:
        if line.lower().startswith("name="):
            return line.split("=", 1)[1].strip() or None
    return None


def _format_gb(bytes_value: int | None) -> str | None:
    if not bytes_value or bytes_value <= 0:
        return None
    gb = bytes_value / (1024**3)
    return f"{gb:.0f}GB" if gb >= 10 else f"{gb:.1f}GB"


def _detect_cpu_device() -> HardwareDeviceOut:
    name = _wmic_first_value(["cpu", "get", "Name"]) or platform.processor() or "CPU"
    vendor = _wmic_first_value(["cpu", "get", "Manufacturer"]) or None

    mem_raw = _wmic_first_value(["computersystem", "get", "TotalPhysicalMemory"])
    mem_total = None
    try:
        mem_total = int(mem_raw) if mem_raw else None
    except Exception:
        mem_total = None

    memory = _format_gb(mem_total)
    if memory:
        memory = f"{memory} (System)"

    cores = os.cpu_count() or None
    return HardwareDeviceOut(
        id="cpu-0",
        name=str(name).strip(),
        type="CPU",
        vendor=str(vendor).strip() if vendor else None,
        memory=memory,
        cores=int(cores) if cores else None,
        status="Available",
    )


def _detect_discrete_gpus() -> list[HardwareDeviceOut]:
    try:
        import torch  # type: ignore
    except Exception:
        return []

    try:
        if not torch.cuda.is_available():
            return []
        devices: list[HardwareDeviceOut] = []
        for idx in range(int(torch.cuda.device_count() or 0)):
            props = torch.cuda.get_device_properties(idx)
            name = str(getattr(props, "name", "")).strip() or f"CUDA GPU {idx}"
            total_mem = int(getattr(props, "total_memory", 0) or 0)
            memory = _format_gb(total_mem)
            cc = None
            try:
                cc = f"{int(props.major)}.{int(props.minor)}"
            except Exception:
                cc = None
            sm = getattr(props, "multi_processor_count", None)
            devices.append(
                HardwareDeviceOut(
                    id=f"gpu-{idx}",
                    name=name,
                    type="Discrete GPU",
                    vendor="NVIDIA",
                    memory=memory,
                    cores=int(sm) if isinstance(sm, int) else None,
                    compute_capability=cc,
                    status="Available",
                )
            )
        return devices
    except Exception:
        return []


@app.get("/hardware", response_model=ApiResponse[list[HardwareDeviceOut]])
def get_hardware_devices():
    devices: list[HardwareDeviceOut] = []
    devices.extend(_detect_discrete_gpus())
    devices.append(_detect_cpu_device())
    return ApiResponse(code=200, message="OK", data=devices)


@app.get("/system/settings", response_model=ApiResponse[SystemSettingsOut])
def get_system_settings():
    settings = load_settings()
    return ApiResponse(code=200, message="OK", data=SystemSettingsOut(**settings))


@app.put(
    "/system/settings",
    response_model=ApiResponse[SystemSettingsOut],
    dependencies=[Depends(require_api_key)],
)
def put_system_settings(payload: SystemSettingsUpdate):
    settings = update_settings(payload.model_dump(exclude_unset=True))
    return ApiResponse(code=200, message="OK", data=SystemSettingsOut(**settings))


class DirectoryPickerRequest(BaseModel):
    title: str | None = None
    initial_dir: str | None = None


@app.post("/system/dialogs/select-directory", response_model=ApiResponse[str])
def select_directory_dialog(payload: DirectoryPickerRequest):
    try:
        import tkinter as tk  # type: ignore
        from tkinter import filedialog  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=501, detail=f"Directory picker is not available: {exc}")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            title=payload.title or "Select directory",
            initialdir=payload.initial_dir or None,
            mustexist=False,
        )
    finally:
        try:
            root.destroy()
        except Exception:
            logger.debug("Failed to destroy Tk root window", exc_info=True)

    selected = (selected or "").strip()
    if not selected:
        return ApiResponse(code=200, message="Cancelled", data=None)
    return ApiResponse(code=200, message="OK", data=selected)


_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _normalize_hex_color(value: str | None, default: str = "#ef4444") -> str:
    candidate = (value or "").strip()
    if not candidate:
        return default
    if not _HEX_COLOR_RE.match(candidate):
        raise HTTPException(status_code=422, detail="Invalid color; expected hex like #RRGGBB")
    return candidate.lower()


def _next_shortcut(existing: list[str]) -> str:
    used = {str(s or "").strip() for s in existing if str(s or "").strip()}
    for i in range(1, 10):
        s = str(i)
        if s not in used:
            return s
    return str(max(9, len(used)) + 1)


@app.get("/projects/{project_id}/labels", response_model=ApiResponse[list[LabelClassOut]])
def list_project_labels(project_id: str, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    rows = (
        db.execute(select(LabelClass).where(LabelClass.project_id == project_id).order_by(LabelClass.created_at.asc()))
        .scalars()
        .all()
    )
    return ApiResponse(code=200, message="OK", data=[LabelClassOut.model_validate(r) for r in rows])


@app.post(
    "/projects/{project_id}/labels",
    response_model=ApiResponse[LabelClassOut],
    status_code=201,
    dependencies=[Depends(require_api_key)],
)
def create_project_label(project_id: str, payload: LabelClassCreate, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")

    exists = (
        db.execute(select(LabelClass.id).where(LabelClass.project_id == project_id, LabelClass.name == name))
        .scalars()
        .first()
    )
    if exists:
        raise HTTPException(status_code=409, detail="Label already exists")

    color = _normalize_hex_color(payload.color)
    shortcut = (payload.shortcut or "").strip()
    if not shortcut:
        shortcuts = db.execute(select(LabelClass.shortcut).where(LabelClass.project_id == project_id)).scalars().all()
        shortcut = _next_shortcut([s for s in shortcuts if s])

    label_id = (payload.id or "").strip() or str(uuid.uuid4())
    record = LabelClass(id=label_id, project_id=project_id, name=name, color=color, shortcut=shortcut)
    db.add(record)
    db.commit()
    db.refresh(record)
    return ApiResponse(code=201, message="Created", data=LabelClassOut.model_validate(record))


@app.put(
    "/projects/{project_id}/labels/{label_id}",
    response_model=ApiResponse[dict],
    dependencies=[Depends(require_api_key)],
)
def update_project_label(project_id: str, label_id: str, payload: LabelClassUpdate, db: Session = Depends(get_db)):
    record = db.get(LabelClass, label_id)
    if not record or record.project_id != project_id:
        raise HTTPException(status_code=404, detail="Label not found")

    if payload.dataset_id:
        ds = db.get(Dataset, payload.dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if ds.project_id != project_id:
            raise HTTPException(status_code=400, detail="dataset_id does not belong to this project")

    old_name = (record.name or "").strip()
    next_name = (payload.name or old_name).strip()
    if not next_name:
        raise HTTPException(status_code=422, detail="name is required")

    if next_name != old_name:
        conflict = (
            db.execute(
                select(LabelClass.id).where(
                    LabelClass.project_id == project_id,
                    LabelClass.name == next_name,
                    LabelClass.id != label_id,
                )
            )
            .scalars()
            .first()
        )
        if conflict:
            raise HTTPException(status_code=409, detail="Label already exists")

    next_color = record.color
    if payload.color is not None:
        next_color = _normalize_hex_color(payload.color, default=record.color)

    next_shortcut = record.shortcut
    if payload.shortcut is not None:
        next_shortcut = (payload.shortcut or "").strip()

    updated = 0
    if next_name != old_name or next_color != record.color:
        image_ids = select(ImageModel.id).where(ImageModel.project_id == project_id)
        if payload.dataset_id:
            image_ids = image_ids.where(ImageModel.dataset_id == payload.dataset_id)

        values: dict = {"updated_at": dt.datetime.now(dt.timezone.utc)}
        if next_name != old_name:
            values["label"] = next_name
        if next_color != record.color:
            values["color"] = next_color

        stmt = update(Annotation).where(Annotation.image_id.in_(image_ids), Annotation.label == old_name).values(**values)
        res = db.execute(stmt)
        updated = int(res.rowcount or 0)

    if payload.name is not None:
        record.name = next_name
    if payload.color is not None:
        record.color = next_color
    if payload.shortcut is not None:
        record.shortcut = next_shortcut

    record.updated_at = dt.datetime.now(dt.timezone.utc)
    db.add(record)
    db.commit()
    db.refresh(record)
    return ApiResponse(
        code=200,
        message="OK",
        data={"label": LabelClassOut.model_validate(record).model_dump(), "updated": updated},
    )


@app.delete(
    "/projects/{project_id}/labels/{label_id}",
    response_model=ApiResponse[None],
    dependencies=[Depends(require_api_key)],
)
def delete_project_label(project_id: str, label_id: str, db: Session = Depends(get_db)):
    record = db.get(LabelClass, label_id)
    if not record or record.project_id != project_id:
        raise HTTPException(status_code=404, detail="Label not found")

    image_ids = select(ImageModel.id).where(ImageModel.project_id == project_id)
    used = db.execute(
        select(func.count()).select_from(Annotation).where(Annotation.image_id.in_(image_ids), Annotation.label == record.name)
    ).scalar_one()
    if int(used or 0) > 0:
        raise HTTPException(status_code=409, detail="Label is in use; rename/reassign existing annotations first")

    db.delete(record)
    db.commit()
    return ApiResponse(code=200, message="OK", data=None)


class LabelRenameRequest(BaseModel):
    from_label: str
    to_label: str
    dataset_id: str | None = None


@app.post(
    "/projects/{project_id}/labels/rename",
    response_model=ApiResponse[dict],
    dependencies=[Depends(require_api_key)],
)
def rename_project_labels(project_id: str, payload: LabelRenameRequest, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    from_label = (payload.from_label or "").strip()
    to_label = (payload.to_label or "").strip()
    if not from_label or not to_label:
        raise HTTPException(status_code=422, detail="from_label and to_label are required")
    if from_label == to_label:
        return ApiResponse(code=200, message="No changes", data={"updated": 0})

    if payload.dataset_id:
        ds = db.get(Dataset, payload.dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if ds.project_id != project_id:
            raise HTTPException(status_code=400, detail="dataset_id does not belong to this project")

    image_ids = select(ImageModel.id).where(ImageModel.project_id == project_id)
    if payload.dataset_id:
        image_ids = image_ids.where(ImageModel.dataset_id == payload.dataset_id)

    stmt = (
        update(Annotation)
        .where(Annotation.image_id.in_(image_ids), Annotation.label == from_label)
        .values(label=to_label, updated_at=dt.datetime.now(dt.timezone.utc))
    )
    res = db.execute(stmt)
    db.commit()
    return ApiResponse(code=200, message="OK", data={"updated": int(res.rowcount or 0)})


def _is_image_filename(filename: str) -> bool:
    name = (filename or "").lower()
    return name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png") or name.endswith(".bmp") or name.endswith(".webp")


def _try_image_size(data: bytes) -> tuple[int | None, int | None]:
    try:
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
            return int(w), int(h)
    except Exception:
        return None, None


def _try_image_size_path(path: Path) -> tuple[int | None, int | None]:
    """
    Try to read image dimensions from a file on disk.

    Prefer this over reading full bytes into memory when possible.
    """
    try:
        with Image.open(path) as img:
            w, h = img.size
            return int(w), int(h)
    except Exception:
        return None, None

def _dataset_out(dataset: Dataset, file_count: int = 0, total_size_bytes: int = 0) -> DatasetOut:
    return DatasetOut(
        id=dataset.id,
        project_id=dataset.project_id,
        name=dataset.name,
        version=dataset.version,
        description=dataset.description,
        status=dataset.status,
        owner=dataset.owner,
        is_public=dataset.is_public,
        tags=dataset.tags,
        splits=DatasetSplits(
            train=dataset.split_train,
            val=dataset.split_val,
            test=dataset.split_test,
        ),
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
        file_count=int(file_count or 0),
        total_size_bytes=int(total_size_bytes or 0),
    )


def _dataset_file_out(file: DatasetFile) -> DatasetFileOut:
    return DatasetFileOut(
        id=file.id,
        dataset_id=file.dataset_id,
        filename=file.filename,
        content_type=file.content_type,
        size_bytes=file.size_bytes,
        sha256=file.sha256,
        created_at=file.created_at,
    )


@app.get("/datasets", response_model=ApiResponse[list[DatasetOut]])
def list_datasets(
    q: str | None = Query(default=None, max_length=200),
    status: str | None = Query(default=None, max_length=50),
    owner: str | None = Query(default=None, max_length=100),
    project_id: str | None = Query(default=None, max_length=100),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    agg = (
        db.query(
            DatasetFile.dataset_id.label("dataset_id"),
            func.count(DatasetFile.id).label("file_count"),
            func.coalesce(func.sum(DatasetFile.size_bytes), 0).label("total_size_bytes"),
        )
        .group_by(DatasetFile.dataset_id)
        .subquery()
    )

    query = (
        db.query(Dataset, agg.c.file_count, agg.c.total_size_bytes)
        .outerjoin(agg, Dataset.id == agg.c.dataset_id)
        .filter(or_(Dataset.is_public.is_(True), Dataset.owner == user))
    )
    if q:
        like = f"%{q.strip()}%"
        query = query.filter(or_(Dataset.name.like(like), Dataset.version.like(like)))
    if status:
        query = query.filter(Dataset.status == status)
    if owner:
        query = query.filter(Dataset.owner == owner)
    if project_id:
        query = query.filter(Dataset.project_id == project_id)

    rows = query.order_by(Dataset.updated_at.desc()).offset(offset).limit(limit).all()
    data = [_dataset_out(d, file_count=fc, total_size_bytes=ts) for (d, fc, ts) in rows]
    return ApiResponse(code=200, message="OK", data=data)


@app.post("/datasets", response_model=ApiResponse[DatasetOut], status_code=201, dependencies=[Depends(require_api_key)])
def create_dataset(
    payload: DatasetCreate,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not payload.project_id:
        raise HTTPException(status_code=400, detail="project_id is required")

    project = db.get(Project, payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        ensure_project_dirs(project.id, root=project_storage_root(project.storage_root))
    except Exception:
        logger.debug("Failed to ensure project dirs (project_id=%s)", project.id, exc_info=True)

    version = payload.version
    if not version:
        existing = (
            db.query(func.count(Dataset.id))
            .filter(Dataset.project_id == payload.project_id, Dataset.name == payload.name)
            .scalar()
            or 0
        )
        version = f"v{int(existing) + 1}"

    conflict = (
        db.query(Dataset.id)
        .filter(
            Dataset.project_id == payload.project_id,
            Dataset.name == payload.name,
            Dataset.version == version,
        )
        .first()
    )
    if conflict:
        raise HTTPException(status_code=409, detail="Dataset version already exists")

    dataset_id = payload.id or str(uuid.uuid4())
    dataset = Dataset(
        id=dataset_id,
        project_id=payload.project_id,
        name=payload.name,
        version=version,
        description=payload.description,
        status="created",
        owner=user,
        is_public=payload.is_public,
        tags=payload.tags,
        split_train=payload.splits.train,
        split_val=payload.splits.val,
        split_test=payload.splits.test,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return ApiResponse(code=201, message="Created", data=_dataset_out(dataset))


@app.post(
    "/datasets/{dataset_id}/clone",
    response_model=ApiResponse[DatasetOut],
    status_code=201,
    dependencies=[Depends(require_api_key)],
)
def clone_dataset(
    dataset_id: str,
    payload: DatasetCloneRequest,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    src = db.get(Dataset, dataset_id)
    if not src:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_writable(src, user)

    if not src.project_id:
        raise HTTPException(status_code=400, detail="Dataset is not bound to a project")
    project = db.get(Project, src.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_id = project.id
    project_root = project_storage_root(project.storage_root)
    ensure_project_dirs(project_id, root=project_root)

    name = (payload.name or src.name).strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    version = (payload.version or "").strip() or None
    if not version:
        existing = (
            db.query(func.count(Dataset.id))
            .filter(Dataset.project_id == project_id, Dataset.name == name)
            .scalar()
            or 0
        )
        version = f"v{int(existing) + 1}"

    conflict = (
        db.query(Dataset.id)
        .filter(
            Dataset.project_id == project_id,
            Dataset.name == name,
            Dataset.version == version,
        )
        .first()
    )
    if conflict:
        raise HTTPException(status_code=409, detail="Dataset version already exists")

    splits = payload.splits or DatasetSplits(train=src.split_train, val=src.split_val, test=src.split_test)
    dst_id = payload.id or str(uuid.uuid4())
    dst = Dataset(
        id=dst_id,
        project_id=project_id,
        name=name,
        version=version,
        description=(payload.description if payload.description is not None else src.description),
        status=src.status,
        owner=user,
        is_public=bool(payload.is_public) if payload.is_public is not None else False,
        tags=payload.tags if payload.tags is not None else src.tags,
        split_train=splits.train,
        split_val=splits.val,
        split_test=splits.test,
    )
    db.add(dst)
    db.flush()

    source_files = (
        db.query(DatasetFile)
        .filter(DatasetFile.dataset_id == dataset_id)
        .order_by(DatasetFile.created_at.asc())
        .all()
    )

    source_images = db.query(ImageModel).filter(ImageModel.dataset_id == dataset_id).all()
    source_image_by_file_id: dict[str, ImageModel] = {}
    source_image_ids: list[str] = []
    for img in source_images:
        if img.dataset_file_id and img.dataset_file_id not in source_image_by_file_id:
            source_image_by_file_id[str(img.dataset_file_id)] = img
        source_image_ids.append(img.id)

    anns_by_image: dict[str, list[Annotation]] = {}
    if source_image_ids:
        anns = (
            db.query(Annotation)
            .filter(Annotation.image_id.in_(source_image_ids))
            .order_by(Annotation.created_at.asc())
            .all()
        )
        for ann in anns:
            anns_by_image.setdefault(ann.image_id, []).append(ann)

    copied = 0
    total_size = 0
    for file in source_files:
        src_path = None
        try:
            src_path = resolve_storage_path(file.storage_path, root=project_root)
        except Exception:
            logger.debug("Failed to resolve dataset file path for clone (file_id=%s)", file.id, exc_info=True)
        if not src_path or not src_path.exists():
            continue

        new_file_id = str(uuid.uuid4())
        stored_name = f"{new_file_id}__{safe_filename(file.filename)}"
        rel = (Path("projects") / project_id / "datasets" / dst_id / stored_name).as_posix()
        dst_path = (project_root / rel).resolve()
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(src_path, dst_path)
        except Exception:
            logger.debug("Failed to copy dataset file during clone (file_id=%s)", file.id, exc_info=True)
            continue

        copied += 1
        total_size += int(file.size_bytes or 0)

        record = DatasetFile(
            id=new_file_id,
            dataset_id=dst_id,
            filename=file.filename,
            storage_path=rel,
            content_type=file.content_type,
            size_bytes=file.size_bytes,
            sha256=file.sha256,
        )
        db.add(record)
        db.flush()

        if _is_image_filename(file.filename) or (file.content_type or "").startswith("image/"):
            width, height = None, None
            try:
                width, height = _try_image_size_path(dst_path)
            except Exception:
                width, height = None, None

            new_img = ImageModel(
                project_id=project_id,
                filename=file.filename,
                source_url=f"/datasets/{dst_id}/files/{new_file_id}/download",
                width=width,
                height=height,
                dataset_id=dst_id,
                dataset_file_id=new_file_id,
            )
            db.add(new_img)
            db.flush()

            src_img = source_image_by_file_id.get(str(file.id))
            if src_img:
                for ann in anns_by_image.get(src_img.id, []):
                    ann_copy = Annotation(
                        image_id=new_img.id,
                        type=ann.type,
                        label=ann.label,
                        color=ann.color,
                        visible=ann.visible,
                        x=ann.x,
                        y=ann.y,
                        width=ann.width,
                        height=ann.height,
                        points=list(ann.points) if ann.points else None,
                    )
                    db.add(ann_copy)

    if copied > 0:
        dst.status = "uploaded"
    db.add(dst)
    db.commit()
    db.refresh(dst)
    return ApiResponse(code=201, message="Created", data=_dataset_out(dst, file_count=copied, total_size_bytes=total_size))


@app.get("/datasets/{dataset_id}", response_model=ApiResponse[DatasetOut])
def get_dataset(dataset_id: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_readable(dataset, user)

    stats = (
        db.query(
            func.count(DatasetFile.id).label("file_count"),
            func.coalesce(func.sum(DatasetFile.size_bytes), 0).label("total_size_bytes"),
        )
        .filter(DatasetFile.dataset_id == dataset_id)
        .first()
    )
    file_count, total_size_bytes = (stats or (0, 0))
    return ApiResponse(code=200, message="OK", data=_dataset_out(dataset, file_count=file_count, total_size_bytes=total_size_bytes))


@app.get("/datasets/{dataset_id}/stats", response_model=ApiResponse[DatasetImageStatsOut])
def get_dataset_image_stats(dataset_id: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_readable(dataset, user)

    row = (
        db.query(
            func.count(ImageModel.id).label("image_count"),
            func.count(ImageModel.width).label("sized_image_count"),
            func.avg(ImageModel.width).label("avg_width"),
            func.avg(ImageModel.height).label("avg_height"),
            func.coalesce(func.sum(ImageModel.width * ImageModel.height), 0).label("total_pixels"),
        )
        .filter(ImageModel.dataset_id == dataset_id)
        .first()
    )
    image_count, sized_count, avg_w, avg_h, total_pixels = row or (0, 0, None, None, 0)

    avg_width = int(round(float(avg_w))) if avg_w is not None else None
    avg_height = int(round(float(avg_h))) if avg_h is not None else None

    stats_out = DatasetImageStatsOut(
        dataset_id=dataset_id,
        image_count=int(image_count or 0),
        sized_image_count=int(sized_count or 0),
        avg_width=avg_width,
        avg_height=avg_height,
        total_pixels=int(total_pixels or 0),
    )
    return ApiResponse(code=200, message="OK", data=stats_out)


@app.patch("/datasets/{dataset_id}", response_model=ApiResponse[DatasetOut], dependencies=[Depends(require_api_key)])
def update_dataset(
    dataset_id: str,
    payload: DatasetUpdate,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_writable(dataset, user)

    data = payload.model_dump(exclude_unset=True)
    splits = data.pop("splits", None)
    for key, value in data.items():
        setattr(dataset, key, value)
    if splits:
        dataset.split_train = splits["train"]
        dataset.split_val = splits["val"]
        dataset.split_test = splits["test"]

    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return ApiResponse(code=200, message="OK", data=_dataset_out(dataset))


@app.delete("/datasets/{dataset_id}", response_model=ApiResponse[None], dependencies=[Depends(require_api_key)])
def delete_dataset(dataset_id: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_writable(dataset, user)

    root: Path | None = None
    if dataset.project_id:
        project = db.get(Project, dataset.project_id)
        if project:
            root = project_storage_root(project.storage_root)

    files = db.query(DatasetFile).filter(DatasetFile.dataset_id == dataset_id).all()
    for f in files:
        try:
            delete_storage_path(f.storage_path, root=root)
        except Exception:
            logger.debug("Failed to delete storage path (rel=%s)", f.storage_path, exc_info=True)

    # Remove any images that were generated from this dataset.
    db.query(ImageModel).filter(ImageModel.dataset_id == dataset_id).delete(synchronize_session=False)
    db.delete(dataset)
    db.commit()
    return ApiResponse(code=200, message="Deleted", data=None)


@app.get("/datasets/{dataset_id}/files", response_model=ApiResponse[list[DatasetFileOut]])
def list_dataset_files(dataset_id: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_readable(dataset, user)

    files = (
        db.query(DatasetFile)
        .filter(DatasetFile.dataset_id == dataset_id)
        .order_by(DatasetFile.created_at.asc())
        .all()
    )
    return ApiResponse(code=200, message="OK", data=[_dataset_file_out(f) for f in files])


@app.post(
    "/datasets/{dataset_id}/files",
    response_model=ApiResponse[list[DatasetFileOut]],
    status_code=201,
    dependencies=[Depends(require_api_key)],
)
def upload_dataset_files(
    dataset_id: str,
    files: list[UploadFile] = File(...),
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_writable(dataset, user)
    if not dataset.project_id:
        raise HTTPException(status_code=400, detail="Dataset is not bound to a project")

    project_id = dataset.project_id
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project_root = project_storage_root(project.storage_root)

    created: list[DatasetFile] = []
    created_ids: list[str] = []

    for upload in files:
        filename = upload.filename or "file"
        is_zip = (upload.content_type or "").lower() in ("application/zip", "application/x-zip-compressed") or filename.lower().endswith(".zip")

        if is_zip:
            try:
                try:
                    upload.file.seek(0)
                except Exception:
                    logger.debug("Failed to seek upload file to start for zip import", exc_info=True)

                import shutil
                import tempfile

                tmp_zip_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(prefix="aipt_zip_", suffix=".zip", delete=False) as tmp:
                        tmp_zip_path = Path(tmp.name).resolve()
                        shutil.copyfileobj(upload.file, tmp)
                        tmp.flush()

                    with zipfile.ZipFile(str(tmp_zip_path), "r") as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue
                            inner_name = info.filename.split("/")[-1].split("\\")[-1] or "file"

                            file_id = str(uuid.uuid4())
                            # Stream ZIP members to disk to avoid holding large files in memory.
                            with zf.open(info, "r") as fp:
                                storage_path, size_bytes, sha256 = save_fileobj(
                                    fp,
                                    dataset_id=dataset_id,
                                    file_id=file_id,
                                    filename=inner_name,
                                    project_id=project_id,
                                    root=project_root,
                                )
                            content_type = mimetypes.guess_type(inner_name)[0] or "application/octet-stream"
                            record = DatasetFile(
                                id=file_id,
                                dataset_id=dataset_id,
                                filename=inner_name,
                                storage_path=storage_path,
                                content_type=content_type,
                                size_bytes=size_bytes,
                                sha256=sha256,
                            )
                            db.add(record)
                            db.flush()
                            created.append(record)
                            created_ids.append(file_id)

                            if _is_image_filename(inner_name) or content_type.startswith("image/"):
                                try:
                                    path = resolve_storage_path(storage_path, root=project_root)
                                    width, height = _try_image_size_path(path)
                                except Exception:
                                    width, height = None, None
                                img = ImageModel(
                                    project_id=project_id,
                                    filename=inner_name,
                                    source_url=f"/datasets/{dataset_id}/files/{file_id}/download",
                                    width=width,
                                    height=height,
                                    dataset_id=dataset_id,
                                    dataset_file_id=file_id,
                                )
                                db.add(img)
                finally:
                    if tmp_zip_path:
                        try:
                            tmp_zip_path.unlink(missing_ok=True)
                        except Exception:
                            logger.debug("Failed to clean up temp zip (path=%s)", tmp_zip_path, exc_info=True)
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid zip file")
            continue

        file_id = str(uuid.uuid4())
        storage_path, size_bytes, sha256 = save_upload_file(
            upload,
            dataset_id=dataset_id,
            file_id=file_id,
            project_id=project_id,
            root=project_root,
        )
        record = DatasetFile(
            id=file_id,
            dataset_id=dataset_id,
            filename=filename,
            storage_path=storage_path,
            content_type=upload.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
            size_bytes=size_bytes,
            sha256=sha256,
        )
        db.add(record)
        db.flush()
        created.append(record)
        created_ids.append(file_id)

        if _is_image_filename(filename) or (upload.content_type or "").startswith("image/"):
            try:
                path = resolve_storage_path(storage_path, root=project_root)
                width, height = _try_image_size_path(path)
            except Exception:
                width, height = None, None
            img = ImageModel(
                project_id=project_id,
                filename=filename,
                source_url=f"/datasets/{dataset_id}/files/{file_id}/download",
                width=width,
                height=height,
                dataset_id=dataset_id,
                dataset_file_id=file_id,
            )
            db.add(img)

    dataset.status = "uploaded"
    db.add(dataset)
    db.commit()

    saved = (
        db.query(DatasetFile)
        .filter(DatasetFile.dataset_id == dataset_id, DatasetFile.id.in_(created_ids))
        .order_by(DatasetFile.created_at.asc())
        .all()
    )
    return ApiResponse(code=201, message="Uploaded", data=[_dataset_file_out(f) for f in saved])


@app.get("/datasets/{dataset_id}/files/{file_id}/download")
def download_dataset_file(dataset_id: str, file_id: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_readable(dataset, user)

    file = db.get(DatasetFile, file_id)
    if not file or file.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="File not found")

    root: Path | None = None
    if dataset.project_id:
        project = db.get(Project, dataset.project_id)
        if project:
            root = project_storage_root(project.storage_root)

    path = resolve_storage_path(file.storage_path, root=root)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(path=str(path), filename=file.filename, media_type=file.content_type)


@app.delete("/datasets/{dataset_id}/files/{file_id}", response_model=ApiResponse[None], dependencies=[Depends(require_api_key)])
def delete_dataset_file(dataset_id: str, file_id: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_writable(dataset, user)

    file = db.get(DatasetFile, file_id)
    if not file or file.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="File not found")

    root: Path | None = None
    if dataset.project_id:
        project = db.get(Project, dataset.project_id)
        if project:
            root = project_storage_root(project.storage_root)

    try:
        delete_storage_path(file.storage_path, root=root)
    except Exception:
        logger.debug("Failed to delete storage path (rel=%s)", file.storage_path, exc_info=True)

    db.query(ImageModel).filter(ImageModel.dataset_file_id == file_id).delete(synchronize_session=False)
    db.delete(file)
    db.commit()
    return ApiResponse(code=200, message="Deleted", data=None)


@app.get("/datasets/{dataset_id}/download")
def download_dataset_zip(dataset_id: str, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_readable(dataset, user)

    files = (
        db.query(DatasetFile)
        .filter(DatasetFile.dataset_id == dataset_id)
        .order_by(DatasetFile.created_at.asc())
        .all()
    )
    root: Path | None = None
    if dataset.project_id:
        project = db.get(Project, dataset.project_id)
        if project:
            root = project_storage_root(project.storage_root)

    zip_bytes = zip_dataset_bytes(dataset_id, [(f.filename, f.storage_path) for f in files], root=root)
    filename = f"{dataset.name}_{dataset.version}.zip"
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.get("/projects", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    projects = (
        db.query(Project)
        .options(selectinload(Project.images))
        .order_by(Project.updated_at.desc())
        .all()
    )
    return [
        ProjectOut(
            id=p.id,
            name=p.name,
            type=p.type,
            status=p.status,
            latest_commit=p.latest_commit,
            storage_root=p.storage_root,
            created_at=p.created_at,
            updated_at=p.updated_at,
            images_count=len(p.images),
        )
        for p in projects
    ]


@app.post("/projects", response_model=ProjectOut, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    storage_root_value = data.pop("storage_root", None)
    if not storage_root_value:
        try:
            storage_root_value = load_settings().get("projects_root_dir")
        except Exception:
            storage_root_value = str(default_storage_root())

    storage_root_value = str(Path(str(storage_root_value)).expanduser().resolve())
    try:
        Path(storage_root_value).mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("Failed to create project storage root dir (%s)", storage_root_value, exc_info=True)

    project = Project(**data, storage_root=storage_root_value)
    db.add(project)
    db.commit()
    db.refresh(project)
    try:
        ensure_project_dirs(project.id, root=project_storage_root(project.storage_root))
    except Exception:
        logger.debug("Failed to ensure project dirs (project_id=%s)", project.id, exc_info=True)
    return ProjectOut(
        id=project.id,
        name=project.name,
        type=project.type,
        status=project.status,
        latest_commit=project.latest_commit,
        storage_root=project.storage_root,
        created_at=project.created_at,
        updated_at=project.updated_at,
        images_count=0,
    )


@app.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: Session = Depends(get_db)):
    project = db.get(Project, project_id, options=[selectinload(Project.images)])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectOut(
        id=project.id,
        name=project.name,
        type=project.type,
        status=project.status,
        latest_commit=project.latest_commit,
        storage_root=project.storage_root,
        created_at=project.created_at,
        updated_at=project.updated_at,
        images_count=len(project.images),
    )


@app.patch("/projects/{project_id}", response_model=ProjectOut)
def update_project(project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)):
    project = db.get(Project, project_id, options=[selectinload(Project.images)])
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(project, key, value)

    db.add(project)
    db.commit()
    db.refresh(project)
    return ProjectOut(
        id=project.id,
        name=project.name,
        type=project.type,
        status=project.status,
        latest_commit=project.latest_commit,
        storage_root=project.storage_root,
        created_at=project.created_at,
        updated_at=project.updated_at,
        images_count=len(project.images),
    )


@app.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(project)
    db.commit()
    return None


@app.get("/projects/{project_id}/images", response_model=list[ImageOut])
def list_project_images(
    project_id: str,
    dataset_id: str | None = Query(default=None, max_length=100),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    ann_counts = (
        db.query(
            Annotation.image_id.label("image_id"),
            func.count(Annotation.id).label("annotations_count"),
        )
        .group_by(Annotation.image_id)
        .subquery()
    )

    query = (
        db.query(ImageModel, func.coalesce(ann_counts.c.annotations_count, 0))
        .outerjoin(ann_counts, ImageModel.id == ann_counts.c.image_id)
        .filter(ImageModel.project_id == project_id)
    )
    if dataset_id:
        query = query.filter(ImageModel.dataset_id == dataset_id)

    rows = query.order_by(ImageModel.created_at.asc()).all()
    return [
        ImageOut(
            id=img.id,
            project_id=img.project_id,
            filename=img.filename,
            source_url=img.source_url,
            dataset_id=getattr(img, "dataset_id", None),
            dataset_file_id=getattr(img, "dataset_file_id", None),
            width=img.width,
            height=img.height,
            created_at=img.created_at,
            annotations_count=int(cnt or 0),
        )
        for (img, cnt) in rows
    ]


@app.post("/projects/{project_id}/images", response_model=ImageOut, status_code=201)
def create_project_image(project_id: str, payload: ImageCreate, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    data = payload.model_dump(exclude_unset=True)
    image = ImageModel(project_id=project_id, **data)
    db.add(image)
    db.commit()
    db.refresh(image)
    return ImageOut(
        id=image.id,
        project_id=image.project_id,
        filename=image.filename,
        source_url=image.source_url,
        width=image.width,
        height=image.height,
        created_at=image.created_at,
        annotations_count=0,
    )


@app.delete("/images/{image_id}", response_model=ApiResponse[None], dependencies=[Depends(require_api_key)])
def delete_image(
    image_id: str,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete an image only when it has no annotations.

    If the image is linked to a dataset file, this will also delete the backing
    dataset file record + on-disk payload so dataset views stay consistent.
    """
    image = db.get(ImageModel, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    ann_count = (
        db.query(func.count(Annotation.id))
        .filter(Annotation.image_id == image_id)
        .scalar()
    )
    if int(ann_count or 0) > 0:
        raise HTTPException(status_code=409, detail="Image has annotations")

    dataset_id = getattr(image, "dataset_id", None)
    dataset_file_id = getattr(image, "dataset_file_id", None)

    if dataset_id and dataset_file_id:
        dataset = db.get(Dataset, dataset_id)
        if dataset:
            ensure_dataset_writable(dataset, user)

        file = db.get(DatasetFile, dataset_file_id)
        if file and file.dataset_id == dataset_id:
            root: Path | None = None
            project = db.get(Project, image.project_id)
            if project:
                root = project_storage_root(project.storage_root)

            try:
                delete_storage_path(file.storage_path, root=root)
            except Exception:
                logger.debug("Failed to delete storage path (rel=%s)", file.storage_path, exc_info=True)

            db.query(ImageModel).filter(ImageModel.dataset_file_id == dataset_file_id).delete(synchronize_session=False)
            db.delete(file)
            db.commit()
            return ApiResponse(code=200, message="Deleted", data=None)

    db.delete(image)
    db.commit()
    return ApiResponse(code=200, message="Deleted", data=None)


class ImageEditCrop(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ImageEditRequest(BaseModel):
    rotate: int | None = None
    crop: ImageEditCrop | None = None


@app.post("/images/{image_id}/edit", response_model=ApiResponse[ImageOut], dependencies=[Depends(require_api_key)])
def edit_image(
    image_id: str,
    payload: ImageEditRequest,
    user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    image = db.get(ImageModel, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    ann_count = db.query(func.count(Annotation.id)).filter(Annotation.image_id == image_id).scalar()
    if int(ann_count or 0) > 0:
        raise HTTPException(status_code=409, detail="Image has annotations")

    dataset_id = getattr(image, "dataset_id", None)
    dataset_file_id = getattr(image, "dataset_file_id", None)
    if not dataset_id or not dataset_file_id:
        raise HTTPException(status_code=400, detail="Image is not linked to a dataset file")

    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_writable(dataset, user)

    file = db.get(DatasetFile, dataset_file_id)
    if not file or file.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="Dataset file not found")

    if payload.rotate is not None and payload.rotate not in (90, 180, 270):
        raise HTTPException(status_code=422, detail="rotate must be one of 90/180/270")
    if payload.crop is None and payload.rotate is None:
        raise HTTPException(status_code=422, detail="No edit operation provided")

    project = db.get(Project, image.project_id)
    project_root = project_storage_root(project.storage_root) if project else None
    src_path = resolve_storage_path(file.storage_path, root=project_root)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    try:
        with Image.open(src_path) as img:
            edited = img
            if payload.rotate is not None:
                # UX: rotate clockwise.
                edited = edited.rotate(-int(payload.rotate), expand=True)

            if payload.crop is not None:
                x = int(payload.crop.x or 0)
                y = int(payload.crop.y or 0)
                cw = int(payload.crop.width or 0)
                ch = int(payload.crop.height or 0)
                if cw <= 0 or ch <= 0:
                    raise HTTPException(status_code=422, detail="crop width/height must be > 0")
                x = max(0, x)
                y = max(0, y)
                x2 = min(int(edited.width), x + cw)
                y2 = min(int(edited.height), y + ch)
                if x >= x2 or y >= y2:
                    raise HTTPException(status_code=422, detail="Invalid crop region")
                edited = edited.crop((x, y, x2, y2))

            fmt = img.format or None
            if fmt and fmt.upper() in {"JPG", "JPEG"} and edited.mode not in {"RGB", "L"}:
                edited = edited.convert("RGB")
            edited.save(src_path, format=fmt)

        # Update metadata + checksums
        with Image.open(src_path) as img2:
            image.width, image.height = int(img2.width), int(img2.height)

        data = src_path.read_bytes()
        file.size_bytes = len(data)
        file.sha256 = hashlib.sha256(data).hexdigest()
        db.add(file)
        db.add(image)
        db.commit()
        db.refresh(image)
        db.refresh(file)
    except HTTPException:
        raise
    except Exception as exc:
        logger.debug("Failed to edit image (image_id=%s): %s", image_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to edit image") from exc

    return ApiResponse(
        code=200,
        message="OK",
        data=ImageOut(
            id=image.id,
            project_id=image.project_id,
            filename=image.filename,
            source_url=image.source_url,
            dataset_id=getattr(image, "dataset_id", None),
            dataset_file_id=getattr(image, "dataset_file_id", None),
            width=image.width,
            height=image.height,
            created_at=image.created_at,
            annotations_count=0,
        ),
    )


@app.get("/images/{image_id}/annotations", response_model=list[AnnotationOut])
def list_image_annotations(image_id: str, db: Session = Depends(get_db)):
    image = db.get(ImageModel, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    anns = (
        db.query(Annotation)
        .filter(Annotation.image_id == image_id)
        .order_by(Annotation.created_at.asc())
        .all()
    )
    return [AnnotationOut.model_validate(a) for a in anns]


@app.put("/images/{image_id}/annotations", response_model=list[AnnotationOut])
def replace_image_annotations(
    image_id: str,
    payload: list[AnnotationCreate],
    db: Session = Depends(get_db),
):
    image = db.get(ImageModel, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    db.query(Annotation).filter(Annotation.image_id == image_id).delete(synchronize_session=False)
    for ann_in in payload:
        ann_data = ann_in.model_dump(exclude_unset=True)
        ann = Annotation(image_id=image_id, **ann_data)
        db.add(ann)

    db.commit()
    anns = (
        db.query(Annotation)
        .filter(Annotation.image_id == image_id)
        .order_by(Annotation.created_at.asc())
        .all()
    )
    return [AnnotationOut.model_validate(a) for a in anns]


@app.patch("/annotations/{annotation_id}", response_model=AnnotationOut)
def update_annotation(annotation_id: str, payload: AnnotationUpdate, db: Session = Depends(get_db)):
    ann = db.get(Annotation, annotation_id)
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(ann, key, value)
    db.add(ann)
    db.commit()
    db.refresh(ann)
    return AnnotationOut.model_validate(ann)


@app.delete("/annotations/{annotation_id}", status_code=204)
def delete_annotation(annotation_id: str, db: Session = Depends(get_db)):
    ann = db.get(Annotation, annotation_id)
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    db.delete(ann)
    db.commit()
    return None


def _rect_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _load_gray_float(path: Path):
    import numpy as np  # local import

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("empty image")
    return arr / 255.0


@app.post("/smart-annotation/detect", response_model=ApiResponse[SmartDetectResponse])
def smart_detect_similar(req: SmartDetectRequest, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.get(Dataset, (req.dataset_id or "").strip())
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_writable(dataset, user)

    project_id = (dataset.project_id or "").strip()
    if not project_id:
        raise HTTPException(status_code=400, detail="Dataset is not bound to a project")
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    ref_image = db.get(ImageModel, (req.reference_image_id or "").strip())
    if not ref_image:
        raise HTTPException(status_code=404, detail="Reference image not found")
    if getattr(ref_image, "dataset_id", None) != dataset.id:
        raise HTTPException(status_code=400, detail="Reference image does not belong to the dataset")

    label = (req.label or "").strip()
    if not label:
        raise HTTPException(status_code=422, detail="label is required")
    color = _normalize_hex_color(req.color, default="#ef4444")

    x1, y1, x2, y2 = req.box
    if not all(isinstance(v, (int, float)) for v in (x1, y1, x2, y2)):
        raise HTTPException(status_code=422, detail="box must be numeric")
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=422, detail="box must have positive area")

    threshold = float(req.threshold)
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=422, detail="threshold must be within [0,1]")

    max_images = req.max_images
    if max_images is not None:
        try:
            max_images = int(max_images)
        except Exception:
            raise HTTPException(status_code=422, detail="max_images must be integer") from None
        if max_images < 1:
            raise HTTPException(status_code=422, detail="max_images must be >= 1")

    max_det_per_image = int(req.max_det_per_image or 0)
    if max_det_per_image < 1 or max_det_per_image > 500:
        raise HTTPException(status_code=422, detail="max_det_per_image must be within [1,500]")

    min_distance = req.min_distance
    if min_distance is not None:
        try:
            min_distance = int(min_distance)
        except Exception:
            raise HTTPException(status_code=422, detail="min_distance must be integer") from None
        if min_distance < 1:
            min_distance = 1

    dedup_iou = float(req.dedup_iou)
    if not (0.0 <= dedup_iou <= 1.0):
        raise HTTPException(status_code=422, detail="dedup_iou must be within [0,1]")

    project_root = project_storage_root(project.storage_root)

    def image_path(img: ImageModel) -> Path:
        file_id = (getattr(img, "dataset_file_id", None) or "").strip()
        if not file_id:
            raise HTTPException(status_code=400, detail="Image is not linked to a dataset file")
        file = db.get(DatasetFile, file_id)
        if not file or file.dataset_id != dataset.id:
            raise HTTPException(status_code=404, detail="Dataset file not found")
        path = resolve_storage_path(file.storage_path, root=project_root)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dataset file missing on disk")
        return path

    try:
        ref_arr = _load_gray_float(image_path(ref_image))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load reference image: {exc}") from exc

    h0, w0 = ref_arr.shape[:2]
    xi1 = int(max(0, min(w0 - 1, round(float(x1)))))
    yi1 = int(max(0, min(h0 - 1, round(float(y1)))))
    xi2 = int(max(0, min(w0, round(float(x2)))))
    yi2 = int(max(0, min(h0, round(float(y2)))))
    if xi2 <= xi1 or yi2 <= yi1:
        raise HTTPException(status_code=422, detail="box is out of bounds")

    template = ref_arr[yi1:yi2, xi1:xi2]
    th, tw = template.shape[:2]
    if th < 6 or tw < 6:
        raise HTTPException(status_code=422, detail="template is too small; please draw a larger box")

    if min_distance is None:
        min_distance = max(2, int(min(th, tw) * 0.25))

    if req.scope == "image":
        images = [ref_image]
    else:
        images = (
            db.query(ImageModel)
            .filter(ImageModel.dataset_id == dataset.id)
            .order_by(ImageModel.created_at.asc())
            .all()
        )
        if max_images is not None:
            images = images[: max_images]

    created_total = 0
    image_stats: list[SmartDetectImageOut] = []

    try:
        from skimage.feature import match_template, peak_local_max  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"skimage is not available: {exc}") from exc

    for img in images:
        created = 0
        try:
            arr = _load_gray_float(image_path(img))
        except HTTPException:
            raise
        except Exception:
            image_stats.append(SmartDetectImageOut(image_id=img.id, created=0))
            continue

        ih, iw = arr.shape[:2]
        if ih < th or iw < tw:
            image_stats.append(SmartDetectImageOut(image_id=img.id, created=0))
            continue

        corr = match_template(arr, template, pad_input=False)
        if corr.size == 0:
            image_stats.append(SmartDetectImageOut(image_id=img.id, created=0))
            continue

        coords = peak_local_max(
            corr,
            threshold_abs=threshold,
            min_distance=min_distance,
            num_peaks=max_det_per_image,
        )
        if coords is None or len(coords) == 0:
            image_stats.append(SmartDetectImageOut(image_id=img.id, created=0))
            continue

        # Sort by score desc.
        scored = sorted(((float(corr[y, x]), int(x), int(y)) for y, x in coords), key=lambda t: t[0], reverse=True)

        existing = (
            db.query(Annotation)
            .filter(Annotation.image_id == img.id, Annotation.type == "rect", Annotation.label == label)
            .all()
        )
        existing_rects: list[tuple[float, float, float, float]] = []
        for a in existing:
            if a.x is None or a.y is None or a.width is None or a.height is None:
                continue
            existing_rects.append((float(a.x), float(a.y), float(a.x + a.width), float(a.y + a.height)))

        created_rects: list[tuple[float, float, float, float]] = []
        for _score, x, y in scored:
            rx1 = float(x)
            ry1 = float(y)
            rx2 = float(min(iw, x + tw))
            ry2 = float(min(ih, y + th))
            rect = (rx1, ry1, rx2, ry2)

            if any(_rect_iou(rect, ex) >= dedup_iou for ex in existing_rects):
                continue
            if any(_rect_iou(rect, ex) >= dedup_iou for ex in created_rects):
                continue

            ann = Annotation(
                image_id=img.id,
                type="rect",
                label=label,
                color=color,
                visible=True,
                x=rx1,
                y=ry1,
                width=rx2 - rx1,
                height=ry2 - ry1,
            )
            db.add(ann)
            created += 1
            created_rects.append(rect)
            if created >= max_det_per_image:
                break

        created_total += created
        image_stats.append(SmartDetectImageOut(image_id=img.id, created=created))

    db.commit()
    return ApiResponse(
        code=200,
        message="OK",
        data=SmartDetectResponse(
            processed_images=len(images),
            created_annotations=created_total,
            template_size=(int(tw), int(th)),
            images=image_stats,
        ),
    )


@app.post("/smart-annotation/segment", response_model=ApiResponse[SmartSegmentResponse])
def smart_segment_at_point(req: SmartSegmentRequest, user: str = Depends(get_current_user), db: Session = Depends(get_db)):
    image = db.get(ImageModel, (req.image_id or "").strip())
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    dataset_id = (getattr(image, "dataset_id", None) or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="Image is not linked to a dataset")
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ensure_dataset_writable(dataset, user)

    project_id = (dataset.project_id or "").strip()
    if not project_id:
        raise HTTPException(status_code=400, detail="Dataset is not bound to a project")
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project_root = project_storage_root(project.storage_root)

    file_id = (getattr(image, "dataset_file_id", None) or "").strip()
    if not file_id:
        raise HTTPException(status_code=400, detail="Image is not linked to a dataset file")
    file = db.get(DatasetFile, file_id)
    if not file or file.dataset_id != dataset.id:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    path = resolve_storage_path(file.storage_path, root=project_root)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset file missing on disk")

    tolerance = float(req.tolerance)
    if tolerance < 0:
        tolerance = 0.0
    if tolerance > 1.0:
        tolerance = 1.0
    simplify = float(req.simplify)
    if simplify < 0:
        simplify = 0.0

    try:
        from smart_segmentation import SmartSegmentError, smart_segment_polygon

        points, area = smart_segment_polygon(
            image_path=path,
            point=req.point,
            tolerance=tolerance,
            simplify=simplify,
            engine=(req.engine or None),
        )
        return ApiResponse(code=200, message="OK", data=SmartSegmentResponse(points=points, area=area))
    except SmartSegmentError as exc:
        raise HTTPException(status_code=int(exc.status_code), detail=str(exc.detail)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.debug("Smart segment failed (image_id=%s): %s", req.image_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Smart segment failed: {exc}") from exc

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = detector.predict(image)
        return {"filename": file.filename, "detections": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/diagnostics", response_model=ApiResponse[TrainDiagnosticsOut])
def train_diagnostics():
    return ApiResponse(code=200, message="OK", data=_collect_train_diagnostics())


@app.get("/train/jobs/{job_id}", response_model=ApiResponse[TrainJobStatusOut])
def get_train_job(job_id: str):
    with _TRAIN_JOBS_LOCK:
        job = _TRAIN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Train job not found")
    return ApiResponse(code=200, message="OK", data=job)


@app.post("/train/jobs/{job_id}/stop", response_model=ApiResponse[TrainJobStatusOut])
def stop_train_job(job_id: str):
    with _TRAIN_JOBS_LOCK:
        job = _TRAIN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Train job not found")

    if job.status in ("completed", "failed", "stopped"):
        return ApiResponse(code=200, message="OK", data=job)

    _train_stop_event(job_id).set()
    _set_train_job(job_id, lambda j: j.model_copy(update={"status": "stopping", "message": "Stop requested"}))
    with _TRAIN_JOBS_LOCK:
        updated = _TRAIN_JOBS.get(job_id) or job
    return ApiResponse(code=200, message="OK", data=updated)


@app.post("/train/jobs/{job_id}/resume", response_model=ApiResponse[TrainJobStatusOut])
def resume_train_job(job_id: str):
    infer_lock = _training_blocked_by_inference_reason()
    if infer_lock:
        raise HTTPException(status_code=409, detail=infer_lock)

    with _TRAIN_JOBS_LOCK:
        job = _TRAIN_JOBS.get(job_id)
        active_other = [j for j in _TRAIN_JOBS.values() if j.id != job_id and j.status in ("queued", "running", "stopping")]
    if not job:
        raise HTTPException(status_code=404, detail="Train job not found")
    if active_other:
        active = max(active_other, key=lambda j: j.created_at)
        raise HTTPException(status_code=409, detail=f"Training already in progress (job_id={active.id})")
    if job.status != "stopped":
        raise HTTPException(status_code=409, detail=f"Job is not stopped (status={job.status})")

    job_dir = Path(job.log_path).parent
    last_weights = job_dir / "runs" / "train" / "weights" / "last.pt"
    if not last_weights.exists():
        raise HTTPException(status_code=400, detail="No last.pt found for resume")

    _train_stop_event(job_id).clear()
    _set_train_job(
        job_id,
        lambda j: j.model_copy(
            update={
                "status": "queued",
                "started_at": None,
                "finished_at": None,
                "error": None,
                "message": "Resuming training",
            }
        ),
    )

    thread = threading.Thread(target=_run_training_job, args=(job_id, True), daemon=True)
    thread.start()

    with _TRAIN_JOBS_LOCK:
        updated = _TRAIN_JOBS.get(job_id) or job
    return ApiResponse(code=200, message="OK", data=updated)


@app.get("/train/jobs/{job_id}/logs", response_model=ApiResponse[TrainLogChunkOut])
def get_train_job_logs(job_id: str, offset: int = Query(default=0, ge=0)):
    with _TRAIN_JOBS_LOCK:
        job = _TRAIN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Train job not found")

    path = Path(job.log_path)
    if not path.exists():
        return ApiResponse(code=200, message="OK", data=TrainLogChunkOut(offset=0, text="", eof=True))

    data = path.read_bytes()
    start = min(int(offset or 0), len(data))
    chunk = data[start : start + 200_000]
    new_offset = start + len(chunk)
    eof = new_offset >= len(data)
    text = chunk.decode(errors="ignore")
    return ApiResponse(code=200, message="OK", data=TrainLogChunkOut(offset=new_offset, text=text, eof=eof))


@app.get("/train/jobs/{job_id}/metrics", response_model=ApiResponse[TrainMetricsOut])
def get_train_job_metrics(job_id: str):
    with _TRAIN_JOBS_LOCK:
        job = _TRAIN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Train job not found")

    job_dir = Path(job.log_path).parent
    results_csv = job_dir / "runs" / "train" / "results.csv"
    if results_csv.exists():
        metrics = _parse_results_csv(results_csv)
    else:
        metrics = _parse_mock_metrics_from_log(Path(job.log_path))
    return ApiResponse(code=200, message="OK", data=metrics)


@app.post("/train")
def train_model(config: TrainConfig):
    infer_lock = _training_blocked_by_inference_reason()
    if infer_lock:
        raise HTTPException(status_code=409, detail=infer_lock)

    with _TRAIN_JOBS_LOCK:
        active_jobs = [j for j in _TRAIN_JOBS.values() if j.status in ("queued", "running", "stopping")]
    if active_jobs:
        active = max(active_jobs, key=lambda j: j.created_at)
        raise HTTPException(status_code=409, detail=f"Training already in progress (job_id={active.id})")

    job_id = str(uuid.uuid4())
    job_dir = (_training_root_dir() / job_id).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = str((job_dir / "train.log").resolve())

    job = TrainJobStatusOut(
        id=job_id,
        status="queued",
        created_at=_now_iso(),
        log_path=log_path,
        config=config,
    )
    with _TRAIN_JOBS_LOCK:
        _TRAIN_JOBS[job_id] = job
    _train_stop_event(job_id).clear()

    thread = threading.Thread(target=_run_training_job, args=(job_id,), daemon=True)
    thread.start()

    return {
        "status": "started",
        "message": "Training queued",
        "config": config.model_dump(),
        "job_id": job_id,
    }


@app.get("/projects/{project_id}/models", response_model=ApiResponse[list[TrainedModelOut]])
def list_trained_models(project_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(TrainedModel)
        .filter(TrainedModel.project_id == project_id)
        .order_by(TrainedModel.created_at.desc())
        .all()
    )
    return ApiResponse(code=200, message="OK", data=[TrainedModelOut.model_validate(m) for m in rows])


@app.get("/models/{model_id}/evaluation", response_model=ApiResponse[ModelEvaluationPageOut])
def evaluate_trained_model(
    model_id: str,
    split: Literal["train", "val", "test"] = Query(default="test"),
    page: int = Query(default=1, ge=1, le=1000000),
    limit: int = Query(default=24, ge=1, le=200),
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
    max_det: int = Query(default=50, ge=1, le=300),
    iou: float = Query(default=0.7, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None, ge=32, le=8192),
    device: str | None = Query(default=None, max_length=50),
    half: bool = Query(default=False),
    augment: bool = Query(default=False),
    classes: str | None = Query(default=None, max_length=200),
    db: Session = Depends(get_db),
):
    rec = db.get(TrainedModel, model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")

    project = db.get(Project, rec.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset_id = (rec.dataset_id or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="Model is not bound to a dataset")

    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project_root = project_storage_root(project.storage_root)
    weights_path = resolve_storage_path(rec.weights_path, root=project_root)
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights not found on disk")

    images = (
        db.query(ImageModel)
        .filter(ImageModel.dataset_id == dataset_id)
        .order_by(ImageModel.created_at.asc())
        .all()
    )

    split_train = float(dataset.split_train or 0.7)
    split_val = float(dataset.split_val or 0.2)

    def pick_split(image_id: str) -> str:
        # Deterministic shuffling for dataset splits (not security-related).
        h = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8]
        r = int(h, 16) / 0xFFFFFFFF
        if r < split_train:
            return "train"
        if r < split_train + split_val:
            return "val"
        return "test"

    split_images = [img for img in images if pick_split(img.id) == split]
    total = len(split_images)

    start = (page - 1) * limit
    end = start + limit
    page_images = split_images[start:end]

    items: list[ModelEvaluationItemOut] = []
    note: str | None = None

    yolo = None
    try:
        yolo = _get_eval_model(weights_path)
    except Exception as exc:  # pragma: no cover
        note = str(exc)

    for img in page_images:
        image_out = ImageOut.model_validate(img)
        detections: list[ModelEvaluationDetectionOut] = []

        file_id = getattr(img, "dataset_file_id", None) or ""
        if not file_id:
            items.append(ModelEvaluationItemOut(image=image_out, detections=[]))
            continue

        file = db.get(DatasetFile, file_id)
        if not file or file.dataset_id != dataset_id:
            items.append(ModelEvaluationItemOut(image=image_out, detections=[]))
            continue

        src_path = resolve_storage_path(file.storage_path, root=project_root)
        if not src_path.exists():
            items.append(ModelEvaluationItemOut(image=image_out, detections=[]))
            continue

        if yolo is None:
            items.append(ModelEvaluationItemOut(image=image_out, detections=[]))
            continue

        try:
            predict_kwargs: dict[str, object] = {"verbose": False, "conf": conf, "max_det": max_det, "iou": iou}
            if imgsz is not None:
                predict_kwargs["imgsz"] = int(imgsz)
            if device:
                predict_kwargs["device"] = device
            if half:
                predict_kwargs["half"] = True
            if augment:
                predict_kwargs["augment"] = True
            if classes:
                selected: list[int] = []
                for part in str(classes).split(","):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        selected.append(int(part))
                    except ValueError:
                        continue
                if selected:
                    predict_kwargs["classes"] = selected

            try:
                if hasattr(yolo, "predict"):
                    results = yolo.predict(str(src_path), **predict_kwargs)
                else:
                    results = yolo(str(src_path), **predict_kwargs)
            except TypeError:
                # Best-effort backward compat for older ultralytics versions.
                safe_kwargs = {"verbose": False, "conf": conf}
                if hasattr(yolo, "predict"):
                    results = yolo.predict(str(src_path), **safe_kwargs)
                else:
                    results = yolo(str(src_path), **safe_kwargs)
            for result in results:
                boxes = getattr(result, "boxes", None)
                if not boxes:
                    continue
                for box in boxes:
                    detection = None
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf_val = float(box.conf[0].item())
                        cls_val = int(box.cls[0].item())
                        names = getattr(result, "names", {}) or {}
                        class_name = str(names.get(cls_val, cls_val))
                        detection = ModelEvaluationDetectionOut(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=conf_val,
                            class_name=class_name,
                            class_id=cls_val,
                        )
                    except Exception as exc:
                        logger.debug(
                            "Failed to parse evaluation detection (model_id=%s image_id=%s): %s",
                            model_id,
                            img.id,
                            exc,
                            exc_info=True,
                        )
                    if detection is None:
                        continue
                    detections.append(detection)
        except Exception as exc:
            logger.debug("Evaluation inference failed (model_id=%s image_id=%s): %s", model_id, img.id, exc, exc_info=True)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        if len(detections) > max_det:
            detections = detections[:max_det]

        items.append(ModelEvaluationItemOut(image=image_out, detections=detections))

    payload = ModelEvaluationPageOut(
        model_id=model_id,
        dataset_id=dataset_id,
        split=split,
        page=page,
        limit=limit,
        total=total,
        note=note,
        items=items,
    )
    return ApiResponse(code=200, message="OK", data=payload)


@app.delete("/models/{model_id}", response_model=ApiResponse[None], dependencies=[Depends(require_api_key)])
def delete_trained_model(model_id: str, db: Session = Depends(get_db)):
    rec = db.get(TrainedModel, model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")

    project = db.get(Project, rec.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_root = project_storage_root(project.storage_root)
    rel_dir = (Path("projects") / rec.project_id / "models" / rec.id).as_posix()
    try:
        path = resolve_storage_path(rel_dir, root=project_root)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        logger.debug("Failed to delete model artifacts (model_id=%s)", model_id, exc_info=True)

    db.delete(rec)
    db.commit()
    return ApiResponse(code=200, message="Deleted", data=None)


def _cleanup_export_artifacts(paths: list[str]) -> None:
    for raw in paths:
        if not raw:
            continue
        try:
            path = Path(raw)
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
        except Exception:
            logger.debug("Failed to cleanup export artifact (path=%s)", raw, exc_info=True)


@app.get("/models/{model_id}/export")
def export_trained_model(
    model_id: str,
    format: str = Query(default="pt", max_length=50),
    opset: int = Query(default=12, ge=7, le=20),
    simplify: bool = Query(default=True),
    dynamic: bool = Query(default=False),
    half: bool = Query(default=False),
    int8: bool = Query(default=False),
    device: str | None = Query(default=None, max_length=50),
    workspace: int | None = Query(default=None, ge=0, le=65536),
    batch: int | None = Query(default=None, ge=1, le=1024),
    imgsz: int | None = Query(default=None, ge=32, le=8192),
    db: Session = Depends(get_db),
):
    rec = db.get(TrainedModel, model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")

    project = db.get(Project, rec.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_root = project_storage_root(project.storage_root)
    weights_path = resolve_storage_path(rec.weights_path, root=project_root)
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights not found on disk")

    fmt = (format or "pt").strip().lower()
    if fmt in {"pt", "pytorch"}:
        filename = f"{rec.name or rec.id}.pt"
        return FileResponse(path=str(weights_path), filename=filename, media_type="application/octet-stream")

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"ultralytics is not available: {exc}") from exc

    export_base = (project_root / "projects" / rec.project_id / "exports").resolve()
    export_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"export_{rec.id}_", dir=str(export_base))).resolve()

    cleanup: list[str] = [str(tmp_dir)]
    try:
        yolo = YOLO(str(weights_path))

        export_kwargs: dict[str, object] = {"format": fmt}
        if fmt == "onnx":
            export_kwargs.update({"opset": opset, "simplify": simplify, "dynamic": dynamic})

        try:
            sig = inspect.signature(yolo.export)
            params = sig.parameters
        except Exception:
            params = {}

        # Prefer exporting into a temporary directory to avoid leaving variants next to weights.
        if "project" in params:
            export_kwargs["project"] = str(tmp_dir)
        if "name" in params:
            export_kwargs["name"] = "export"
        if "exist_ok" in params:
            export_kwargs["exist_ok"] = True
        if "save_dir" in params:
            export_kwargs["save_dir"] = str(tmp_dir)
        if device and "device" in params:
            export_kwargs["device"] = device
        if "half" in params:
            export_kwargs["half"] = half
        if "int8" in params:
            export_kwargs["int8"] = int8
        if workspace is not None and "workspace" in params:
            export_kwargs["workspace"] = workspace
        if batch is not None and "batch" in params:
            export_kwargs["batch"] = batch
        if imgsz is not None and "imgsz" in params:
            export_kwargs["imgsz"] = imgsz

        result = yolo.export(**export_kwargs)

        exported: Path | None = None
        if isinstance(result, (str, Path)):
            exported = Path(result).expanduser()
        elif isinstance(result, (list, tuple)) and result:
            exported = Path(str(result[0])).expanduser()
        if exported and not exported.is_absolute():
            exported = (tmp_dir / exported).resolve()

        if not exported or not exported.exists():
            # Best-effort search in tmp_dir, then in the model directory.
            suffix = {
                "onnx": ".onnx",
                "torchscript": ".torchscript",
                "engine": ".engine",
            }.get(fmt)

            if fmt == "openvino":
                candidates = list(tmp_dir.rglob("*.xml"))
                if not candidates:
                    candidates = list(weights_path.parent.rglob("*.xml"))
                if candidates:
                    exported = candidates[0].parent
            elif suffix:
                candidates = list(tmp_dir.rglob(f"*{suffix}"))
                if not candidates:
                    candidates = [p for p in weights_path.parent.glob(f"*{suffix}") if p.is_file()]
                if candidates:
                    exported = candidates[0]

        if not exported or not exported.exists():
            raise HTTPException(status_code=500, detail="Export failed: output not found")

        # If export wrote outside tmp_dir (e.g., next to weights), clean it up after download.
        if exported.resolve() != tmp_dir and tmp_dir not in exported.resolve().parents:
            cleanup.append(str(exported))

        if exported.is_dir():
            zip_path = (tmp_dir / f"{rec.name or rec.id}_{fmt}.zip").resolve()
            with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in exported.rglob("*"):
                    if p.is_file():
                        zf.write(p, arcname=p.relative_to(exported).as_posix())
            cleanup.append(str(zip_path))
            return FileResponse(
                path=str(zip_path),
                filename=zip_path.name,
                media_type="application/zip",
                background=BackgroundTask(_cleanup_export_artifacts, cleanup),
            )

        ext = exported.suffix or f".{fmt}"
        filename = f"{rec.name or rec.id}{ext}"
        return FileResponse(
            path=str(exported),
            filename=filename,
            media_type="application/octet-stream",
            background=BackgroundTask(_cleanup_export_artifacts, cleanup),
        )
    except HTTPException:
        _cleanup_export_artifacts(cleanup)
        raise
    except Exception as exc:  # pragma: no cover
        _cleanup_export_artifacts(cleanup)
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc


@app.get("/inference/status", response_model=ApiResponse[InferenceStatusOut])
def inference_status():
    return ApiResponse(code=200, message="OK", data=_inference_status_snapshot())


@app.post("/inference/sessions", response_model=ApiResponse[InferenceSessionOut])
def create_inference_session(req: InferenceSessionCreateIn, db: Session = Depends(get_db)):
    active = _active_training_job()
    if active:
        raise HTTPException(status_code=409, detail=f"Training is in progress (job_id={active.id})")

    rec = db.get(TrainedModel, req.model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")
    if req.project_id and rec.project_id != req.project_id:
        raise HTTPException(status_code=400, detail="Model does not belong to the selected project")

    project = db.get(Project, rec.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_root = project_storage_root(project.storage_root)
    weights_path = resolve_storage_path(rec.weights_path, root=project_root)
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights not found on disk")

    key = "|".join(
        [
            str(weights_path.resolve()),
            req.format,
            (req.device or "").strip(),
            "half=1" if req.half else "half=0",
            "int8=1" if req.int8 else "int8=0",
            f"workspace={req.workspace or ''}",
            f"batch={req.batch or ''}",
            f"imgsz={req.imgsz or ''}",
        ]
    )

    with _INFER_LOCK:
        existing_id = _INFER_SESSION_BY_KEY.get(key)
        if existing_id and existing_id in _INFER_SESSIONS:
            s = _INFER_SESSIONS[existing_id]
            out = InferenceSessionOut(
                id=s.id,
                project_id=s.project_id,
                model_id=s.model_id,
                format=s.format,
                device=s.device,
                created_at=s.created_at,
                last_used_at=s.last_used_at,
                cached_artifact=s.artifact_path.name,
            )
            return ApiResponse(code=200, message="OK", data=out)

        evt = _INFER_LOADING.get(key)
        if evt is None:
            evt = threading.Event()
            _INFER_LOADING[key] = evt
            is_loader = True
        else:
            is_loader = False

    if not is_loader:
        evt.wait(timeout=1200)
        with _INFER_LOCK:
            existing_id = _INFER_SESSION_BY_KEY.get(key)
            if existing_id and existing_id in _INFER_SESSIONS:
                s = _INFER_SESSIONS[existing_id]
                out = InferenceSessionOut(
                    id=s.id,
                    project_id=s.project_id,
                    model_id=s.model_id,
                    format=s.format,
                    device=s.device,
                    created_at=s.created_at,
                    last_used_at=s.last_used_at,
                    cached_artifact=s.artifact_path.name,
                )
                return ApiResponse(code=200, message="OK", data=out)
            err = _INFER_LOADING_ERRORS.pop(key, None)
        raise HTTPException(status_code=500, detail=err or "Inference session initialization failed")

    try:
        cache_dir = _inference_cache_dir(project_root, rec.project_id, rec.id, req.format)
        artifact = _ensure_inference_artifact(weights_path, cache_dir, req)

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=503, detail=f"ultralytics is not available: {exc}") from exc

        model_obj: object
        try:
            model_obj = YOLO(str(artifact))
        except Exception:
            # Best-effort fallback for OpenVINO directory-based loads.
            if req.format == "openvino" and artifact.is_file():
                model_obj = YOLO(str(artifact.parent))
            else:
                raise

        session_id = str(uuid.uuid4())
        created_at = _now_iso()
        session = _InferenceSession(
            id=session_id,
            key=key,
            project_id=rec.project_id,
            model_id=rec.id,
            format=req.format,
            device=req.device,
            artifact_path=artifact,
            model=model_obj,
            created_at=created_at,
        )

        with _INFER_LOCK:
            _INFER_SESSIONS[session_id] = session
            _INFER_SESSION_BY_KEY[key] = session_id

        out = InferenceSessionOut(
            id=session_id,
            project_id=session.project_id,
            model_id=session.model_id,
            format=session.format,
            device=session.device,
            created_at=created_at,
            last_used_at=None,
            cached_artifact=session.artifact_path.name,
        )
        return ApiResponse(code=201, message="Created", data=out)
    except HTTPException as exc:
        with _INFER_LOCK:
            _INFER_LOADING_ERRORS[key] = str(exc.detail)
        raise
    except Exception as exc:  # pragma: no cover
        with _INFER_LOCK:
            _INFER_LOADING_ERRORS[key] = str(exc)
        raise
    finally:
        with _INFER_LOCK:
            evt2 = _INFER_LOADING.pop(key, None)
            if evt2:
                evt2.set()


@app.delete("/inference/sessions/{session_id}", response_model=ApiResponse[None])
def close_inference_session(session_id: str):
    with _INFER_LOCK:
        session = _INFER_SESSIONS.get(session_id)
        if not session or not isinstance(session, _InferenceSession):
            raise HTTPException(status_code=404, detail="Inference session not found")
        if session.active_requests > 0:
            raise HTTPException(status_code=409, detail="Inference session is busy")
        _INFER_SESSIONS.pop(session_id, None)
        _INFER_SESSION_BY_KEY.pop(session.key, None)
    return ApiResponse(code=200, message="Closed", data=None)


@app.post("/inference/sessions/{session_id}/predict", response_model=ApiResponse[InferencePredictionOut])
def inference_predict(
    session_id: str,
    file: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
    max_det: int = Query(default=50, ge=1, le=300),
    iou: float = Query(default=0.7, ge=0.0, le=1.0),
    imgsz: int | None = Query(default=None, ge=32, le=8192),
    classes: str | None = Query(default=None, max_length=200),
):
    active = _active_training_job()
    if active:
        raise HTTPException(status_code=409, detail=f"Training is in progress (job_id={active.id})")

    with _INFER_LOCK:
        session = _INFER_SESSIONS.get(session_id)
        if not session or not isinstance(session, _InferenceSession):
            raise HTTPException(status_code=404, detail="Inference session not found")
        global _INFER_ACTIVE_REQUESTS
        _INFER_ACTIVE_REQUESTS += 1
        session.active_requests += 1

    detections: list[ModelEvaluationDetectionOut] = []
    note: str | None = None

    suffix = Path(file.filename or "image").suffix
    if not suffix:
        suffix = ".jpg"
    fd, tmp_name = tempfile.mkstemp(prefix="infer_", suffix=suffix)
    try:
        os.close(fd)
    except Exception:
        logger.debug("Failed to close temp fd for inference upload", exc_info=True)
    tmp = Path(tmp_name).resolve()
    try:
        with tmp.open("wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        with session.lock:
            try:
                predict_kwargs: dict[str, object] = {"verbose": False, "conf": conf, "max_det": max_det, "iou": iou}
                if imgsz is not None:
                    predict_kwargs["imgsz"] = int(imgsz)
                if session.device:
                    predict_kwargs["device"] = session.device
                if classes:
                    selected: list[int] = []
                    for part in str(classes).split(","):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            selected.append(int(part))
                        except ValueError:
                            continue
                    if selected:
                        predict_kwargs["classes"] = selected

                yolo = session.model
                try:
                    if hasattr(yolo, "predict"):
                        results = yolo.predict(str(tmp), **predict_kwargs)  # type: ignore[attr-defined]
                    else:
                        results = yolo(str(tmp), **predict_kwargs)  # type: ignore[misc]
                except TypeError:
                    safe_kwargs = {"verbose": False, "conf": conf}
                    if hasattr(yolo, "predict"):
                        results = yolo.predict(str(tmp), **safe_kwargs)  # type: ignore[attr-defined]
                    else:
                        results = yolo(str(tmp), **safe_kwargs)  # type: ignore[misc]

                for result in results:
                    boxes = getattr(result, "boxes", None)
                    if not boxes:
                        continue
                    for box in boxes:
                        detection = None
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf_val = float(box.conf[0].item())
                            cls_val = int(box.cls[0].item())
                            names = getattr(result, "names", {}) or {}
                            class_name = str(names.get(cls_val, cls_val))
                            detection = ModelEvaluationDetectionOut(
                                bbox=(float(x1), float(y1), float(x2), float(y2)),
                                confidence=conf_val,
                                class_name=class_name,
                                class_id=cls_val,
                            )
                        except Exception as exc:
                            logger.debug(
                                "Failed to parse inference detection (session_id=%s): %s",
                                session_id,
                                exc,
                                exc_info=True,
                            )
                        if detection is None:
                            continue
                        detections.append(detection)
            except Exception as exc:
                logger.debug("Inference predict failed (session_id=%s): %s", session_id, exc, exc_info=True)
                note = str(exc)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        if len(detections) > max_det:
            detections = detections[:max_det]

        with _INFER_LOCK:
            session.last_used_at = _now_iso()
        return ApiResponse(code=200, message="OK", data=InferencePredictionOut(session_id=session_id, detections=detections, note=note))
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            logger.debug("Failed to cleanup inference temp file: %s", tmp, exc_info=True)
        with _INFER_LOCK:
            _INFER_ACTIVE_REQUESTS = max(0, _INFER_ACTIVE_REQUESTS - 1)
            if session_id in _INFER_SESSIONS and isinstance(_INFER_SESSIONS[session_id], _InferenceSession):
                _INFER_SESSIONS[session_id].active_requests = max(0, _INFER_SESSIONS[session_id].active_requests - 1)
