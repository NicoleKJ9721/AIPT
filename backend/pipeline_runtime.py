from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from fastapi import HTTPException
from PIL import Image
from sqlalchemy.orm import Session

from db_models import Project, TrainedModel
from schemas import (
    ModelEvaluationDetectionOut,
    PipelineConnectorSpec,
    PipelineRunOut,
    PipelineRunStepOut,
    PipelineStepSpec,
)
from storage import project_storage_root, resolve_storage_path


logger = logging.getLogger(__name__)


class PipelineCachedModel(Protocol):
    model: object
    lock: threading.Lock


@dataclass
class _Ctx:
    img: Image.Image
    offset_x: float = 0.0
    offset_y: float = 0.0


@dataclass
class _Region:
    ctx: _Ctx
    bbox: tuple[float, float, float, float]
    conf: float
    class_id: int
    class_name: str


def _clamp_float(v: float, lo: float, hi: float) -> float:
    try:
        fv = float(v)
    except Exception:
        return lo
    if fv < lo:
        return lo
    if fv > hi:
        return hi
    return fv


def _resolve_pipeline_classes(model_names: dict[int, str] | None, classes: list[int | str] | None) -> list[int] | None:
    if not classes:
        return None

    names = model_names or {}
    name_to_id = {str(v).strip().lower(): int(k) for k, v in names.items() if v is not None}
    out: list[int] = []

    for raw in classes:
        if isinstance(raw, bool):  # bool is a subclass of int
            continue
        if isinstance(raw, int):
            out.append(int(raw))
            continue
        text = str(raw or "").strip()
        if not text:
            continue
        if text.isdigit():
            out.append(int(text))
            continue
        idx = name_to_id.get(text.lower())
        if idx is not None:
            out.append(int(idx))

    seen: set[int] = set()
    uniq: list[int] = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    return uniq or None


def _expand_crop_bbox(
    *,
    bbox: tuple[float, float, float, float],
    pad: float,
    w: int,
    h: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    pad = _clamp_float(pad, 0.0, 1.0)
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    px = bw * pad
    py = bh * pad

    nx1 = int(_clamp_float(x1 - px, 0.0, max(0.0, float(w - 1))))
    ny1 = int(_clamp_float(y1 - py, 0.0, max(0.0, float(h - 1))))
    nx2 = int(_clamp_float(x2 + px, 0.0, float(w)))
    ny2 = int(_clamp_float(y2 + py, 0.0, float(h)))

    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def _legacy_connector_from_crop(step: PipelineStepSpec) -> PipelineConnectorSpec | None:
    if not bool(step.crop):
        return None
    return PipelineConnectorSpec(
        source="prev_detections",
        min_conf=0.0,
        classes=None,
        padding=float(step.crop_padding or 0.0),
        max_regions=step.crop_max_regions,
        on_empty="stop",
    )


def _step_connector(step: PipelineStepSpec) -> PipelineConnectorSpec | None:
    if step.connector is not None:
        return step.connector
    return _legacy_connector_from_crop(step)


def _safe_class_name(names: dict[int, str] | None, cls_id: int, fallback: str = "object") -> str:
    if names and cls_id in names:
        return str(names.get(cls_id))
    if cls_id >= 0:
        return str(cls_id)
    return fallback


def run_pipeline_steps(
    *,
    project_id: str,
    steps: list[PipelineStepSpec],
    image: Image.Image,
    db: Session,
    get_pipeline_model: Callable[[Path], PipelineCachedModel],
    active_training_job: Callable[[], object | None] | None = None,
    pipeline_id: str | None = None,
) -> PipelineRunOut:
    if active_training_job:
        active = active_training_job()
        if active:
            active_id = getattr(active, "id", "-")
            raise HTTPException(status_code=409, detail=f"Training is in progress (job_id={active_id})")

    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project_root = project_storage_root(project.storage_root)

    root_ctx = _Ctx(img=image, offset_x=0.0, offset_y=0.0)
    ctxs: list[_Ctx] = [root_ctx]
    out_steps: list[PipelineRunStepOut] = []
    note: str | None = None

    for idx, step in enumerate(steps):
        model_rec = db.get(TrainedModel, step.model_id)
        if not model_rec:
            raise HTTPException(status_code=404, detail=f"Model not found (model_id={step.model_id})")
        if model_rec.project_id != project_id:
            raise HTTPException(status_code=400, detail=f"Model does not belong to project (model_id={step.model_id})")

        weights_path = resolve_storage_path(model_rec.weights_path, root=project_root)
        if not weights_path.exists():
            raise HTTPException(status_code=404, detail=f"Weights not found on disk (model_id={step.model_id})")

        cached = get_pipeline_model(weights_path)
        model_obj = cached.model
        model_names = getattr(model_obj, "names", None)

        conf = _clamp_float(step.conf, 0.0, 1.0)
        iou = _clamp_float(step.iou, 0.0, 1.0)
        max_det = max(1, min(300, int(step.max_det or 50)))

        class_indices = _resolve_pipeline_classes(model_names, step.classes)
        predict_kwargs: dict[str, object] = {"verbose": False, "conf": conf, "iou": iou, "max_det": max_det}
        if class_indices:
            predict_kwargs["classes"] = class_indices

        connector = _step_connector(step)
        connector_allowed_classes = (
            _resolve_pipeline_classes(model_names, connector.classes)
            if connector and connector.classes
            else None
        )
        connector_min_conf = float(connector.min_conf) if connector else 0.0

        step_dets: list[ModelEvaluationDetectionOut] = []
        local_det_regions: list[_Region] = []
        local_seg_regions: list[_Region] = []

        started = time.perf_counter()
        step_note: str | None = None

        with cached.lock:
            for ctx in ctxs:
                try:
                    try:
                        if hasattr(model_obj, "predict"):
                            results = model_obj.predict(ctx.img, **predict_kwargs)  # type: ignore[attr-defined]
                        else:
                            results = model_obj(ctx.img, **predict_kwargs)  # type: ignore[misc]
                    except TypeError:
                        safe_kwargs = {"verbose": False, "conf": conf}
                        if hasattr(model_obj, "predict"):
                            results = model_obj.predict(ctx.img, **safe_kwargs)  # type: ignore[attr-defined]
                        else:
                            results = model_obj(ctx.img, **safe_kwargs)  # type: ignore[misc]
                except Exception as exc:
                    logger.debug("Pipeline predict failed (model_id=%s step=%s): %s", step.model_id, step.id, exc, exc_info=True)
                    step_note = str(exc)
                    continue

                for result in results:
                    boxes = getattr(result, "boxes", None)
                    names = getattr(result, "names", None) or (model_names or {}) or {}

                    # Boxes (detection/segmentation heads with box outputs).
                    if boxes:
                        for box in boxes:
                            try:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                conf_val = float(box.conf[0].item())
                                cls_val = int(box.cls[0].item())
                                class_name = _safe_class_name(names, cls_val)
                            except Exception:
                                continue

                            step_dets.append(
                                ModelEvaluationDetectionOut(
                                    bbox=(
                                        float(x1 + ctx.offset_x),
                                        float(y1 + ctx.offset_y),
                                        float(x2 + ctx.offset_x),
                                        float(y2 + ctx.offset_y),
                                    ),
                                    confidence=conf_val,
                                    class_name=class_name,
                                    class_id=cls_val,
                                )
                            )
                            local_det_regions.append(
                                _Region(
                                    ctx=ctx,
                                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                                    conf=conf_val,
                                    class_id=cls_val,
                                    class_name=class_name,
                                )
                            )

                    # Masks (segmentation). Extract tight bbox from polygons for connector chaining.
                    masks = getattr(result, "masks", None)
                    polys = getattr(masks, "xy", None) if masks is not None else None
                    if polys:
                        boxes_list = list(boxes) if boxes is not None else []
                        has_boxes = bool(boxes_list)
                        for poly_idx, poly in enumerate(polys):
                            try:
                                xs = [float(p[0]) for p in poly]
                                ys = [float(p[1]) for p in poly]
                            except Exception:
                                continue
                            if not xs or not ys:
                                continue
                            x1 = float(min(xs))
                            y1 = float(min(ys))
                            x2 = float(max(xs))
                            y2 = float(max(ys))
                            if x2 <= x1 or y2 <= y1:
                                continue

                            cls_val = -1
                            conf_val = 1.0
                            class_name = "segment"
                            if poly_idx < len(boxes_list):
                                try:
                                    b = boxes_list[poly_idx]
                                    cls_val = int(b.cls[0].item())
                                    conf_val = float(b.conf[0].item())
                                    class_name = _safe_class_name(names, cls_val, fallback="segment")
                                except Exception:
                                    pass

                            local_seg_regions.append(
                                _Region(
                                    ctx=ctx,
                                    bbox=(x1, y1, x2, y2),
                                    conf=conf_val,
                                    class_id=cls_val,
                                    class_name=class_name,
                                )
                            )
                            # If the model only outputs masks (no boxes), expose mask bbox as detection result.
                            if not has_boxes:
                                step_dets.append(
                                    ModelEvaluationDetectionOut(
                                        bbox=(
                                            float(x1 + ctx.offset_x),
                                            float(y1 + ctx.offset_y),
                                            float(x2 + ctx.offset_x),
                                            float(y2 + ctx.offset_y),
                                        ),
                                        confidence=conf_val,
                                        class_name=class_name,
                                        class_id=cls_val,
                                    )
                                )

        step_dets.sort(key=lambda d: d.confidence, reverse=True)
        if len(step_dets) > max_det:
            step_dets = step_dets[:max_det]

        duration_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
        out_steps.append(
            PipelineRunStepOut(
                step_id=step.id,
                title=step.title,
                model_id=step.model_id,
                detections=step_dets,
                duration_ms=duration_ms,
                note=step_note,
            )
        )

        is_last = idx >= len(steps) - 1
        if connector is None or is_last:
            continue

        source_regions = local_det_regions if connector.source == "prev_detections" else local_seg_regions
        source_regions = [r for r in source_regions if r.conf >= connector_min_conf]
        if connector_allowed_classes:
            allowed = set(connector_allowed_classes)
            source_regions = [r for r in source_regions if r.class_id in allowed]

        source_regions.sort(key=lambda r: r.conf, reverse=True)
        if connector.max_regions is not None:
            source_regions = source_regions[: int(connector.max_regions)]

        next_ctxs: list[_Ctx] = []
        for region in source_regions:
            w, h = region.ctx.img.size
            crop_box = _expand_crop_bbox(
                bbox=region.bbox,
                pad=float(connector.padding or 0.0),
                w=int(w),
                h=int(h),
            )
            if crop_box is None:
                continue
            cx1, cy1, cx2, cy2 = crop_box
            try:
                cropped = region.ctx.img.crop((cx1, cy1, cx2, cy2))
            except Exception:
                continue
            next_ctxs.append(
                _Ctx(
                    img=cropped,
                    offset_x=region.ctx.offset_x + float(cx1),
                    offset_y=region.ctx.offset_y + float(cy1),
                )
            )

        if next_ctxs:
            ctxs = next_ctxs
            continue

        if connector.on_empty == "fallback_full":
            ctxs = [root_ctx]
            if note is None:
                note = f"Step '{step.title}' produced no connector regions; fallback to full image."
            continue

        if connector.on_empty == "skip":
            if note is None:
                note = f"Step '{step.title}' produced no connector regions; skipped connector."
            continue

        # Default: stop.
        note = f"Step '{step.title}' produced no connector regions; pipeline stopped early."
        break

    final = out_steps[-1].detections if out_steps else []
    merged: list[ModelEvaluationDetectionOut] = []
    for s in out_steps:
        merged.extend(s.detections)
    merged.sort(key=lambda d: d.confidence, reverse=True)
    return PipelineRunOut(
        project_id=project_id,
        pipeline_id=pipeline_id,
        steps=out_steps,
        final_detections=final,
        merged_detections=merged,
        note=note,
    )
