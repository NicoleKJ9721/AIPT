from __future__ import annotations

import importlib
import logging
import os
import threading
from dataclasses import dataclass
from collections import deque
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmartSegmentError(Exception):
    status_code: int
    detail: str

    def __str__(self) -> str:  # pragma: no cover
        return self.detail


_MODEL_LOCK = threading.Lock()
_MODELS: dict[str, object] = {}


def _clamp01(v: float) -> float:
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return float(v)


def _normalize_engine(engine: str | None) -> str:
    raw = (engine or "").strip().lower()
    if not raw:
        raw = (os.getenv("AIPT_SMART_SEG_ENGINE") or "auto").strip().lower()

    aliases = {
        "classic": "flood",
        "default": "auto",
        "sam_trt": "sam_tensorrt",
        "fastsam_trt": "fastsam_tensorrt",
        "trt": "fastsam_tensorrt",
        "tensorrt": "fastsam_tensorrt",
    }
    raw = aliases.get(raw, raw)

    allowed = {"auto", "flood", "sam", "sam_tensorrt", "fastsam", "fastsam_tensorrt"}
    if raw not in allowed:
        raw = "auto"
    return raw


def _cuda_available() -> bool:
    try:
        torch = importlib.import_module("torch")
        return bool(getattr(torch, "cuda").is_available())
    except Exception:
        return False


def _tensorrt_available() -> bool:
    try:
        return importlib.util.find_spec("tensorrt") is not None
    except Exception:
        return False


def _ultralytics_available() -> bool:
    try:
        return importlib.util.find_spec("ultralytics") is not None
    except Exception:
        return False


def _auto_engine() -> str:
    if not _ultralytics_available():
        return "flood"
    if _tensorrt_available() and _cuda_available():
        return "fastsam_tensorrt"
    return "fastsam"


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        area += (xj + xi) * (yj - yi)
        j = i
    return float(area / 2.0)


def _simplify_polygon(poly: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if tolerance <= 0 or len(poly) < 3:
        return poly

    try:
        import numpy as np  # local import
        from skimage.measure import approximate_polygon  # type: ignore
    except Exception:
        return poly

    # skimage expects (row, col) = (y, x)
    arr = np.asarray([(y, x) for x, y in poly], dtype=np.float32)
    approx = approximate_polygon(arr, tolerance=tolerance)
    if approx is None or len(approx) < 3:
        return poly
    out = [(float(col), float(row)) for row, col in approx]

    # Drop the duplicated closing point if present.
    if len(out) >= 2 and abs(out[0][0] - out[-1][0]) < 1e-6 and abs(out[0][1] - out[-1][1]) < 1e-6:
        out = out[:-1]
    return out


def _mask_to_polygon(mask: "object", simplify: float) -> tuple[list[float], int]:
    import numpy as np  # local import

    arr = np.asarray(mask, dtype=np.uint8)
    if arr.ndim != 2:
        raise SmartSegmentError(status_code=422, detail="Invalid mask shape")

    mask_bool = arr.astype(bool)
    area = int(mask_bool.sum())
    if area < 20:
        raise SmartSegmentError(status_code=422, detail="No region found; try another point or adjust threshold")

    try:
        from skimage.measure import approximate_polygon, find_contours  # type: ignore
        from skimage.morphology import closing, disk, opening  # type: ignore

        # Light smoothing.
        mask_bool = opening(mask_bool, disk(1))
        mask_bool = closing(mask_bool, disk(2))

        contours = find_contours(mask_bool.astype(np.uint8), 0.5)
        if not contours:
            raise SmartSegmentError(status_code=422, detail="Failed to extract contour")

        contour = max(contours, key=lambda c: c.shape[0])
        approx = approximate_polygon(contour, tolerance=simplify) if simplify > 0 else contour
        if approx is None or len(approx) < 3:
            raise SmartSegmentError(status_code=422, detail="Polygon too small")

        points: list[float] = []
        for row, col in approx:
            points.append(float(col))
            points.append(float(row))

        if len(points) < 6:
            raise SmartSegmentError(status_code=422, detail="Polygon too small")
        return points, int(mask_bool.sum())
    except SmartSegmentError:
        raise
    except Exception:
        # Fallback without skimage: use tight bbox polygon.
        ys, xs = np.where(mask_bool)
        if ys.size == 0 or xs.size == 0:
            raise SmartSegmentError(status_code=422, detail="No region found; try another point or adjust threshold")
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max())
        y2 = float(ys.max())
        points = [x1, y1, x2, y1, x2, y2, x1, y2]
        return points, area


def _flood_fill_tolerance(arr: "object", seed_y: int, seed_x: int, tol: float):
    import numpy as np  # local import

    img = np.asarray(arr, dtype=np.float32)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    base = float(img[seed_y, seed_x])
    q: deque[tuple[int, int]] = deque([(seed_y, seed_x)])
    visited[seed_y, seed_x] = True

    while q:
        y, x = q.popleft()
        if abs(float(img[y, x]) - base) > tol:
            continue
        mask[y, x] = True
        if y > 0 and not visited[y - 1, x]:
            visited[y - 1, x] = True
            q.append((y - 1, x))
        if y + 1 < h and not visited[y + 1, x]:
            visited[y + 1, x] = True
            q.append((y + 1, x))
        if x > 0 and not visited[y, x - 1]:
            visited[y, x - 1] = True
            q.append((y, x - 1))
        if x + 1 < w and not visited[y, x + 1]:
            visited[y, x + 1] = True
            q.append((y, x + 1))

    return mask


def _segment_flood(image_path: Path, point: tuple[float, float], tolerance: float, simplify: float) -> tuple[list[float], int]:
    try:
        import numpy as np  # local import
        from PIL import Image
        flood = None
        try:
            from skimage.segmentation import flood as sk_flood  # type: ignore

            flood = sk_flood
        except Exception:
            flood = None
    except Exception as exc:
        raise SmartSegmentError(status_code=503, detail=f"Smart segment dependency missing: {exc}") from exc

    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.size == 0:
        raise SmartSegmentError(status_code=422, detail="Empty image")

    h, w = arr.shape[:2]
    px, py = point
    x = int(max(0, min(w - 1, round(float(px)))))
    y = int(max(0, min(h - 1, round(float(py)))))

    tol = _clamp01(float(tolerance))
    if flood is not None:
        mask = flood(arr, (y, x), tolerance=tol)
    else:
        mask = _flood_fill_tolerance(arr, y, x, tol)
    if mask is None:
        raise SmartSegmentError(status_code=422, detail="No region found; try another point or increase tolerance")

    if int(mask.sum()) > int(h * w * 0.9):
        raise SmartSegmentError(status_code=422, detail="Region too large; decrease tolerance or click a different point")

    return _mask_to_polygon(mask, simplify=float(simplify))


def _ensure_ultralytics_model(engine: str) -> object:
    # Decide model family + default weights.
    if engine.startswith("sam"):
        cls_name = "SAM"
        default_weights = "sam_b.pt"
    else:
        cls_name = "FastSAM"
        # Prefer a smaller default for interactive UX; users can override via env.
        default_weights = "FastSAM-s.pt"

    weights = (os.getenv("AIPT_SMART_SEG_WEIGHTS") or default_weights).strip() or default_weights
    device = (os.getenv("AIPT_SMART_SEG_DEVICE") or ("0" if _cuda_available() else "cpu")).strip() or "cpu"
    cache_key = "|".join([engine, weights, device])

    with _MODEL_LOCK:
        cached = _MODELS.get(cache_key)
        if cached is not None:
            return cached

    try:
        ul = importlib.import_module("ultralytics")
        ModelCls = getattr(ul, cls_name)
    except Exception as exc:
        raise SmartSegmentError(status_code=503, detail=f"ultralytics is not available: {exc}") from exc

    model = ModelCls(weights)

    # Best-effort TensorRT export when requested.
    if engine.endswith("_tensorrt") and not str(weights).lower().endswith(".engine"):
        if not _tensorrt_available():
            logger.info("TensorRT not available; falling back to PyTorch weights for smart segmentation.")
        elif not _cuda_available():
            logger.info("CUDA not available; falling back to PyTorch weights for smart segmentation.")
        else:
            try:
                exported = model.export(format="engine", device=device, half=True)
                if exported and str(exported).lower().endswith(".engine"):
                    model = ModelCls(str(exported))
            except Exception as exc:
                logger.warning("Smart segment TensorRT export failed; falling back to PyTorch (%s).", exc)

    with _MODEL_LOCK:
        _MODELS[cache_key] = model
    return model


def _segment_ultralytics(
    engine: str, image_path: Path, point: tuple[float, float], tolerance: float, simplify: float
) -> tuple[list[float], int]:
    model = _ensure_ultralytics_model(engine)

    px, py = point
    conf = _clamp01(float(tolerance))

    try:
        results = model.predict(str(image_path), points=[[float(px), float(py)]], labels=[1], conf=conf, verbose=False)  # type: ignore[attr-defined]
    except TypeError:
        # Some versions may not accept conf.
        results = model.predict(str(image_path), points=[[float(px), float(py)]], labels=[1], verbose=False)  # type: ignore[attr-defined]

    if not results:
        raise SmartSegmentError(status_code=422, detail="No region found; try another point or adjust threshold")
    r0 = results[0]
    masks = getattr(r0, "masks", None)
    if masks is None:
        raise SmartSegmentError(status_code=422, detail="No mask found; try another point or adjust threshold")

    polys = getattr(masks, "xy", None)
    if not polys:
        raise SmartSegmentError(status_code=422, detail="No mask found; try another point or adjust threshold")

    candidates: list[list[tuple[float, float]]] = []
    for poly in polys:
        try:
            pts = [(float(p[0]), float(p[1])) for p in poly]
            if len(pts) >= 3:
                candidates.append(pts)
        except Exception:
            continue

    if not candidates:
        raise SmartSegmentError(status_code=422, detail="No usable polygon found")

    # Prefer the polygon that contains the clicked point; otherwise fall back to the largest one.
    selected: list[tuple[float, float]] | None = None
    for poly in candidates:
        if _point_in_polygon(float(px), float(py), poly):
            selected = poly
            break
    if selected is None:
        selected = max(candidates, key=lambda p: abs(_polygon_area(p)))

    selected = _simplify_polygon(selected, tolerance=float(simplify))
    if len(selected) < 3:
        raise SmartSegmentError(status_code=422, detail="Polygon too small")

    points: list[float] = []
    for x, y in selected:
        points.append(float(x))
        points.append(float(y))
    if len(points) < 6:
        raise SmartSegmentError(status_code=422, detail="Polygon too small")

    area = int(abs(_polygon_area(selected)))
    return points, area


def smart_segment_polygon(
    image_path: Path,
    point: tuple[float, float],
    tolerance: float,
    simplify: float,
    engine: str | None = None,
) -> tuple[list[float], int]:
    chosen = _normalize_engine(engine)
    if chosen == "auto":
        chosen = _auto_engine()

    if chosen == "flood":
        return _segment_flood(image_path=image_path, point=point, tolerance=tolerance, simplify=simplify)

    try:
        return _segment_ultralytics(engine=chosen, image_path=image_path, point=point, tolerance=tolerance, simplify=simplify)
    except SmartSegmentError:
        raise
    except Exception as exc:
        logger.warning("Smart segment failed with engine=%s; falling back to flood (%s).", chosen, exc)
        return _segment_flood(image_path=image_path, point=point, tolerance=tolerance, simplify=simplify)
