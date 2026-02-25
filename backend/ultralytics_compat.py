from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def import_yolo():  # type: ignore[no-untyped-def]
    """
    Import Ultralytics' YOLO class with best-effort compatibility across versions.

    Prefer the internal model import path to avoid importing optional subsystems
    (e.g. onnxruntime) too early on Windows.
    """
    try:
        from ultralytics.models.yolo.model import YOLO  # type: ignore

        return YOLO
    except Exception:
        from ultralytics import YOLO  # type: ignore

        return YOLO


def filter_kwargs_by_signature(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filter kwargs to only include parameters accepted by the callable.

    This protects against Ultralytics version mismatches where some keyword
    arguments may not be supported.
    """
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return dict(kwargs)

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    return {k: v for k, v in kwargs.items() if k in params}

