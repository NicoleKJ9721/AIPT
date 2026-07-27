from __future__ import annotations

import inspect

from PIL import Image

class ObjectDetector:
    def __init__(self, model_path="yolov8s.pt"):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is not installed; run `pip install ultralytics` "
                f"(import error: {e})"
            ) from e
        self.model = YOLO(model_path)

    def predict(
        self,
        image: Image.Image,
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 100,
        imgsz: int | None = None,
        classes: list[int] | None = None,
    ):
        # Run inference with configurable filtering knobs for interactive auto-labeling.
        predict_kwargs: dict[str, object] = {
            "verbose": False,
            "conf": float(conf),
            "iou": float(iou),
            "max_det": int(max_det),
        }
        if imgsz is not None:
            predict_kwargs["imgsz"] = int(imgsz)
        if classes:
            predict_kwargs["classes"] = [int(c) for c in classes]

        try:
            call_fn = self.model.predict if hasattr(self.model, "predict") else self.model
            sig = inspect.signature(call_fn)
            safe_kwargs = {k: v for k, v in predict_kwargs.items() if k in sig.parameters}
            results = call_fn(image, **safe_kwargs)
        except TypeError:
            results = self.model(image)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                label = result.names[int(cls)]
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": label,
                    "class_id": int(cls)
                })
        return detections

    def train(self, data_yaml, epochs=1):
        # Start training
        results = self.model.train(data=data_yaml, epochs=epochs)
        return results
