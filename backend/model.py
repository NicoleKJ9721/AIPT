from __future__ import annotations

from PIL import Image

try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]
    _ULTRALYTICS_IMPORT_ERROR = e

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        if YOLO is None:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is not installed; run `pip install ultralytics` "
                f"(import error: {_ULTRALYTICS_IMPORT_ERROR})"
            )
        self.model = YOLO(model_path)

    def predict(self, image: Image.Image):
        # Run inference
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
