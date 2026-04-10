"""YOLO inference wrapper loaded once at startup (singleton)."""

import io
import time

from PIL import Image
from ultralytics import YOLO

from .config import settings
from .schemas import BBox, Detection, ImageResult


class Detector:
    _instance: "Detector | None" = None
    _model: YOLO | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_path: str | None = None):
        path = model_path or settings.model_path
        self._model = YOLO(path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image_bytes: bytes, image_id: str, conf: float = 0.25) -> ImageResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        start = time.perf_counter()
        results = self._model(img, conf=conf, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        detections: list[Detection] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf_val = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                detections.append(Detection(
                    class_name=self._model.names.get(cls_id, f"class_{cls_id}"),
                    class_id=cls_id,
                    confidence=round(conf_val, 4),
                    bbox=BBox(x1=round(x1, 1), y1=round(y1, 1),
                              x2=round(x2, 1), y2=round(y2, 1)),
                ))

        return ImageResult(image_id=image_id, detections=detections, inference_time_ms=round(elapsed_ms, 2))


detector = Detector()
