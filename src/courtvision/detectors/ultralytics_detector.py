from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from courtvision.detectors.base import Detection


def get_device(preferred: str = "auto") -> str:
    """Return an Ultralytics-friendly device string."""
    if preferred != "auto":
        return preferred

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_ultralytics_model(model_path: str | Path, model_type: str = "auto") -> Any:
    """Load a YOLO or RT-DETR model while keeping ultralytics as an optional dep."""
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed. Install it with `pip install -e '.[gpu]'` "
            "or `pip install ultralytics`."
        ) from exc

    model_path = str(model_path)
    normalized_type = model_type.lower()
    if normalized_type == "auto":
        normalized_name = Path(model_path).name.lower()
        normalized_type = "rtdetr" if "rtdetr" in normalized_name else "yolo"

    if normalized_type in {"rtdetr", "rt-detr"}:
        return RTDETR(model_path)
    if normalized_type == "yolo":
        return YOLO(model_path)

    raise ValueError(f"Unsupported Ultralytics model type: {model_type}")


class UltralyticsDetector:
    """YOLO/RT-DETR adapter that normalizes Ultralytics results to Detection."""

    def __init__(
        self,
        model_path: str | Path,
        model_type: str = "auto",
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int | tuple[int, int] | None = None,
        classes: Iterable[int] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model = load_ultralytics_model(model_path, model_type)
        self.model_path = str(model_path)
        self.model_name = model_name or Path(model_path).stem
        self.device = get_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = list(classes) if classes is not None else None

    def predict_frame(self, frame, frame_index: int) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            verbose=False,
        )
        return detections_from_result(
            results[0],
            frame_index=frame_index,
            model_name=self.model_name,
        )

    def track_frame(
        self,
        frame,
        frame_index: int,
        persist: bool = True,
        tracker: str = "botsort.yaml",
    ) -> list[Detection]:
        results = self.model.track(
            source=frame,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            persist=persist,
            tracker=tracker,
            verbose=False,
        )
        return detections_from_result(
            results[0],
            frame_index=frame_index,
            model_name=self.model_name,
        )


def detections_from_result(
    result: Any,
    frame_index: int,
    model_name: str,
) -> list[Detection]:
    if result.boxes is None:
        return []

    names = getattr(result, "names", {}) or {}
    detections = []
    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().tolist()
        confidence = float(box.conf[0].detach().cpu())
        class_id = int(box.cls[0].detach().cpu())
        class_name = names.get(class_id, str(class_id))
        detections.append(
            Detection(
                frame=frame_index,
                class_name=class_name,
                confidence=confidence,
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                model_name=model_name,
            )
        )
    return detections
