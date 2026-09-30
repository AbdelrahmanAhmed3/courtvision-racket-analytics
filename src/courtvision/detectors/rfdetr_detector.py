"""Local player detection with RF-DETR's COCO-pretrained weights.

RF-DETR and its Nano/Small/Medium/Base weights are Apache-2.0 (ADR 0002). The
weights download once on first use and are cached by the ``rfdetr`` package.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from courtvision.detectors.base import Detection

DEFAULT_SIZE = "small"
DEFAULT_CONFIDENCE = 0.3
PLAYER_COCO_CLASS = "person"
RFDETR_SIZES = ("nano", "small", "medium", "base")


def load_rfdetr_model(size: str = DEFAULT_SIZE) -> Any:
    """Load a COCO-pretrained RF-DETR model, keeping rfdetr an optional dependency."""
    if size not in RFDETR_SIZES:
        raise ValueError(f"size must be one of {', '.join(RFDETR_SIZES)}")
    try:
        import rfdetr
    except ImportError as exc:
        raise ImportError(
            "RF-DETR is not installed. Install it with `pip install -e '.[local]'`."
        ) from exc
    model_class = getattr(rfdetr, f"RFDETR{size.capitalize()}")
    return model_class()


def coco_class_names() -> dict[int, str]:
    """Map RF-DETR's COCO category ids (1-based, with gaps) to names."""
    from rfdetr.assets.coco_classes import COCO_CLASSES

    return dict(COCO_CLASSES)


class RFDetrDetector:
    """Detect players locally; keeps only COCO ``person`` boxes."""

    def __init__(
        self,
        size: str = DEFAULT_SIZE,
        confidence: float = DEFAULT_CONFIDENCE,
        model: Any | None = None,
        class_names: dict[int, str] | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        self.confidence = confidence
        self.model_name = f"rf-detr-{size}"
        self.model = model if model is not None else load_rfdetr_model(size)
        self.class_names = (
            class_names if class_names is not None else coco_class_names()
        )

    def predict_frame(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        # OpenCV frames are BGR; RF-DETR expects RGB arrays.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        result = self.model.predict(rgb, threshold=self.confidence)
        detections = []
        for box, confidence, class_id in zip(
            result.xyxy, result.confidence, result.class_id, strict=True
        ):
            if self.class_names.get(int(class_id)) != PLAYER_COCO_CLASS:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            detections.append(
                Detection(
                    frame=frame_index,
                    class_name=PLAYER_COCO_CLASS,
                    confidence=float(confidence),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    model_name=self.model_name,
                )
            )
        return detections
