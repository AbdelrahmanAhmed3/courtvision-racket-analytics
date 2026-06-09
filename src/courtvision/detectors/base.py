from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    frame: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    model_name: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Detector:
    model_name: str

    def predict_frame(self, frame, frame_index: int) -> list[Detection]:
        raise NotImplementedError
