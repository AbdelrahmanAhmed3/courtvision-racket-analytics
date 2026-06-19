from __future__ import annotations

from dataclasses import dataclass

PLAYER_CLASS_NAMES = frozenset({"person", "player"})
BALL_CLASS_NAMES = frozenset({"ball", "tennis-ball", "tennis ball", "sports ball"})


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


def normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace("_", "-")


def is_player_class(class_name: str) -> bool:
    return normalize_class_name(class_name) in PLAYER_CLASS_NAMES


def is_ball_class(class_name: str) -> bool:
    return normalize_class_name(class_name) in BALL_CLASS_NAMES


def is_player_or_ball_class(class_name: str) -> bool:
    return is_player_class(class_name) or is_ball_class(class_name)


def filter_player_detections(detections: list[Detection]) -> list[Detection]:
    return [
        detection for detection in detections if is_player_class(detection.class_name)
    ]


def filter_player_ball_detections(detections: list[Detection]) -> list[Detection]:
    return [
        detection
        for detection in detections
        if is_player_or_ball_class(detection.class_name)
    ]
