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


def filter_player_detections(
    detections: list[Detection],
    max_players: int | None = None,
) -> list[Detection]:
    """Keep player classes, optionally retaining the highest-confidence players."""
    players = [
        detection for detection in detections if is_player_class(detection.class_name)
    ]
    if max_players is None:
        return players
    if max_players < 1:
        raise ValueError("max_players must be at least one")
    return sorted(players, key=lambda detection: detection.confidence, reverse=True)[
        :max_players
    ]


def expected_player_count(court_type: str) -> int:
    """Return the player cap for the supported racket-sport court types."""
    normalized = court_type.strip().lower()
    if normalized == "tennis":
        return 2
    if normalized in {"padel", "paddle"}:
        return 4
    raise ValueError(f"Unsupported court type: {court_type}")


def filter_player_ball_detections(detections: list[Detection]) -> list[Detection]:
    return [
        detection
        for detection in detections
        if is_player_or_ball_class(detection.class_name)
    ]
