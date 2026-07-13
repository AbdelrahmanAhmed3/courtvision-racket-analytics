from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LandmarkDefinition:
    name: str
    template: tuple[float, float]
    required: bool = False


_TENNIS_SINGLE_MARGIN = (10.97 - 8.23) / (2 * 10.97)
_TENNIS_SERVICE_Y = 5.485 / 23.77

TENNIS_LANDMARKS = (
    LandmarkDefinition("far_left_baseline_corner", (0.0, 0.0), required=True),
    LandmarkDefinition("far_right_baseline_corner", (1.0, 0.0), required=True),
    LandmarkDefinition("near_right_baseline_corner", (1.0, 1.0), required=True),
    LandmarkDefinition("near_left_baseline_corner", (0.0, 1.0), required=True),
    LandmarkDefinition("left_net_point", (0.0, 0.5)),
    LandmarkDefinition("right_net_point", (1.0, 0.5)),
    LandmarkDefinition(
        "far_left_service_intersection",
        (_TENNIS_SINGLE_MARGIN, _TENNIS_SERVICE_Y),
    ),
    LandmarkDefinition(
        "far_right_service_intersection",
        (1.0 - _TENNIS_SINGLE_MARGIN, _TENNIS_SERVICE_Y),
    ),
    LandmarkDefinition(
        "near_left_service_intersection",
        (_TENNIS_SINGLE_MARGIN, 1.0 - _TENNIS_SERVICE_Y),
    ),
    LandmarkDefinition(
        "near_right_service_intersection",
        (1.0 - _TENNIS_SINGLE_MARGIN, 1.0 - _TENNIS_SERVICE_Y),
    ),
)

PADEL_LANDMARKS = (
    LandmarkDefinition("far_left_baseline_corner", (0.0, 0.0), required=True),
    LandmarkDefinition("far_right_baseline_corner", (1.0, 0.0), required=True),
    LandmarkDefinition("near_right_baseline_corner", (1.0, 1.0), required=True),
    LandmarkDefinition("near_left_baseline_corner", (0.0, 1.0), required=True),
    LandmarkDefinition("left_net_point", (0.0, 0.5)),
    LandmarkDefinition("right_net_point", (1.0, 0.5)),
    LandmarkDefinition("far_left_service_intersection", (0.0, 0.15)),
    LandmarkDefinition("far_right_service_intersection", (1.0, 0.15)),
    LandmarkDefinition("near_left_service_intersection", (0.0, 0.85)),
    LandmarkDefinition("near_right_service_intersection", (1.0, 0.85)),
)


def landmarks_for_court(court_type: str) -> tuple[LandmarkDefinition, ...]:
    normalized = court_type.strip().lower()
    if normalized == "tennis":
        return TENNIS_LANDMARKS
    if normalized in {"padel", "paddle"}:
        return PADEL_LANDMARKS
    raise ValueError(f"Unsupported court type: {court_type}")


def landmark_by_name(court_type: str, name: str) -> LandmarkDefinition:
    for landmark in landmarks_for_court(court_type):
        if landmark.name == name:
            return landmark
    raise KeyError(f"Unknown {court_type} landmark: {name}")


def default_landmark_names(court_type: str) -> list[str]:
    normalized = court_type.strip().lower()
    if normalized in {"tennis", "padel", "paddle"}:
        return [
            "far_left_baseline_corner",
            "far_right_baseline_corner",
            "near_right_baseline_corner",
            "near_left_baseline_corner",
            "far_left_service_intersection",
            "far_right_service_intersection",
            "near_right_service_intersection",
            "near_left_service_intersection",
        ]
    return [landmark.name for landmark in landmarks_for_court(court_type)]


def display_landmark_name(name: str) -> str:
    labels = {
        "far_left_baseline_corner": "Far left baseline corner (top-left)",
        "far_right_baseline_corner": "Far right baseline corner (top-right)",
        "near_right_baseline_corner": "Near right baseline corner (bottom-right)",
        "near_left_baseline_corner": "Near left baseline corner (bottom-left)",
        "left_net_point": "Left net point",
        "right_net_point": "Right net point",
        "far_left_service_intersection": "Far left service-line intersection",
        "far_right_service_intersection": "Far right service-line intersection",
        "near_left_service_intersection": "Near left service-line intersection",
        "near_right_service_intersection": "Near right service-line intersection",
    }
    return labels.get(name, name.replace("_", " ").title())
