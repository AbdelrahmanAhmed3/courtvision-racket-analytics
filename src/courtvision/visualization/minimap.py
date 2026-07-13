from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CourtSpec:
    name: str
    width_m: float
    length_m: float
    service_line_from_baseline_m: float
    singles_width_m: float | None = None


TENNIS_COURT = CourtSpec(
    name="tennis",
    width_m=10.97,
    length_m=23.77,
    service_line_from_baseline_m=5.485,
    singles_width_m=8.23,
)
PADEL_COURT = CourtSpec(
    name="padel",
    width_m=10.0,
    length_m=20.0,
    service_line_from_baseline_m=3.0,
)


def get_court_spec(court_type: str) -> CourtSpec:
    normalized = court_type.strip().lower()
    if normalized == "tennis":
        return TENNIS_COURT
    if normalized in {"padel", "paddle"}:
        return PADEL_COURT
    raise ValueError(f"Unsupported court type: {court_type}")


def world_to_canvas(
    point: tuple[float, float],
    spec: CourtSpec,
    canvas_size: tuple[int, int],
    margin: int,
) -> tuple[int, int]:
    canvas_width, canvas_height = canvas_size
    usable_width = canvas_width - 2 * margin
    usable_height = canvas_height - 2 * margin
    x = margin + point[0] / spec.width_m * usable_width
    y = margin + point[1] / spec.length_m * usable_height
    return int(round(x)), int(round(y))


def draw_court_map(
    court_type: str,
    canvas_size: tuple[int, int] = (500, 900),
    margin: int = 45,
) -> np.ndarray:
    spec = get_court_spec(court_type)
    canvas_width, canvas_height = canvas_size
    image = np.full((canvas_height, canvas_width, 3), (31, 91, 59), dtype=np.uint8)

    top_left = world_to_canvas((0, 0), spec, canvas_size, margin)
    bottom_right = world_to_canvas(
        (spec.width_m, spec.length_m),
        spec,
        canvas_size,
        margin,
    )
    cv2.rectangle(image, top_left, bottom_right, (235, 240, 235), 3)

    net_y = spec.length_m / 2
    _draw_world_line(
        image,
        (0, net_y),
        (spec.width_m, net_y),
        spec,
        canvas_size,
        margin,
    )

    service = spec.service_line_from_baseline_m
    _draw_world_line(
        image,
        (0, service),
        (spec.width_m, service),
        spec,
        canvas_size,
        margin,
    )
    _draw_world_line(
        image,
        (0, spec.length_m - service),
        (spec.width_m, spec.length_m - service),
        spec,
        canvas_size,
        margin,
    )
    _draw_world_line(
        image,
        (spec.width_m / 2, service),
        (spec.width_m / 2, spec.length_m - service),
        spec,
        canvas_size,
        margin,
    )

    if spec.singles_width_m is not None:
        side_margin = (spec.width_m - spec.singles_width_m) / 2
        _draw_world_line(
            image,
            (side_margin, 0),
            (side_margin, spec.length_m),
            spec,
            canvas_size,
            margin,
        )
        _draw_world_line(
            image,
            (spec.width_m - side_margin, 0),
            (spec.width_m - side_margin, spec.length_m),
            spec,
            canvas_size,
            margin,
        )

    label = f"{spec.name.upper()} COURT"
    cv2.putText(
        image,
        label,
        (margin, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 240, 235),
        2,
        cv2.LINE_AA,
    )
    return image


def draw_projected_point(
    image: np.ndarray,
    point: tuple[float, float],
    court_type: str,
    label: str,
    color: tuple[int, int, int],
    canvas_size: tuple[int, int] = (500, 900),
    margin: int = 45,
) -> np.ndarray:
    spec = get_court_spec(court_type)
    x, y = world_to_canvas(point, spec, canvas_size, margin)
    cv2.circle(image, (x, y), 8, color, -1)
    cv2.putText(
        image,
        label,
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def _draw_world_line(
    image: np.ndarray,
    first: tuple[float, float],
    second: tuple[float, float],
    spec: CourtSpec,
    canvas_size: tuple[int, int],
    margin: int,
) -> None:
    cv2.line(
        image,
        world_to_canvas(first, spec, canvas_size, margin),
        world_to_canvas(second, spec, canvas_size, margin),
        (235, 240, 235),
        2,
        cv2.LINE_AA,
    )
