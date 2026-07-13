from __future__ import annotations

import cv2
import numpy as np

from courtvision.calibration.homography import (
    HomographyEstimate,
    project_template_points,
)
from courtvision.calibration.io import CalibrationRecord
from courtvision.calibration.validation import CalibrationValidation
from courtvision.visualization.minimap import CourtSpec, get_court_spec

TemplateLine = tuple[tuple[float, float], tuple[float, float]]


def draw_calibration_overlay(
    frame: np.ndarray,
    calibration: CalibrationRecord,
    estimate: HomographyEstimate,
    validation: CalibrationValidation,
) -> np.ndarray:
    image = frame.copy()
    _draw_projected_court(image, calibration.court_type, estimate)

    projected = project_template_points(estimate.template_points, estimate)
    for index, name in enumerate(estimate.landmark_names):
        observed = estimate.image_points[index]
        expected = projected[index]
        error = validation.per_landmark_error_px[name]
        inlier = name in estimate.inlier_names
        color = (0, 220, 0) if inlier else (0, 0, 255)
        x, y = [int(round(value)) for value in observed]
        cv2.circle(image, (x, y), 6, color, -1)
        cv2.line(
            image,
            (x, y),
            tuple(int(round(value)) for value in expected),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{name} {error:.1f}px",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    _draw_status(image, validation)
    return image


def _draw_projected_court(
    image: np.ndarray,
    court_type: str,
    estimate: HomographyEstimate,
) -> None:
    spec = get_court_spec(court_type)
    lines = _template_lines(court_type, spec)
    for start, end in lines:
        projected = project_template_points(np.asarray([start, end]), estimate)
        cv2.line(
            image,
            tuple(int(round(value)) for value in projected[0]),
            tuple(int(round(value)) for value in projected[1]),
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )


def _template_lines(
    court_type: str,
    spec: CourtSpec,
) -> list[TemplateLine]:
    service_y = spec.service_line_from_baseline_m / spec.length_m
    lines = [
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (1.0, 1.0)),
        ((1.0, 1.0), (0.0, 1.0)),
        ((0.0, 1.0), (0.0, 0.0)),
        ((0.0, 0.5), (1.0, 0.5)),
        ((0.0, service_y), (1.0, service_y)),
        ((0.0, 1.0 - service_y), (1.0, 1.0 - service_y)),
        ((0.5, service_y), (0.5, 1.0 - service_y)),
    ]
    if court_type.lower() == "tennis" and spec.singles_width_m is not None:
        margin = (spec.width_m - spec.singles_width_m) / (2 * spec.width_m)
        lines.extend(
            [
                ((margin, 0.0), (margin, 1.0)),
                ((1.0 - margin, 0.0), (1.0 - margin, 1.0)),
            ]
        )
    return lines


def _draw_status(image: np.ndarray, validation: CalibrationValidation) -> None:
    status_color = {
        "good": (0, 220, 0),
        "acceptable": (0, 220, 255),
        "warning": (0, 165, 255),
        "bad": (0, 0, 255),
    }[validation.status]
    text = (
        f"Calibration {validation.status.upper()} | "
        f"inliers {validation.inlier_count}/{validation.landmark_count} | "
        f"mean {validation.mean_reprojection_error_px:.1f}px | "
        f"max {validation.max_reprojection_error_px:.1f}px"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 36), (18, 18, 18), -1)
    cv2.putText(
        image,
        text,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA,
    )
