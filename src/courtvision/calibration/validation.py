from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtvision.calibration.homography import (
    HomographyEstimate,
    project_template_points,
)


@dataclass(frozen=True)
class CalibrationValidation:
    status: str
    mean_reprojection_error_px: float
    max_reprojection_error_px: float
    per_landmark_error_px: dict[str, float]
    inlier_count: int
    landmark_count: int


def validate_homography(estimate: HomographyEstimate) -> CalibrationValidation:
    projected = project_template_points(estimate.template_points, estimate)
    errors = np.linalg.norm(projected - estimate.image_points, axis=1)
    per_landmark = {
        name: float(error)
        for name, error in zip(estimate.landmark_names, errors, strict=True)
    }
    landmark_count = len(estimate.landmark_names)
    inlier_count = len(estimate.inlier_names)
    inliers = np.asarray(
        [name in estimate.inlier_names for name in estimate.landmark_names],
        dtype=bool,
    )
    inlier_errors = errors[inliers]
    mean_inlier_error = (
        float(inlier_errors.mean()) if len(inlier_errors) else float("inf")
    )
    max_inlier_error = (
        float(inlier_errors.max()) if len(inlier_errors) else float("inf")
    )

    if landmark_count == 4:
        status = "warning"
    elif inlier_count < 4:
        status = "bad"
    elif mean_inlier_error <= 5 and max_inlier_error <= 8:
        status = "good"
    elif mean_inlier_error <= 15:
        status = "acceptable"
    else:
        status = "bad"

    return CalibrationValidation(
        status=status,
        mean_reprojection_error_px=mean_inlier_error,
        max_reprojection_error_px=max_inlier_error,
        per_landmark_error_px=per_landmark,
        inlier_count=inlier_count,
        landmark_count=landmark_count,
    )
