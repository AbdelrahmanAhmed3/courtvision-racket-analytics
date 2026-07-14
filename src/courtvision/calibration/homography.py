from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from courtvision.calibration.io import CalibrationRecord

BASELINE_CORNER_NAMES = (
    "far_left_baseline_corner",
    "far_right_baseline_corner",
    "near_right_baseline_corner",
    "near_left_baseline_corner",
)


@dataclass(frozen=True)
class HomographyEstimate:
    template_to_image: np.ndarray
    image_to_template: np.ndarray
    inlier_names: tuple[str, ...]
    image_points: np.ndarray
    template_points: np.ndarray
    landmark_names: tuple[str, ...]


def estimate_template_homography(
    calibration: CalibrationRecord,
    ransac_reprojection_threshold_px: float = 6.0,
    preserve_manual_baseline_corners: bool = True,
) -> HomographyEstimate:
    names = tuple(
        name
        for name, observation in calibration.landmarks.items()
        if observation.visible
    )
    if len(names) < 4:
        raise ValueError("At least four visible landmarks are required")

    image_points = np.asarray(
        [calibration.landmarks[name].image for name in names],
        dtype=np.float32,
    )
    template_points = np.asarray(
        [calibration.landmarks[name].template for name in names],
        dtype=np.float32,
    )
    template_to_image, inlier_mask = cv2.findHomography(
        template_points,
        image_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reprojection_threshold_px,
    )
    if template_to_image is None or inlier_mask is None:
        raise ValueError("Could not estimate a homography from the landmarks")

    inliers = inlier_mask.reshape(-1).astype(bool)
    required_indexes = [
        names.index(name) for name in BASELINE_CORNER_NAMES if name in names
    ]

    # Manual baseline corners are the geometric anchors. Optional landmarks such
    # as a net post must not be allowed to flip or collapse the court if their
    # physical meaning was clicked incorrectly.
    if (
        preserve_manual_baseline_corners
        and len(required_indexes) == 4
        and not inliers[required_indexes].all()
    ):
        corner_template = template_points[required_indexes].astype(np.float32)
        corner_image = image_points[required_indexes].astype(np.float32)
        template_to_image = cv2.getPerspectiveTransform(corner_template, corner_image)
        projected = cv2.perspectiveTransform(
            template_points.reshape(-1, 1, 2),
            template_to_image,
        ).reshape(-1, 2)
        residuals = np.linalg.norm(projected - image_points, axis=1)
        inliers = residuals <= ransac_reprojection_threshold_px

    image_to_template = np.linalg.inv(template_to_image)
    return HomographyEstimate(
        template_to_image=template_to_image,
        image_to_template=image_to_template,
        inlier_names=tuple(
            name for name, inlier in zip(names, inliers, strict=True) if inlier
        ),
        image_points=image_points,
        template_points=template_points,
        landmark_names=names,
    )


def project_template_points(
    points: np.ndarray,
    estimate: HomographyEstimate,
) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        estimate.template_to_image,
    ).reshape(-1, 2)


def project_image_points(
    points: np.ndarray,
    estimate: HomographyEstimate,
) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        estimate.image_to_template,
    ).reshape(-1, 2)
