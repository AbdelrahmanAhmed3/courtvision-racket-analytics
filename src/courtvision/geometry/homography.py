from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CourtCalibration:
    court_type: str
    image_points: np.ndarray
    court_points: np.ndarray


def load_calibration(path: str | Path) -> CourtCalibration:
    """Load corresponding image and court-plane points from a JSON file."""
    data = json.loads(Path(path).read_text())
    image_points = np.asarray(data["image_points"], dtype=np.float32)
    court_points = np.asarray(data["court_points"], dtype=np.float32)
    if image_points.shape != court_points.shape:
        raise ValueError("image_points and court_points must have the same shape")
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("Calibration points must be Nx2 coordinates")
    if len(image_points) < 4:
        raise ValueError("At least four point correspondences are required")

    return CourtCalibration(
        court_type=data["court_type"],
        image_points=image_points,
        court_points=court_points,
    )


def estimate_homography(
    image_points: np.ndarray,
    court_points: np.ndarray,
    ransac_reprojection_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate image-to-court homography using RANSAC."""
    homography, inlier_mask = cv2.findHomography(
        image_points,
        court_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reprojection_threshold,
    )
    if homography is None or inlier_mask is None:
        raise ValueError("Could not estimate a valid homography")
    return homography, inlier_mask.reshape(-1).astype(bool)


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def bottom_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Return the image point that approximates a player's foot position."""
    return ((x1 + x2) / 2, y2)


def reprojection_error(
    image_points: np.ndarray,
    court_points: np.ndarray,
    homography: np.ndarray,
    inliers: np.ndarray | None = None,
) -> float:
    projected = project_points(image_points, homography)
    errors = np.linalg.norm(projected - court_points, axis=1)
    if inliers is not None and inliers.any():
        errors = errors[inliers]
    return float(errors.mean()) if len(errors) else float("inf")
