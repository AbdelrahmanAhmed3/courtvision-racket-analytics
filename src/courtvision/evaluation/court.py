"""Where the court is in a labelled segment, from four points clicked by hand.

The points are where the service lines meet the side walls: painted, sharp and
well inside broadcast frames, unlike the near floor corners, which boards and
glass often hide. Four points fit a homography exactly, so nothing can measure
the clicks: the tool draws the court from the fit, and its baselines, net and
centre line must sit on the real ones.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from courtvision.calibration.homography import (
    HomographyEstimate,
    project_template_points,
)
from courtvision.calibration.landmarks import landmark_by_name

# Clicked in this order; each name is a padel calibration landmark.
LANDMARK_NAMES = (
    "far_left_service_intersection",
    "far_right_service_intersection",
    "near_right_service_intersection",
    "near_left_service_intersection",
)
POINT_NAMES = (
    "far service line, left end",
    "far service line, right end",
    "near service line, right end",
    "near service line, left end",
)
TEMPLATE_POINTS = np.float32(
    [landmark_by_name("padel", name).template for name in LANDMARK_NAMES]
)


def court_estimate(points: Sequence[Sequence[float]]) -> HomographyEstimate:
    """The image-to-court homography that fits the four clicked points."""
    image_points = np.asarray(points, dtype=np.float32)
    template_to_image = cv2.getPerspectiveTransform(TEMPLATE_POINTS, image_points)
    return HomographyEstimate(
        template_to_image=template_to_image,
        image_to_template=np.linalg.inv(template_to_image),
        inlier_names=LANDMARK_NAMES,
        image_points=image_points,
        template_points=TEMPLATE_POINTS,
        landmark_names=LANDMARK_NAMES,
    )


def court_lines(points: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """Image end points of the court outline, net and centre line, for checking.

    The net line is where the net meets the floor, not its top.
    """
    template_lines = [
        [[0, 0], [1, 0]],  # far baseline
        [[1, 0], [1, 1]],  # right side
        [[1, 1], [0, 1]],  # near baseline
        [[0, 1], [0, 0]],  # left side
        [[0, 0.5], [1, 0.5]],  # net
        [[0.5, 0.15], [0.5, 0.85]],  # centre service line
    ]
    estimate = court_estimate(points)
    return [
        project_template_points(np.float32(line), estimate) for line in template_lines
    ]
