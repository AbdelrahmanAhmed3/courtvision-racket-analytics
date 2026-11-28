"""Where the court is in a labelled segment, from points clicked by hand.

The court is fit from any four or more padel landmarks the camera sees. In
broadcast frames these are the four service-line ends, where the service lines
meet the side walls: painted, sharp and well inside the picture, unlike the near
floor corners, which boards and glass often hide. From a phone on the back glass
the near service line runs off both sides of the picture, so the near "T", the
net's ends on the floor and the far half are clicked instead.

Four points fit a homography exactly, so nothing can measure the clicks: the tool
draws the court from the fit, and its lines must sit on the real ones. With five
or more the fit is least squares, and each click's distance from the fitted court
shows a bad click.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import cv2
import numpy as np

from courtvision.calibration.homography import (
    HomographyEstimate,
    project_template_points,
)
from courtvision.calibration.landmarks import landmark_by_name

# Asked for in this order when marking; the first four are the default.
COURT_LANDMARKS = (
    ("far_left_service_intersection", "far service line, left end"),
    ("far_right_service_intersection", "far service line, right end"),
    ("near_right_service_intersection", "near service line, right end"),
    ("near_left_service_intersection", "near service line, left end"),
    ("far_service_center", "far service line, centre (the T)"),
    ("near_service_center", "near service line, centre (the T)"),
    ("left_net_point", "net, left end on the floor"),
    ("right_net_point", "net, right end on the floor"),
    ("net_center", "net, centre on the floor"),
    ("far_left_baseline_corner", "far left corner"),
    ("far_right_baseline_corner", "far right corner"),
    ("near_right_baseline_corner", "near right corner"),
    ("near_left_baseline_corner", "near left corner"),
)
DESCRIPTIONS = dict(COURT_LANDMARKS)
# Four unnamed points are these, in this order (labels saved before names).
LANDMARK_NAMES = tuple(name for name, _ in COURT_LANDMARKS[:4])
MIN_POINTS = 4
# A line's part closer to the camera than this share of the net's distance is
# not drawn: it is far outside the picture, or behind the camera.
MIN_DEPTH_SHARE = 0.05

CourtPoints = Sequence[Sequence[float]] | Mapping[str, Sequence[float]]


def named_points(points: CourtPoints) -> dict[str, tuple[float, float]]:
    """The clicked points by landmark name, checked."""
    if isinstance(points, Mapping):
        named = {name: tuple(point) for name, point in points.items()}
    elif len(points) == len(LANDMARK_NAMES):
        named = dict(zip(LANDMARK_NAMES, map(tuple, points), strict=True))
    else:
        raise ValueError("Unnamed court points must be the four service-line ends")
    unknown = sorted(set(named) - set(DESCRIPTIONS))
    if unknown:
        raise ValueError(f"Unknown court landmarks: {unknown}")
    if any(len(point) != 2 for point in named.values()):
        raise ValueError("Court points must be (x, y) points")
    if len(named) < MIN_POINTS:
        raise ValueError(f"The court needs at least {MIN_POINTS} points")
    if not _spread(named):
        raise ValueError("No four of these points are free of three on one line")
    return {name: (float(x), float(y)) for name, (x, y) in named.items()}


def court_estimate(points: CourtPoints) -> HomographyEstimate:
    """The image-to-court homography that fits the clicked points.

    ``points`` is either four points, the service-line ends in the order of
    ``LANDMARK_NAMES``, or a mapping from landmark name to point.
    """
    named = named_points(points)
    names = tuple(named)
    image_points = np.float32(list(named.values()))
    template_points = np.float32([_template(name) for name in names])
    if len(names) == MIN_POINTS:
        template_to_image = cv2.getPerspectiveTransform(template_points, image_points)
    else:
        # Least squares, refined to the smallest distances in the image.
        template_to_image, _ = cv2.findHomography(template_points, image_points, 0)
    return HomographyEstimate(
        template_to_image=template_to_image,
        image_to_template=np.linalg.inv(template_to_image),
        inlier_names=names,
        image_points=image_points,
        template_points=template_points,
        landmark_names=names,
    )


def court_half(
    points: CourtPoints, x: float, y: float, margin: float = 0.0
) -> int | None:
    """0 if image point (x, y) on the floor is in the near half, 1 if the far.

    None if it is within ``margin`` of the net, as a share of the court's
    length, where a click a few pixels off could put it on either side.
    """
    estimate = court_estimate(points)
    point = np.float32([[[x, y]]])
    _, court_y = cv2.perspectiveTransform(point, estimate.image_to_template)[0, 0]
    if abs(court_y - 0.5) < margin:
        return None
    return 0 if court_y > 0.5 else 1


def court_residuals(points: CourtPoints) -> dict[str, float]:
    """How far each click is from where the fitted court puts its landmark, in pixels.

    Zero for four points, which fit exactly.
    """
    estimate = court_estimate(points)
    fitted = project_template_points(estimate.template_points, estimate)
    distances = np.linalg.norm(fitted - estimate.image_points, axis=1)
    return dict(zip(estimate.landmark_names, distances.tolist(), strict=True))


def court_lines(points: CourtPoints) -> list[np.ndarray]:
    """Image end points of the court outline, net, centre and service lines.

    The net line is where the net meets the floor, not its top. A line's part
    near or behind the camera is cut off; a line with nothing left is empty.
    """
    template_lines = [
        [[0, 0], [1, 0]],  # far baseline
        [[1, 0], [1, 1]],  # right side
        [[1, 1], [0, 1]],  # near baseline
        [[0, 1], [0, 0]],  # left side
        [[0, 0.5], [1, 0.5]],  # net
        [[0.5, 0.15], [0.5, 0.85]],  # centre service line
        [[0, 0.15], [1, 0.15]],  # far service line
        [[0, 0.85], [1, 0.85]],  # near service line
    ]
    estimate = court_estimate(points)
    depth = estimate.template_to_image[2]
    depth = depth if depth @ [0.5, 0.5, 1] > 0 else -depth
    min_depth = MIN_DEPTH_SHARE * (depth @ [0.5, 0.5, 1])
    lines = []
    for line in template_lines:
        visible = _part_in_front(np.float64(line), depth, min_depth)
        lines.append(
            np.empty((0, 2), np.float32)
            if visible is None
            else project_template_points(np.float32(visible), estimate)
        )
    return lines


def _part_in_front(line: np.ndarray, depth: np.ndarray, min_depth: float):
    """The part of a court line at least ``min_depth`` from the camera, or None.

    Depth (the homography's third row) is linear in court coordinates, so the cut
    falls where it equals ``min_depth``.
    """
    start, end = line
    near_start, near_end = (depth @ [*point, 1] for point in (start, end))
    if near_start < min_depth and near_end < min_depth:
        return None
    if near_start < min_depth:
        start = start + (end - start) * _share(near_start, near_end, min_depth)
    elif near_end < min_depth:
        end = end + (start - end) * _share(near_end, near_start, min_depth)
    return np.stack([start, end])


def _share(near: float, far: float, min_depth: float) -> float:
    """How far from the near end, as a share of the line, depth reaches the minimum."""
    return (min_depth - near) / (far - near)


def _template(name: str) -> tuple[float, float]:
    return landmark_by_name("padel", name).template


def _spread(named: Mapping[str, Sequence[float]]) -> bool:
    """Whether some four of the points have no three on one court line.

    Only such four fix a homography: the far service line's ends and T plus one
    more point leave it free.
    """
    templates = [_template(name) for name in named]
    return any(
        not any(_on_one_line(*three) for three in combinations(four, 3))
        for four in combinations(templates, 4)
    )


def _on_one_line(first, second, third) -> bool:
    cross = (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )
    return abs(cross) < 1e-9
