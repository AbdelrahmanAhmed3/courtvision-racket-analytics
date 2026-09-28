"""Model-guided court-keypoint refinement from painted line intersections."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import cv2
import numpy as np

from courtvision.calibration.homography import BASELINE_CORNER_NAMES
from courtvision.calibration.pixel_masks import near_white_pixel_mask

MINIMUM_AUTO_REFINED_LANDMARKS = 6

PADEL_HORIZONTAL_ENDPOINTS = {
    "far_baseline": (
        "far_left_baseline_corner",
        "far_right_baseline_corner",
    ),
    "near_baseline": (
        "near_left_baseline_corner",
        "near_right_baseline_corner",
    ),
    "far_service": (
        "far_left_service_intersection",
        "far_right_service_intersection",
    ),
    "near_service": (
        "near_left_service_intersection",
        "near_right_service_intersection",
    ),
}
PADEL_ENDPOINT_LANDMARKS = frozenset(
    landmark_name
    for landmark_names in PADEL_HORIZONTAL_ENDPOINTS.values()
    for landmark_name in landmark_names
)


@dataclass(frozen=True)
class KeypointRefinementSettings:
    """Thresholds for model-guided court-line fitting."""

    minimum_value: int = 170
    maximum_saturation: int = 80
    corridor_width_ratio: float = 0.045
    minimum_corridor_width_px: float = 18.0
    line_distance_px: float = 2.5
    maximum_angle_degrees: float = 10.0
    ransac_trials_per_line: int = 500
    minimum_span_ratio: float = 0.35
    maximum_intersection_shift_ratio: float = 1.35


@dataclass(frozen=True)
class FittedCourtLine:
    """One fitted image line and the evidence supporting it."""

    name: str
    points: tuple[float, float, float, float]
    support: float
    offset_px: float


@dataclass(frozen=True)
class KeypointRefinementResult:
    """Refined landmarks plus enough evidence for UI and Auto-mode gating."""

    landmarks: dict[str, tuple[float, float]]
    original_landmarks: dict[str, tuple[float, float]]
    refined_landmark_names: tuple[str, ...]
    refinement_distances_px: dict[str, float]
    fitted_lines: tuple[FittedCourtLine, ...]
    line_mask: np.ndarray

    @property
    def refined_count(self) -> int:
        return len(self.refined_landmark_names)

    @property
    def required_corners_refined(self) -> bool:
        refined = set(self.refined_landmark_names)
        return all(name in refined for name in BASELINE_CORNER_NAMES)

    @property
    def auto_ready(self) -> bool:
        return (
            self.required_corners_refined
            and self.refined_count >= MINIMUM_AUTO_REFINED_LANDMARKS
        )

    @property
    def mean_line_support(self) -> float:
        if not self.fitted_lines:
            return 0.0
        return float(np.mean([line.support for line in self.fitted_lines]))


def refine_keypoint_landmarks(
    frame: np.ndarray,
    landmarks: dict[str, tuple[float, float]],
    court_type: str,
    settings: KeypointRefinementSettings | None = None,
) -> KeypointRefinementResult:
    """Snap model landmarks to nearby intersections of painted court lines."""
    settings = settings or KeypointRefinementSettings()
    normalized_court_type = court_type.strip().lower()
    if normalized_court_type == "paddle":
        normalized_court_type = "padel"
    if normalized_court_type not in {"tennis", "padel"}:
        raise ValueError(f"Unsupported court type: {court_type}")

    mask = near_white_pixel_mask(
        frame,
        minimum_value=settings.minimum_value,
        maximum_saturation=settings.maximum_saturation,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    height, width = frame.shape[:2]
    corridor_width = max(
        settings.minimum_corridor_width_px,
        min(width, height) * settings.corridor_width_ratio,
    )
    line_landmarks, intersection_lines = _court_line_topology(
        landmarks,
        normalized_court_type,
    )
    fitted_lines: dict[str, FittedCourtLine] = {}
    rng = np.random.default_rng(41)
    for line_name, endpoint_names in line_landmarks.items():
        expected_points = [
            landmarks[name] for name in endpoint_names if name in landmarks
        ]
        if len(expected_points) < 2:
            continue
        fitted = _fit_model_guided_line(
            mask,
            np.asarray(expected_points, dtype=np.float64),
            line_name,
            corridor_width,
            settings,
            rng,
            extend_along_line=(
                normalized_court_type == "padel"
                and line_name in PADEL_HORIZONTAL_ENDPOINTS
            ),
        )
        if fitted is not None:
            fitted_lines[line_name] = fitted

    refined = dict(landmarks)
    refined_names: list[str] = []
    distances: dict[str, float] = {}
    if normalized_court_type == "padel":
        _refine_padel_horizontal_endpoints(
            fitted_lines,
            landmarks,
            refined,
            refined_names,
            distances,
            mask,
            settings,
            width,
            height,
            ("far_baseline", "near_baseline"),
        )
    maximum_shift = corridor_width * settings.maximum_intersection_shift_ratio
    for landmark_name, (
        first_line_name,
        second_line_name,
    ) in intersection_lines.items():
        if (
            landmark_name not in landmarks
            or landmark_name in refined_names
            or (
                normalized_court_type == "padel"
                and landmark_name in PADEL_ENDPOINT_LANDMARKS
            )
        ):
            continue
        first_line = fitted_lines.get(first_line_name)
        second_line = fitted_lines.get(second_line_name)
        if first_line is None or second_line is None:
            continue
        intersection = _line_intersection(first_line.points, second_line.points)
        original = np.asarray(landmarks[landmark_name], dtype=np.float64)
        distance = float(np.linalg.norm(intersection - original))
        if (
            not np.isfinite(intersection).all()
            or distance > maximum_shift
            or not (0 <= intersection[0] < width and 0 <= intersection[1] < height)
        ):
            continue
        refined[landmark_name] = (float(intersection[0]), float(intersection[1]))
        refined_names.append(landmark_name)
        distances[landmark_name] = distance

    if normalized_court_type == "padel":
        _refine_padel_horizontal_endpoints(
            fitted_lines,
            landmarks,
            refined,
            refined_names,
            distances,
            mask,
            settings,
            width,
            height,
            ("far_service", "near_service"),
            geometry_landmarks=refined,
        )

    if not _valid_refined_boundary(refined, refined_names):
        for name in BASELINE_CORNER_NAMES:
            refined[name] = landmarks[name]
            distances.pop(name, None)
        refined_names = [
            name for name in refined_names if name not in BASELINE_CORNER_NAMES
        ]

    return KeypointRefinementResult(
        landmarks=refined,
        original_landmarks=dict(landmarks),
        refined_landmark_names=tuple(refined_names),
        refinement_distances_px=distances,
        fitted_lines=tuple(fitted_lines.values()),
        line_mask=mask,
    )


def draw_keypoint_refinement(
    frame: np.ndarray,
    result: KeypointRefinementResult,
) -> np.ndarray:
    """Draw fitted lines, raw model points, and accepted intersections."""
    rendered = frame.copy()
    palette = (
        (255, 175, 0),
        (0, 210, 255),
        (255, 90, 170),
        (80, 220, 80),
    )
    for index, line in enumerate(result.fitted_lines):
        x1, y1, x2, y2 = (round(value) for value in line.points)
        color = palette[index % len(palette)]
        cv2.line(rendered, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            rendered,
            f"{line.name} {line.support:.0%}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    refined_names = set(result.refined_landmark_names)
    for name, original in result.original_landmarks.items():
        original_pixel = tuple(round(value) for value in original)
        cv2.circle(rendered, original_pixel, 5, (255, 0, 255), 2, cv2.LINE_AA)
        if name not in refined_names:
            continue
        refined_pixel = tuple(round(value) for value in result.landmarks[name])
        cv2.line(
            rendered,
            original_pixel,
            refined_pixel,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.drawMarker(
            rendered,
            refined_pixel,
            (0, 220, 0),
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )
    return rendered


def _court_line_topology(
    landmarks: dict[str, tuple[float, float]],
    court_type: str,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, str]]]:
    lines: dict[str, tuple[str, ...]] = {
        "far_baseline": (
            "far_left_baseline_corner",
            "far_right_baseline_corner",
        ),
        "near_baseline": (
            "near_left_baseline_corner",
            "near_right_baseline_corner",
        ),
        "left_boundary": (
            "far_left_baseline_corner",
            "near_left_baseline_corner",
        ),
        "right_boundary": (
            "far_right_baseline_corner",
            "near_right_baseline_corner",
        ),
        "far_service": (
            "far_left_service_intersection",
            "far_service_center",
            "far_right_service_intersection",
        ),
        "near_service": (
            "near_left_service_intersection",
            "near_service_center",
            "near_right_service_intersection",
        ),
        "service_center": ("far_service_center", "near_service_center"),
        "net": ("left_net_point", "net_center", "right_net_point"),
    }
    if court_type == "tennis":
        lines.update(
            {
                "left_service_side": (
                    "far_left_service_intersection",
                    "near_left_service_intersection",
                ),
                "right_service_side": (
                    "far_right_service_intersection",
                    "near_right_service_intersection",
                ),
            }
        )
        service_sides = ("left_service_side", "right_service_side")
    else:
        service_sides = ("left_boundary", "right_boundary")

    intersections = {
        "far_left_baseline_corner": ("far_baseline", "left_boundary"),
        "far_right_baseline_corner": ("far_baseline", "right_boundary"),
        "near_right_baseline_corner": ("near_baseline", "right_boundary"),
        "near_left_baseline_corner": ("near_baseline", "left_boundary"),
        "far_left_service_intersection": ("far_service", service_sides[0]),
        "far_right_service_intersection": ("far_service", service_sides[1]),
        "near_right_service_intersection": ("near_service", service_sides[1]),
        "near_left_service_intersection": ("near_service", service_sides[0]),
        "far_service_center": ("far_service", "service_center"),
        "near_service_center": ("near_service", "service_center"),
        "left_net_point": ("net", "left_boundary"),
        "right_net_point": ("net", "right_boundary"),
        "net_center": ("net", "service_center"),
    }
    # Missing endpoints are handled by the fitter; retaining the full topology
    # makes partial keypoint schemas work without per-model branches.
    return lines, intersections


def _fit_model_guided_line(
    mask: np.ndarray,
    expected_points: np.ndarray,
    line_name: str,
    corridor_width: float,
    settings: KeypointRefinementSettings,
    rng: np.random.Generator,
    extend_along_line: bool = False,
) -> FittedCourtLine | None:
    first = expected_points[0]
    second = expected_points[-1]
    expected_direction = second - first
    expected_length = float(np.linalg.norm(expected_direction))
    if expected_length < 20:
        return None
    expected_unit = expected_direction / expected_length
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    pixels = np.column_stack((xs, ys)).astype(np.float64)
    relative = pixels - first
    projections = relative @ expected_unit
    perpendicular_distances = np.abs(
        expected_unit[0] * relative[:, 1] - expected_unit[1] * relative[:, 0]
    )
    projection_extension = corridor_width
    if extend_along_line:
        projection_extension = max(
            projection_extension,
            mask.shape[1] * 0.12,
            expected_length * 0.5,
        )
    corridor = (
        (perpendicular_distances <= corridor_width)
        & (projections >= -projection_extension)
        & (projections <= expected_length + projection_extension)
    )
    candidates = pixels[corridor]
    if len(candidates) < 20:
        return None
    if len(candidates) > 8_000:
        candidates = candidates[rng.choice(len(candidates), 8_000, replace=False)]

    expected_angle = float(
        np.arctan2(expected_direction[1], expected_direction[0]) % pi
    )
    maximum_angle = np.deg2rad(settings.maximum_angle_degrees)
    minimum_inliers = max(20, round(expected_length * 0.08))
    best_inliers: np.ndarray | None = None
    best_score = 0.0
    for _ in range(settings.ransac_trials_per_line):
        sample = candidates[rng.choice(len(candidates), 2, replace=False)]
        direction = sample[1] - sample[0]
        length = float(np.linalg.norm(direction))
        if length < expected_length * 0.18:
            continue
        angle = float(np.arctan2(direction[1], direction[0]) % pi)
        if _angle_distance(angle, expected_angle) > maximum_angle:
            continue
        distances = _point_line_distances(candidates, sample[0], sample[1])
        inliers = distances <= settings.line_distance_px
        if int(inliers.sum()) < minimum_inliers:
            continue
        inlier_projections = (candidates[inliers] - first) @ expected_unit
        span = float(np.ptp(inlier_projections))
        span_ratio = span / expected_length
        if span_ratio < settings.minimum_span_ratio:
            continue
        offset = float(
            _point_line_distances(expected_points, sample[0], sample[1]).mean()
        )
        prior = np.exp(-offset / max(1.0, corridor_width * 0.75))
        score = float(inliers.sum()) * min(1.5, span_ratio) * prior
        if score > best_score:
            best_score = score
            best_inliers = inliers
    if best_inliers is None:
        return None

    inlier_points = candidates[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(
        inlier_points.astype(np.float32),
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)
    direction = np.asarray((vx, vy), dtype=np.float64)
    origin = np.asarray((x0, y0), dtype=np.float64)
    if np.dot(direction, expected_unit) < 0:
        direction *= -1
    if extend_along_line:
        inlier_points = _select_guided_line_run(
            inlier_points,
            origin,
            direction,
            expected_points,
            expected_length,
        )
        vx, vy, x0, y0 = cv2.fitLine(
            inlier_points.astype(np.float32),
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        ).reshape(-1)
        direction = np.asarray((vx, vy), dtype=np.float64)
        origin = np.asarray((x0, y0), dtype=np.float64)
        if np.dot(direction, expected_unit) < 0:
            direction *= -1
    inlier_projections = (inlier_points - origin) @ direction
    endpoint_percentiles = (0.25, 99.75) if extend_along_line else (1, 99)
    start, end = np.percentile(inlier_projections, endpoint_percentiles)
    fitted_first = origin + start * direction
    fitted_second = origin + end * direction
    span_ratio = float(end - start) / expected_length
    offset = float(
        _point_line_distances(expected_points, fitted_first, fitted_second).mean()
    )
    density = len(inlier_points) / max(1.0, expected_length * 0.25)
    support = float(min(1.0, density) * min(1.0, span_ratio / 0.7))
    return FittedCourtLine(
        name=line_name,
        points=tuple(float(value) for value in (*fitted_first, *fitted_second)),
        support=support,
        offset_px=offset,
    )


def _select_guided_line_run(
    inlier_points: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    expected_points: np.ndarray,
    expected_length: float,
) -> np.ndarray:
    """Keep the continuous painted run that best overlaps the model prior."""
    projections = (inlier_points - origin) @ direction
    expected_projections = (expected_points - origin) @ direction
    expected_start = float(expected_projections.min())
    expected_end = float(expected_projections.max())
    occupied = np.unique(np.rint(projections).astype(np.int32))
    if len(occupied) < 2:
        return inlier_points

    maximum_gap = max(8.0, min(36.0, expected_length * 0.055))
    split_indexes = np.flatnonzero(np.diff(occupied) > maximum_gap) + 1
    runs = np.split(occupied, split_indexes)

    def run_score(run: np.ndarray) -> tuple[float, float, float]:
        start = float(run[0])
        end = float(run[-1])
        overlap = max(
            0.0,
            min(end, expected_end) - max(start, expected_start),
        )
        center_distance = abs(
            (start + end) * 0.5 - (expected_start + expected_end) * 0.5
        )
        return overlap, -center_distance, end - start

    selected = max(runs, key=run_score)
    selected_start = float(selected[0]) - 1.0
    selected_end = float(selected[-1]) + 1.0
    selected_points = inlier_points[
        (projections >= selected_start) & (projections <= selected_end)
    ]
    return selected_points if len(selected_points) >= 20 else inlier_points


def _refine_padel_horizontal_endpoints(
    fitted_lines: dict[str, FittedCourtLine],
    original: dict[str, tuple[float, float]],
    refined: dict[str, tuple[float, float]],
    refined_names: list[str],
    distances: dict[str, float],
    line_mask: np.ndarray,
    settings: KeypointRefinementSettings,
    width: int,
    height: int,
    line_names: tuple[str, ...],
    geometry_landmarks: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Use painted horizontal endpoints when padel side lines are not visible."""
    maximum_shift = width * 0.16
    geometry_landmarks = geometry_landmarks or original
    for line_name in line_names:
        landmark_names = PADEL_HORIZONTAL_ENDPOINTS[line_name]
        fitted = fitted_lines.get(line_name)
        if fitted is None:
            continue
        endpoints = None
        if line_name in {"far_service", "near_service"}:
            expected_points = _padel_service_geometry_points(
                geometry_landmarks,
                line_name,
            )
            if expected_points is not None:
                endpoints = _snap_line_evidence_to_expected_endpoints(
                    line_mask,
                    fitted,
                    expected_points,
                    settings.line_distance_px * 1.5,
                )
        if endpoints is None:
            endpoints = sorted(
                (
                    np.asarray(fitted.points[:2], dtype=np.float64),
                    np.asarray(fitted.points[2:], dtype=np.float64),
                ),
                key=lambda point: point[0],
            )
        for endpoint_index, (landmark_name, endpoint) in enumerate(
            zip(landmark_names, endpoints, strict=True)
        ):
            if landmark_name not in original:
                continue
            original_point = np.asarray(original[landmark_name], dtype=np.float64)
            contracts_model_span = (
                endpoint_index == 0 and endpoint[0] > original_point[0] + 3.0
            ) or (
                endpoint_index == 1 and endpoint[0] < original_point[0] - 3.0
            )
            if contracts_model_span:
                continue
            distance = float(
                np.linalg.norm(endpoint - original_point)
            )
            if (
                distance > maximum_shift
                or not np.isfinite(endpoint).all()
                or not (0 <= endpoint[0] < width and 0 <= endpoint[1] < height)
            ):
                continue
            refined[landmark_name] = (float(endpoint[0]), float(endpoint[1]))
            refined_names.append(landmark_name)
            distances[landmark_name] = distance


def _padel_service_geometry_points(
    landmarks: dict[str, tuple[float, float]],
    line_name: str,
) -> np.ndarray | None:
    if not all(name in landmarks for name in BASELINE_CORNER_NAMES):
        return None
    template = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float32,
    )
    image = np.asarray(
        [landmarks[name] for name in BASELINE_CORNER_NAMES],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(template, image)
    service_y = 0.15 if line_name == "far_service" else 0.85
    return cv2.perspectiveTransform(
        np.asarray(((0.0, service_y), (1.0, service_y)), dtype=np.float32).reshape(
            -1, 1, 2
        ),
        transform,
    ).reshape(-1, 2).astype(np.float64)


def _snap_line_evidence_to_expected_endpoints(
    line_mask: np.ndarray,
    fitted: FittedCourtLine,
    expected_points: np.ndarray,
    maximum_distance_px: float,
) -> list[np.ndarray] | None:
    """Recover a painted line's robust full span after RANSAC fits its axis."""
    first = np.asarray(fitted.points[:2], dtype=np.float64)
    second = np.asarray(fitted.points[2:], dtype=np.float64)
    direction = second - first
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None
    unit = direction / length
    ys, xs = np.nonzero(line_mask)
    if len(xs) < 30:
        return None
    pixels = np.column_stack((xs, ys)).astype(np.float64)
    projections = (pixels - first) @ unit
    expected_projections = (expected_points - first) @ unit
    in_corridor = _point_line_distances(
        pixels,
        first,
        second,
    ) <= maximum_distance_px
    evidence_projections = projections[in_corridor]
    if len(evidence_projections) < 20:
        return None
    expected_span = float(np.ptp(expected_projections))
    search_radius = max(line_mask.shape[1] * 0.08, expected_span * 0.12)
    selected_projections: list[float] = []
    for expected_projection in expected_projections:
        local = evidence_projections[
            np.abs(evidence_projections - expected_projection) <= search_radius
        ]
        if not len(local):
            return None
        selected_projections.append(
            float(local[np.argmin(np.abs(local - expected_projection))])
        )
    if abs(selected_projections[1] - selected_projections[0]) < expected_span * 0.55:
        return None
    endpoints = [first + projection * unit for projection in selected_projections]
    return sorted(endpoints, key=lambda point: point[0])


def _point_line_distances(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    direction = second - first
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return np.full(len(points), np.inf)
    relative = points - first
    return (
        np.abs(direction[0] * relative[:, 1] - direction[1] * relative[:, 0])
        / length
    )


def _line_intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> np.ndarray:
    x1, y1, x2, y2 = first
    x3, y3, x4, y4 = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-8:
        return np.asarray((np.nan, np.nan), dtype=np.float64)
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    x = (
        determinant_first * (x3 - x4)
        - (x1 - x2) * determinant_second
    ) / denominator
    y = (
        determinant_first * (y3 - y4)
        - (y1 - y2) * determinant_second
    ) / denominator
    return np.asarray((x, y), dtype=np.float64)


def _angle_distance(first: float, second: float) -> float:
    return abs((first - second + pi / 2) % pi - pi / 2)


def _valid_refined_boundary(
    landmarks: dict[str, tuple[float, float]],
    refined_names: list[str],
) -> bool:
    if not all(name in landmarks for name in BASELINE_CORNER_NAMES):
        return False
    if not all(name in refined_names for name in BASELINE_CORNER_NAMES):
        return True
    quad = np.asarray(
        [landmarks[name] for name in BASELINE_CORNER_NAMES],
        dtype=np.float32,
    )
    return bool(
        cv2.isContourConvex(quad.reshape(-1, 1, 2))
        and cv2.contourArea(quad.reshape(-1, 1, 2)) > 100
    )
