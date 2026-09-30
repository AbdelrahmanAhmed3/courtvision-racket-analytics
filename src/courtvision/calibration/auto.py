"""Automatic court-boundary proposals from near-white pixels and RANSAC lines.

Experimental: works on the near side of the court but is unreliable on the far
side. The app uses it in its Assisted and Auto calibration modes; Manual
remains the reliable path. See docs/roadmap.md (v4.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import pi

import cv2
import numpy as np

from courtvision.calibration.homography import BASELINE_CORNER_NAMES
from courtvision.calibration.landmarks import (
    default_landmark_names,
    landmark_by_name,
)
from courtvision.calibration.pixel_masks import near_white_pixel_mask

DEFAULT_MAXIMUM_LINE_COUNT = 20
DEFAULT_RANSAC_TRIALS_PER_LINE = 700
RANSAC_HYPOTHESIS_TRIALS = 1_200
MIN_CONFIDENCE = 0.70
MIN_SINGLES_ANCHOR_CONFIDENCE = 0.55
MIN_TEMPLATE_LINE_SUPPORT = 0.24
MIN_TENNIS_SINGLES_SUPPORT = 0.18


@dataclass(frozen=True)
class AutoCalibrationSettings:
    """Tunable near-white, RANSAC, and search-region parameters for Auto mode."""

    minimum_value: int = 170
    maximum_saturation: int = 80
    min_line_length_ratio: float = 1 / 12
    line_distance_px: float = 2.5
    ransac_trials_per_line: int = DEFAULT_RANSAC_TRIALS_PER_LINE
    maximum_line_count: int = DEFAULT_MAXIMUM_LINE_COUNT
    crop_top_ratio: float = 0.0
    crop_bottom_ratio: float = 1.0
    crop_left_ratio: float = 0.0
    crop_right_ratio: float = 1.0


@dataclass(frozen=True)
class AutoCalibrationProposal:
    """A confidence-scored doubles-court proposal from doubles or singles paint."""

    points: tuple[tuple[float, float], ...]
    confidence: float
    line_count: int
    line_support: float
    geometry_score: float
    boundary_anchor: str
    landmark_points: tuple[tuple[str, tuple[float, float]], ...] = ()

    @property
    def landmarks(self) -> dict[str, tuple[float, float]]:
        landmarks = dict(zip(BASELINE_CORNER_NAMES, self.points, strict=True))
        landmarks.update(self.landmark_points)
        return landmarks


@dataclass(frozen=True)
class AutoCalibrationDebug:
    """Intermediate evidence produced by automatic court calibration."""

    proposal: AutoCalibrationProposal | None
    line_mask: np.ndarray
    lines: tuple[np.ndarray, ...]
    search_bounds: tuple[int, int, int, int]


def detect_auto_calibration(
    frame: np.ndarray,
    court_type: str = "tennis",
    minimum_confidence: float = MIN_CONFIDENCE,
    settings: AutoCalibrationSettings | None = None,
) -> AutoCalibrationProposal | None:
    """Detect a four-corner court proposal from fitted pixel-consensus lines."""
    return analyze_auto_calibration(
        frame,
        court_type,
        minimum_confidence,
        settings,
    ).proposal


def analyze_auto_calibration(
    frame: np.ndarray,
    court_type: str = "tennis",
    minimum_confidence: float = MIN_CONFIDENCE,
    settings: AutoCalibrationSettings | None = None,
) -> AutoCalibrationDebug:
    """Run Auto calibration and retain the mask and fitted lines for inspection."""
    settings = settings or AutoCalibrationSettings()
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _search_bounds(settings, width, height)
    crop = frame[y1:y2, x1:x2]
    crop_mask = _court_line_mask(crop, settings)
    line_mask = np.zeros((height, width), dtype=np.uint8)
    line_mask[y1:y2, x1:x2] = crop_mask
    lines = _ransac_lines(crop_mask, x2 - x1, settings)
    lines = [line + np.asarray((x1, y1, x1, y1)) for line in lines]
    if len(lines) < 4:
        return AutoCalibrationDebug(None, line_mask, tuple(lines), (x1, y1, x2, y2))

    distance_to_line = cv2.distanceTransform(255 - line_mask, cv2.DIST_L2, 3)
    best: tuple[float, np.ndarray, float, float, str] | None = None
    rng = np.random.default_rng(19)
    for indexes in _sample_line_sets(lines, rng):
        selected = [lines[index] for index in indexes]
        for baselines, sidelines in _court_boundary_pairings(selected):
            quad = _quadrilateral_from_lines(baselines, sidelines, width, height)
            if quad is None:
                continue
            candidates = [("doubles", quad, quad)]
            if court_type.strip().lower() == "tennis":
                candidates.append(("singles", quad, _singles_to_doubles_quad(quad)))
            for boundary_anchor, observed_quad, doubles_quad in candidates:
                line_support, geometry_score = _score_quadrilateral(
                    observed_quad,
                    doubles_quad,
                    distance_to_line,
                    width,
                    height,
                    court_type,
                    boundary_anchor,
                )
                score = 0.72 * line_support + 0.28 * geometry_score
                # Prefer direct doubles evidence when both interpretations fit.
                if boundary_anchor == "singles":
                    score *= 0.98
                if best is None or score > best[0]:
                    best = (
                        score,
                        doubles_quad,
                        line_support,
                        geometry_score,
                        boundary_anchor,
                    )

    proposal = None
    if best is not None:
        score, quad, line_support, geometry_score, boundary_anchor = best
        required_confidence = minimum_confidence
        if boundary_anchor == "singles":
            required_confidence = min(minimum_confidence, MIN_SINGLES_ANCHOR_CONFIDENCE)
        if score >= required_confidence:
            proposal = AutoCalibrationProposal(
                points=tuple((float(x), float(y)) for x, y in quad),
                confidence=float(score),
                line_count=len(lines),
                line_support=float(line_support),
                geometry_score=float(geometry_score),
                boundary_anchor=boundary_anchor,
                landmark_points=tuple(
                    _observed_interior_landmarks(
                        quad,
                        lines,
                        court_type,
                        width,
                        height,
                    ).items()
                ),
            )
    return AutoCalibrationDebug(proposal, line_mask, tuple(lines), (x1, y1, x2, y2))


def draw_auto_proposal(
    frame: np.ndarray,
    proposal: AutoCalibrationProposal,
) -> np.ndarray:
    """Draw the proposed outer court and its confidence on a calibration frame."""
    image = frame.copy()
    points = np.asarray(proposal.points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [points], True, (0, 220, 255), 3, cv2.LINE_AA)
    for index, (x, y) in enumerate(proposal.points, start=1):
        cv2.circle(image, (round(x), round(y)), 7, (0, 220, 255), -1)
        cv2.putText(
            image,
            str(index),
            (round(x) + 10, round(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    boundary_anchor = getattr(proposal, "boundary_anchor", "doubles")
    label = (
        f"AUTO CONFIDENCE {proposal.confidence:.0%} ({boundary_anchor.upper()} ANCHOR)"
    )
    cv2.putText(
        image,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def draw_ransac_debug(
    frame: np.ndarray,
    debug: AutoCalibrationDebug,
) -> np.ndarray:
    """Render the search region and every line fitted from near-white pixels."""
    image = frame.copy()
    x1, y1, x2, y2 = debug.search_bounds
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 165, 255), 2)
    for index, line in enumerate(debug.lines, start=1):
        x_start, y_start, x_end, y_end = (round(value) for value in line)
        color = (255, 170, 0) if index <= 20 else (130, 90, 0)
        cv2.line(image, (x_start, y_start), (x_end, y_end), color, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index),
            (x_start, y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def _court_line_mask(
    frame: np.ndarray,
    settings: AutoCalibrationSettings,
) -> np.ndarray:
    near_white = near_white_pixel_mask(
        frame,
        minimum_value=settings.minimum_value,
        maximum_saturation=settings.maximum_saturation,
    )
    return cv2.morphologyEx(
        near_white,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )


def _ransac_lines(
    line_mask: np.ndarray,
    width: int,
    settings: AutoCalibrationSettings,
) -> list[np.ndarray]:
    """Fit full lines directly to near-white pixels with repeated RANSAC.

    A player may erase the middle of a baseline. Because a RANSAC hypothesis is
    supported by pixels on both visible sides, the fitted line still crosses the
    gap; it is not restricted to one continuous segment as HoughLinesP is.
    """
    ys, xs = np.nonzero(line_mask)
    if len(xs) < 2:
        return []
    points = np.column_stack((xs, ys)).astype(np.float64)
    rng = np.random.default_rng(23)
    minimum_length = max(25.0, width * settings.min_line_length_ratio)
    minimum_inliers = max(60, round(minimum_length / 3))
    lines: list[np.ndarray] = []
    remaining = points

    while (
        len(lines) < settings.maximum_line_count and len(remaining) >= minimum_inliers
    ):
        sample = remaining
        if len(sample) > 12_000:
            indexes = rng.choice(len(sample), size=12_000, replace=False)
            sample = sample[indexes]

        best_inliers: np.ndarray | None = None
        best_score = 0.0
        for _ in range(settings.ransac_trials_per_line):
            first, second = sample[rng.choice(len(sample), size=2, replace=False)]
            direction = second - first
            direction_length = np.linalg.norm(direction)
            if direction_length < minimum_length * 0.35:
                continue
            distances = (
                np.abs(
                    direction[0] * (sample[:, 1] - first[1])
                    - direction[1] * (sample[:, 0] - first[0])
                )
                / direction_length
            )
            inliers = distances <= settings.line_distance_px
            if inliers.sum() < minimum_inliers:
                continue
            projections = (sample[inliers] - first) @ (direction / direction_length)
            span = float(projections.max() - projections.min())
            score = inliers.sum() * min(2.0, span / minimum_length)
            if span >= minimum_length and score > best_score:
                best_inliers = inliers
                best_score = score

        if best_inliers is None:
            break
        line = _refine_ransac_line(sample[best_inliers])
        inliers = _line_inliers(remaining, line, settings.line_distance_px)
        if inliers.sum() < minimum_inliers or _line_length(line) < minimum_length:
            break
        lines.append(line)
        # Remove the full paint stripe, not only its centre. Otherwise RANSAC
        # repeatedly rediscovers the two edges of one thick court marking.
        paint_inliers = _line_inliers(
            remaining,
            line,
            max(6.0, settings.line_distance_px * 2.5),
        )
        remaining = remaining[~paint_inliers]
    return lines


def _refine_ransac_line(points: np.ndarray) -> np.ndarray:
    """Refit a consensus line and return its visible extent in the point cloud."""
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).reshape(-1)
    direction = np.asarray((vx, vy), dtype=np.float64)
    origin = np.asarray((x0, y0), dtype=np.float64)
    projections = (points - origin) @ direction
    start, end = np.percentile(projections, (1, 99))
    first = origin + start * direction
    second = origin + end * direction
    return np.asarray((*first, *second), dtype=np.float64)


def _line_inliers(
    points: np.ndarray,
    line: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    first = line[:2]
    second = line[2:]
    direction = second - first
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.zeros(len(points), dtype=bool)
    distances = (
        np.abs(
            direction[0] * (points[:, 1] - first[1])
            - direction[1] * (points[:, 0] - first[0])
        )
        / length
    )
    projections = (points - first) @ (direction / length)
    return (
        (distances <= distance_threshold)
        & (projections >= -distance_threshold)
        & (projections <= length + distance_threshold)
    )


def _search_bounds(
    settings: AutoCalibrationSettings,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = round(width * settings.crop_left_ratio)
    right = round(width * settings.crop_right_ratio)
    top = round(height * settings.crop_top_ratio)
    bottom = round(height * settings.crop_bottom_ratio)
    if right - left < 50 or bottom - top < 50:
        raise ValueError("Auto-calibration search region is too small")
    return left, top, right, bottom


def _sample_line_sets(
    lines: list[np.ndarray],
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    """Build court hypotheses from line intersections instead of line length.

    A valid court has two approximately parallel baseline candidates. For every
    RANSAC trial we select such a pair, then select two lines that cross it. The
    four resulting intersections are a candidate court. This gives the template
    scorer useful candidates even when a crowd line is longer than a court line.
    """
    parallel_pairs = [
        (first, second)
        for first, second in combinations(range(len(lines)), 2)
        if _angle_distance(_line_angle(lines[first]), _line_angle(lines[second]))
        <= 0.18
    ]
    if not parallel_pairs:
        return []

    hypotheses: set[tuple[int, int, int, int]] = set()
    attempts = 0
    maximum_attempts = RANSAC_HYPOTHESIS_TRIALS * 8
    while len(hypotheses) < RANSAC_HYPOTHESIS_TRIALS and attempts < maximum_attempts:
        attempts += 1
        first, second = parallel_pairs[rng.integers(len(parallel_pairs))]
        crossing = [
            index
            for index, line in enumerate(lines)
            if index not in (first, second)
            and _angle_distance(_line_angle(lines[first]), _line_angle(line)) >= 0.20
        ]
        if len(crossing) < 2:
            continue
        third, fourth = rng.choice(crossing, size=2, replace=False).tolist()
        hypotheses.add((first, second, third, fourth))
    return list(hypotheses)


def _court_boundary_pairings(
    lines: list[np.ndarray],
) -> list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    pairings = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))
    result = []
    for first_a, first_b, second_a, second_b in pairings:
        first = (lines[first_a], lines[first_b])
        second = (lines[second_a], lines[second_b])
        if _angle_distance(_line_angle(first[0]), _line_angle(first[1])) > 0.18:
            continue
        if _angle_distance(_line_angle(first[0]), _line_angle(second[0])) < 0.20:
            continue
        if _angle_distance(_line_angle(first[0]), _line_angle(second[1])) < 0.20:
            continue
        result.append((first, second))
    return result


def _quadrilateral_from_lines(
    baselines: tuple[np.ndarray, np.ndarray],
    sidelines: tuple[np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray | None:
    intersections = np.asarray(
        [
            [_line_intersection(baseline, sideline) for sideline in sidelines]
            for baseline in baselines
        ],
        dtype=np.float64,
    )
    if not np.isfinite(intersections).all():
        return None
    mean_y = intersections[:, :, 1].mean(axis=1)
    top_index, bottom_index = np.argsort(mean_y)
    top = intersections[top_index]
    bottom = intersections[bottom_index]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    quad = np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    margin_x = width * 0.2
    margin_y = height * 0.2
    if (
        (quad[:, 0] < -margin_x).any()
        or (quad[:, 0] > width + margin_x).any()
        or (quad[:, 1] < -margin_y).any()
        or (quad[:, 1] > height + margin_y).any()
    ):
        return None
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return None
    return quad


def _singles_to_doubles_quad(singles_quad: np.ndarray) -> np.ndarray:
    """Extrapolate outer doubles corners from an observed tennis singles quad."""
    singles_margin = (10.97 - 8.23) / (2 * 10.97)
    singles_template = np.asarray(
        [
            [singles_margin, 0.0],
            [1.0 - singles_margin, 0.0],
            [1.0 - singles_margin, 1.0],
            [singles_margin, 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(singles_template, singles_quad)
    doubles_template = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    return cv2.perspectiveTransform(
        doubles_template.reshape(-1, 1, 2), transform
    ).reshape(-1, 2)


def _observed_interior_landmarks(
    doubles_quad: np.ndarray,
    lines: list[np.ndarray],
    court_type: str,
    width: int,
    height: int,
) -> dict[str, tuple[float, float]]:
    """Return service intersections supported by independently fitted lines.

    The outer corners establish a provisional court transform. We use it only
    to ask which RANSAC line should be the relevant sideline/service line; the
    returned point is the intersection of those fitted image lines, not a
    template point projected from the four corners.
    """
    template = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(template, doubles_quad.astype(np.float32))
    landmarks: dict[str, tuple[float, float]] = {}
    for name in default_landmark_names(court_type)[4:]:
        definition = landmark_by_name(court_type, name)
        template_x, template_y = definition.template
        vertical = _match_template_line(
            ((template_x, 0.0), (template_x, 1.0)), transform, lines
        )
        horizontal = _match_template_line(
            ((0.0, template_y), (1.0, template_y)), transform, lines
        )
        if vertical is None or horizontal is None:
            continue
        point = _line_intersection(vertical, horizontal)
        expected = cv2.perspectiveTransform(
            np.asarray([[template_x, template_y]], dtype=np.float32).reshape(-1, 1, 2),
            transform,
        ).reshape(2)
        if (
            not np.isfinite(point).all()
            or np.linalg.norm(np.asarray(point) - expected) > max(15.0, width * 0.03)
            or not (0 <= point[0] < width and 0 <= point[1] < height)
        ):
            continue
        landmarks[name] = (float(point[0]), float(point[1]))
    return landmarks


def _match_template_line(
    template_line: tuple[tuple[float, float], tuple[float, float]],
    transform: np.ndarray,
    lines: list[np.ndarray],
) -> np.ndarray | None:
    projected = cv2.perspectiveTransform(
        np.asarray(template_line, dtype=np.float32).reshape(-1, 1, 2), transform
    ).reshape(-1, 2)
    expected_direction = projected[1] - projected[0]
    expected_length = np.linalg.norm(expected_direction)
    if expected_length < 1e-6:
        return None
    expected_angle = float(
        np.arctan2(expected_direction[1], expected_direction[0]) % pi
    )
    samples = _sample_edge(projected[0], projected[1])
    best_line: np.ndarray | None = None
    best_score = float("inf")
    for line in lines:
        if _angle_distance(expected_angle, _line_angle(line)) > 0.13:
            continue
        distances = _distance_to_infinite_line(samples, line)
        mean_distance = float(distances.mean())
        if mean_distance > 6.0:
            continue
        overlap = _line_overlap_ratio(projected, line)
        if overlap < 0.25:
            continue
        score = mean_distance + (1.0 - overlap) * 4.0
        if score < best_score:
            best_line = line
            best_score = score
    return best_line


def _distance_to_infinite_line(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    first = line[:2]
    second = line[2:]
    direction = second - first
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.full(len(points), np.inf)
    return (
        np.abs(
            direction[0] * (points[:, 1] - first[1])
            - direction[1] * (points[:, 0] - first[0])
        )
        / length
    )


def _line_overlap_ratio(expected: np.ndarray, observed: np.ndarray) -> float:
    direction = expected[1] - expected[0]
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return 0.0
    unit_direction = direction / length
    observed_projection = (observed.reshape(2, 2) - expected[0]) @ unit_direction
    overlap_start = max(0.0, float(observed_projection.min()))
    overlap_end = min(length, float(observed_projection.max()))
    return max(0.0, overlap_end - overlap_start) / length


def _score_quadrilateral(
    observed_quad: np.ndarray,
    doubles_quad: np.ndarray,
    distance_to_line: np.ndarray,
    width: int,
    height: int,
    court_type: str,
    boundary_anchor: str,
) -> tuple[float, float]:
    area = abs(cv2.contourArea(observed_quad.reshape(-1, 1, 2)))
    area_ratio = area / (width * height)
    top_width = np.linalg.norm(observed_quad[1] - observed_quad[0])
    bottom_width = np.linalg.norm(observed_quad[2] - observed_quad[3])
    height_left = np.linalg.norm(observed_quad[3] - observed_quad[0])
    height_right = np.linalg.norm(observed_quad[2] - observed_quad[1])
    # Short inner boxes and service boxes often have strong line support too.
    # A usable full-court proposal needs a meaningful fraction of the frame.
    if (
        area_ratio < 0.16
        or min(top_width, bottom_width, height_left, height_right) < 40
    ):
        return 0.0, 0.0
    samples = np.concatenate(
        [
            _sample_edge(observed_quad[index], observed_quad[(index + 1) % 4])
            for index in range(4)
        ]
    )
    xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, height - 1)
    boundary_support = float(np.exp(-distance_to_line[ys, xs] / 4.0).mean())
    template_support = _template_line_support(
        doubles_quad,
        distance_to_line,
        width,
        height,
        court_type,
        boundary_anchor,
    )
    # A false trapezoid in the crowd can have convincing outer edges. The net,
    # service, and singles lines are the evidence that makes it a court.
    line_support = 0.20 * boundary_support + 0.80 * template_support
    shape_score = min(1.0, area_ratio / 0.42)
    width_score = min(top_width, bottom_width) / max(top_width, bottom_width)
    height_score = min(height_left, height_right) / max(height_left, height_right)
    geometry_score = float(0.5 * shape_score + 0.25 * width_score + 0.25 * height_score)
    return line_support, geometry_score


def _template_line_support(
    quad: np.ndarray,
    distance_to_line: np.ndarray,
    width: int,
    height: int,
    court_type: str,
    boundary_anchor: str,
) -> float:
    template = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(template, quad.astype(np.float32))
    normalized_court_type = court_type.strip().lower()
    lines = _court_template_lines(court_type)
    if boundary_anchor == "singles":
        required_indexes = set(range(4, len(lines)))
        minimum_required_lines = 4
    else:
        required_indexes = set(range(8))
        minimum_required_lines = 6
    supports = []
    for index, (first, second) in enumerate(lines):
        projected = cv2.perspectiveTransform(
            np.asarray([first, second], dtype=np.float32).reshape(-1, 1, 2),
            transform,
        ).reshape(-1, 2)
        points = _sample_edge(projected[0], projected[1])
        # Clamping out-of-frame points would accidentally score image borders
        # as court lines. Auto calibration requires a complete visible court.
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] >= height).any()
        ):
            if index in required_indexes:
                return 0.0
            supports.append(0.0)
            continue
        xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
        supports.append(float(np.exp(-distance_to_line[ys, xs] / 4.0).mean()))

    # Require broadly distributed evidence. A crowd trapezoid may explain one
    # or two template lines, but it should not pass without court-line support
    # across the entire template.
    required_supports = [supports[index] for index in sorted(required_indexes)]
    if (
        sum(support >= MIN_TEMPLATE_LINE_SUPPORT for support in required_supports)
        < minimum_required_lines
    ):
        return 0.0
    if (
        boundary_anchor == "doubles"
        and normalized_court_type == "tennis"
        and max(supports[8:]) < (MIN_TENNIS_SINGLES_SUPPORT)
    ):
        # If the proposed boundary is actually the singles court, the template's
        # predicted inner singles sidelines have no matching paint. Requiring one
        # prevents a well-detected singles rectangle from becoming the doubles
        # homography boundary.
        return 0.0
    return float(
        0.35 * np.mean(required_supports) + 0.65 * np.percentile(required_supports, 25)
    )


def _court_template_lines(
    court_type: str,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Return court markings in a template bounded by doubles sidelines.

    The four template corners are always the outer doubles-court corners. For
    tennis, singles sidelines are internal validation markings only and must
    never redefine the homography boundary.
    """
    normalized = court_type.strip().lower()
    service_y = 5.485 / 23.77 if normalized == "tennis" else 0.15
    lines = [
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
        ((1, 1), (0, 1)),
        ((0, 1), (0, 0)),
        ((0, 0.5), (1, 0.5)),
        ((0, service_y), (1, service_y)),
        ((0, 1 - service_y), (1, 1 - service_y)),
        ((0.5, service_y), (0.5, 1 - service_y)),
    ]
    if normalized == "tennis":
        singles_margin = (10.97 - 8.23) / (2 * 10.97)
        lines.extend(
            [
                ((singles_margin, 0), (singles_margin, 1)),
                ((1 - singles_margin, 0), (1 - singles_margin, 1)),
            ]
        )
    return tuple(lines)


def _sample_edge(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, 60, dtype=np.float32)[:, None]
    return first + fractions * (second - first)


def _line_angle(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line
    return float(np.arctan2(y2 - y1, x2 - x1) % pi)


def _angle_distance(first: float, second: float) -> float:
    return abs((first - second + pi / 2) % pi - pi / 2)


def _line_length(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line
    return float(np.hypot(x2 - x1, y2 - y1))


def _line_intersection(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = first
    x3, y3, x4, y4 = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return float("nan"), float("nan")
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    return (
        (determinant_first * (x3 - x4) - (x1 - x2) * determinant_second) / denominator,
        (determinant_first * (y3 - y4) - (y1 - y2) * determinant_second) / denominator,
    )
