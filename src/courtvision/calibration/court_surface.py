"""Model-seeded, full-frame court-surface color diagnostics.

Experimental, padel only: needs a model's corner proposal as a seed and assumes
the baselines are roughly horizontal in the image. See docs/roadmap.md (v4.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from courtvision.calibration.homography import BASELINE_CORNER_NAMES

SAMPLE_TEMPLATE_POINTS = np.asarray(
    (
        (0.20, 0.30),
        (0.50, 0.30),
        (0.80, 0.30),
        (0.20, 0.70),
        (0.50, 0.70),
        (0.80, 0.70),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class CourtSurfaceSettings:
    """Controls for court-color sampling and full-frame similarity."""

    color_distance_threshold: float = 24.0
    lightness_weight: float = 0.20
    sample_radius_ratio: float = 0.012
    minimum_sample_radius_px: int = 4


@dataclass(frozen=True)
class CourtSurfaceAnalysis:
    """Intermediate color evidence retained for inspection in the UI."""

    sample_points: tuple[tuple[float, float], ...]
    sample_radius_px: int
    court_lab: tuple[float, float, float]
    court_bgr: tuple[int, int, int]
    color_distance: np.ndarray
    surface_mask: np.ndarray

    @property
    def court_rgb_hex(self) -> str:
        blue, green, red = self.court_bgr
        return f"#{red:02X}{green:02X}{blue:02X}"

    @property
    def coverage_ratio(self) -> float:
        return float(np.count_nonzero(self.surface_mask) / self.surface_mask.size)


@dataclass(frozen=True)
class BoundaryScanSettings:
    """Controls for inside-out scans from the model court into the color mask."""

    scans_per_boundary: int = 24
    maximum_gap_px: int = 6
    anchor_search_radius_px: int = 12
    minimum_scan_length_px: float = 20.0


@dataclass(frozen=True)
class BoundaryScan:
    """One accepted path from a safe court interior point to a mask boundary."""

    boundary: str
    anchor: tuple[float, float]
    endpoint: tuple[float, float]


@dataclass(frozen=True)
class SurfaceBoundaryCandidates:
    """Raw boundary evidence shown before any line fitting or point movement."""

    scans: tuple[BoundaryScan, ...]

    def count(self, boundary: str) -> int:
        return sum(scan.boundary == boundary for scan in self.scans)

    def endpoints(self, boundary: str) -> tuple[tuple[float, float], ...]:
        return tuple(
            scan.endpoint for scan in self.scans if scan.boundary == boundary
        )


@dataclass(frozen=True)
class SurfaceBoundarySettings:
    """Controls for recovering the two outer ground-level court spans."""

    vertical_search_ratio: float = 0.18
    minimum_central_support: float = 0.65
    sustained_rows: int = 4
    interior_sample_depth_px: int = 12
    horizontal_close_px: int = 15
    endpoint_percentile: float = 20.0
    minimum_frame_margin_ratio: float = 0.01
    minimum_transition_gap_rows: int = 2


@dataclass(frozen=True)
class SurfaceBoundaryRefinement:
    """Outer padel corners recovered from the full-frame court-color mask."""

    landmarks: dict[str, tuple[float, float]]
    original_landmarks: dict[str, tuple[float, float]]
    refined_landmark_names: tuple[str, ...]
    far_row: int
    near_row: int
    far_span: tuple[float, float]
    near_span: tuple[float, float]
    confidence: float

    @property
    def mean_shift_px(self) -> float:
        shifts = [
            np.linalg.norm(
                np.asarray(self.landmarks[name])
                - np.asarray(self.original_landmarks[name])
            )
            for name in self.refined_landmark_names
        ]
        return float(np.mean(shifts)) if shifts else 0.0


@dataclass(frozen=True)
class SurfaceThresholdStability:
    """How much the recovered corners move across nearby color tolerances."""

    tested_thresholds: tuple[float, ...]
    successful_thresholds: tuple[float, ...]
    maximum_corner_change_px: float
    stable: bool


def analyze_court_surface_color(
    frame: np.ndarray,
    landmarks: dict[str, tuple[float, float]],
    settings: CourtSurfaceSettings | None = None,
) -> CourtSurfaceAnalysis:
    """Sample model-guided interior colors and classify the complete frame."""
    settings = settings or CourtSurfaceSettings()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image")
    if settings.color_distance_threshold <= 0:
        raise ValueError("color_distance_threshold must be positive")
    if not 0 <= settings.lightness_weight <= 1:
        raise ValueError("lightness_weight must be between zero and one")
    if not all(name in landmarks for name in BASELINE_CORNER_NAMES):
        raise ValueError("four baseline corners are required to sample court color")

    height, width = frame.shape[:2]
    template_corners = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float32,
    )
    image_corners = np.asarray(
        [landmarks[name] for name in BASELINE_CORNER_NAMES],
        dtype=np.float32,
    )
    template_to_image = cv2.getPerspectiveTransform(
        template_corners,
        image_corners,
    )
    sample_points = cv2.perspectiveTransform(
        SAMPLE_TEMPLATE_POINTS.reshape(-1, 1, 2),
        template_to_image,
    ).reshape(-1, 2)
    sample_radius = max(
        settings.minimum_sample_radius_px,
        round(min(width, height) * settings.sample_radius_ratio),
    )

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    patch_medians = []
    for point in sample_points:
        patch = _sample_patch(lab, point, sample_radius)
        if len(patch):
            patch_medians.append(np.median(patch, axis=0))
    if not patch_medians:
        raise ValueError("model court lies outside the frame")
    court_lab_array = np.median(np.asarray(patch_medians), axis=0)

    delta = lab - court_lab_array
    delta[..., 0] *= settings.lightness_weight
    distance = np.linalg.norm(delta, axis=2)
    surface_mask = np.where(
        distance <= settings.color_distance_threshold,
        255,
        0,
    ).astype(np.uint8)
    court_bgr = cv2.cvtColor(
        np.rint(court_lab_array).clip(0, 255).astype(np.uint8).reshape(1, 1, 3),
        cv2.COLOR_LAB2BGR,
    ).reshape(3)
    return CourtSurfaceAnalysis(
        sample_points=tuple(
            (float(point[0]), float(point[1])) for point in sample_points
        ),
        sample_radius_px=sample_radius,
        court_lab=tuple(float(value) for value in court_lab_array),
        court_bgr=tuple(int(value) for value in court_bgr),
        color_distance=distance,
        surface_mask=surface_mask,
    )


def draw_court_color_samples(
    frame: np.ndarray,
    analysis: CourtSurfaceAnalysis,
) -> np.ndarray:
    """Mark the interior patches used to estimate court color."""
    rendered = frame.copy()
    for index, point in enumerate(analysis.sample_points, start=1):
        center = tuple(round(value) for value in point)
        cv2.circle(
            rendered,
            center,
            analysis.sample_radius_px,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            str(index),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )
    return rendered


def draw_court_color_distance(
    analysis: CourtSurfaceAnalysis,
    maximum_distance: float = 60.0,
) -> np.ndarray:
    """Render a heatmap where warmer pixels are closer to the sampled color."""
    normalized = np.clip(
        analysis.color_distance / max(1.0, maximum_distance),
        0.0,
        1.0,
    )
    similarity = np.rint((1.0 - normalized) * 255).astype(np.uint8)
    return cv2.applyColorMap(similarity, cv2.COLORMAP_TURBO)


def refine_surface_boundaries(
    analysis: CourtSurfaceAnalysis,
    landmarks: dict[str, tuple[float, float]],
    settings: SurfaceBoundarySettings | None = None,
) -> SurfaceBoundaryRefinement:
    """Move padel baseline corners to outer spans in the court-color mask.

    The keypoint model supplies approximate far and near levels. Within those
    local bands, the first sustained court-colored row is the far boundary and
    the last sustained row is the near boundary. Endpoint extraction only uses
    rows immediately inside the court, avoiding unrelated same-color regions.
    """
    settings = settings or SurfaceBoundarySettings()
    _validate_surface_boundary_settings(settings)
    if not all(name in landmarks for name in BASELINE_CORNER_NAMES):
        raise ValueError("four baseline corners are required for surface refinement")

    mask = analysis.surface_mask
    height, width = mask.shape
    original = dict(landmarks)
    far_left = np.asarray(landmarks[BASELINE_CORNER_NAMES[0]], dtype=np.float64)
    far_right = np.asarray(landmarks[BASELINE_CORNER_NAMES[1]], dtype=np.float64)
    near_right = np.asarray(landmarks[BASELINE_CORNER_NAMES[2]], dtype=np.float64)
    near_left = np.asarray(landmarks[BASELINE_CORNER_NAMES[3]], dtype=np.float64)
    far_y = float((far_left[1] + far_right[1]) * 0.5)
    near_y = float((near_left[1] + near_right[1]) * 0.5)
    court_height = near_y - far_y
    if court_height < 20:
        raise ValueError("model court boundary is vertically degenerate")

    search_radius = max(8, round(court_height * settings.vertical_search_ratio))
    far_interval = _central_x_interval(far_left, far_right, width)
    near_interval = _central_x_interval(near_left, near_right, width)
    far_row, far_support = _find_surface_transition_row(
        mask,
        far_interval,
        round(far_y),
        search_radius,
        settings,
        boundary="far",
    )
    near_row, near_support = _find_surface_transition_row(
        mask,
        near_interval,
        round(near_y),
        search_radius,
        settings,
        boundary="near",
    )
    if far_row >= near_row:
        raise ValueError("surface boundary rows are reversed")

    close_width = settings.horizontal_close_px
    if close_width % 2 == 0:
        close_width += 1
    connected_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3, close_width), dtype=np.uint8),
    )
    far_span = _boundary_span_from_interior_rows(
        connected_mask,
        far_row,
        far_interval,
        settings,
        boundary="far",
    )
    near_span = _boundary_span_from_interior_rows(
        connected_mask,
        near_row,
        near_interval,
        settings,
        boundary="near",
    )

    refined = dict(landmarks)
    refined["far_left_baseline_corner"] = (far_span[0], float(far_row))
    refined["far_right_baseline_corner"] = (far_span[1], float(far_row))
    refined["near_right_baseline_corner"] = (near_span[1], float(near_row))
    refined["near_left_baseline_corner"] = (near_span[0], float(near_row))
    _validate_refined_quad(refined, width, height)

    span_consistency = min(
        1.0,
        (far_span[1] - far_span[0]) / max(1.0, far_right[0] - far_left[0]),
        (near_span[1] - near_span[0]) / max(1.0, near_right[0] - near_left[0]),
    )
    confidence = float(
        np.clip((far_support + near_support + span_consistency) / 3.0, 0.0, 1.0)
    )
    return SurfaceBoundaryRefinement(
        landmarks=refined,
        original_landmarks=original,
        refined_landmark_names=BASELINE_CORNER_NAMES,
        far_row=far_row,
        near_row=near_row,
        far_span=far_span,
        near_span=near_span,
        confidence=confidence,
    )


def evaluate_surface_threshold_stability(
    frame: np.ndarray,
    landmarks: dict[str, tuple[float, float]],
    surface_settings: CourtSurfaceSettings,
    boundary_settings: SurfaceBoundarySettings,
    threshold_delta: float = 6.0,
) -> SurfaceThresholdStability:
    """Compare surface corners at the current and nearby color tolerances."""
    if threshold_delta <= 0:
        raise ValueError("threshold_delta must be positive")
    center = surface_settings.color_distance_threshold
    thresholds = tuple(
        sorted({max(1.0, center - threshold_delta), center, center + threshold_delta})
    )
    results: dict[float, SurfaceBoundaryRefinement] = {}
    for threshold in thresholds:
        trial_settings = CourtSurfaceSettings(
            color_distance_threshold=threshold,
            lightness_weight=surface_settings.lightness_weight,
            sample_radius_ratio=surface_settings.sample_radius_ratio,
            minimum_sample_radius_px=surface_settings.minimum_sample_radius_px,
        )
        try:
            analysis = analyze_court_surface_color(frame, landmarks, trial_settings)
            results[threshold] = refine_surface_boundaries(
                analysis,
                landmarks,
                boundary_settings,
            )
        except ValueError:
            continue

    center_result = results.get(center)
    maximum_change = float("inf")
    if center_result is not None:
        center_corners = np.asarray(
            [center_result.landmarks[name] for name in BASELINE_CORNER_NAMES],
            dtype=np.float64,
        )
        changes = []
        for result in results.values():
            corners = np.asarray(
                [result.landmarks[name] for name in BASELINE_CORNER_NAMES],
                dtype=np.float64,
            )
            corner_changes = np.linalg.norm(corners - center_corners, axis=1)
            changes.append(float(corner_changes.max()))
        maximum_change = max(changes, default=0.0)
    allowed_change = max(6.0, float(np.hypot(*frame.shape[:2])) * 0.01)
    return SurfaceThresholdStability(
        tested_thresholds=thresholds,
        successful_thresholds=tuple(
            threshold for threshold in thresholds if threshold in results
        ),
        maximum_corner_change_px=maximum_change,
        stable=(
            len(results) == len(thresholds)
            and np.isfinite(maximum_change)
            and maximum_change <= allowed_change
        ),
    )


def draw_surface_boundary_refinement(
    frame: np.ndarray,
    result: SurfaceBoundaryRefinement,
) -> np.ndarray:
    """Show model corners, selected spans, and accepted surface corners."""
    rendered = frame.copy()
    for row, span, label in (
        (result.far_row, result.far_span, "far surface boundary"),
        (result.near_row, result.near_span, "near surface boundary"),
    ):
        first = (round(span[0]), row)
        second = (round(span[1]), row)
        cv2.line(rendered, first, second, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(
            rendered,
            label,
            (first[0], max(18, row - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 220, 0),
            1,
            cv2.LINE_AA,
        )
    for name in result.refined_landmark_names:
        original = tuple(round(value) for value in result.original_landmarks[name])
        refined = tuple(round(value) for value in result.landmarks[name])
        cv2.circle(rendered, original, 5, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(rendered, original, refined, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(
            rendered,
            refined,
            (0, 220, 0),
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )
    return rendered


def _validate_surface_boundary_settings(
    settings: SurfaceBoundarySettings,
) -> None:
    if not 0 < settings.vertical_search_ratio <= 0.5:
        raise ValueError("vertical_search_ratio must be between zero and 0.5")
    if not 0 < settings.minimum_central_support <= 1:
        raise ValueError("minimum_central_support must be between zero and one")
    if settings.sustained_rows < 2:
        raise ValueError("sustained_rows must be at least two")
    if settings.interior_sample_depth_px < 2:
        raise ValueError("interior_sample_depth_px must be at least two")
    if settings.horizontal_close_px < 1:
        raise ValueError("horizontal_close_px must be positive")
    if not 0 <= settings.endpoint_percentile < 50:
        raise ValueError("endpoint_percentile must be between zero and 50")
    if not 0 <= settings.minimum_frame_margin_ratio <= 0.1:
        raise ValueError("minimum_frame_margin_ratio must be between zero and 0.1")
    if settings.minimum_transition_gap_rows < 1:
        raise ValueError("minimum_transition_gap_rows must be positive")


def _central_x_interval(
    left: np.ndarray,
    right: np.ndarray,
    frame_width: int,
) -> tuple[int, int]:
    start = round(float(left[0] * 0.65 + right[0] * 0.35))
    end = round(float(left[0] * 0.35 + right[0] * 0.65))
    start = int(np.clip(start, 0, frame_width - 2))
    end = int(np.clip(end, start + 1, frame_width - 1))
    return start, end


def _find_surface_transition_row(
    mask: np.ndarray,
    x_interval: tuple[int, int],
    expected_y: int,
    search_radius: int,
    settings: SurfaceBoundarySettings,
    boundary: str,
) -> tuple[int, float]:
    height = mask.shape[0]
    first_row = max(0, expected_y - search_radius)
    last_row = min(height - 1, expected_y + search_radius)
    start_x, end_x = x_interval
    supports = np.asarray(
        [
            np.count_nonzero(mask[row, start_x : end_x + 1])
            / (end_x - start_x + 1)
            for row in range(first_row, last_row + 1)
        ],
        dtype=np.float64,
    )
    supported = supports >= settings.minimum_central_support
    run_length = settings.sustained_rows
    padded = np.concatenate(([False], supported, [False])).astype(np.int8)
    changes = np.diff(padded)
    run_starts = np.flatnonzero(changes == 1)
    run_ends = np.flatnonzero(changes == -1) - 1
    runs = [
        (int(start), int(end))
        for start, end in zip(run_starts, run_ends, strict=True)
        if end - start + 1 >= run_length
    ]
    if not runs:
        raise ValueError(f"no sustained {boundary} court-color boundary was found")
    gap_rows = settings.minimum_transition_gap_rows
    if boundary == "far":
        candidates = [
            (start, end)
            for start, end in runs
            if start >= gap_rows
            and not supported[start - gap_rows : start].any()
        ]
        if not candidates:
            raise ValueError("far court-color entry is not visible in the frame")
        row_index, _ = min(
            candidates,
            key=lambda run: abs(first_row + run[0] - expected_y),
        )
        support_slice = supports[row_index : row_index + run_length]
        row = first_row + row_index
    elif boundary == "near":
        candidates = [
            (start, end)
            for start, end in runs
            if end + gap_rows < len(supported)
            and not supported[end + 1 : end + 1 + gap_rows].any()
        ]
        if not candidates:
            raise ValueError("near court-color exit is not visible in the frame")
        _, row_index = min(
            candidates,
            key=lambda run: abs(first_row + run[1] - expected_y),
        )
        support_slice = supports[row_index - run_length + 1 : row_index + 1]
        row = first_row + row_index
    else:
        raise ValueError(f"unsupported boundary: {boundary}")
    frame_margin = max(
        run_length,
        round(height * settings.minimum_frame_margin_ratio),
    )
    if boundary == "far" and row < frame_margin:
        raise ValueError("far court-color boundary is too close to the frame edge")
    if boundary == "near" and row > height - 1 - frame_margin:
        raise ValueError("near court-color boundary is too close to the frame edge")
    return row, float(np.mean(support_slice))


def _boundary_span_from_interior_rows(
    mask: np.ndarray,
    boundary_row: int,
    central_interval: tuple[int, int],
    settings: SurfaceBoundarySettings,
    boundary: str,
) -> tuple[float, float]:
    height = mask.shape[0]
    depth = settings.interior_sample_depth_px
    if boundary == "far":
        rows = range(boundary_row, min(height, boundary_row + depth))
    elif boundary == "near":
        rows = range(max(0, boundary_row - depth + 1), boundary_row + 1)
    else:
        raise ValueError(f"unsupported boundary: {boundary}")

    left_edges: list[int] = []
    right_edges: list[int] = []
    for row in rows:
        run = _best_row_run(mask[row], central_interval)
        if run is None:
            continue
        left_edges.append(run[0])
        right_edges.append(run[1])
    minimum_rows = max(2, settings.sustained_rows // 2)
    if len(left_edges) < minimum_rows:
        raise ValueError(f"not enough {boundary} rows for endpoint refinement")
    lower = settings.endpoint_percentile
    upper = 100.0 - lower
    return (
        float(np.percentile(left_edges, lower)),
        float(np.percentile(right_edges, upper)),
    )


def _best_row_run(
    row_mask: np.ndarray,
    central_interval: tuple[int, int],
) -> tuple[int, int] | None:
    xs = np.flatnonzero(row_mask)
    if not len(xs):
        return None
    split_indexes = np.flatnonzero(np.diff(xs) > 1) + 1
    runs = np.split(xs, split_indexes)
    center_start, center_end = central_interval

    def score(run: np.ndarray) -> tuple[int, int]:
        overlap = max(
            0,
            min(int(run[-1]), center_end) - max(int(run[0]), center_start) + 1,
        )
        return overlap, len(run)

    selected = max(runs, key=score)
    if score(selected)[0] == 0:
        return None
    return int(selected[0]), int(selected[-1])


def _validate_refined_quad(
    landmarks: dict[str, tuple[float, float]],
    width: int,
    height: int,
) -> None:
    quad = np.asarray(
        [landmarks[name] for name in BASELINE_CORNER_NAMES],
        dtype=np.float32,
    )
    if not np.isfinite(quad).all():
        raise ValueError("surface boundary contains non-finite points")
    if (
        np.any(quad[:, 0] < 0)
        or np.any(quad[:, 0] >= width)
        or np.any(quad[:, 1] < 0)
        or np.any(quad[:, 1] >= height)
    ):
        raise ValueError("surface boundary lies outside the frame")
    contour = quad.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= 100:
        raise ValueError("surface boundary is not a valid convex court")


def find_surface_boundary_candidates(
    analysis: CourtSurfaceAnalysis,
    landmarks: dict[str, tuple[float, float]],
    settings: BoundaryScanSettings | None = None,
) -> SurfaceBoundaryCandidates:
    """Scan from model-guided interior anchors to full-frame mask boundaries."""
    settings = settings or BoundaryScanSettings()
    if settings.scans_per_boundary < 2:
        raise ValueError("scans_per_boundary must be at least two")
    if settings.maximum_gap_px < 0:
        raise ValueError("maximum_gap_px cannot be negative")
    if not all(name in landmarks for name in BASELINE_CORNER_NAMES):
        raise ValueError("four baseline corners are required for boundary scans")

    transform = _template_to_image_transform(landmarks)
    scans: list[BoundaryScan] = []
    positions = np.linspace(0.08, 0.92, settings.scans_per_boundary)

    for court_y in positions:
        left_edge, right_edge = _project_template_points(
            transform,
            ((0.0, court_y), (1.0, court_y)),
        )
        width_direction = _unit_vector(right_edge - left_edge)
        left_anchor, right_anchor = _project_template_points(
            transform,
            ((0.35, court_y), (0.65, court_y)),
        )
        _append_boundary_scan(
            scans,
            "left",
            analysis.surface_mask,
            left_anchor,
            -width_direction,
            settings,
        )
        _append_boundary_scan(
            scans,
            "right",
            analysis.surface_mask,
            right_anchor,
            width_direction,
            settings,
        )

    for court_x in positions:
        far_edge, near_edge = _project_template_points(
            transform,
            ((court_x, 0.0), (court_x, 1.0)),
        )
        length_direction = _unit_vector(near_edge - far_edge)
        far_anchor, near_anchor = _project_template_points(
            transform,
            ((court_x, 0.30), (court_x, 0.70)),
        )
        _append_boundary_scan(
            scans,
            "far",
            analysis.surface_mask,
            far_anchor,
            -length_direction,
            settings,
        )
        _append_boundary_scan(
            scans,
            "near",
            analysis.surface_mask,
            near_anchor,
            length_direction,
            settings,
        )

    return SurfaceBoundaryCandidates(scans=tuple(scans))


def draw_surface_boundary_candidates(
    frame: np.ndarray,
    candidates: SurfaceBoundaryCandidates,
) -> np.ndarray:
    """Draw accepted scan paths and their raw boundary endpoints."""
    rendered = frame.copy()
    colors = {
        "left": (60, 220, 70),
        "right": (0, 165, 255),
        "far": (230, 80, 230),
        "near": (0, 230, 255),
    }
    for scan in candidates.scans:
        anchor = tuple(round(value) for value in scan.anchor)
        endpoint = tuple(round(value) for value in scan.endpoint)
        color = colors[scan.boundary]
        cv2.line(rendered, anchor, endpoint, color, 1, cv2.LINE_AA)
        cv2.circle(rendered, anchor, 2, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(rendered, endpoint, 4, color, -1, cv2.LINE_AA)
    return rendered


def _sample_patch(
    image: np.ndarray,
    point: np.ndarray,
    radius: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    center_x = int(round(float(point[0])))
    center_y = int(round(float(point[1])))
    left = max(0, center_x - radius)
    right = min(width, center_x + radius + 1)
    top = max(0, center_y - radius)
    bottom = min(height, center_y + radius + 1)
    if left >= right or top >= bottom:
        return np.empty((0, 3), dtype=image.dtype)
    patch = image[top:bottom, left:right]
    local_y, local_x = np.ogrid[
        top - center_y : bottom - center_y,
        left - center_x : right - center_x,
    ]
    circle = local_x * local_x + local_y * local_y <= radius * radius
    return patch[circle]


def _template_to_image_transform(
    landmarks: dict[str, tuple[float, float]],
) -> np.ndarray:
    template_corners = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float32,
    )
    image_corners = np.asarray(
        [landmarks[name] for name in BASELINE_CORNER_NAMES],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(template_corners, image_corners)


def _project_template_points(
    transform: np.ndarray,
    points: tuple[tuple[float, float], ...],
) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        transform,
    ).reshape(-1, 2)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        raise ValueError("model court boundary is degenerate")
    return vector.astype(np.float64) / length


def _append_boundary_scan(
    scans: list[BoundaryScan],
    boundary: str,
    mask: np.ndarray,
    anchor: np.ndarray,
    direction: np.ndarray,
    settings: BoundaryScanSettings,
) -> None:
    endpoint_and_anchor = _scan_to_mask_boundary(
        mask,
        anchor.astype(np.float64),
        direction,
        settings,
    )
    if endpoint_and_anchor is None:
        return
    accepted_anchor, endpoint = endpoint_and_anchor
    scans.append(
        BoundaryScan(
            boundary=boundary,
            anchor=(float(accepted_anchor[0]), float(accepted_anchor[1])),
            endpoint=(float(endpoint[0]), float(endpoint[1])),
        )
    )


def _scan_to_mask_boundary(
    mask: np.ndarray,
    anchor: np.ndarray,
    direction: np.ndarray,
    settings: BoundaryScanSettings,
) -> tuple[np.ndarray, np.ndarray] | None:
    accepted_anchor = _nearest_mask_point(
        mask,
        anchor,
        settings.anchor_search_radius_px,
    )
    if accepted_anchor is None:
        return None

    height, width = mask.shape
    maximum_distance = round(float(np.hypot(width, height)))
    last_match = accepted_anchor.copy()
    gap = 0
    for distance in range(1, maximum_distance + 1):
        point = accepted_anchor + direction * distance
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if not (0 <= x < width and 0 <= y < height):
            break
        if mask[y, x]:
            last_match = np.asarray((x, y), dtype=np.float64)
            gap = 0
        else:
            gap += 1
            if gap > settings.maximum_gap_px:
                break
    if np.linalg.norm(last_match - accepted_anchor) < settings.minimum_scan_length_px:
        return None
    return accepted_anchor, last_match


def _nearest_mask_point(
    mask: np.ndarray,
    point: np.ndarray,
    radius: int,
) -> np.ndarray | None:
    height, width = mask.shape
    center_x = int(round(float(point[0])))
    center_y = int(round(float(point[1])))
    left = max(0, center_x - radius)
    right = min(width, center_x + radius + 1)
    top = max(0, center_y - radius)
    bottom = min(height, center_y + radius + 1)
    ys, xs = np.nonzero(mask[top:bottom, left:right])
    if not len(xs):
        return None
    xs = xs + left
    ys = ys + top
    distances = (xs - center_x) ** 2 + (ys - center_y) ** 2
    index = int(np.argmin(distances))
    return np.asarray((xs[index], ys[index]), dtype=np.float64)
