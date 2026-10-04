from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courtvision.calibration.homography import HomographyEstimate, project_image_points
from courtvision.detectors.base import Detection
from courtvision.detectors.tracknet_adapter import BallPoint
from courtvision.geometry.homography import bottom_center
from courtvision.visualization.minimap import CourtSpec

# How far outside the painted court (side, end) a player may stand, in metres.
# Padel is enclosed by walls; tennis players often play several metres behind
# the baseline.
COURT_MARGINS_M = {"padel": (0.5, 0.5), "tennis": (4.0, 7.0)}


@dataclass(frozen=True)
class CourtCoordinate:
    frame: int
    track_id: int
    object_type: str
    image_x: float
    image_y: float
    template_x: float
    template_y: float
    court_x_m: float
    court_y_m: float
    in_bounds: bool
    confidence: float


def project_player_detection(
    detection: Detection,
    track_id: int,
    estimate: HomographyEstimate,
    spec: CourtSpec,
) -> CourtCoordinate:
    image_x, image_y = bottom_center(
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
    )
    template_x, template_y = project_image_points(
        np.asarray([[image_x, image_y]]),
        estimate,
    )[0]
    in_bounds = bool(0 <= template_x <= 1 and 0 <= template_y <= 1)
    return CourtCoordinate(
        frame=detection.frame,
        track_id=track_id,
        object_type="player",
        image_x=float(image_x),
        image_y=float(image_y),
        template_x=float(template_x),
        template_y=float(template_y),
        court_x_m=float(template_x * spec.width_m),
        court_y_m=float(template_y * spec.length_m),
        in_bounds=in_bounds,
        confidence=detection.confidence,
    )


def project_ball_point(
    point: BallPoint,
    estimate: HomographyEstimate,
    spec: CourtSpec,
) -> CourtCoordinate | None:
    """Project a TrackNet source-pixel ball center into court coordinates."""
    if not point.visible:
        return None
    template_x, template_y = project_image_points(
        np.asarray([[point.x, point.y]]),
        estimate,
    )[0]
    in_bounds = bool(0 <= template_x <= 1 and 0 <= template_y <= 1)
    return CourtCoordinate(
        frame=point.frame,
        track_id=0,
        object_type="ball",
        image_x=float(point.x),
        image_y=float(point.y),
        template_x=float(template_x),
        template_y=float(template_y),
        court_x_m=float(template_x * spec.width_m),
        court_y_m=float(template_y * spec.length_m),
        in_bounds=in_bounds,
        confidence=point.confidence,
    )


def court_margins_m(spec: CourtSpec) -> tuple[float, float]:
    """Return how far beyond the side and end lines players may stand."""
    try:
        return COURT_MARGINS_M[spec.name]
    except KeyError:
        raise ValueError(f"No court margins defined for {spec.name!r}") from None


def filter_detections_on_court(
    detections: list[Detection],
    estimate: HomographyEstimate,
    spec: CourtSpec,
) -> list[Detection]:
    """Keep people whose feet project inside the court plus its margins.

    Drops spectators, ball kids and people on neighbouring courts.
    """
    if not detections:
        return []
    side_margin_m, end_margin_m = court_margins_m(spec)
    feet = np.asarray(
        [
            bottom_center(detection.x1, detection.y1, detection.x2, detection.y2)
            for detection in detections
        ],
        dtype=float,
    )
    template_points = project_image_points(feet, estimate)
    court_x_m = template_points[:, 0] * spec.width_m
    court_y_m = template_points[:, 1] * spec.length_m
    on_court = (
        (court_x_m >= -side_margin_m)
        & (court_x_m <= spec.width_m + side_margin_m)
        & (court_y_m >= -end_margin_m)
        & (court_y_m <= spec.length_m + end_margin_m)
    )
    return [
        detection for detection, keep in zip(detections, on_court, strict=True) if keep
    ]
