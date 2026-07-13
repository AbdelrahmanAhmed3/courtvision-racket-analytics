from __future__ import annotations

from courtvision.analytics.court_coordinates import project_ball_point
from courtvision.calibration.homography import estimate_template_homography
from courtvision.calibration.io import (
    CalibrationRecord,
    LandmarkObservation,
)
from courtvision.detectors.tracknet_adapter import BallPoint
from courtvision.visualization.minimap import get_court_spec


def test_projects_visible_ball_center_into_court_coordinates() -> None:
    calibration = CalibrationRecord(
        video_id="clip.mp4",
        frame_index=0,
        frame_width=100,
        frame_height=200,
        court_type="tennis",
        landmarks={
            "far_left_baseline_corner": _landmark((0, 0), (0, 0)),
            "far_right_baseline_corner": _landmark((100, 0), (1, 0)),
            "near_right_baseline_corner": _landmark((100, 200), (1, 1)),
            "near_left_baseline_corner": _landmark((0, 200), (0, 1)),
        },
    )

    projected = project_ball_point(
        BallPoint(frame=12, x=50, y=100, confidence=0.8),
        estimate_template_homography(calibration),
        get_court_spec("tennis"),
    )

    assert projected is not None
    assert projected.frame == 12
    assert projected.track_id == 0
    assert projected.object_type == "ball"
    assert projected.in_bounds
    assert projected.template_x == 0.5
    assert projected.template_y == 0.5
    assert projected.confidence == 0.8


def test_ignores_missing_ball_point() -> None:
    calibration = CalibrationRecord(
        video_id="clip.mp4",
        frame_index=0,
        frame_width=100,
        frame_height=200,
        court_type="padel",
        landmarks={
            "far_left_baseline_corner": _landmark((0, 0), (0, 0)),
            "far_right_baseline_corner": _landmark((100, 0), (1, 0)),
            "near_right_baseline_corner": _landmark((100, 200), (1, 1)),
            "near_left_baseline_corner": _landmark((0, 200), (0, 1)),
        },
    )

    projected = project_ball_point(
        BallPoint(frame=12, x=None, y=None, confidence=0.0),
        estimate_template_homography(calibration),
        get_court_spec("padel"),
    )

    assert projected is None


def _landmark(
    image: tuple[float, float],
    template: tuple[float, float],
) -> LandmarkObservation:
    return LandmarkObservation(
        image=image,
        template=template,
        visible=True,
        confidence=1.0,
        source="test",
    )
