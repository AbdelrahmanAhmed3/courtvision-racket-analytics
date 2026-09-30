import numpy as np

from courtvision.analytics.court_coordinates import (
    court_margins_m,
    filter_detections_on_court,
    project_player_detection,
)
from courtvision.calibration.homography import HomographyEstimate
from courtvision.detectors.base import Detection
from courtvision.visualization.minimap import PADEL_COURT, TENNIS_COURT


def test_project_player_detection_uses_bbox_bottom_center() -> None:
    estimate = HomographyEstimate(
        template_to_image=np.eye(3),
        image_to_template=np.eye(3),
        inlier_names=(),
        image_points=np.empty((0, 2)),
        template_points=np.empty((0, 2)),
        landmark_names=(),
    )
    detection = Detection(
        frame=7,
        class_name="player",
        confidence=0.9,
        x1=0.1,
        y1=0.2,
        x2=0.3,
        y2=0.6,
        model_name="test",
    )

    point = project_player_detection(detection, 4, estimate, TENNIS_COURT)

    assert point.image_x == 0.2
    assert point.image_y == 0.6
    assert np.isclose(point.template_x, 0.2)
    assert np.isclose(point.template_y, 0.6)
    assert point.in_bounds is True


def identity_estimate() -> HomographyEstimate:
    return HomographyEstimate(
        template_to_image=np.eye(3),
        image_to_template=np.eye(3),
        inlier_names=(),
        image_points=np.empty((0, 2)),
        template_points=np.empty((0, 2)),
        landmark_names=(),
    )


def person_with_feet_at(template_x: float, template_y: float) -> Detection:
    return Detection(
        frame=0,
        class_name="person",
        confidence=0.9,
        x1=template_x - 0.01,
        y1=template_y - 0.05,
        x2=template_x + 0.01,
        y2=template_y,
        model_name="test",
    )


def test_court_filter_drops_people_standing_beside_a_padel_court() -> None:
    player = person_with_feet_at(0.5, 0.5)
    next_court = person_with_feet_at(1.2, 0.5)  # 2 m beyond the side wall

    kept = filter_detections_on_court(
        [player, next_court], identity_estimate(), PADEL_COURT
    )

    assert kept == [player]


def test_court_filter_keeps_tennis_players_well_behind_the_baseline() -> None:
    behind_baseline = person_with_feet_at(0.5, 1.2)  # about 4.8 m behind it
    spectator = person_with_feet_at(0.5, 1.5)  # about 11.9 m behind it

    kept = filter_detections_on_court(
        [behind_baseline, spectator], identity_estimate(), TENNIS_COURT
    )

    assert kept == [behind_baseline]
    assert court_margins_m(TENNIS_COURT) == (4.0, 7.0)
    assert court_margins_m(PADEL_COURT) == (0.5, 0.5)
