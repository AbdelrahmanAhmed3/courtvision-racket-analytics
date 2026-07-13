import numpy as np

from courtvision.analytics.court_coordinates import project_player_detection
from courtvision.calibration.homography import HomographyEstimate
from courtvision.detectors.base import Detection
from courtvision.visualization.minimap import TENNIS_COURT


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
