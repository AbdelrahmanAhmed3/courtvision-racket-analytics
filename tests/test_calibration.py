import numpy as np

from courtvision.calibration.homography import estimate_template_homography
from courtvision.calibration.io import CalibrationRecord, LandmarkObservation
from courtvision.calibration.validation import validate_homography


def make_calibration() -> CalibrationRecord:
    points = {
        "far_left_baseline_corner": ((0, 0), (0, 0)),
        "far_right_baseline_corner": ((100, 0), (1, 0)),
        "near_right_baseline_corner": ((100, 200), (1, 1)),
        "near_left_baseline_corner": ((0, 200), (0, 1)),
        "left_net_point": ((0, 100), (0, 0.5)),
        "right_net_point": ((100, 100), (1, 0.5)),
    }
    landmarks = {
        name: LandmarkObservation(
            image=image,
            template=template,
            visible=True,
            confidence=1.0,
            source="manual",
        )
        for name, (image, template) in points.items()
    }
    return CalibrationRecord(
        video_id="test.mp4",
        frame_index=0,
        frame_width=100,
        frame_height=200,
        court_type="tennis",
        landmarks=landmarks,
    )


def test_named_landmark_homography_and_validation() -> None:
    estimate = estimate_template_homography(make_calibration())
    validation = validate_homography(estimate)

    assert np.allclose(estimate.image_to_template[0, 0], 0.01, atol=1e-5)
    assert validation.status == "good"
    assert validation.inlier_count == 6
    assert validation.mean_reprojection_error_px < 1e-4
