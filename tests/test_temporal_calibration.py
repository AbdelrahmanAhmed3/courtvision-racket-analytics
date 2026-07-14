from __future__ import annotations

import cv2
import numpy as np

from courtvision.calibration.io import CalibrationRecord, LandmarkObservation
from courtvision.calibration.temporal import TemporalCourtCalibrator


def test_optical_flow_tracks_landmarks_and_refreshes_homography() -> None:
    calibration, frame = _calibration_frame()
    moved = cv2.warpAffine(
        frame,
        np.float32([[1, 0, 5], [0, 1, 4]]),
        (frame.shape[1], frame.shape[0]),
    )
    tracker = TemporalCourtCalibrator(calibration, frame, min_inliers=6)

    result = tracker.update(moved, frame_index=1)

    assert result.valid
    assert result.calibration is not None
    assert result.estimate is not None
    assert result.validation is not None
    assert result.tracked_landmark_count == 8
    assert result.validation.inlier_count == 8
    point = result.calibration.landmarks["far_left_baseline_corner"].image
    assert np.allclose(point, (55, 34), atol=1.5)


def _calibration_frame() -> tuple[CalibrationRecord, np.ndarray]:
    templates = {
        "far_left_baseline_corner": (0.0, 0.0),
        "far_right_baseline_corner": (1.0, 0.0),
        "near_right_baseline_corner": (1.0, 1.0),
        "near_left_baseline_corner": (0.0, 1.0),
        "far_left_service_intersection": (0.25, 0.25),
        "far_right_service_intersection": (0.75, 0.25),
        "near_right_service_intersection": (0.75, 0.75),
        "near_left_service_intersection": (0.25, 0.75),
    }
    frame = np.zeros((220, 320, 3), dtype=np.uint8)
    landmarks = {}
    for index, (name, template) in enumerate(templates.items()):
        x = 50 + template[0] * 200
        y = 30 + template[1] * 140
        center = (int(round(x)), int(round(y)))
        cv2.drawMarker(
            frame,
            center,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=13 + index % 3,
            thickness=2,
        )
        landmarks[name] = LandmarkObservation(
            image=(x, y),
            template=template,
            visible=True,
            confidence=1.0,
            source="manual",
        )
    return (
        CalibrationRecord(
            video_id="synthetic.mp4",
            frame_index=0,
            frame_width=320,
            frame_height=220,
            court_type="tennis",
            landmarks=landmarks,
        ),
        frame,
    )
