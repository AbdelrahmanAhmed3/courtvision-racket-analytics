import cv2
import numpy as np

from courtvision.calibration.court_surface import CourtSurfaceSettings
from courtvision.calibration.keypoints import (
    detect_keypoint_calibration,
    keypoint_proposal_from_response,
    landmark_audit_entries,
)


def test_maps_tennis_keypoints_to_canonical_landmarks() -> None:
    response = {
        "predictions": [
            {
                "confidence": 0.9,
                "keypoints": [
                    {"class": "1", "x": 10, "y": 90, "confidence": 0.9},
                    {"class": "4", "x": 190, "y": 90, "confidence": 0.9},
                    {"class": "5", "x": 140, "y": 10, "confidence": 0.9},
                    {"class": "8", "x": 60, "y": 10, "confidence": 0.9},
                    {"class": "9", "x": 45, "y": 50, "confidence": 0.9},
                    {"class": "11", "x": 155, "y": 50, "confidence": 0.9},
                ],
            }
        ]
    }

    proposal = keypoint_proposal_from_response(
        response, "tennis", "tennis-court-detection-onesd/10"
    )

    assert proposal is not None
    assert proposal.landmark_count == 6
    assert proposal.landmarks["far_left_baseline_corner"] == (60.0, 10.0)
    assert proposal.landmarks["near_right_baseline_corner"] == (190.0, 90.0)
    assert proposal.landmarks["near_left_service_intersection"] == (45.0, 50.0)


def test_maps_padel_keypoints_and_ignores_low_confidence_points() -> None:
    response = {
        "predictions": [
            {
                "confidence": 0.8,
                "keypoints": [
                    {"class": "FCL", "x": 10, "y": 100, "confidence": 0.9},
                    {"class": "FCR", "x": 190, "y": 100, "confidence": 0.9},
                    {"class": "BCR", "x": 160, "y": 10, "confidence": 0.9},
                    {"class": "BCL", "x": 40, "y": 10, "confidence": 0.9},
                    {"class": "NC", "x": 100, "y": 55, "confidence": 0.1},
                ],
            }
        ]
    }

    proposal = keypoint_proposal_from_response(
        response, "padel", "padel-court-fmfv8/15", minimum_keypoint_confidence=0.2
    )

    assert proposal is not None
    assert proposal.points == (
        (40.0, 10.0),
        (160.0, 10.0),
        (190.0, 100.0),
        (10.0, 100.0),
    )
    assert proposal.landmarks["near_left_baseline_corner"] == (10.0, 100.0)
    assert "net_center" not in proposal.landmarks


def test_padel_inference_moves_model_corners_to_surface_boundary(monkeypatch) -> None:
    frame = np.full((240, 320, 3), (20, 20, 20), dtype=np.uint8)
    cv2.fillConvexPoly(
        frame,
        np.asarray(((70, 25), (250, 25), (310, 225), (10, 225)), np.int32),
        (190, 105, 35),
    )
    response = {
        "predictions": [
            {
                "confidence": 0.9,
                "keypoints": [
                    {"class": "FCL", "x": 45, "y": 205, "confidence": 0.9},
                    {"class": "FCR", "x": 275, "y": 205, "confidence": 0.9},
                    {"class": "BCR", "x": 228, "y": 43, "confidence": 0.9},
                    {"class": "BCL", "x": 92, "y": 43, "confidence": 0.9},
                ],
            }
        ]
    }
    monkeypatch.setattr(
        "courtvision.calibration.keypoints.build_inference_client",
        lambda *_: object(),
    )
    monkeypatch.setattr(
        "courtvision.calibration.keypoints.get_api_key",
        lambda: "test-key",
    )
    monkeypatch.setattr(
        "courtvision.calibration.keypoints.infer_with_retries",
        lambda *_: response,
    )

    proposal = detect_keypoint_calibration(
        frame,
        "padel",
        refine_with_court_lines=False,
        court_surface_settings=CourtSurfaceSettings(color_distance_threshold=18.0),
    )

    assert proposal is not None
    assert proposal.surface_refinement is not None
    np.testing.assert_allclose(
        proposal.landmarks["far_left_baseline_corner"], (70.0, 25.0), atol=3.0
    )
    np.testing.assert_allclose(
        proposal.landmarks["near_right_baseline_corner"], (310.0, 225.0), atol=3.0
    )
    audit = landmark_audit_entries(proposal)
    assert len(audit) == 4
    assert {entry.source for entry in audit} == {"Court-color outer boundary"}
