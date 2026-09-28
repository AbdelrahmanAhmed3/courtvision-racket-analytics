import cv2
import numpy as np

from courtvision.calibration.auto import (
    AutoCalibrationSettings,
    analyze_auto_calibration,
    detect_auto_calibration,
)


def test_detect_auto_calibration_finds_synthetic_court_boundary() -> None:
    image = np.full((720, 1280, 3), (42, 104, 70), dtype=np.uint8)
    expected = np.asarray(
        [[480, 250], [800, 250], [1160, 660], [100, 660]], dtype=np.int32
    )
    cv2.polylines(image, [expected], True, (245, 245, 245), 6, cv2.LINE_AA)
    transform = cv2.getPerspectiveTransform(
        np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        expected.astype(np.float32),
    )
    for first, second in (
        ((0, 0.5), (1, 0.5)),
        ((0, 5.485 / 23.77), (1, 5.485 / 23.77)),
        ((0, 1 - 5.485 / 23.77), (1, 1 - 5.485 / 23.77)),
        ((0.5, 5.485 / 23.77), (0.5, 1 - 5.485 / 23.77)),
        ((0.125, 0), (0.125, 1)),
        ((0.875, 0), (0.875, 1)),
    ):
        projected = cv2.perspectiveTransform(
            np.asarray([first, second], dtype=np.float32).reshape(-1, 1, 2),
            transform,
        ).reshape(-1, 2)
        cv2.line(
            image,
            tuple(np.rint(projected[0]).astype(int)),
            tuple(np.rint(projected[1]).astype(int)),
            (245, 245, 245),
            6,
            cv2.LINE_AA,
        )

    proposal = detect_auto_calibration(image)

    assert proposal is not None
    assert proposal.confidence >= 0.75
    assert proposal.line_count >= 4
    assert set(proposal.landmarks) == {
        "far_left_baseline_corner",
        "far_right_baseline_corner",
        "near_right_baseline_corner",
        "near_left_baseline_corner",
        "far_left_service_intersection",
        "far_right_service_intersection",
        "near_right_service_intersection",
        "near_left_service_intersection",
    }
    np.testing.assert_allclose(
        np.asarray(proposal.points),
        expected,
        atol=18,
    )


def test_detect_auto_calibration_rejects_frame_without_line_evidence() -> None:
    image = np.full((720, 1280, 3), (42, 104, 70), dtype=np.uint8)

    assert detect_auto_calibration(image) is None


def test_auto_calibration_prefers_full_court_template_over_crowd_trapezoid() -> None:
    image = np.full((720, 1280, 3), (42, 104, 70), dtype=np.uint8)
    expected = np.asarray(
        [[480, 250], [800, 250], [1160, 660], [100, 660]], dtype=np.int32
    )
    crowd_trapezoid = np.asarray(
        [[80, 45], [1200, 45], [1270, 205], [10, 205]], dtype=np.int32
    )
    cv2.polylines(image, [crowd_trapezoid], True, (245, 245, 245), 6, cv2.LINE_AA)
    cv2.polylines(image, [expected], True, (245, 245, 245), 6, cv2.LINE_AA)
    transform = cv2.getPerspectiveTransform(
        np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        expected.astype(np.float32),
    )
    for first, second in (
        ((0, 0.5), (1, 0.5)),
        ((0, 5.485 / 23.77), (1, 5.485 / 23.77)),
        ((0, 1 - 5.485 / 23.77), (1, 1 - 5.485 / 23.77)),
        ((0.5, 5.485 / 23.77), (0.5, 1 - 5.485 / 23.77)),
        ((0.125, 0), (0.125, 1)),
        ((0.875, 0), (0.875, 1)),
    ):
        projected = cv2.perspectiveTransform(
            np.asarray([first, second], dtype=np.float32).reshape(-1, 1, 2),
            transform,
        ).reshape(-1, 2)
        cv2.line(
            image,
            tuple(np.rint(projected[0]).astype(int)),
            tuple(np.rint(projected[1]).astype(int)),
            (245, 245, 245),
            6,
            cv2.LINE_AA,
        )

    proposal = detect_auto_calibration(image)

    assert proposal is not None
    np.testing.assert_allclose(np.asarray(proposal.points), expected, atol=18)


def test_auto_calibration_keeps_the_requested_number_of_ransac_lines() -> None:
    image = np.full((720, 1280, 3), (42, 104, 70), dtype=np.uint8)
    for offset in range(12):
        cv2.line(
            image,
            (40, 50 + offset * 45),
            (1240, 50 + offset * 45),
            (255, 255, 255),
            3,
        )

    debug = analyze_auto_calibration(
        image,
        settings=AutoCalibrationSettings(maximum_line_count=6),
    )

    assert len(debug.lines) <= 6


def test_auto_calibration_expands_singles_anchor_to_doubles_corners() -> None:
    image = np.full((720, 1280, 3), (42, 104, 70), dtype=np.uint8)
    expected_doubles = np.asarray(
        [[480, 250], [800, 250], [1160, 660], [100, 660]], dtype=np.int32
    )
    template = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(
        template, expected_doubles.astype(np.float32)
    )
    singles_margin = (10.97 - 8.23) / (2 * 10.97)
    for first, second in (
        ((singles_margin, 0), (1 - singles_margin, 0)),
        ((1 - singles_margin, 0), (1 - singles_margin, 1)),
        ((1 - singles_margin, 1), (singles_margin, 1)),
        ((singles_margin, 1), (singles_margin, 0)),
        ((singles_margin, 0.5), (1 - singles_margin, 0.5)),
        ((singles_margin, 5.485 / 23.77), (1 - singles_margin, 5.485 / 23.77)),
        (
            (singles_margin, 1 - 5.485 / 23.77),
            (1 - singles_margin, 1 - 5.485 / 23.77),
        ),
        ((0.5, 5.485 / 23.77), (0.5, 1 - 5.485 / 23.77)),
    ):
        projected = cv2.perspectiveTransform(
            np.asarray([first, second], dtype=np.float32).reshape(-1, 1, 2),
            transform,
        ).reshape(-1, 2)
        cv2.line(
            image,
            tuple(np.rint(projected[0]).astype(int)),
            tuple(np.rint(projected[1]).astype(int)),
            (245, 245, 245),
            6,
            cv2.LINE_AA,
        )

    proposal = detect_auto_calibration(image)

    assert proposal is not None
    assert proposal.boundary_anchor == "singles"
    np.testing.assert_allclose(np.asarray(proposal.points), expected_doubles, atol=24)
