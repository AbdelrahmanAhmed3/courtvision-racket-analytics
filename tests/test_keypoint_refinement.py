import cv2
import numpy as np

from courtvision.calibration.keypoint_refinement import refine_keypoint_landmarks


def _synthetic_court() -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    image = np.full((720, 1280, 3), (45, 105, 65), dtype=np.uint8)
    quad = np.asarray(
        [[470, 180], [810, 180], [1100, 650], [180, 650]],
        dtype=np.float32,
    )
    template = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(template, quad)
    service_y = 5.485 / 23.77
    singles_margin = (10.97 - 8.23) / (2 * 10.97)
    definitions = {
        "far_left_baseline_corner": (0.0, 0.0),
        "far_right_baseline_corner": (1.0, 0.0),
        "near_right_baseline_corner": (1.0, 1.0),
        "near_left_baseline_corner": (0.0, 1.0),
        "far_left_service_intersection": (singles_margin, service_y),
        "far_right_service_intersection": (1 - singles_margin, service_y),
        "near_right_service_intersection": (
            1 - singles_margin,
            1 - service_y,
        ),
        "near_left_service_intersection": (singles_margin, 1 - service_y),
        "far_service_center": (0.5, service_y),
        "near_service_center": (0.5, 1 - service_y),
    }
    line_segments = (
        ((0, 0), (1, 0)),
        ((1, 0), (1, 1)),
        ((1, 1), (0, 1)),
        ((0, 1), (0, 0)),
        ((singles_margin, 0), (singles_margin, 1)),
        ((1 - singles_margin, 0), (1 - singles_margin, 1)),
        ((0, service_y), (1, service_y)),
        ((0, 1 - service_y), (1, 1 - service_y)),
        ((0.5, service_y), (0.5, 1 - service_y)),
    )
    for first, second in line_segments:
        projected = cv2.perspectiveTransform(
            np.asarray([first, second], dtype=np.float32).reshape(-1, 1, 2),
            transform,
        ).reshape(-1, 2)
        cv2.line(
            image,
            tuple(np.rint(projected[0]).astype(int)),
            tuple(np.rint(projected[1]).astype(int)),
            (245, 245, 245),
            5,
            cv2.LINE_AA,
        )
    template_points = np.asarray(list(definitions.values()), dtype=np.float32)
    projected = cv2.perspectiveTransform(
        template_points.reshape(-1, 1, 2),
        transform,
    ).reshape(-1, 2)
    expected = {
        name: tuple(float(value) for value in point)
        for name, point in zip(definitions, projected, strict=True)
    }
    return image, expected


def test_refines_shifted_model_points_to_painted_intersections() -> None:
    image, expected = _synthetic_court()
    model_points = {
        name: (point[0] + 16.0, point[1] + 7.0) for name, point in expected.items()
    }

    result = refine_keypoint_landmarks(image, model_points, "tennis")

    assert result.auto_ready
    assert result.refined_count == len(expected)
    for name, point in expected.items():
        np.testing.assert_allclose(result.landmarks[name], point, atol=4.0)


def test_keeps_model_points_when_no_court_line_evidence_exists() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    landmarks = {
        "far_left_baseline_corner": (470.0, 180.0),
        "far_right_baseline_corner": (810.0, 180.0),
        "near_right_baseline_corner": (1100.0, 650.0),
        "near_left_baseline_corner": (180.0, 650.0),
    }

    result = refine_keypoint_landmarks(image, landmarks, "tennis")

    assert result.landmarks == landmarks
    assert result.refined_count == 0
    assert not result.auto_ready


def test_refines_padel_service_points_against_outer_boundaries() -> None:
    image = np.full((720, 1280, 3), (50, 100, 70), dtype=np.uint8)
    quad = np.asarray(
        [[480, 190], [800, 190], [1080, 650], [200, 650]],
        dtype=np.float32,
    )
    template = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(template, quad)
    definitions = {
        "far_left_baseline_corner": (0.0, 0.0),
        "far_right_baseline_corner": (1.0, 0.0),
        "near_right_baseline_corner": (1.0, 1.0),
        "near_left_baseline_corner": (0.0, 1.0),
        "far_left_service_intersection": (0.0, 0.15),
        "far_right_service_intersection": (1.0, 0.15),
        "near_right_service_intersection": (1.0, 0.85),
        "near_left_service_intersection": (0.0, 0.85),
        "far_service_center": (0.5, 0.15),
        "near_service_center": (0.5, 0.85),
    }
    for first, second in (
        ((0, 0), (1, 0)),
        ((1, 1), (0, 1)),
        ((0, 0.15), (1, 0.15)),
        ((0, 0.85), (1, 0.85)),
        ((0.5, 0.15), (0.5, 0.85)),
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
            5,
            cv2.LINE_AA,
        )
    projected = cv2.perspectiveTransform(
        np.asarray(list(definitions.values()), dtype=np.float32).reshape(-1, 1, 2),
        transform,
    ).reshape(-1, 2)
    expected = {
        name: tuple(float(value) for value in point)
        for name, point in zip(definitions, projected, strict=True)
    }
    horizontal_biases = {
        "far_left_baseline_corner": (70.0, 7.0),
        "far_right_baseline_corner": (-60.0, 7.0),
        "near_right_baseline_corner": (-135.0, -7.0),
        "near_left_baseline_corner": (55.0, -7.0),
        "far_left_service_intersection": (60.0, -10.0),
        "far_right_service_intersection": (-65.0, -10.0),
        "near_right_service_intersection": (-105.0, 12.0),
        "near_left_service_intersection": (55.0, 12.0),
        "far_service_center": (0.0, -10.0),
        "near_service_center": (0.0, 12.0),
    }
    model_points = {
        name: (
            point[0] + horizontal_biases[name][0],
            point[1] + horizontal_biases[name][1],
        )
        for name, point in expected.items()
    }

    result = refine_keypoint_landmarks(image, model_points, "padel")

    assert result.auto_ready
    for name, point in expected.items():
        np.testing.assert_allclose(result.landmarks[name], point, atol=4.0)


def test_padel_endpoint_refinement_never_contracts_the_model_court() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    for y, start_x, end_x in (
        (180, 330, 690),
        (240, 320, 700),
        (530, 240, 780),
        (640, 220, 800),
    ):
        cv2.line(image, (start_x, y), (end_x, y), (245, 245, 245), 5)
    landmarks = {
        "far_left_baseline_corner": (300.0, 180.0),
        "far_right_baseline_corner": (720.0, 180.0),
        "near_right_baseline_corner": (850.0, 640.0),
        "near_left_baseline_corner": (170.0, 640.0),
        "far_left_service_intersection": (290.0, 240.0),
        "far_right_service_intersection": (730.0, 240.0),
        "near_right_service_intersection": (830.0, 530.0),
        "near_left_service_intersection": (190.0, 530.0),
    }

    result = refine_keypoint_landmarks(image, landmarks, "padel")

    assert result.landmarks == landmarks
    assert not result.auto_ready
