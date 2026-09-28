import cv2
import numpy as np

from courtvision.calibration.court_surface import (
    BoundaryScanSettings,
    CourtSurfaceSettings,
    SurfaceBoundarySettings,
    analyze_court_surface_color,
    evaluate_surface_threshold_stability,
    find_surface_boundary_candidates,
    refine_surface_boundaries,
)


def test_surface_color_mask_searches_beyond_model_corners() -> None:
    frame = np.full((240, 320, 3), (20, 20, 20), dtype=np.uint8)
    actual_court = np.asarray(
        ((70, 25), (250, 25), (310, 225), (10, 225)),
        dtype=np.int32,
    )
    court_bgr = (190, 105, 35)
    cv2.fillConvexPoly(frame, actual_court, court_bgr)
    cv2.rectangle(frame, (145, 80), (175, 150), (30, 30, 210), -1)
    model_landmarks = {
        "far_left_baseline_corner": (90.0, 45.0),
        "far_right_baseline_corner": (230.0, 45.0),
        "near_right_baseline_corner": (275.0, 205.0),
        "near_left_baseline_corner": (45.0, 205.0),
    }

    result = analyze_court_surface_color(
        frame,
        model_landmarks,
        CourtSurfaceSettings(color_distance_threshold=18.0),
    )

    np.testing.assert_allclose(result.court_bgr, court_bgr, atol=3)
    assert result.surface_mask[190, 30] == 255
    assert result.surface_mask[10, 10] == 0
    assert result.surface_mask[110, 160] == 0


def test_surface_color_requires_all_boundary_corners() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        analyze_court_surface_color(frame, {})
    except ValueError as error:
        assert "four baseline corners" in str(error)
    else:
        raise AssertionError("missing corners should fail")


def test_inside_out_scans_expand_beyond_inset_model_boundary() -> None:
    frame = np.full((240, 320, 3), (20, 20, 20), dtype=np.uint8)
    actual_court = np.asarray(
        ((70, 25), (250, 25), (310, 225), (10, 225)),
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, actual_court, (190, 105, 35))
    cv2.line(frame, (0, 120), (319, 120), (10, 10, 10), 5)
    model_landmarks = {
        "far_left_baseline_corner": (90.0, 45.0),
        "far_right_baseline_corner": (230.0, 45.0),
        "near_right_baseline_corner": (275.0, 205.0),
        "near_left_baseline_corner": (45.0, 205.0),
    }
    analysis = analyze_court_surface_color(
        frame,
        model_landmarks,
        CourtSurfaceSettings(color_distance_threshold=18.0),
    )

    candidates = find_surface_boundary_candidates(
        analysis,
        model_landmarks,
        BoundaryScanSettings(scans_per_boundary=16, maximum_gap_px=2),
    )

    assert candidates.count("left") >= 10
    assert candidates.count("right") >= 10
    assert candidates.count("far") >= 10
    assert candidates.count("near") >= 10
    assert min(point[0] for point in candidates.endpoints("left")) < 45
    assert max(point[0] for point in candidates.endpoints("right")) > 275
    assert min(point[1] for point in candidates.endpoints("far")) < 45
    assert max(point[1] for point in candidates.endpoints("near")) > 205


def test_surface_boundary_refinement_moves_corners_to_outer_ground_spans() -> None:
    frame = np.full((240, 320, 3), (20, 20, 20), dtype=np.uint8)
    actual_court = np.asarray(
        ((70, 25), (250, 25), (310, 225), (10, 225)),
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, actual_court, (190, 105, 35))
    model_landmarks = {
        "far_left_baseline_corner": (92.0, 43.0),
        "far_right_baseline_corner": (228.0, 43.0),
        "near_right_baseline_corner": (275.0, 205.0),
        "near_left_baseline_corner": (45.0, 205.0),
        "far_left_service_intersection": (82.0, 65.0),
        "far_right_service_intersection": (238.0, 65.0),
    }
    analysis = analyze_court_surface_color(
        frame,
        model_landmarks,
        CourtSurfaceSettings(color_distance_threshold=18.0),
    )

    result = refine_surface_boundaries(
        analysis,
        model_landmarks,
        SurfaceBoundarySettings(
            minimum_central_support=0.8,
            sustained_rows=3,
            interior_sample_depth_px=8,
        ),
    )

    np.testing.assert_allclose(
        result.landmarks["far_left_baseline_corner"], (70.0, 25.0), atol=3.0
    )
    np.testing.assert_allclose(
        result.landmarks["far_right_baseline_corner"], (250.0, 25.0), atol=3.0
    )
    np.testing.assert_allclose(
        result.landmarks["near_right_baseline_corner"], (310.0, 225.0), atol=3.0
    )
    np.testing.assert_allclose(
        result.landmarks["near_left_baseline_corner"], (10.0, 225.0), atol=3.0
    )
    assert result.landmarks["far_left_service_intersection"] == (82.0, 65.0)
    assert result.confidence > 0.8


def test_surface_boundary_refinement_ignores_disconnected_color_distractor() -> None:
    frame = np.full((240, 320, 3), (20, 20, 20), dtype=np.uint8)
    actual_court = np.asarray(
        ((80, 40), (240, 40), (300, 220), (20, 220)),
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, actual_court, (190, 105, 35))
    cv2.rectangle(frame, (0, 0), (319, 15), (190, 105, 35), -1)
    model_landmarks = {
        "far_left_baseline_corner": (95.0, 50.0),
        "far_right_baseline_corner": (225.0, 50.0),
        "near_right_baseline_corner": (270.0, 205.0),
        "near_left_baseline_corner": (50.0, 205.0),
    }
    analysis = analyze_court_surface_color(
        frame,
        model_landmarks,
        CourtSurfaceSettings(color_distance_threshold=18.0),
    )

    result = refine_surface_boundaries(analysis, model_landmarks)

    assert result.far_row >= 38
    assert result.near_row >= 218


def test_surface_boundary_reports_stable_nearby_color_tolerances() -> None:
    frame = np.full((240, 320, 3), (20, 20, 20), dtype=np.uint8)
    cv2.fillConvexPoly(
        frame,
        np.asarray(((70, 25), (250, 25), (310, 225), (10, 225)), np.int32),
        (190, 105, 35),
    )
    model_landmarks = {
        "far_left_baseline_corner": (92.0, 43.0),
        "far_right_baseline_corner": (228.0, 43.0),
        "near_right_baseline_corner": (275.0, 205.0),
        "near_left_baseline_corner": (45.0, 205.0),
    }

    stability = evaluate_surface_threshold_stability(
        frame,
        model_landmarks,
        CourtSurfaceSettings(color_distance_threshold=18.0),
        SurfaceBoundarySettings(),
        threshold_delta=5.0,
    )

    assert stability.stable
    assert stability.successful_thresholds == (13.0, 18.0, 23.0)
    assert stability.maximum_corner_change_px == 0.0


def test_surface_boundary_rejects_near_court_continuing_out_of_frame() -> None:
    frame = np.full((240, 320, 3), (20, 20, 20), dtype=np.uint8)
    cv2.fillConvexPoly(
        frame,
        np.asarray(((80, 30), (240, 30), (340, 270), (-20, 270)), np.int32),
        (190, 105, 35),
    )
    model_landmarks = {
        "far_left_baseline_corner": (90.0, 40.0),
        "far_right_baseline_corner": (230.0, 40.0),
        "near_right_baseline_corner": (305.0, 230.0),
        "near_left_baseline_corner": (15.0, 230.0),
    }
    analysis = analyze_court_surface_color(
        frame,
        model_landmarks,
        CourtSurfaceSettings(color_distance_threshold=18.0),
    )

    try:
        refine_surface_boundaries(analysis, model_landmarks)
    except ValueError as error:
        assert "near" in str(error)
    else:
        raise AssertionError("an off-frame near boundary must not be accepted")


def test_far_boundary_uses_floor_below_same_color_back_wall() -> None:
    frame = np.full((240, 320, 3), (20, 20, 20), dtype=np.uint8)
    court_color = (190, 105, 35)
    cv2.rectangle(frame, (70, 10), (250, 35), court_color, -1)
    cv2.fillConvexPoly(
        frame,
        np.asarray(((70, 41), (250, 41), (310, 225), (10, 225)), np.int32),
        court_color,
    )
    model_landmarks = {
        "far_left_baseline_corner": (90.0, 39.0),
        "far_right_baseline_corner": (230.0, 39.0),
        "near_right_baseline_corner": (275.0, 205.0),
        "near_left_baseline_corner": (45.0, 205.0),
    }
    analysis = analyze_court_surface_color(
        frame,
        model_landmarks,
        CourtSurfaceSettings(color_distance_threshold=18.0),
    )

    result = refine_surface_boundaries(analysis, model_landmarks)

    assert 40 <= result.far_row <= 42
    assert result.far_span[0] >= 65
    assert result.far_span[1] <= 255
