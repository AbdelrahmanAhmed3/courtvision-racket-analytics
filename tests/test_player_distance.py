from courtvision.analytics.court_coordinates import CourtCoordinate
from courtvision.analytics.player_distance import PlayerDistanceTracker


def make_player_point(track_id: int, x: float, y: float) -> CourtCoordinate:
    return CourtCoordinate(
        frame=0,
        track_id=track_id,
        object_type="player",
        image_x=x,
        image_y=y,
        template_x=x,
        template_y=y,
        court_x_m=x,
        court_y_m=y,
        in_bounds=True,
        confidence=0.9,
    )


def test_player_distance_accumulates_projected_metres() -> None:
    tracker = PlayerDistanceTracker(
        smoothing_alpha=1.0,
        max_speed_mps=20.0,
        max_gap_seconds=2.0,
    )

    tracker.update([make_player_point(4, 0.0, 0.0)], 0.0)
    tracker.update([make_player_point(4, 3.0, 4.0)], 1.0)

    assert tracker.distances_m == {4: 5.0}


def test_player_distance_rejects_implausible_projection_jump() -> None:
    tracker = PlayerDistanceTracker(smoothing_alpha=1.0, max_speed_mps=10.0)

    tracker.update([make_player_point(4, 0.0, 0.0)], 0.0)
    tracker.update([make_player_point(4, 30.0, 0.0)], 1.0)

    assert tracker.distances_m == {4: 0.0}


def test_player_distance_resumes_after_a_rejected_jump() -> None:
    tracker = PlayerDistanceTracker(smoothing_alpha=1.0, max_speed_mps=12.0)

    tracker.update([make_player_point(4, 0.0, 0.0)], 0.0)
    tracker.update([make_player_point(4, 0.5, 0.0)], 0.1)
    tracker.update([make_player_point(4, 3.0, 0.0)], 0.2)
    tracker.update([make_player_point(4, 3.5, 0.0)], 0.3)
    tracker.update([make_player_point(4, 4.0, 0.0)], 0.4)

    assert tracker.distances_m == {4: 1.5}
