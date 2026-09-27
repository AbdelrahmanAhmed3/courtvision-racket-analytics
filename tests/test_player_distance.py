import numpy as np

from courtvision.analytics.court_coordinates import CourtCoordinate
from courtvision.analytics.player_distance import PlayerDistanceTracker
from courtvision.visualization.minimap import draw_shot_speed


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


def test_shot_speed_label_draws_on_map() -> None:
    image = np.zeros((900, 500, 3), dtype=np.uint8)

    draw_shot_speed(
        image,
        event_type="transit",
        hitter_id=2,
        receiver_id=3,
        speed_kmh=84.5,
    )

    assert image.any()
