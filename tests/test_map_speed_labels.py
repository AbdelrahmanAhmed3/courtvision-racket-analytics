import numpy as np

from courtvision.analytics.ball_events import Shot
from courtvision.analytics.movement import MovementStats
from courtvision.analytics.report import build_report
from scripts.run_full_pipeline import map_speed_labels


def stats() -> MovementStats:
    return MovementStats(10.0, 20.0, 7.2, 12.0, 0.0, 0.0, 0.0, np.zeros((20, 10)))


def shot(frame, hitter, receiver, speed=50.0, distance=10.0, received=None) -> Shot:
    """A shot; one with no receiver is a winner, timed to its bounce."""
    if received is None and receiver is not None:
        received = frame + 30
    return Shot(frame, hitter, receiver, received, None, distance, 1.0, speed)


def test_a_shot_to_the_hitters_own_team_gets_no_speed_label() -> None:
    teams = {1: "near", 2: "far", 3: "near", 4: "far"}
    shots = [
        shot(10, 1, 2),  # across the net
        shot(40, 2, 4),  # to a partner: one of the impacts is false
        shot(70, 4, 4),  # to themself
        shot(100, 3, None),  # no receiver
        shot(130, 1, 2, speed=None),  # no speed
        shot(160, 2, 1, distance=24.2),  # longer than the court: a wrong pair
    ]

    labels = map_speed_labels(shots, teams)

    assert [label.frame for label in labels] == [10, 100]
    assert [label.frame for label in map_speed_labels(shots)] == [10, 40, 70, 100]


def test_the_fastest_shot_skips_speeds_from_wrong_pairs() -> None:
    teams = {1: "near", 2: "far", 3: "near"}
    shots = [
        shot(10, 1, 2, speed=60.0),
        shot(40, 2, 1, speed=232.0, distance=24.2),  # longer than the court
        shot(70, 1, 3, speed=150.0),  # to a partner
        shot(100, 1, None, speed=180.0, received=110),  # next impact has no player
    ]
    movement = {track_id: stats() for track_id in teams}

    report = build_report("clip.mp4", 25.0, 250, teams, movement, shots)

    assert report["highlights"]["fastest_shot"]["approx_court_speed_kmh"] == 60.0
