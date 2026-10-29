from courtvision.analytics.ball_events import Shot
from scripts.run_full_pipeline import map_speed_labels


def shot(frame, hitter, receiver, speed=50.0) -> Shot:
    return Shot(frame, hitter, receiver, frame + 30, None, 10.0, 1.0, speed)


def test_a_shot_to_the_hitters_own_team_gets_no_speed_label() -> None:
    teams = {1: "near", 2: "far", 3: "near", 4: "far"}
    shots = [
        shot(10, 1, 2),  # across the net
        shot(40, 2, 4),  # to a partner: one of the impacts is false
        shot(70, 4, 4),  # to themself
        shot(100, 3, None),  # no receiver
        shot(130, 1, 2, speed=None),  # no speed
    ]

    labels = map_speed_labels(shots, teams)

    assert [label.frame for label in labels] == [10, 100]
    assert [label.frame for label in map_speed_labels(shots)] == [10, 40, 70, 100]
