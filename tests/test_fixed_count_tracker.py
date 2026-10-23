from courtvision.detectors.base import Detection
from courtvision.tracking.fixed_count_tracker import FixedCountTracker


def person(x, y=100.0, width=60.0, height=160.0) -> Detection:
    return Detection(0, "person", 0.9, x, y, x + width, y + height, "test")


def ids(tracked) -> dict[float, int]:
    """Track ID by the box's left edge."""
    return {t.detection.x1: t.track_id for t in tracked}


def test_a_box_that_changes_shape_keeps_its_id() -> None:
    tracker = FixedCountTracker()
    first = tracker.update([person(100), person(500)], timestamp_seconds=0.0)
    # The left player crouches for a smash: the box shrinks and moves, no overlap.
    second = tracker.update(
        [person(90, y=200, height=60), person(502)], timestamp_seconds=0.04
    )

    assert ids(first) == {100: 1, 500: 2}
    assert ids(second) == {90: 1, 502: 2}


def test_a_hidden_player_comes_back_with_the_nearest_missing_id() -> None:
    tracker = FixedCountTracker()
    tracker.update([person(x) for x in (100, 400, 700, 1000)], timestamp_seconds=0.0)
    tracker.update([person(x) for x in (100, 400, 1000)], timestamp_seconds=0.04)
    for t in range(2, 50):  # player 3 stays hidden for two seconds
        tracker.update(
            [person(x) for x in (100, 400, 1000)], timestamp_seconds=t * 0.04
        )

    back = tracker.update(
        [person(x) for x in (100, 400, 820, 1000)], timestamp_seconds=2.0
    )

    assert ids(back) == {100: 1, 400: 2, 820: 3, 1000: 4}


def test_no_more_ids_than_players() -> None:
    tracker = FixedCountTracker(max_players=4)
    seen = set()
    for step in range(20):  # people appear in a new place every frame
        tracked = tracker.update(
            [person(50 + 300 * ((step + k) % 4) + step) for k in range(2)],
            timestamp_seconds=step * 0.04,
        )
        seen |= {t.track_id for t in tracked}

    assert seen <= {1, 2, 3, 4}
