import numpy as np
import pytest

from courtvision.analytics.movement import movement_stats

FPS = 25.0


def walk(track_id, start_frame, seconds, start, velocity, jitter=0.0):
    """Positions moving at a constant velocity (m/s), optionally wobbling ±jitter."""
    frames = round(seconds * FPS)
    return [
        (
            track_id,
            start_frame + i,
            start[0] + velocity[0] * i / FPS + (jitter if i % 2 else -jitter),
            start[1] + velocity[1] * i / FPS,
        )
        for i in range(frames + 1)
    ]


def test_a_steady_run_gives_its_distance_and_speeds() -> None:
    # 10 m along the court in 5 s: 2 m/s, 7.2 km/h.
    (stats,) = movement_stats(walk(1, 0, 5, (5.0, 4.0), (0.0, 2.0)), FPS).values()

    assert stats.distance_m == pytest.approx(10.0, rel=0.01)
    assert stats.average_speed_kmh == pytest.approx(7.2, rel=0.03)
    assert stats.top_speed_kmh == pytest.approx(7.2, rel=0.01)


def test_wobble_from_tracking_does_not_inflate_the_distance() -> None:
    # The same run, with the feet position wobbling ±15 cm every frame: summing
    # raw steps would give about 39 m.
    positions = walk(1, 0, 5, (5.0, 4.0), (0.0, 2.0), jitter=0.15)

    (stats,) = movement_stats(positions, FPS).values()

    assert stats.distance_m == pytest.approx(10.0, rel=0.1)
    assert stats.top_speed_kmh < 9.0


def test_a_sprint_sets_the_top_speed() -> None:
    # Walking at 1 m/s, a 1 s sprint at 6 m/s (21.6 km/h), walking again.
    slow1 = walk(1, 0, 2, (2.0, 12.0), (1.0, 0.0))
    sprint = walk(1, 51, 1, (4.04, 12.0), (6.0, 0.0))
    slow2 = walk(1, 77, 2, (10.08, 12.0), (-1.0, 0.0))

    (stats,) = movement_stats(slow1 + sprint + slow2, FPS).values()

    assert stats.top_speed_kmh == pytest.approx(21.6, rel=0.05)


def test_time_at_the_net_and_at_the_back() -> None:
    # 2 s standing 2 m from the net, lost for 1 s, then 3 s behind the service
    # line (8.5 m from the net).
    at_net = walk(3, 0, 2, (5.0, 8.0), (0.0, 0.0))
    at_back = walk(3, 76, 3, (5.0, 1.5), (0.0, 0.0))

    (stats,) = movement_stats(at_net + at_back, FPS).values()

    assert stats.seconds_at_net == pytest.approx(2.04, abs=0.01)
    assert stats.seconds_at_back == pytest.approx(3.04, abs=0.01)
    assert stats.seconds_in_middle == 0.0
    assert stats.distance_m == 0.0  # the jump while lost is not running


def test_the_heatmap_counts_seconds_in_each_square_metre() -> None:
    still = walk(2, 0, 4, (2.5, 15.5), (0.0, 0.0))

    (stats,) = movement_stats(still, FPS).values()

    assert stats.heatmap_seconds[15, 2] == pytest.approx(4.04)
    assert stats.heatmap_seconds.sum() == pytest.approx(4.04)
    assert np.count_nonzero(stats.heatmap_seconds) == 1


def test_each_player_gets_their_own_stats() -> None:
    positions = walk(1, 0, 1, (1.0, 1.0), (1.0, 0.0)) + walk(
        2, 0, 1, (1.0, 19.0), (0.0, 0.0)
    )

    stats = movement_stats(positions, FPS)

    assert stats[1].distance_m == pytest.approx(1.0, rel=0.02)
    assert stats[2].distance_m == 0.0


def test_an_id_swap_is_not_a_sprint() -> None:
    # Walking at 1 m/s, the track suddenly lands on the other far player 7 m
    # away (an ID swap), then walks on from there.
    before = walk(4, 0, 2, (0.5, 3.0), (1.0, 0.0))
    after = walk(4, 51, 2, (7.5, 7.5), (-1.0, 0.0))

    (stats,) = movement_stats(before + after, FPS).values()

    assert stats.distance_m == pytest.approx(4.0, rel=0.05)
    assert stats.top_speed_kmh == pytest.approx(3.6, rel=0.05)
