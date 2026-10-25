import pytest

from courtvision.analytics.ball_events import BallEvent, Shot, shots_from_events
from courtvision.detectors.tracknet_adapter import BallPoint
from courtvision.evaluation.court import court_estimate

# Service line ends on a square image grid: x px / 1000 across, y px / 1000 along,
# so 1 px is 1 cm across (10 m wide) and 2 cm along the court (20 m long).
COURT = court_estimate([[0, 150], [1000, 150], [1000, 850], [0, 850]])
FPS = 25.0


def ball_line(start_frame, start, end_frame, end) -> list[BallPoint]:
    """The ball moving in a straight line between two image positions."""
    steps = end_frame - start_frame
    return [
        BallPoint(
            start_frame + i,
            start[0] + (end[0] - start[0]) * i / steps,
            start[1] + (end[1] - start[1]) * i / steps,
        )
        for i in range(steps + 1)
    ]


def impact(frame, track_id) -> BallEvent:
    return BallEvent("impact", frame, track_id)


def bounce(frame) -> BallEvent:
    return BallEvent("bounce", frame, None)


def test_a_serve_bounces_in_the_service_box_before_the_return() -> None:
    # The near server (track 1) hits at frame 0; the ball travels 8 m along the
    # court and bounces at 18; the far receiver (track 3) returns at 26.
    ball = ball_line(0, (500, 700), 26, (500, 300))
    events = [impact(0, 1), bounce(18), impact(26, 3)]

    serve, _ = shots_from_events(events, ball, COURT, FPS)

    assert (serve.hitter_track_id, serve.receiver_track_id) == (1, 3)
    assert (serve.receive_frame, serve.bounce_frame) == (26, 18)
    assert serve.distance_m == pytest.approx(8.0, abs=0.01)
    assert serve.duration_seconds == pytest.approx(26 / FPS)
    assert serve.court_speed_kmh == pytest.approx(8.0 / (26 / FPS) * 3.6, abs=0.1)


def test_a_volley_is_hit_back_before_it_bounces() -> None:
    ball = ball_line(0, (400, 520), 12, (450, 470))
    events = [impact(0, 2), impact(12, 4)]

    volley, _ = shots_from_events(events, ball, COURT, FPS)

    assert (volley.receiver_track_id, volley.bounce_frame) == (4, None)


def test_a_lob_two_and_a_half_seconds_long_is_still_one_exchange() -> None:
    ball = ball_line(0, (300, 400), 62, (600, 800))
    events = [impact(0, 3), bounce(50), impact(62, 1)]

    lob, _ = shots_from_events(events, ball, COURT, FPS)

    assert (lob.receiver_track_id, lob.receive_frame, lob.bounce_frame) == (1, 62, 50)


def test_the_last_shot_of_a_point_has_no_receiver() -> None:
    # The winner bounces at 20; the next impact is a new point 5 s later.
    ball = ball_line(0, (500, 300), 20, (500, 700))
    events = [impact(0, 3), bounce(20), impact(125, 1)]

    winner, next_point = shots_from_events(events, ball, COURT, FPS)

    assert winner == Shot(
        frame=0,
        hitter_track_id=3,
        receiver_track_id=None,
        receive_frame=None,
        bounce_frame=20,
        distance_m=pytest.approx(8.0, abs=0.01),
        duration_seconds=pytest.approx(0.8),
        court_speed_kmh=pytest.approx(36.0, abs=0.1),
    )
    assert next_point.distance_m is None  # no ball seen after it
