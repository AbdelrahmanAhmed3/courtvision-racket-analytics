from courtvision.analytics.ball_events import BallEvent, Shot
from courtvision.analytics.rallies import Rally, segment_rallies

FPS = 25.0


def event(kind, frame, track_id=None) -> BallEvent:
    return BallEvent(kind, frame, track_id)


def shot(frame) -> Shot:
    return Shot(frame, 1, None, None, None, None, None, None)


def test_a_long_silence_ends_a_rally_and_each_starts_at_its_serve() -> None:
    first = [
        event("bounce", 0),  # the server bounces the ball before serving
        event("impact", 30, 1),  # the serve
        event("bounce", 48),
        event("impact", 56, 3),
        event("wall_rebound", 90),  # the point ends at the glass
    ]
    second = [event("impact", 400, 2), event("bounce", 420), event("impact", 430, 4)]

    rallies = segment_rallies(
        first + second, FPS, shots=[shot(30), shot(56), shot(400)]
    )

    assert rallies == [
        Rally(30, 90, (shot(30), shot(56))),
        Rally(400, 430, (shot(400),)),
    ]


def test_a_lone_impact_is_not_a_rally() -> None:
    events = [event("bounce", 10), event("impact", 20, 1), event("bounce", 40)]

    assert segment_rallies(events, FPS) == []


def test_a_short_pause_inside_a_rally_does_not_split_it() -> None:
    # A lob: 4 s without a detected event, under the 5 s limit.
    events = [event("impact", 0, 1), event("impact", 100, 3), event("impact", 120, 1)]

    assert segment_rallies(events, FPS) == [Rally(0, 120, ())]
