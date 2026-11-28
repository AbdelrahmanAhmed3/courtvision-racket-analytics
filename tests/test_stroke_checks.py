"""A labelled stroke must fit the bounces before it, so n can step to doubtful ones."""

from courtvision.evaluation.checks import find_problems
from courtvision.evaluation.labels import BallEvent, VideoLabels
from tests.test_labelling import make_labels

# The service-line ends in a 1280x720 picture: the far line at y 200, the near at 500.
COURT = [[400.0, 200.0], [880.0, 200.0], [1080.0, 500.0], [200.0, 500.0]]
NEAR = (640.0, 600.0)  # a floor point in the near half
FAR = (640.0, 150.0)  # and in the far half


def rally(*events: BallEvent) -> VideoLabels:
    """One labelled segment, 0-99, whose rally runs the whole segment."""
    labels = make_labels(box_interval=None)
    segment = labels.segments[0]
    segment.status = "label"
    segment.rally_start, segment.rally_end = 0, 99
    segment.court_points = COURT
    segment.reviewed_box_frames = list(labels.box_frames(segment))  # not tested here
    segment.events = list(events)
    return labels


def problems(labels: VideoLabels) -> list[tuple[int, str]]:
    return [(p.frame, p.text) for p in find_problems(labels, labels.segments[0])]


def hit(frame: int, player: int, stroke: str | None = None) -> BallEvent:
    return BallEvent("impact", frame, player=player, stroke=stroke)


def bounce(frame: int, point: tuple[float, float]) -> BallEvent:
    return BallEvent("bounce", frame, x=point[0], y=point[1])


def test_strokes_that_fit_their_bounces_are_no_problem() -> None:
    labels = rally(
        hit(0, 1, "serve"),
        bounce(10, FAR),
        hit(20, 3),
        bounce(30, NEAR),
        hit(40, 2, "backhand"),
        hit(50, 4),
        hit(60, 1, "forehand_volley"),
        BallEvent("hidden_bounce", 70),
        hit(80, 3, "forehand"),
    )
    assert problems(labels) == []


def test_a_volley_after_a_bounce_is_a_problem() -> None:
    labels = rally(hit(0, 3), bounce(10, NEAR), hit(20, 1, "backhand_volley"))
    assert problems(labels) == [
        (
            20,
            "Volley after the bounce at frame 10. "
            "Forehand or backhand, or not a bounce?",
        )
    ]


def test_a_forehand_with_no_bounce_before_it_is_a_problem() -> None:
    labels = rally(hit(0, 3), hit(20, 2, "forehand"))
    assert problems(labels) == [
        (
            20,
            "Forehand with no bounce before it. "
            "A volley, or a bounce out of sight (b twice) missing?",
        )
    ]


def test_a_serve_after_the_first_hit_is_a_problem() -> None:
    labels = rally(hit(0, 3), bounce(10, NEAR), hit(20, 1, "serve"))
    assert problems(labels) == [(20, "A serve that is not the rally's first hit")]


def test_a_hit_after_a_bounce_on_the_other_side_names_the_wrong_player() -> None:
    labels = rally(hit(0, 3), bounce(10, FAR), hit(20, 1))
    assert problems(labels) == [
        (20, "P1 hits after the ball bounced on the far side. Wrong player?")
    ]


def test_a_bounce_out_of_sight_leaves_the_side_unknown() -> None:
    labels = rally(
        hit(0, 3), bounce(10, FAR), BallEvent("hidden_bounce", 12), hit(20, 1)
    )
    sides = [text for _, text in problems(labels) if "side" in text]
    # Two bounces end the point, which the older rule reports; the side is unknown.
    assert sides == []


def at_court_depth(depth: float) -> tuple[float, float]:
    """The image point on the middle of the court, ``depth`` from the far baseline."""
    import cv2
    import numpy as np

    from courtvision.evaluation.court import court_estimate

    estimate = court_estimate(COURT)
    point = np.float32([[[0.5, depth]]])
    x, y = cv2.perspectiveTransform(point, estimate.template_to_image)[0, 0]
    return float(x), float(y)


def test_a_bounce_beside_the_net_is_on_no_side() -> None:
    just_far = at_court_depth(0.48)  # a few pixels off could be the near half
    labels = rally(hit(0, 4), bounce(10, just_far), hit(20, 2))
    assert problems(labels) == []

    clearly_far = at_court_depth(0.4)
    labels = rally(hit(0, 4), bounce(10, clearly_far), hit(20, 2))
    assert [frame for frame, _ in problems(labels)] == [20]


def fed_rally(*events: BallEvent, rally_start: int) -> VideoLabels:
    """A rally starting mid-segment, as when a coach feeds the first ball."""
    labels = rally(*events)
    segment = labels.segments[0]
    segment.rally_start = rally_start
    segment.reviewed_box_frames = list(labels.box_frames(segment))
    return labels


def test_a_video_the_coach_feeds_needs_no_serve() -> None:
    # The coach's feed is not labelled, so the rally starts before the first hit,
    # and the fed ball can be volleyed straight back.
    labels = fed_rally(hit(12, 3, "forehand"), hit(20, 1), rally_start=10)
    assert [text for _, text in problems(labels)] == [
        "The rally starts mid-segment but not on an impact (serve)"
    ]
    labels.coach_feeds = True
    assert problems(labels) == []


def test_a_serve_in_a_video_the_coach_feeds_is_a_problem() -> None:
    labels = fed_rally(hit(10, 1, "serve"), bounce(15, FAR), hit(20, 3), rally_start=10)
    labels.coach_feeds = True
    assert problems(labels) == [
        (10, "A serve in a video the coach feeds. Not a serve?")
    ]


def test_a_far_players_glass_needs_a_wall_rebound() -> None:
    labels = rally(hit(0, 1), bounce(10, FAR), hit(20, 3, "glass"))
    assert problems(labels) == [
        (20, "Glass with no wall rebound before it. Missed rebound, or another stroke?")
    ]
    labels.segments[0].events.append(BallEvent("wall_rebound", 15, x=640, y=100))
    assert problems(labels) == []
    # The near back glass is behind the camera, so near players aren't checked.
    labels = rally(hit(0, 3), bounce(10, NEAR), hit(20, 1, "bajada"))
    assert problems(labels) == []


def test_a_bounce_on_the_frame_of_a_hit_is_a_problem() -> None:
    labels = rally(
        hit(0, 3), BallEvent("hidden_bounce", 20), hit(20, 1, "smash"), hit(40, 4)
    )
    assert problems(labels) == [
        (20, "A hit and a hidden bounce on the same frame. Move one?")
    ]


def test_a_hit_before_the_rally_start_doesnt_make_the_serve_a_second_one() -> None:
    labels = fed_rally(
        hit(2, 3), hit(5, 1, "serve"), bounce(10, FAR), hit(20, 3), rally_start=5
    )
    assert problems(labels) == [(2, "impact by P3 is outside the rally")]
