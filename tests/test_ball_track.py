from courtvision.analytics.ball_track import (
    Impact,
    detect_impacts,
    event_candidates,
    split_into_tracklets,
)
from courtvision.detectors.base import Detection
from courtvision.detectors.tracknet_adapter import BallPoint


def ball(frame, x, y=300.0) -> BallPoint:
    return BallPoint(frame, x, y)


def frames(tracklets) -> list[list[int]]:
    return [[point.frame for point in tracklet] for tracklet in tracklets]


def test_an_impossible_jump_is_a_tracklet_of_its_own() -> None:
    # The example from the Madrid rally: TrackNet jumps 400 px for one frame.
    points = [ball(1, 667), ball(2, 1065), ball(3, 693), ball(4, 695), ball(5, 689)]

    assert frames(split_into_tracklets(points)) == [[1], [2], [3, 4, 5]]


def test_a_gap_ends_a_tracklet_and_short_gaps_allow_longer_moves() -> None:
    points = [
        ball(1, 100),
        BallPoint(2, None, None, 0.0),
        ball(3, 340),  # 240 px in 2 frames: possible
        ball(7, 360),  # 4 frames after the last: lost in between
    ]

    assert frames(split_into_tracklets(points)) == [[1, 3], [7]]


def test_tracklet_ends_next_to_a_player_are_one_impact() -> None:
    player = [(7, Detection(0, "person", 0.9, 600, 200, 680, 400, "test"))]
    players = {frame: player for frame in range(30)}
    incoming = [ball(f, 300 + 30 * f) for f in range(10)]  # reaches the player
    outgoing = [ball(f, 570 - 30 * (f - 12)) for f in range(12, 20)]  # hit back
    blip = [ball(25, 640)]  # one wrong position on the player: too short

    impacts = detect_impacts([*incoming, *outgoing, *blip], players)

    # Frame 9 ends the incoming tracklet next to the player and frame 12 starts
    # the outgoing one; together they are one impact by track 7. The first and
    # last positions are far from the player and are not impacts.
    assert impacts == [Impact(frame=10, track_id=7)]


def test_event_candidates_are_hidden_gaps_free_ends_and_sharp_turns() -> None:
    before_hit = [ball(f, 100 + 20 * f) for f in range(0, 6)]  # ends at frame 5
    after_hit = [ball(f, 300 - 20 * (f - 11)) for f in range(11, 16)]  # hidden 6 frames
    # Much later the ball bounces: moving down the image, then up, without a gap.
    bounce = [ball(f, 500, 200 + 15 * abs(f - 43)) for f in range(40, 47)]

    candidates = event_candidates([*before_hit, *after_hit, *bounce])

    # 0: the first tracklet starts; 10: where the paths before and after the
    # hidden hit meet (200 + 20 per frame from frame 5 meets 300 - 20 per frame
    # back from frame 11 at 10.5); 15 and 40: tracklet ends far from the next;
    # 43: the bounce; 46: the end.
    assert candidates == [0, 10, 15, 40, 43, 46]


def test_a_hidden_event_is_placed_where_the_paths_meet() -> None:
    from courtvision.analytics.ball_track import meeting_frame

    # The ball comes in fast and leaves slowly, hidden from frame 4 to 12. The
    # paths meet at x = 300 on frame 10 (the middle of the gap would be 8).
    before = [ball(f, 100 + 20 * f) for f in range(0, 4)]  # reaches 300 at 10
    after = [ball(f, 300 + 5 * (f - 10)) for f in range(13, 17)]

    assert meeting_frame(before, after) == 10


def test_a_lone_tracklet_start_is_followed_back_to_the_hitter() -> None:
    # The ball leaves the edge of the player's box at frame 10, but TrackNet
    # first finds it at 14. Followed back, its path reaches the box at 10.
    player = [(3, Detection(0, "person", 0.9, 600, 200, 680, 400, "test"))]
    players = {frame: player for frame in range(30)}
    after_hit = [ball(f, 680 + 40 * (f - 10)) for f in range(14, 20)]

    assert event_candidates(after_hit)[0] == 14
    assert event_candidates(after_hit, players)[0] == 10
