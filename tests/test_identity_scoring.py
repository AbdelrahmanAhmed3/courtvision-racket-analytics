import pytest

from courtvision.detectors.base import Detection
from courtvision.evaluation.identity import score_identities
from courtvision.evaluation.labels import PlayerBox, VideoLabels, segments_from_cuts

LEFT, RIGHT = (100, 100, 160, 260), (500, 100, 560, 260)


def box(coordinates) -> Detection:
    return Detection(0, "person", 0.9, *coordinates, "test")


def labels_with(frames: dict[int, list[tuple[tuple, int]]]) -> VideoLabels:
    labels = VideoLabels(
        video="clip.mp4",
        source_url="",
        fps=25.0,
        width=1280,
        height=720,
        frame_count=1000,
        segments=segments_from_cuts([], 1000),
    )
    (segment,) = labels.segments
    segment.status, segment.rally_start, segment.rally_end = "label", 0, 999
    for frame, players in frames.items():
        segment.boxes[frame] = [PlayerBox(*xyxy, player=p) for xyxy, p in players]
        segment.reviewed_box_frames.append(frame)
    return labels


def test_a_tracker_that_keeps_both_players_scores_one() -> None:
    labels = labels_with({f: [(LEFT, 1), (RIGHT, 2)] for f in (25, 50, 75)})
    tracks = {0: {f: [(7, box(LEFT)), (9, box(RIGHT))] for f in (25, 50, 75)}}

    score = score_identities(labels, tracks)

    assert score.idf1 == 1.0
    assert (score.switches, score.ids_per_segment) == (0, (2,))


def test_a_new_id_after_a_hidden_moment_is_a_switch() -> None:
    labels = labels_with({f: [(LEFT, 1), (RIGHT, 2)] for f in (25, 50, 75, 100)})
    left_ids = {25: 7, 50: 7, 75: 12, 100: 12}  # player 1 comes back as ID 12
    tracks = {0: {f: [(left_ids[f], box(LEFT)), (9, box(RIGHT))] for f in left_ids}}

    score = score_identities(labels, tracks)

    # Player 1 is paired with one of its two IDs for 2 of its 4 frames.
    assert score.id_hits == 6
    assert score.idf1 == pytest.approx(2 * 6 / (8 + 8))
    assert (score.switches, score.ids_per_segment) == (1, (3,))


def test_players_are_paired_to_maximise_shared_frames_not_pairs() -> None:
    # Player 1 has ID 7 on 10 frames and ID 8 once; player 2 has ID 7 once.
    frames = {25 * i: [(LEFT, 1)] for i in range(1, 12)}
    frames[300] = [(RIGHT, 2)]
    ids = {25 * i: 7 for i in range(1, 11)} | {275: 8, 300: 7}
    tracks = {0: {f: [(ids[f], box(frames[f][0][0]))] for f in frames}}

    score = score_identities(labels_with(frames), tracks)

    # Player 1 keeps ID 7 (10 frames). Pairing for the most pairs instead would
    # give player 1 ID 8 and player 2 ID 7: 2 frames.
    assert score.id_hits == 10


def test_team_scoring_checks_each_box_against_its_players_team() -> None:
    from courtvision.evaluation.identity import score_teams

    labels = labels_with({f: [(LEFT, 1), (RIGHT, 3)] for f in (25, 50)})
    tracks = {0: {f: [(7, box(LEFT)), (9, box(RIGHT))] for f in (25, 50)}}

    # Track 9 is player 3, a far-team player, but was put on the near team.
    assert score_teams(labels, tracks, {0: {7: "near", 9: "near"}}) == (2, 4)
