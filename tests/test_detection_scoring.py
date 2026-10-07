import numpy as np
import pytest

from courtvision.detectors.base import Detection
from courtvision.evaluation.detection import (
    box_iou,
    load_detections,
    match_boxes,
    save_detections,
    score_detections,
)
from courtvision.evaluation.labels import PlayerBox, VideoLabels, segments_from_cuts

COURT_POINTS = [[400, 250], [880, 250], [1100, 550], [180, 550]]


def person(x1, y1, x2, y2, confidence=0.9, frame=25) -> Detection:
    return Detection(frame, "person", confidence, x1, y1, x2, y2, "test")


def test_iou_of_boxes() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)

    assert box_iou(a, b)[0] == pytest.approx([1.0, 50 / 150, 0.0])


def test_each_labelled_box_takes_one_detection() -> None:
    labelled = [PlayerBox(0, 0, 10, 20, player=1)]
    detected = [person(1, 0, 11, 20), person(0, 1, 10, 21), person(50, 0, 60, 20)]

    assert match_boxes(labelled, detected) == [(0, 1)]  # IoU 0.90 beats 0.82
    assert match_boxes(labelled, [person(6, 0, 16, 20)]) == []  # IoU 0.25


def test_detections_are_scored_as_the_pipeline_keeps_them() -> None:
    labels = make_labels()
    detections = [
        person(600, 300, 660, 420),  # player 1, feet on the court
        person(20, 30, 60, 120),  # spectator: feet off the court
        person(300, 300, 340, 400, confidence=0.5),  # on court, but not a player
    ]

    pipeline = score_detections(labels, detections)
    everything = score_detections(labels, detections, as_the_pipeline=False)

    # Player 2 stands outside the court: labelled, so a miss for both.
    assert (pipeline.hits, pipeline.predicted, pipeline.labelled) == (1, 2, 2)
    assert (everything.hits, everything.predicted, everything.labelled) == (1, 3, 2)


def test_ball_and_court_boxes_are_not_player_detections() -> None:
    labels = make_labels()
    court = Detection(25, "court", 0.9, 400, 250, 880, 420, "test")

    assert score_detections(labels, [court], as_the_pipeline=False).predicted == 0


def test_only_the_four_most_confident_players_are_kept() -> None:
    labels = make_labels()
    on_court = [
        person(500 + 30 * i, 300, 520 + 30 * i, 420, 0.3 + 0.1 * i) for i in range(5)
    ]

    assert score_detections(labels, on_court).predicted == 4


def test_detections_survive_a_save_and_load(tmp_path) -> None:
    detections = [person(1.5, 2, 3, 4)]
    path = tmp_path / "detections.json"

    save_detections(path, detections)

    assert load_detections(path) == detections


def make_labels() -> VideoLabels:
    """One labelled segment with a court and one finished box frame, 25."""
    labels = VideoLabels(
        video="clip.mp4",
        source_url="",
        fps=25.0,
        width=1280,
        height=720,
        frame_count=100,
        segments=segments_from_cuts([], 100),
    )
    (segment,) = labels.segments
    segment.status, segment.rally_start, segment.rally_end = "label", 0, 99
    segment.court_points = COURT_POINTS
    segment.boxes[25] = [
        PlayerBox(600, 300, 660, 420, player=1),
        PlayerBox(20, 600, 60, 700, player=2),  # outside the court, near the camera
    ]
    segment.reviewed_box_frames = [25]
    return labels
