import json

import pytest

from courtvision.evaluation.labels import BallEvent, VideoLabels, segments_from_cuts
from courtvision.evaluation.scoring import (
    PredictedEvent,
    load_predictions,
    match_frames,
    predictions_from_shots,
    save_predictions,
    score_events,
)


def test_each_label_matches_one_prediction_within_the_tolerance() -> None:
    assert match_frames([100, 108], [105, 110], tolerance=5) == [(100, 105), (108, 110)]
    assert match_frames([100, 104], [102], tolerance=5) == [(100, 102)]
    assert match_frames([100], [106], tolerance=5) == []
    assert match_frames([], [100], tolerance=5) == []


def test_the_nearest_prediction_is_not_always_the_right_one() -> None:
    # Label 100 is nearer to 103, but taking it would leave 108 with nothing.
    assert match_frames([100, 108], [96, 103], tolerance=5) == [(100, 96), (108, 103)]


def test_repeated_predictions_of_one_event_are_false_alarms() -> None:
    labels = make_labels()

    scores = score_events(
        labels, [PredictedEvent("impact", frame) for frame in (20, 21, 22)], tolerance=2
    )

    impact = scores["impact"]
    assert (impact.hits, impact.predicted, impact.labelled) == (1, 3, 2)
    assert impact.precision == pytest.approx(1 / 3)
    assert impact.recall == pytest.approx(1 / 2)
    assert impact.frame_errors == (0,)  # the exact prediction is the one paired


def test_predictions_outside_labelled_rallies_are_ignored() -> None:
    labels = make_labels()
    predictions = [
        PredictedEvent("bounce", 9),  # just before the rally starts: within tolerance
        PredictedEvent("bounce", 5),  # before the rally
        PredictedEvent("bounce", 150),  # in a skipped segment
    ]

    bounce = score_events(labels, predictions, tolerance=2)["bounce"]

    assert (bounce.hits, bounce.predicted, bounce.labelled) == (0, 1, 1)


def test_predictions_files_name_known_event_kinds(tmp_path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"events": [{"kind": "impact", "frame": 12}]}))
    assert load_predictions(path) == [PredictedEvent("impact", 12)]

    path.write_text(json.dumps({"events": [{"kind": "smash", "frame": 12}]}))
    with pytest.raises(ValueError, match="smash"):
        load_predictions(path)


def make_labels() -> VideoLabels:
    """Segment 0-100 holds a rally from 10 to 80; segment 100-200 is skipped."""
    labels = VideoLabels(
        video="clip.mp4",
        source_url="",
        fps=25.0,
        width=1280,
        height=720,
        frame_count=200,
        segments=segments_from_cuts([100], 200),
    )
    rally, skipped = labels.segments
    rally.status, rally.rally_start, rally.rally_end = "label", 10, 80
    rally.events = [
        BallEvent("impact", 10, player=1),
        BallEvent("bounce", 30, x=600.0, y=400.0),
        BallEvent("impact", 21, player=3),
    ]
    skipped.status = "skip"
    return labels


def test_shots_become_impacts_and_bounces(tmp_path) -> None:
    from courtvision.analytics.shots import ShotEvent

    def shot(frame, receive_frame, bounce_frame):
        return ShotEvent("transit", frame, 1, 3, receive_frame, bounce_frame, 0, 0, 0)

    predictions = predictions_from_shots([shot(10, 40, 30), shot(40, 70, None)])

    assert predictions == [
        PredictedEvent("impact", 10),
        PredictedEvent("impact", 40),  # the receiver's impact starts the next shot
        PredictedEvent("impact", 70),
        PredictedEvent("bounce", 30),
    ]
    path = tmp_path / "predictions.json"
    save_predictions(path, predictions)
    assert load_predictions(path) == predictions
