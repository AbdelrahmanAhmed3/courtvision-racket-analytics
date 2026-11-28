"""The stroke of each near-team hit, labelled with t in the labelling tool (#80)."""

import json

import pytest

from courtvision.evaluation.labels import (
    STROKE_GROUPS,
    BallEvent,
    load_labels,
    save_labels,
)
from courtvision.evaluation.session import STROKE_KEYS, LabelSession
from tests.test_labelling import make_labels


def near_team_hits() -> LabelSession:
    """Two labelled segments (0-99, 100-199) at 25 fps; hits by P1, P3, P2, P2."""
    labels = make_labels()
    first, second = labels.segments
    first.status = second.status = "label"
    first.events = [
        BallEvent("impact", 30, player=1),
        BallEvent("impact", 50, player=3),
        BallEvent("bounce", 55, x=600.0, y=400.0),
        BallEvent("impact", 70, player=2),
    ]
    second.events = [BallEvent("impact", 120, player=2)]
    return LabelSession(labels, frame=0)


def stroke_at(session: LabelSession, frame: int) -> str | None:
    events = session.labels.segment_at(frame).events
    return next(e.stroke for e in events if e.kind == "impact" and e.frame == frame)


def test_t_asks_for_each_near_team_hit_from_just_before_it() -> None:
    session = near_team_hits()

    session.key("t")
    assert session.stroke_for == 30
    assert session.frame == 18  # half a second (12 frames at 25 fps) before
    session.key("f")
    assert stroke_at(session, 30) == "forehand"
    assert session.stroke_for == 70  # the far player's hit at 50 is passed by
    session.key("j")
    assert session.stroke_for == 120 and session.frame == 108
    session.key("m")

    assert [stroke_at(session, f) for f in (30, 50, 70, 120)] == [
        "forehand",
        None,
        "bandeja",
        "smash",
    ]
    assert session.stroke_for is None
    assert "Every near-team hit has a stroke" in session.message


def test_t_skips_a_hit_and_t_on_a_hit_changes_its_stroke() -> None:
    session = near_team_hits()
    session.key("t")
    session.key("t")
    assert session.stroke_for == 70

    session.go_to(30)
    session.key("t")
    assert session.stroke_for == 30 and session.frame == 30
    session.key("b")
    assert stroke_at(session, 30) == "backhand"
    session.go_to(30)
    session.key("t")
    assert "now backhand" in session.message
    session.key("f")

    assert stroke_at(session, 30) == "forehand"


def test_undo_goes_back_to_the_hit_and_x_clears_its_stroke() -> None:
    session = near_team_hits()
    session.key("t")
    session.key("v")
    assert session.stroke_for == 70

    session.key("u")
    assert stroke_at(session, 30) is None
    assert session.stroke_for == 30 and session.frame == 18
    session.key("c")
    assert stroke_at(session, 30) == "backhand_volley"
    session.go_to(30)
    session.key("t")
    session.key("x")

    assert stroke_at(session, 30) is None


def test_after_esc_the_stroke_keys_do_their_usual_work() -> None:
    session = near_team_hits()
    session.key("t")
    assert session.key_help()[:2] == (("s", "serve"), ("f", "forehand (ground stroke)"))

    session.key("esc")
    session.key("o")  # skips the segment, instead of labelling a drop shot

    assert session.stroke_for is None
    assert session.segment.status == "skip"
    assert stroke_at(session, 30) is None
    assert session.key_help()[0] == ("a / d", "previous / next frame")


def test_a_new_hitter_keeps_the_stroke() -> None:
    session = near_team_hits()
    session.key("t")
    session.key("l")
    session.key("esc")
    session.go_to(30)

    session.key("2")
    session.key("2")

    events = session.labels.segment_at(30).events
    assert BallEvent("impact", 30, player=2, stroke="lob") in events


def test_the_checklist_counts_near_team_hits_with_a_stroke() -> None:
    session = near_team_hits()
    session.key("t")
    session.key("s")

    texts = [item.text for item in session.checklist()]

    assert "Strokes: 1 of 3 near-team hits (t)" in texts
    assert "Stroke of P2's hit at frame 70: none yet" in texts


def test_strokes_are_saved_and_older_files_load_without_them(tmp_path) -> None:
    labels = near_team_hits().labels
    labels.segments[0].events[0].stroke = "vibora"
    path = tmp_path / "labels.json"
    save_labels(labels, path)
    assert load_labels(path) == labels

    data = json.loads(path.read_text())
    data["schema_version"] = 3
    for segment in data["segments"]:
        for event in segment["events"]:
            del event["stroke"]
    path.write_text(json.dumps(data))

    loaded = load_labels(path)
    assert all(e.stroke is None for s in loaded.segments for e in s.events)


def test_a_stroke_is_known_and_only_on_an_impact() -> None:
    with pytest.raises(ValueError, match="Unknown stroke"):
        BallEvent("impact", 3, player=1, stroke="tweener")
    with pytest.raises(ValueError, match="Only an impact"):
        BallEvent("bounce", 3, x=1.0, y=2.0, stroke="lob")


def test_every_stroke_has_one_key_that_leaves_frame_stepping_free() -> None:
    keys = [key for key, _ in STROKE_KEYS]

    assert sorted(stroke for _, stroke in STROKE_KEYS) == sorted(STROKE_GROUPS)
    assert len(set(keys)) == len(keys)
    assert not set(keys) & {"a", "d", "t", "x", "u", "esc", "space", "1", "2"}


def test_far_strokes_asks_for_every_players_hits() -> None:
    labels = near_team_hits().labels
    session = LabelSession(labels, stroke_players=(1, 2, 3, 4))

    session.key("t")
    session.key("f")

    assert session.stroke_for == 50  # the far player's hit is asked for too
    texts = [item.text for item in session.checklist()]
    assert "Strokes: 1 of 4 hits (t)" in texts


def test_h_marks_a_video_the_coach_feeds_and_t_then_refuses_a_serve(tmp_path) -> None:
    session = near_team_hits()
    session.key("h")
    assert session.labels.coach_feeds
    path = tmp_path / "labels.json"
    save_labels(session.labels, path)
    assert load_labels(path).coach_feeds

    session.key("t")
    session.key("s")  # serve
    assert stroke_at(session, 30) is None
    assert session.stroke_for == 30
    assert "No serves" in session.message
    session.key("f")
    assert stroke_at(session, 30) == "forehand"

    session.key("esc")
    session.key("h")
    assert not session.labels.coach_feeds
