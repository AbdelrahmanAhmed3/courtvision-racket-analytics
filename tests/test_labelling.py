import pytest

from courtvision.evaluation.labels import (
    BallEvent,
    PlayerBox,
    VideoLabels,
    load_labels,
    save_labels,
    segments_from_cuts,
)
from courtvision.evaluation.session import LabelSession


def make_labels(cuts=(100,), frame_count=200, box_interval=25) -> VideoLabels:
    return VideoLabels(
        video="clip.mp4",
        source_url="https://example.test/clip",
        fps=25.0,
        width=1280,
        height=720,
        frame_count=frame_count,
        box_interval=box_interval,
        segments=segments_from_cuts(list(cuts), frame_count),
    )


def labelled_session(frame=10) -> LabelSession:
    session = LabelSession(make_labels(), frame=frame)
    session.key("l")
    return session


def test_cuts_split_the_video_into_segments() -> None:
    segments = segments_from_cuts([0, 120, 50, 52, 999, 198], frame_count=200)

    # 52 and 198 would leave segments shorter than 5 frames: flashes, not cuts.
    assert [(segment.start, segment.end) for segment in segments] == [
        (0, 50),
        (50, 120),
        (120, 200),
    ]


def test_labels_survive_a_save_and_load(tmp_path) -> None:
    labels = make_labels()
    segment = labels.segments[0]
    segment.status = "label"
    segment.rally_start = 5
    segment.events.append(BallEvent("impact", 12, player=3))
    segment.events.append(BallEvent("bounce", 20, x=640.0, y=400.0))
    segment.boxes[25] = [PlayerBox(1, 2, 30, 80, player=2)]
    segment.reviewed_box_frames.append(25)
    path = tmp_path / "labels.json"

    save_labels(labels, path)

    assert load_labels(path) == labels


def test_box_frames_follow_the_rally_and_interval() -> None:
    labels = make_labels(cuts=(), frame_count=200)
    segment = labels.segments[0]
    segment.rally_start, segment.rally_end = 30, 100

    assert labels.box_frames(segment) == [50, 75, 100]


def test_an_impact_needs_a_labelled_segment_and_replaces_one_on_the_same_frame():
    session = LabelSession(make_labels(), frame=10)
    session.key("2")
    assert session.segment.events == []

    session.key("l")
    session.key("2")
    session.key("4")

    assert session.segment.events == [BallEvent("impact", 10, player=4)]


def test_bounce_and_wall_rebound_record_the_clicked_ball_position() -> None:
    session = labelled_session()

    session.key("b")
    session.click(640, 400)
    session.key("w")
    session.click(10, 300)

    assert session.segment.events == [
        BallEvent("bounce", 10, x=640, y=400),
        BallEvent("wall_rebound", 10, x=10, y=300),
    ]


def test_boxes_are_prefilled_assigned_and_marked_done() -> None:
    session = labelled_session(frame=25)
    assert session.needs_prefill()
    session.prefill([(0, 0, 50, 100), (200, 0, 260, 100), (205, 10, 215, 30)])

    session.click(210, 20)  # inside two boxes: the smaller one is selected
    session.key("2")
    session.click(10, 10)
    session.key("1")
    session.click(210, 50)
    session.key("1")  # player 1 moves to this box; the first box loses it
    session.key("y")

    assert session.boxes() == [
        PlayerBox(200, 0, 260, 100, player=1),
        PlayerBox(205, 10, 215, 30, player=2),
    ]
    assert 25 in session.segment.reviewed_box_frames
    assert session.next_box_frame() == 50
    session.go_to(90)
    assert session.next_box_frame() == 0  # wraps back to a skipped frame


def test_undo_brings_back_boxes_removed_when_done() -> None:
    session = labelled_session(frame=25)
    session.prefill([(0, 0, 50, 100), (200, 0, 260, 100)])
    session.key("y")
    assert session.boxes() == []

    session.key("u")

    assert len(session.boxes()) == 2


def test_drawing_a_box_needs_a_box_frame() -> None:
    session = labelled_session(frame=26)
    session.drag(0, 0, 40, 90)
    assert session.boxes() == []

    session.go_to(25)
    session.prefill([])
    session.drag(40, 90, 0, 0)

    assert session.boxes() == [PlayerBox(0, 0, 40, 90)]
    assert session.selected == 0


def test_a_cut_moves_later_events_into_a_new_segment() -> None:
    session = labelled_session(frame=10)
    session.key("1")
    session.go_to(60)
    session.key("3")

    session.key("c")

    first, second = session.labels.segments[:2]
    assert (first.start, first.end, second.start, second.end) == (0, 60, 60, 100)
    assert [event.frame for event in first.events] == [10]
    assert [event.frame for event in second.events] == [60]
    assert second.status == "label"


def test_undo_restores_the_previous_labels() -> None:
    session = labelled_session()
    session.key("1")

    session.key("u")

    assert session.segment.events == []


def test_the_rally_cannot_end_before_it_starts() -> None:
    session = labelled_session(frame=50)
    session.key("r")
    session.go_to(40)

    session.key("e")

    assert session.segment.rally_end is None


def test_checklist_reports_what_is_missing() -> None:
    session = labelled_session(frame=25)
    session.prefill([(0, 0, 50, 100)])

    items = {item.text: item.done for item in session.checklist()}

    assert items["Rally start marked (r)"] is False
    assert items["Box frame: 0/1 boxes assigned, press y when done"] is False


def test_an_impact_requires_a_player_from_one_to_four() -> None:
    with pytest.raises(ValueError):
        BallEvent("impact", 3, player=5)


def test_cuts_are_found_where_the_picture_changes() -> None:
    import numpy as np

    from courtvision.evaluation.cuts import detect_cuts

    blue = np.zeros((90, 160, 3), np.uint8)
    blue[..., 0] = 200
    red = np.zeros((90, 160, 3), np.uint8)
    red[..., 2] = 200

    assert detect_cuts([blue, blue, red, red, red, blue]) == [2, 5]


def test_a_cut_mid_rally_moves_every_label_with_its_frame() -> None:
    labels = make_labels(cuts=(), frame_count=200)
    session = LabelSession(labels, frame=10)
    session.key("l")
    session.key("r")
    session.go_to(150)
    session.key("e")
    session.prefill([(0, 0, 40, 90)])
    session.click(10, 10)
    session.key("1")
    session.key("y")

    session.go_to(100)
    session.key("c")

    head, tail = session.labels.segments
    assert (head.rally_start, head.rally_end) == (10, None)
    assert (tail.rally_start, tail.rally_end) == (None, 150)
    assert tail.reviewed_box_frames == [150] and head.reviewed_box_frames == []
    assert labels_box_frames_inside(session.labels)


def labels_box_frames_inside(labels) -> bool:
    return all(
        segment.contains(frame)
        for segment in labels.segments
        for frame in labels.box_frames(segment)
    )


def test_merging_undoes_a_false_cut() -> None:
    session = labelled_session(frame=10)
    session.key("r")
    session.go_to(120)
    session.key("e")  # segment 2 is unreviewed: the rally end needs a label first
    assert session.segment.rally_end is None
    session.key("l")
    session.key("e")

    session.key("m")

    (segment,) = [s for s in session.labels.segments if s.contains(120)]
    assert (segment.start, segment.end) == (0, 200)
    assert (segment.rally_start, segment.rally_end) == (10, 120)


def test_the_rally_cannot_start_after_it_ends() -> None:
    session = labelled_session(frame=40)
    session.key("e")
    session.go_to(60)

    session.key("r")

    assert session.segment.rally_start is None


def test_boxes_done_twice_is_recorded_once() -> None:
    session = labelled_session(frame=25)
    session.prefill([])

    session.key("y")
    session.key("y")

    assert session.segment.reviewed_box_frames == [25]


def test_dragging_with_a_box_selected_resizes_it() -> None:
    session = labelled_session(frame=25)
    session.prefill([(0, 0, 40, 90)])
    session.click(10, 10)
    session.key("2")
    session.click(10, 10)

    session.drag(5, 5, 50, 100)

    assert session.boxes() == [PlayerBox(5, 5, 50, 100, player=2)]


def test_events_outside_the_rally_are_kept_with_a_warning() -> None:
    session = labelled_session(frame=50)
    session.key("r")
    session.go_to(30)

    session.key("1")

    assert "outside the rally" in session.message
    assert len(session.segment.events) == 1


def test_invalid_labels_are_rejected(tmp_path) -> None:
    from courtvision.evaluation.labels import Segment

    with pytest.raises(ValueError):
        BallEvent("bounce", 3)  # no ball position
    with pytest.raises(ValueError):
        Segment(0, 10, status="lable")
    path = tmp_path / "labels.json"
    save_labels(make_labels(), path)
    path.write_text(
        path.read_text().replace('"schema_version": 1', '"schema_version": 9')
    )
    with pytest.raises(ValueError, match="schema"):
        load_labels(path)


def test_saving_leaves_no_temporary_file(tmp_path) -> None:
    save_labels(make_labels(), tmp_path / "labels.json")

    assert [path.name for path in tmp_path.iterdir()] == ["labels.json"]
