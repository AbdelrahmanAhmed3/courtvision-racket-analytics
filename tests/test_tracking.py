from courtvision.detectors.base import Detection
from courtvision.tracking.simple_tracker import SimpleIouTracker, iou


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame: int = 0,
) -> Detection:
    return Detection(
        frame=frame,
        class_name="player",
        confidence=0.9,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        model_name="test",
    )


def test_detection_center() -> None:
    detection = make_detection(10, 20, 30, 60)

    assert detection.center == (20, 40)


def test_iou() -> None:
    assert iou((0, 0, 100, 100), (50, 0, 150, 100)) == 1 / 3
    assert iou((0, 0, 100, 100), (120, 0, 220, 100)) == 0


def test_tracker_reuses_id_above_threshold() -> None:
    tracker = SimpleIouTracker(iou_threshold=0.3)

    first = tracker.update([make_detection(0, 0, 100, 100, frame=0)])
    second = tracker.update([make_detection(40, 0, 140, 100, frame=1)])

    assert first[0].track_id == 1
    assert second[0].track_id == 1
    assert second[0].hits == 2


def test_tracker_creates_new_id_below_threshold() -> None:
    tracker = SimpleIouTracker(iou_threshold=0.3)

    first = tracker.update([make_detection(0, 0, 100, 100, frame=0)])
    second = tracker.update([make_detection(60, 0, 160, 100, frame=1)])

    assert first[0].track_id == 1
    assert second[0].track_id == 2


def test_tracker_removes_missing_tracks() -> None:
    tracker = SimpleIouTracker(iou_threshold=0.3, max_missing_frames=1)

    tracker.update([make_detection(0, 0, 100, 100, frame=0)])
    tracker.update([])
    tracker.update([])

    assert tracker.tracks == []
