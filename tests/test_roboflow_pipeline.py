from courtvision.detectors.base import Detection
from courtvision.tracking.simple_tracker import SimpleIouTracker
from scripts.run_roboflow_pipeline import tracked_items_from_detections


def make_detection(
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame: int,
) -> Detection:
    return Detection(
        frame=frame,
        class_name=class_name,
        confidence=0.9,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        model_name="test",
    )


def test_tracked_items_assign_player_ids_and_drop_balls() -> None:
    tracker = SimpleIouTracker(iou_threshold=0.3)

    first = tracked_items_from_detections(
        [
            make_detection("player", 0, 0, 100, 100, frame=0),
            make_detection("ball", 200, 200, 210, 210, frame=0),
        ],
        tracker,
    )
    second = tracked_items_from_detections(
        [
            make_detection("player", 40, 0, 140, 100, frame=1),
            make_detection("ball", 205, 205, 215, 215, frame=1),
        ],
        tracker,
    )

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == 1
    assert second[0].track_id == 1
