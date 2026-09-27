from courtvision.detectors.base import Detection
from courtvision.tracking import kalman_iou_tracker
from courtvision.tracking.kalman_iou_tracker import KalmanIouTracker
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

    first = tracker.update([make_detection(0, 0, 100, 100, frame=0)], 0.0)
    second = tracker.update([make_detection(40, 0, 140, 100, frame=1)], 0.1)

    assert first[0].track_id == 1
    assert second[0].track_id == 1
    assert second[0].hits == 2


def test_tracker_creates_new_id_below_threshold() -> None:
    tracker = SimpleIouTracker(iou_threshold=0.3)

    first = tracker.update([make_detection(0, 0, 100, 100, frame=0)], 0.0)
    second = tracker.update([make_detection(60, 0, 160, 100, frame=1)], 0.1)

    assert first[0].track_id == 1
    assert second[0].track_id == 2


def test_tracker_reassociates_a_returning_player_within_time_limit() -> None:
    tracker = SimpleIouTracker(
        iou_threshold=0.3,
        max_missing_seconds=3.0,
        reassociation_distance_px=100.0,
    )

    first = tracker.update([make_detection(0, 0, 100, 100, frame=0)], 0.0)
    tracker.update([], 1.0)
    returned = tracker.update([make_detection(70, 0, 170, 100, frame=60)], 2.0)

    assert first[0].track_id == 1
    assert returned[0].track_id == 1


def test_tracker_removes_tracks_after_time_limit() -> None:
    tracker = SimpleIouTracker(iou_threshold=0.3, max_missing_seconds=1.0)

    tracker.update([make_detection(0, 0, 100, 100, frame=0)], 0.0)
    tracker.update([], 0.5)
    tracker.update([], 1.1)

    assert tracker.tracks == []


def test_kalman_tracker_predicts_motion_across_a_missed_detection() -> None:
    tracker = KalmanIouTracker(
        iou_threshold=0.3,
        max_missing_seconds=3.0,
        reassociation_distance_px=100.0,
    )

    first = tracker.update([make_detection(0, 0, 100, 100, frame=0)], 0.0)
    tracker.update([make_detection(20, 0, 120, 100, frame=1)], 0.1)
    tracker.update([], 0.2)
    returned = tracker.update([make_detection(60, 0, 160, 100, frame=3)], 0.3)

    assert first[0].track_id == 1
    assert returned[0].track_id == 1
    assert tracker.last_update_stats.dormant_recoveries == 1


def test_kalman_tracker_expires_after_dormant_window() -> None:
    tracker = KalmanIouTracker(max_missing_seconds=1.0)

    tracker.update([make_detection(0, 0, 100, 100)], 0.0)
    tracker.update([], 1.1)

    assert tracker.tracks == []


def test_kalman_tracker_uses_global_hungarian_assignment(monkeypatch) -> None:
    tracker = KalmanIouTracker(iou_threshold=0.3)
    tracker.update(
        [
            make_detection(0, 0, 100, 100),
            make_detection(200, 0, 300, 100),
        ],
        0.0,
    )

    def controlled_iou(track_box, detection_box) -> float:
        track_is_left = track_box[0] < 100
        detection_is_left = detection_box[0] < 100
        scores = {
            (True, True): 0.9,
            (True, False): 0.8,
            (False, True): 0.85,
            (False, False): 0.0,
        }
        return scores[(track_is_left, detection_is_left)]

    monkeypatch.setattr(kalman_iou_tracker, "iou", controlled_iou)
    tracks = tracker.update(
        [
            make_detection(10, 0, 110, 100),
            make_detection(210, 0, 310, 100),
        ],
        0.1,
    )

    ids_by_detection_x1 = {track.detection.x1: track.track_id for track in tracks}
    assert ids_by_detection_x1 == {10: 2, 210: 1}
