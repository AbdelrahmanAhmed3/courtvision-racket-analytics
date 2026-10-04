from types import SimpleNamespace

import numpy as np

from courtvision.detectors.rfdetr_detector import RFDetrDetector, resolve_device

COCO_NAMES = {1: "person", 37: "sports ball", 43: "tennis racket"}


class FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, image, threshold):
        self.calls.append((image, threshold))
        return self.result


def make_result(boxes, confidences, class_ids):
    return SimpleNamespace(
        xyxy=np.asarray(boxes, dtype=float),
        confidence=np.asarray(confidences, dtype=float),
        class_id=np.asarray(class_ids, dtype=int),
    )


def test_keeps_only_people_as_player_detections() -> None:
    model = FakeModel(
        make_result(
            boxes=[[10, 20, 50, 120], [60, 60, 64, 64], [70, 20, 90, 60]],
            confidences=[0.9, 0.8, 0.7],
            class_ids=[1, 37, 43],
        )
    )
    detector = RFDetrDetector(model=model, class_names=COCO_NAMES, size="small")

    detections = detector.predict_frame(np.zeros((200, 200, 3), np.uint8), 7)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == "person"
    assert detection.frame == 7
    assert detection.confidence == 0.9
    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (10, 20, 50, 120)
    assert detection.model_name == "rf-detr-small"


def test_sends_rgb_frames_and_the_confidence_threshold() -> None:
    model = FakeModel(make_result(np.zeros((0, 4)), [], []))
    detector = RFDetrDetector(model=model, class_names=COCO_NAMES, confidence=0.35)
    bgr_frame = np.zeros((2, 2, 3), np.uint8)
    bgr_frame[..., 0] = 255

    assert detector.predict_frame(bgr_frame, 0) == []

    image, threshold = model.calls[0]
    assert threshold == 0.35
    assert image[0, 0].tolist() == [0, 0, 255]


def test_an_explicit_device_is_used_as_given() -> None:
    assert resolve_device("cpu") == "cpu"
