from courtvision.detectors.base import Detection


def test_detection_center() -> None:
    detection = Detection(
        frame=0,
        class_name="player",
        confidence=0.9,
        x1=10,
        y1=20,
        x2=30,
        y2=60,
        model_name="test",
    )

    assert detection.center == (20, 40)
