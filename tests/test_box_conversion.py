from courtvision.detectors.roboflow_detector import roboflow_box_to_xyxy


def test_roboflow_box_to_xyxy() -> None:
    prediction = {"x": 100, "y": 100, "width": 40, "height": 20}

    assert roboflow_box_to_xyxy(prediction) == (80, 90, 120, 110)
