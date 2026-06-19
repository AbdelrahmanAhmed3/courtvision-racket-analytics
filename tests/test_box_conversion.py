from courtvision.detectors.base import (
    Detection,
    filter_player_ball_detections,
    filter_player_detections,
    is_ball_class,
    is_player_class,
)
from courtvision.detectors.roboflow_detector import (
    detections_from_response,
    roboflow_box_to_xyxy,
)


def test_roboflow_box_to_xyxy() -> None:
    prediction = {"x": 100, "y": 100, "width": 40, "height": 20}

    assert roboflow_box_to_xyxy(prediction) == (80, 90, 120, 110)


def test_detections_from_roboflow_response() -> None:
    response = {
        "predictions": [
            {
                "x": 100,
                "y": 100,
                "width": 40,
                "height": 20,
                "confidence": 0.92,
                "class": "tennis-ball",
            }
        ]
    }

    detections = detections_from_response(
        response,
        frame_index=12,
        model_name="tennis-ball-model/1",
    )

    assert len(detections) == 1
    assert detections[0].frame == 12
    assert detections[0].class_name == "tennis-ball"
    assert detections[0].confidence == 0.92
    assert detections[0].center == (100, 100)


def test_player_ball_class_helpers() -> None:
    assert is_player_class("person")
    assert is_player_class("player")
    assert is_ball_class("ball")
    assert is_ball_class("tennis_ball")
    assert is_ball_class("tennis ball")


def test_filter_player_ball_detections() -> None:
    detections = [
        Detection(0, "court", 0.9, 0, 0, 10, 10, "test"),
        Detection(0, "net", 0.9, 0, 0, 10, 10, "test"),
        Detection(0, "player", 0.9, 0, 0, 10, 10, "test"),
        Detection(0, "tennis-ball", 0.9, 0, 0, 10, 10, "test"),
    ]

    filtered = filter_player_ball_detections(detections)

    assert [detection.class_name for detection in filtered] == [
        "player",
        "tennis-ball",
    ]


def test_filter_player_detections() -> None:
    detections = [
        Detection(0, "court", 0.9, 0, 0, 10, 10, "test"),
        Detection(0, "player", 0.9, 0, 0, 10, 10, "test"),
        Detection(0, "person", 0.9, 0, 0, 10, 10, "test"),
        Detection(0, "ball", 0.9, 0, 0, 10, 10, "test"),
    ]

    filtered = filter_player_detections(detections)

    assert [detection.class_name for detection in filtered] == ["player", "person"]
