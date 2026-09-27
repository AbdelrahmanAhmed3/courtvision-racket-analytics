from courtvision.detectors.base import (
    Detection,
    expected_player_count,
    filter_player_ball_detections,
    filter_player_detections,
    is_ball_class,
    is_player_class,
)
from courtvision.detectors.roboflow_detector import (
    detections_from_response,
    filter_by_confidence,
    infer_with_retries,
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


def test_filters_roboflow_detections_locally_by_confidence() -> None:
    detections = [
        Detection(0, "player", 0.29, 0, 0, 10, 10, "test"),
        Detection(0, "player", 0.30, 0, 0, 10, 10, "test"),
    ]

    filtered = filter_by_confidence(detections, minimum_confidence=0.30)

    assert [detection.confidence for detection in filtered] == [0.30]


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


def test_filter_player_detections_limits_to_highest_confidence() -> None:
    detections = [
        Detection(0, "player", 0.4, 0, 0, 10, 10, "test"),
        Detection(0, "player", 0.9, 0, 0, 10, 10, "test"),
        Detection(0, "person", 0.7, 0, 0, 10, 10, "test"),
    ]

    filtered = filter_player_detections(detections, max_players=2)

    assert [detection.confidence for detection in filtered] == [0.9, 0.7]
    assert expected_player_count("tennis") == 2
    assert expected_player_count("padel") == 4


def test_retries_temporary_hosted_inference_failure(monkeypatch) -> None:
    class TimeoutError(Exception):
        status_code = 524

    class Client:
        calls = 0

        def infer(self, source, model_id):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            return {"predictions": []}

    monkeypatch.setattr(
        "courtvision.detectors.roboflow_detector.time.sleep",
        lambda _: None,
    )
    client = Client()

    response = infer_with_retries(
        client,
        source="frame",
        model_id="tennis-v4d0h/2",
        max_retries=1,
    )

    assert response == {"predictions": []}
    assert client.calls == 2
