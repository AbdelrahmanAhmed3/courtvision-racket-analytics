from courtvision.detectors.ultralytics_detector import (
    detections_from_result,
    get_device,
)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)


class FakeBox:
    xyxy = [FakeTensor([10, 20, 30, 40])]
    conf = [FakeTensor(0.8)]
    cls = [FakeTensor(0)]


class FakeResult:
    boxes = [FakeBox()]
    names = {0: "person"}


def test_get_device_explicit_cpu() -> None:
    assert get_device("cpu") == "cpu"


def test_detections_from_result() -> None:
    detections = detections_from_result(
        FakeResult(),
        frame_index=7,
        model_name="yolo11n",
    )

    assert len(detections) == 1
    assert detections[0].frame == 7
    assert detections[0].class_name == "person"
    assert detections[0].confidence == 0.8
    assert detections[0].center == (20, 30)
