from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LandmarkObservation:
    image: tuple[float, float]
    template: tuple[float, float]
    visible: bool
    confidence: float
    source: str


@dataclass(frozen=True)
class CalibrationRecord:
    video_id: str
    frame_index: int
    frame_width: int
    frame_height: int
    court_type: str
    landmarks: dict[str, LandmarkObservation]


def save_calibration(path: str | Path, calibration: CalibrationRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration), indent=2) + "\n")


def load_calibration(path: str | Path) -> CalibrationRecord:
    data = json.loads(Path(path).read_text())
    landmarks = {
        name: LandmarkObservation(
            image=tuple(observation["image"]),
            template=tuple(observation["template"]),
            visible=bool(observation["visible"]),
            confidence=float(observation["confidence"]),
            source=str(observation["source"]),
        )
        for name, observation in data["landmarks"].items()
    }
    return CalibrationRecord(
        video_id=str(data["video_id"]),
        frame_index=int(data["frame_index"]),
        frame_width=int(data["frame_width"]),
        frame_height=int(data["frame_height"]),
        court_type=str(data["court_type"]),
        landmarks=landmarks,
    )
