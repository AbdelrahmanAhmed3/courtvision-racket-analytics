from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_polygon(path: str | Path) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    return np.asarray(data["polygon"], dtype=np.float32)


def point_inside_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    polygon = np.asarray(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0
