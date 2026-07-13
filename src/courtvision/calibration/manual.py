from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from courtvision.calibration.landmarks import (
    LandmarkDefinition,
    display_landmark_name,
)


def collect_manual_landmarks(
    frame: np.ndarray,
    landmarks: Iterable[LandmarkDefinition],
    window_name: str = "CourtVision manual calibration",
) -> dict[str, tuple[float, float]]:
    """Collect landmarks with left-click; Enter saves, U undoes, Esc cancels."""
    ordered_landmarks = list(landmarks)
    selected: list[tuple[float, float]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < len(ordered_landmarks):
            selected.append((float(x), float(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        image = frame.copy()
        _draw_instructions(image, ordered_landmarks, selected)
        cv2.imshow(window_name, image)
        key = cv2.waitKey(20) & 0xFF
        if key in {13, 10}:
            if len(selected) >= 4:
                break
        elif key in {ord("u"), ord("U")} and selected:
            selected.pop()
        elif key in {ord("r"), ord("R")}:
            selected.clear()
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Manual calibration cancelled")

    cv2.destroyWindow(window_name)
    return {
        landmark.name: point
        for landmark, point in zip(ordered_landmarks, selected, strict=True)
    }


def _draw_instructions(
    image: np.ndarray,
    landmarks: list[LandmarkDefinition],
    selected: list[tuple[float, float]],
) -> None:
    prompt = (
        f"Click {len(selected) + 1}/{len(landmarks)}: "
        f"{display_landmark_name(landmarks[len(selected)].name)}"
        if len(selected) < len(landmarks)
        else "All selected. Press Enter to save."
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 102), (18, 18, 18), -1)
    cv2.putText(
        image,
        prompt,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Far = top/opposite camera end | Near = bottom/camera-side end",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Enter: save (4+ points) | U: undo | R: reset | Esc: cancel",
        (16, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    for index, point in enumerate(selected):
        label = landmarks[index].name
        x, y = [int(round(value)) for value in point]
        cv2.circle(image, (x, y), 6, (255, 0, 255), -1)
        cv2.putText(
            image,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
