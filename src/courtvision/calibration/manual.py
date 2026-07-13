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


def collect_matplotlib_landmarks(
    frame: np.ndarray,
    landmarks: Iterable[LandmarkDefinition],
) -> dict[str, tuple[float, float]]:
    """Collect landmarks from an interactive Matplotlib notebook canvas.

    This is intended for notebook environments such as Kaggle. The caller must
    enable an interactive Matplotlib backend (``%matplotlib widget``) first.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required for notebook calibration. Install matplotlib "
            "and ipympl, then run %matplotlib widget."
        ) from error

    backend = matplotlib.get_backend().lower()
    if "inline" in backend or "agg" in backend:
        raise RuntimeError(
            "Notebook calibration needs an interactive Matplotlib backend. In Kaggle, "
            "run `%matplotlib widget` in one cell, then invoke this script with `%run` "
            "instead of `!python`."
        )

    ordered_landmarks = list(landmarks)
    selected: list[tuple[float, float]] = []
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.imshow(image_rgb)
    axis.set_axis_off()

    for index, landmark in enumerate(ordered_landmarks):
        axis.set_title(
            f"Click {index + 1}/{len(ordered_landmarks)}: "
            f"{display_landmark_name(landmark.name)}\n"
            "Far = top/opposite camera end | Near = bottom/camera-side end"
        )
        figure.canvas.draw_idle()
        point = plt.ginput(1, timeout=-1)
        if not point:
            plt.close(figure)
            raise RuntimeError(
                "Notebook calibration cancelled before all points were set"
            )
        x, y = point[0]
        selected.append((float(x), float(y)))
        axis.plot(x, y, "mo", markersize=7)
        axis.annotate(
            str(index + 1),
            (x, y),
            xytext=(7, -7),
            textcoords="offset points",
            color="magenta",
            weight="bold",
        )

    axis.set_title("Calibration complete")
    figure.canvas.draw_idle()
    plt.close(figure)
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
