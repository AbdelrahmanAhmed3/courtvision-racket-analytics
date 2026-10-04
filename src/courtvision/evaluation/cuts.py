"""Find camera cuts, where the picture changes abruptly between two frames."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

DEFAULT_CUT_THRESHOLD = 0.45


def _colour_histogram(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram)


def detect_cuts(
    frames: Iterable[np.ndarray], threshold: float = DEFAULT_CUT_THRESHOLD
) -> list[int]:
    """Return the indexes of frames that start a new shot.

    Compares hue-saturation histograms of consecutive frames; a Bhattacharyya
    distance above ``threshold`` is a cut. Fades and fast pans can be missed or
    split, so the labelling tool lets you add cuts by hand.
    """
    cuts = []
    previous = None
    for index, frame in enumerate(frames):
        histogram = _colour_histogram(frame)
        if previous is not None:
            distance = cv2.compareHist(previous, histogram, cv2.HISTCMP_BHATTACHARYYA)
            if distance > threshold:
                cuts.append(index)
        previous = histogram
    return cuts
