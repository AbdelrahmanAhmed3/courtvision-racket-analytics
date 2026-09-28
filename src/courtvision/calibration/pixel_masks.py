"""Pixel-selection helpers used to inspect court-line evidence."""

from __future__ import annotations

import cv2
import numpy as np


def near_white_pixel_mask(
    frame: np.ndarray,
    minimum_value: int = 170,
    maximum_saturation: int = 80,
) -> np.ndarray:
    """Return a binary mask for bright, low-saturation pixels in a BGR frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.asarray((0, 0, minimum_value), dtype=np.uint8),
        np.asarray((180, maximum_saturation, 255), dtype=np.uint8),
    )
