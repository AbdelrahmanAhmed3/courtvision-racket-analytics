"""Court heatmaps: where each player spent their time, one small court each."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

CELL_PX = 16  # one square metre
LINE = (235, 235, 235)


def draw_heatmaps(
    heatmaps: Mapping[int, np.ndarray], teams: Mapping[int, str] | None = None
) -> np.ndarray:
    """Side-by-side courts, far baseline at the top, warmer where a player stood.

    Each heatmap is seconds per square metre: 20 rows from the far baseline,
    10 columns from the left side.
    """
    teams = teams or {}
    courts = [
        _court(seconds, f"P{track_id} {teams.get(track_id, '')}".strip())
        for track_id, seconds in sorted(heatmaps.items())
    ]
    if not courts:
        return np.zeros((40, 200, 3), np.uint8)
    gap = np.full((courts[0].shape[0], 12, 3), 255, np.uint8)
    row = [courts[0]]
    for court in courts[1:]:
        row += [gap, court]
    return np.hstack(row)


def _court(seconds: np.ndarray, title: str) -> np.ndarray:
    rows, columns = seconds.shape
    scaled = seconds / seconds.max() if seconds.max() > 0 else seconds
    cells = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    court = cv2.resize(
        cells, (columns * CELL_PX, rows * CELL_PX), interpolation=cv2.INTER_NEAREST
    )
    width, height = columns * CELL_PX, rows * CELL_PX
    cv2.rectangle(court, (0, 0), (width - 1, height - 1), LINE, 1)
    net = height // 2
    cv2.line(court, (0, net), (width, net), LINE, 2)
    for service in (3, rows - 3):  # service lines 3 m from each back wall
        y = service * CELL_PX
        cv2.line(court, (0, y), (width, y), LINE, 1)
    cv2.line(
        court, (width // 2, 3 * CELL_PX), (width // 2, height - 3 * CELL_PX), LINE, 1
    )
    header = np.full((28, width, 3), 255, np.uint8)
    cv2.putText(header, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    return np.vstack([header, court])
