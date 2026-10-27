"""Movement stats per player from their positions on the court.

A player's position is their feet projected onto the court, and it wobbles from
frame to frame by a few pixels, which far from the camera is tens of
centimetres. Summing raw frame-to-frame steps on a 15-second broadcast clip gives
the far players 58 and 73 m, two to three times what they ran
(reports/player_movement.md). So positions are first smoothed with a centred
moving average over ``smoothing_seconds``: short enough to keep a sprint or a
change of direction, long enough to absorb the wobble. A step faster than
``max_speed_kmh`` is a tracking jump, not running: an ID swap or a false
detection. The track is cut there, like at a gap, so neither the distance, the
smoothing nor the top speed runs across it.

Court coordinates are metres: x across the court (0 to 10), y along it from the
far baseline (0) to the near one (20), with the net at y = 10.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

DEFAULT_SMOOTHING_SECONDS = 0.5
DEFAULT_MAX_SPEED_KMH = 30.0  # faster than any padel player runs
DEFAULT_MAX_GAP_SECONDS = 0.5  # a longer gap starts a new run of positions
NET_Y_M = 10.0
NET_ZONE_M = 4.0  # "at the net": within 4 m of it, where volleys are played
SERVICE_LINE_FROM_NET_M = 6.95  # "at the back": behind the service line
COURT_WIDTH_M, COURT_LENGTH_M = 10, 20


@dataclass(frozen=True)
class MovementStats:
    seconds_tracked: float
    distance_m: float
    average_speed_kmh: float  # distance over time tracked
    top_speed_kmh: float  # fastest 0.5 s, from smoothed positions
    seconds_at_net: float
    seconds_in_middle: float
    seconds_at_back: float
    # Seconds spent in each 1 m cell: 20 rows from the far baseline, 10 columns
    # from the left side. Positions outside the court count in the edge cells.
    heatmap_seconds: np.ndarray


def movement_stats(
    positions: Iterable[tuple[int, int, float, float]],
    fps: float,
    smoothing_seconds: float = DEFAULT_SMOOTHING_SECONDS,
    max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> dict[int, MovementStats]:
    """Stats by track ID from (track ID, frame, x metres, y metres) positions."""
    by_track: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for track_id, frame, x, y in positions:
        by_track[track_id].append((frame, x, y))
    return {
        track_id: _stats(
            sorted(points), fps, smoothing_seconds, max_speed_kmh, max_gap_seconds
        )
        for track_id, points in by_track.items()
    }


def _stats(points, fps, smoothing_seconds, max_speed_kmh, max_gap_seconds):
    window = max(1, round(smoothing_seconds * fps))
    max_step = max_speed_kmh / 3.6 / fps  # metres per frame
    span = max(1, round(smoothing_seconds * fps))  # frames over which top speed runs
    distance = 0.0
    top_speed = 0.0
    frames_tracked = 0
    zone_frames = {"net": 0, "middle": 0, "back": 0}
    heatmap = np.zeros((COURT_LENGTH_M, COURT_WIDTH_M))
    for run in _runs(points, max(1, round(max_gap_seconds * fps)), max_step):
        frames = np.array([frame for frame, _, _ in run])
        smooth = _smooth(np.array([(x, y) for _, x, y in run], dtype=float), window)
        frames_tracked += len(run)
        steps = np.linalg.norm(np.diff(smooth, axis=0), axis=1)
        per_frame = steps / np.diff(frames)
        distance += float(steps[per_frame <= max_step].sum())
        if len(run) > span:
            moved = np.linalg.norm(smooth[span:] - smooth[:-span], axis=1)
            speeds = moved / ((frames[span:] - frames[:-span]) / fps) * 3.6
            plausible = speeds[speeds <= max_speed_kmh]
            if plausible.size:
                top_speed = max(top_speed, float(plausible.max()))
        from_net = np.abs(smooth[:, 1] - NET_Y_M)
        zone_frames["net"] += int((from_net <= NET_ZONE_M).sum())
        zone_frames["back"] += int((from_net >= SERVICE_LINE_FROM_NET_M).sum())
        zone_frames["middle"] += int(
            ((from_net > NET_ZONE_M) & (from_net < SERVICE_LINE_FROM_NET_M)).sum()
        )
        rows = np.clip(smooth[:, 1].astype(int), 0, COURT_LENGTH_M - 1)
        columns = np.clip(smooth[:, 0].astype(int), 0, COURT_WIDTH_M - 1)
        np.add.at(heatmap, (rows, columns), 1)
    seconds = frames_tracked / fps
    return MovementStats(
        seconds_tracked=seconds,
        distance_m=distance,
        average_speed_kmh=distance / seconds * 3.6 if seconds else 0.0,
        top_speed_kmh=top_speed,
        seconds_at_net=zone_frames["net"] / fps,
        seconds_in_middle=zone_frames["middle"] / fps,
        seconds_at_back=zone_frames["back"] / fps,
        heatmap_seconds=heatmap / fps,
    )


def _runs(points, max_gap_frames, max_step):
    """Split frame-sorted points where the track is lost or jumps."""
    runs = [[points[0]]]
    for previous, point in zip(points, points[1:], strict=False):
        frames = point[0] - previous[0]
        moved = np.hypot(point[1] - previous[1], point[2] - previous[2])
        if frames > max_gap_frames or moved > max_step * frames:
            runs.append([])
        runs[-1].append(point)
    return runs


def _smooth(xy: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average; near the ends the window shrinks to fit."""
    if window <= 1 or len(xy) < 3:
        return xy
    half = window // 2
    smooth = np.empty_like(xy)
    for i in range(len(xy)):
        reach = min(half, i, len(xy) - 1 - i)
        smooth[i] = xy[i - reach : i + reach + 1].mean(axis=0)
    return smooth
