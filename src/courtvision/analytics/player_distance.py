from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from courtvision.analytics.court_coordinates import CourtCoordinate

DEFAULT_SMOOTHING_ALPHA = 0.4
DEFAULT_MAX_SPEED_MPS = 12.0
DEFAULT_MAX_GAP_SECONDS = 0.5


@dataclass
class PlayerDistanceState:
    total_distance_m: float
    smoothed_point_m: tuple[float, float]
    last_timestamp_seconds: float


class PlayerDistanceTracker:
    """Accumulate plausible player movement in calibrated court metres."""

    def __init__(
        self,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        max_speed_mps: float = DEFAULT_MAX_SPEED_MPS,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    ) -> None:
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be greater than zero")
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be greater than zero")
        self.smoothing_alpha = smoothing_alpha
        self.max_speed_mps = max_speed_mps
        self.max_gap_seconds = max_gap_seconds
        self._states: dict[int, PlayerDistanceState] = {}

    @property
    def distances_m(self) -> dict[int, float]:
        return {
            track_id: state.total_distance_m
            for track_id, state in self._states.items()
        }

    def update(
        self,
        points: list[CourtCoordinate],
        timestamp_seconds: float,
    ) -> None:
        for point in points:
            if point.object_type != "player":
                continue
            raw_point = (point.court_x_m, point.court_y_m)
            if not all(isfinite(value) for value in raw_point):
                continue
            self._update_player(point.track_id, raw_point, timestamp_seconds)

    def _update_player(
        self,
        track_id: int,
        raw_point_m: tuple[float, float],
        timestamp_seconds: float,
    ) -> None:
        state = self._states.get(track_id)
        if state is None:
            self._states[track_id] = PlayerDistanceState(
                total_distance_m=0.0,
                smoothed_point_m=raw_point_m,
                last_timestamp_seconds=timestamp_seconds,
            )
            return

        delta_seconds = timestamp_seconds - state.last_timestamp_seconds
        if delta_seconds <= 0:
            return
        if delta_seconds > self.max_gap_seconds:
            state.smoothed_point_m = raw_point_m
            state.last_timestamp_seconds = timestamp_seconds
            return

        previous_x, previous_y = state.smoothed_point_m
        alpha = self.smoothing_alpha
        smoothed_point = (
            previous_x + alpha * (raw_point_m[0] - previous_x),
            previous_y + alpha * (raw_point_m[1] - previous_y),
        )
        distance_m = hypot(
            smoothed_point[0] - previous_x,
            smoothed_point[1] - previous_y,
        )
        state.last_timestamp_seconds = timestamp_seconds
        if distance_m / delta_seconds > self.max_speed_mps:
            return
        state.total_distance_m += distance_m
        state.smoothed_point_m = smoothed_point
