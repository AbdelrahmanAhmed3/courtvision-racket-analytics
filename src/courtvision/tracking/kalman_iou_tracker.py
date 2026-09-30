from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.detectors.base import Detection
from courtvision.tracking.simple_tracker import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_MISSING_SECONDS,
    DEFAULT_REASSOCIATION_DISTANCE_PX,
    TrackedDetection,
    TrackerUpdateStats,
    center_distance,
    detection_bbox,
    iou,
)

INVALID_COST = 1_000_000.0


@dataclass
class KalmanTrackState:
    track_id: int
    detection: Detection
    state: np.ndarray
    covariance: np.ndarray
    last_seen_seconds: float
    last_update_seconds: float
    age: int = 1
    hits: int = 1
    missing_seconds: float = 0.0

    @property
    def predicted_bbox(self) -> tuple[float, float, float, float]:
        center_x, center_y, width, height = self.state[:4]
        return (
            float(center_x - width / 2),
            float(center_y - height / 2),
            float(center_x + width / 2),
            float(center_y + height / 2),
        )

    def to_tracked_detection(self) -> TrackedDetection:
        return TrackedDetection(
            track_id=self.track_id,
            detection=self.detection,
            age=self.age,
            hits=self.hits,
            missing_seconds=self.missing_seconds,
        )


class KalmanIouTracker:
    """Time-aware constant-velocity tracker with global IoU assignment.

    Experimental: kept for comparison with SimpleIouTracker. On the tested
    padel clip it produced more tracks and ID switches, so it is not used by
    any pipeline. See issue #5.

    State is ``[center_x, center_y, width, height, vx, vy, vw, vh]``. Each
    frame predicts every live or dormant track, then Hungarian assignment finds
    the lowest-cost one-to-one set of detection matches.
    """

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_missing_seconds: float = DEFAULT_MAX_MISSING_SECONDS,
        reassociation_distance_px: float = DEFAULT_REASSOCIATION_DISTANCE_PX,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between zero and one")
        if max_missing_seconds <= 0:
            raise ValueError("max_missing_seconds must be greater than zero")
        if reassociation_distance_px <= 0:
            raise ValueError("reassociation_distance_px must be greater than zero")

        self.iou_threshold = iou_threshold
        self.max_missing_seconds = max_missing_seconds
        self.reassociation_distance_px = reassociation_distance_px
        self._next_track_id = 1
        self._tracks: dict[int, KalmanTrackState] = {}
        self._last_timestamp_seconds: float | None = None
        self.last_update_stats = TrackerUpdateStats()

    @property
    def tracks(self) -> list[KalmanTrackState]:
        return list(self._tracks.values())

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()
        self._last_timestamp_seconds = None
        self.last_update_stats = TrackerUpdateStats()

    def update(
        self,
        detections: list[Detection],
        timestamp_seconds: float,
    ) -> list[TrackedDetection]:
        if (
            self._last_timestamp_seconds is not None
            and timestamp_seconds < self._last_timestamp_seconds
        ):
            raise ValueError("timestamp_seconds must be monotonic")

        self._predict_tracks(timestamp_seconds)
        self._expire_tracks(timestamp_seconds)
        matches, dormant_recoveries = self._assign(detections)
        matched_track_ids = set(matches.values())

        for detection_index, track_id in matches.items():
            track = self._tracks[track_id]
            self._correct(track, detections[detection_index])
            track.detection = detections[detection_index]
            track.last_seen_seconds = timestamp_seconds
            track.last_update_seconds = timestamp_seconds
            track.age += 1
            track.hits += 1
            track.missing_seconds = 0.0

        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.age += 1
                track.missing_seconds = timestamp_seconds - track.last_seen_seconds

        visible_track_ids = set(matched_track_ids)
        new_tracks = 0
        for detection_index, detection in enumerate(detections):
            if detection_index in matches:
                continue
            track_id = self._allocate_track_id()
            self._tracks[track_id] = self._new_track(
                track_id, detection, timestamp_seconds
            )
            visible_track_ids.add(track_id)
            new_tracks += 1

        self._last_timestamp_seconds = timestamp_seconds
        self.last_update_stats = TrackerUpdateStats(
            iou_matches=len(matches) - dormant_recoveries,
            dormant_recoveries=dormant_recoveries,
            new_tracks=new_tracks,
        )
        return [
            self._tracks[track_id].to_tracked_detection()
            for track_id in sorted(visible_track_ids)
        ]

    def _new_track(
        self, track_id: int, detection: Detection, timestamp_seconds: float
    ) -> KalmanTrackState:
        x1, y1, x2, y2 = detection_bbox(detection)
        state = np.array(
            [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0, 0, 0, 0],
            dtype=float,
        )
        return KalmanTrackState(
            track_id=track_id,
            detection=detection,
            state=state,
            covariance=np.diag([100, 100, 100, 100, 400, 400, 100, 100]).astype(float),
            last_seen_seconds=timestamp_seconds,
            last_update_seconds=timestamp_seconds,
        )

    def _allocate_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _predict_tracks(self, timestamp_seconds: float) -> None:
        for track in self._tracks.values():
            delta_seconds = timestamp_seconds - track.last_update_seconds
            if delta_seconds <= 0:
                continue
            transition = np.eye(8)
            for coordinate in range(4):
                transition[coordinate, coordinate + 4] = delta_seconds
            process_noise = np.diag([4, 4, 2, 2, 25, 25, 9, 9]).astype(float) * max(
                delta_seconds, 0.001
            )
            track.state = transition @ track.state
            track.state[2:4] = np.maximum(track.state[2:4], 1.0)
            track.covariance = (
                transition @ track.covariance @ transition.T + process_noise
            )
            track.last_update_seconds = timestamp_seconds

    def _expire_tracks(self, timestamp_seconds: float) -> None:
        for track_id, track in list(self._tracks.items()):
            if timestamp_seconds - track.last_seen_seconds > self.max_missing_seconds:
                del self._tracks[track_id]

    def _assign(self, detections: list[Detection]) -> tuple[dict[int, int], int]:
        if not detections or not self._tracks:
            return {}, 0

        track_ids = sorted(self._tracks)
        costs = np.full((len(track_ids), len(detections)), INVALID_COST)
        for track_row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            for detection_index, detection in enumerate(detections):
                overlap = iou(track.predicted_bbox, detection_bbox(detection))
                if overlap >= self.iou_threshold:
                    costs[track_row, detection_index] = 1.0 - overlap
                    continue
                if track.missing_seconds > 0:
                    distance = center_distance(
                        track.predicted_bbox, detection_bbox(detection)
                    )
                    if distance <= self.reassociation_distance_px:
                        costs[track_row, detection_index] = 1.0 + (
                            distance / self.reassociation_distance_px
                        )

        rows, columns = linear_sum_assignment(costs)
        matches: dict[int, int] = {}
        dormant_recoveries = 0
        for track_row, detection_index in zip(rows, columns, strict=True):
            if costs[track_row, detection_index] >= INVALID_COST:
                continue
            matches[detection_index] = track_ids[track_row]
            if self._tracks[track_ids[track_row]].missing_seconds > 0:
                dormant_recoveries += 1
        return matches, dormant_recoveries

    @staticmethod
    def _correct(track: KalmanTrackState, detection: Detection) -> None:
        x1, y1, x2, y2 = detection_bbox(detection)
        measurement = np.array(
            [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float
        )
        observation = np.zeros((4, 8))
        observation[:, :4] = np.eye(4)
        measurement_noise = np.diag([25, 25, 16, 16]).astype(float)
        innovation = measurement - observation @ track.state
        innovation_covariance = (
            observation @ track.covariance @ observation.T + measurement_noise
        )
        gain = track.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        track.state = track.state + gain @ innovation
        track.covariance = (np.eye(8) - gain @ observation) @ track.covariance
