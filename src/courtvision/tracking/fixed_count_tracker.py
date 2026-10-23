"""A tracker for a known number of players: it never gives out more IDs.

Padel has exactly four players for a whole rally. A general tracker gives a
new ID whenever it cannot match a box, so a player whose box changes shape in
one frame (a crouch, a smash) or who was hidden comes back as player 5 or 6
(reports/player_identity.md). Here IDs 1 to ``max_players`` are given out as
players first appear, and after that a box that matches no ID by overlap takes
the ID, among those not seen this frame, whose last box is nearest.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.detectors.base import Detection
from courtvision.tracking.simple_tracker import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_REASSOCIATION_DISTANCE_PX,
    TrackedDetection,
    TrackState,
    center_distance,
    detection_bbox,
    iou,
)


class FixedCountTracker:
    """Track at most ``max_players`` people, each keeping one ID.

    Each update first matches boxes to IDs by overlap (IoU at least
    ``iou_threshold``): greedily from the largest overlap, or, with
    ``assignment="hungarian"``, as the best one-to-one set. A box left over
    takes a new ID while fewer than ``max_players`` exist and no missing ID was
    last seen within ``new_id_distance_px``; once all IDs exist, it takes the
    nearest missing ID, however far.
    """

    def __init__(
        self,
        max_players: int = 4,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        assignment: str = "greedy",
        new_id_distance_px: float = DEFAULT_REASSOCIATION_DISTANCE_PX,
    ) -> None:
        if max_players < 1:
            raise ValueError("max_players must be at least one")
        if assignment not in ("greedy", "hungarian"):
            raise ValueError(f"Unknown assignment: {assignment}")
        self.max_players = max_players
        self.iou_threshold = iou_threshold
        self.assignment = assignment
        self.new_id_distance_px = new_id_distance_px
        self._tracks: dict[int, TrackState] = {}
        self._last_timestamp_seconds: float | None = None

    @property
    def tracks(self) -> list[TrackState]:
        return list(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._last_timestamp_seconds = None

    def update(
        self, detections: list[Detection], timestamp_seconds: float
    ) -> list[TrackedDetection]:
        if (
            self._last_timestamp_seconds is not None
            and timestamp_seconds < self._last_timestamp_seconds
        ):
            raise ValueError("timestamp_seconds must be monotonic")
        self._last_timestamp_seconds = timestamp_seconds
        matches = self._match_by_overlap(detections)
        matches.update(self._match_left_over(detections, matches))

        for track in self._tracks.values():
            track.age += 1
        for detection_index, track_id in matches.items():
            detection = detections[detection_index]
            track = self._tracks.get(track_id)
            if track is None:
                self._tracks[track_id] = TrackState(
                    track_id, detection, last_seen_seconds=timestamp_seconds
                )
                continue
            track.detection = detection
            track.hits += 1
            track.last_seen_seconds = timestamp_seconds
        for track in self._tracks.values():
            track.missing_seconds = timestamp_seconds - track.last_seen_seconds
        return [
            self._tracks[track_id].to_tracked_detection()
            for track_id in sorted(matches.values())
        ]

    def _match_by_overlap(self, detections: list[Detection]) -> dict[int, int]:
        """Detection index -> track ID, for boxes overlapping an ID's last box."""
        track_ids = sorted(self._tracks)
        if not detections or not track_ids:
            return {}
        overlaps = np.array(
            [
                [iou(detection_bbox(d), self._tracks[t].bbox) for t in track_ids]
                for d in detections
            ]
        )
        allowed = overlaps >= self.iou_threshold
        if self.assignment == "hungarian":
            rows, columns = linear_sum_assignment(
                np.where(allowed, overlaps, 0.0), maximize=True
            )
            pairs = zip(rows, columns, strict=True)
        else:
            order = np.argsort(-overlaps, axis=None)
            pairs = (np.unravel_index(i, overlaps.shape) for i in order)
        matches: dict[int, int] = {}
        for row, column in pairs:
            track_id = track_ids[column]
            if (
                allowed[row, column]
                and row not in matches
                and track_id not in matches.values()
            ):
                matches[int(row)] = track_id
        return matches

    def _match_left_over(
        self, detections: list[Detection], matches: dict[int, int]
    ) -> dict[int, int]:
        """Boxes left over take the nearest missing IDs, or new ones while room."""
        left_over = [i for i in range(len(detections)) if i not in matches]
        missing = [t for t in sorted(self._tracks) if t not in matches.values()]
        assigned: dict[int, int] = {}
        if left_over and missing:
            distances = np.array(
                [
                    [
                        center_distance(
                            detection_bbox(detections[i]), self._tracks[t].bbox
                        )
                        for t in missing
                    ]
                    for i in left_over
                ]
            )
            full = len(self._tracks) >= self.max_players
            for row, column in zip(*linear_sum_assignment(distances), strict=True):
                if full or distances[row, column] <= self.new_id_distance_px:
                    assigned[left_over[row]] = missing[column]
        next_id = max(self._tracks, default=0) + 1
        for index in left_over:
            if index not in assigned and next_id <= self.max_players:
                assigned[index] = next_id
                next_id += 1
        return assigned
