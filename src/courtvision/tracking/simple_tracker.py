from __future__ import annotations

from dataclasses import dataclass

from courtvision.detectors.base import Detection

DEFAULT_IOU_THRESHOLD = 0.3


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    detection: Detection
    age: int
    hits: int
    misses: int

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            self.detection.x1,
            self.detection.y1,
            self.detection.x2,
            self.detection.y2,
        )


@dataclass
class TrackState:
    track_id: int
    detection: Detection
    age: int = 1
    hits: int = 1
    misses: int = 0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            self.detection.x1,
            self.detection.y1,
            self.detection.x2,
            self.detection.y2,
        )

    def to_tracked_detection(self) -> TrackedDetection:
        return TrackedDetection(
            track_id=self.track_id,
            detection=self.detection,
            age=self.age,
            hits=self.hits,
            misses=self.misses,
        )


class SimpleIouTracker:
    """Greedy IoU tracker for short, detector-produced bbox sequences."""

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_missing_frames: int = 10,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._next_track_id = 1
        self._tracks: dict[int, TrackState] = {}

    @property
    def tracks(self) -> list[TrackState]:
        return list(self._tracks.values())

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        matches = self._match_detections(detections)
        matched_track_ids = set(matches.values())
        matched_detection_indexes = set(matches)

        for detection_index, track_id in matches.items():
            track = self._tracks[track_id]
            track.detection = detections[detection_index]
            track.age += 1
            track.hits += 1
            track.misses = 0

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_track_ids:
                track.age += 1
                track.misses += 1
                if track.misses > self.max_missing_frames:
                    del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indexes:
                continue
            track_id = self._allocate_track_id()
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                detection=detection,
            )

        visible_track_ids = {
            *matched_track_ids,
            *[
                track.track_id
                for track in self._tracks.values()
                if track.detection in detections and track.misses == 0
            ],
        }
        return [
            self._tracks[track_id].to_tracked_detection()
            for track_id in sorted(visible_track_ids)
            if track_id in self._tracks
        ]

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()

    def _allocate_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _match_detections(self, detections: list[Detection]) -> dict[int, int]:
        candidates = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                score = iou(detection_bbox(detection), track.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, detection_index, track_id))

        candidates.sort(reverse=True)
        matches: dict[int, int] = {}
        used_tracks = set()
        for _, detection_index, track_id in candidates:
            if detection_index in matches or track_id in used_tracks:
                continue
            matches[detection_index] = track_id
            used_tracks.add(track_id)
        return matches


def detection_bbox(detection: Detection) -> tuple[float, float, float, float]:
    return detection.x1, detection.y1, detection.x2, detection.y2


def iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second

    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    intersection = box_area((inter_x1, inter_y1, inter_x2, inter_y2))

    union = box_area(first) + box_area(second) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
