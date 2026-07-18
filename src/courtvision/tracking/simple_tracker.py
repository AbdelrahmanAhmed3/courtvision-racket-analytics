from __future__ import annotations

from dataclasses import dataclass

from courtvision.detectors.base import Detection

DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_MISSING_SECONDS = 3.0
DEFAULT_REASSOCIATION_DISTANCE_PX = 250.0


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    detection: Detection
    age: int
    hits: int
    missing_seconds: float

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
    last_seen_seconds: float = 0.0
    missing_seconds: float = 0.0

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
            missing_seconds=self.missing_seconds,
        )


class SimpleIouTracker:
    """Greedy IoU tracker with time-based dormant-track reassociation."""

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_missing_seconds: float = DEFAULT_MAX_MISSING_SECONDS,
        reassociation_distance_px: float = DEFAULT_REASSOCIATION_DISTANCE_PX,
    ) -> None:
        if max_missing_seconds <= 0:
            raise ValueError("max_missing_seconds must be greater than zero")
        if reassociation_distance_px <= 0:
            raise ValueError("reassociation_distance_px must be greater than zero")
        self.iou_threshold = iou_threshold
        self.max_missing_seconds = max_missing_seconds
        self.reassociation_distance_px = reassociation_distance_px
        self._next_track_id = 1
        self._tracks: dict[int, TrackState] = {}
        self._last_timestamp_seconds: float | None = None

    @property
    def tracks(self) -> list[TrackState]:
        return list(self._tracks.values())

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
        self._expire_tracks(timestamp_seconds)
        matches = self._match_detections(detections)
        matches.update(self._reassociate_dormant_tracks(detections, matches))
        matched_track_ids = set(matches.values())
        matched_detection_indexes = set(matches)

        for detection_index, track_id in matches.items():
            track = self._tracks[track_id]
            track.detection = detections[detection_index]
            track.age += 1
            track.hits += 1
            track.last_seen_seconds = timestamp_seconds
            track.missing_seconds = 0.0

        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track.age += 1
                track.missing_seconds = timestamp_seconds - track.last_seen_seconds

        visible_track_ids = set(matched_track_ids)
        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indexes:
                continue
            track_id = self._allocate_track_id()
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                detection=detection,
                last_seen_seconds=timestamp_seconds,
            )
            visible_track_ids.add(track_id)
        self._last_timestamp_seconds = timestamp_seconds
        return [
            self._tracks[track_id].to_tracked_detection()
            for track_id in sorted(visible_track_ids)
            if track_id in self._tracks
        ]

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()
        self._last_timestamp_seconds = None

    def _expire_tracks(self, timestamp_seconds: float) -> None:
        for track_id, track in list(self._tracks.items()):
            missing_seconds = timestamp_seconds - track.last_seen_seconds
            if missing_seconds > self.max_missing_seconds:
                del self._tracks[track_id]

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

    def _reassociate_dormant_tracks(
        self,
        detections: list[Detection],
        matches: dict[int, int],
    ) -> dict[int, int]:
        used_track_ids = set(matches.values())
        candidates = []
        for detection_index, detection in enumerate(detections):
            if detection_index in matches:
                continue
            for track_id, track in self._tracks.items():
                if track_id in used_track_ids or track.missing_seconds <= 0:
                    continue
                distance = center_distance(detection_bbox(detection), track.bbox)
                if distance <= self.reassociation_distance_px:
                    candidates.append((distance, detection_index, track_id))

        candidates.sort()
        reassociated = {}
        for _, detection_index, track_id in candidates:
            if detection_index in reassociated or track_id in used_track_ids:
                continue
            reassociated[detection_index] = track_id
            used_track_ids.add(track_id)
        return reassociated


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


def center_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    return (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5
