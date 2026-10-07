"""Hand labels for evaluating CourtVision on a source video.

A source video is split into segments at camera cuts. Each labelled segment holds
one rally: its start and end frames, every impact (with the hitter), every bounce
and wall rebound (with the ball's image position), player boxes with identities
on every ``box_interval``-th frame, and where the court is in the image. See
docs/evaluation.md for the file format.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2  # 2 added court points; version 1 files load without them
READABLE_SCHEMA_VERSIONS = (1, 2)
PLAYER_IDS = (1, 2, 3, 4)
EVENT_KINDS = ("impact", "bounce", "wall_rebound")
SEGMENT_STATUSES = ("unreviewed", "label", "skip")
# Cuts closer together than this are flashes or fades, not new camera shots.
MIN_SEGMENT_FRAMES = 5


@dataclass
class BallEvent:
    """An impact (player set) or a bounce or wall rebound (ball position set)."""

    kind: str
    frame: int
    player: int | None = None
    x: float | None = None
    y: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"Unknown event kind: {self.kind}")
        if self.kind == "impact" and self.player not in PLAYER_IDS:
            raise ValueError("An impact needs a player from 1 to 4")
        if self.kind != "impact" and (self.x is None or self.y is None):
            raise ValueError(f"A {self.kind} needs the ball's x and y")


@dataclass
class PlayerBox:
    """A player's box on one frame; ``player`` is None until assigned."""

    x1: float
    y1: float
    x2: float
    y2: float
    player: int | None = None

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Segment:
    """Frames between two camera cuts, ``start`` inclusive and ``end`` exclusive."""

    start: int
    end: int
    status: str = "unreviewed"
    rally_start: int | None = None
    rally_end: int | None = None
    events: list[BallEvent] = field(default_factory=list)
    boxes: dict[int, list[PlayerBox]] = field(default_factory=dict)
    reviewed_box_frames: list[int] = field(default_factory=list)
    # Where the service lines meet the side walls, in image pixels: far left, far
    # right, near right, near left (see courtvision.evaluation.court).
    court_points: list[list[float]] | None = None

    def __post_init__(self) -> None:
        if self.status not in SEGMENT_STATUSES:
            raise ValueError(f"Unknown segment status: {self.status}")
        points = self.court_points
        if points is not None and [len(point) for point in points] != [2] * 4:
            raise ValueError("Court points must be four (x, y) points")

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.end

    def split_at(self, frame: int) -> Segment:
        """Shorten this segment to end at ``frame``; return the part after it.

        Every label moves with the frame it belongs to, so a rally split by the
        cut leaves its start on this segment and its end on the new one. Court
        points stay here: after a cut the camera, and so the court, differ.
        """
        if not self.start < frame < self.end:
            raise ValueError(f"Frame {frame} is not inside segment {self.start}")
        tail = Segment(frame, self.end, status=self.status)
        if self.rally_start is not None and self.rally_start >= frame:
            tail.rally_start, self.rally_start = self.rally_start, None
        if self.rally_end is not None and self.rally_end >= frame:
            tail.rally_end, self.rally_end = self.rally_end, None
        tail.events = [event for event in self.events if event.frame >= frame]
        self.events = [event for event in self.events if event.frame < frame]
        tail.boxes = {f: boxes for f, boxes in self.boxes.items() if f >= frame}
        self.boxes = {f: boxes for f, boxes in self.boxes.items() if f < frame}
        tail.reviewed_box_frames = [f for f in self.reviewed_box_frames if f >= frame]
        self.reviewed_box_frames = [f for f in self.reviewed_box_frames if f < frame]
        self.end = frame
        return tail

    def absorb(self, later: Segment) -> None:
        """Merge the segment that directly follows this one into it."""
        if later.start != self.end:
            raise ValueError("Only the next segment can be merged")
        if self.status == "unreviewed":
            self.status = later.status
        if self.rally_start is None:
            self.rally_start = later.rally_start
        if self.court_points is None:
            self.court_points = later.court_points
        if later.rally_end is not None:
            self.rally_end = later.rally_end
        self.events += later.events
        self.boxes.update(later.boxes)
        self.reviewed_box_frames += later.reviewed_box_frames
        self.end = later.end


@dataclass
class VideoLabels:
    video: str
    source_url: str
    fps: float
    width: int
    height: int
    frame_count: int
    box_interval: int = 25
    segments: list[Segment] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def segment_at(self, frame: int) -> Segment:
        for segment in self.segments:
            if segment.contains(frame):
                return segment
        raise ValueError(f"Frame {frame} is outside every segment")

    def box_frames(self, segment: Segment) -> list[int]:
        """Frames in a segment's rally (or whole segment) that need player boxes."""
        start = segment.start if segment.rally_start is None else segment.rally_start
        end = segment.end if segment.rally_end is None else segment.rally_end + 1
        interval = self.box_interval
        first_multiple = (start + interval - 1) // interval * interval
        return list(range(first_multiple, min(end, segment.end), interval))


def segments_from_cuts(
    cuts: list[int], frame_count: int, min_frames: int = MIN_SEGMENT_FRAMES
) -> list[Segment]:
    """Split ``frame_count`` frames into segments at the given cut frames.

    A cut that would leave a segment shorter than ``min_frames`` is ignored.
    """
    bounds = [0]
    for cut in sorted(set(cuts)):
        if cut - bounds[-1] >= min_frames and frame_count - cut >= min_frames:
            bounds.append(cut)
    bounds.append(frame_count)
    return [Segment(start, end) for start, end in zip(bounds, bounds[1:], strict=False)]


def save_labels(labels: VideoLabels, path: str | Path) -> None:
    """Write labels atomically, so a crash never leaves a half-written file."""
    data = asdict(labels)
    for segment in data["segments"]:
        segment["boxes"] = {
            str(frame): boxes for frame, boxes in segment["boxes"].items()
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(temporary, path)


def load_labels(path: str | Path) -> VideoLabels:
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") not in READABLE_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported labels schema: {data.get('schema_version')}")
    segments = [
        Segment(
            start=item["start"],
            end=item["end"],
            status=item["status"],
            rally_start=item["rally_start"],
            rally_end=item["rally_end"],
            events=[BallEvent(**event) for event in item["events"]],
            boxes={
                int(frame): [PlayerBox(**box) for box in boxes]
                for frame, boxes in item["boxes"].items()
            },
            reviewed_box_frames=list(item["reviewed_box_frames"]),
            court_points=item.get("court_points"),
        )
        for item in data["segments"]
    ]
    known = {"video", "source_url", "fps", "width", "height", "frame_count"}
    known |= {"box_interval", "schema_version"}
    unknown = set(data) - known - {"segments"}
    if unknown:
        raise ValueError(f"Unknown fields in labels file: {sorted(unknown)}")
    fields = {key: value for key, value in data.items() if key != "segments"}
    fields["schema_version"] = SCHEMA_VERSION  # saved in the current schema
    return VideoLabels(**fields, segments=segments)
