"""Hand labels for evaluating CourtVision on a clip.

A clip is split into segments at camera cuts. Each labelled segment holds one
rally: its start and end frames, every impact (with the hitter), every bounce and
wall rebound (with the ball's image position), and player boxes with identities on
every ``box_interval``-th frame. See docs/evaluation.md for the file format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
PLAYER_IDS = (1, 2, 3, 4)
EVENT_KINDS = ("impact", "bounce", "wall_rebound")


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
    status: str = "unreviewed"  # "label", "skip" or "unreviewed"
    rally_start: int | None = None
    rally_end: int | None = None
    events: list[BallEvent] = field(default_factory=list)
    boxes: dict[int, list[PlayerBox]] = field(default_factory=dict)
    reviewed_box_frames: list[int] = field(default_factory=list)

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.end


@dataclass
class ClipLabels:
    clip: str
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
        start = (
            segment.rally_start if segment.rally_start is not None else segment.start
        )
        end = segment.rally_end + 1 if segment.rally_end is not None else segment.end
        first = -(-start // self.box_interval) * self.box_interval
        return list(range(first, end, self.box_interval))


def segments_from_cuts(cuts: list[int], frame_count: int) -> list[Segment]:
    """Split ``frame_count`` frames into segments at the given cut frames."""
    bounds = sorted({0, frame_count, *(cut for cut in cuts if 0 < cut < frame_count)})
    return [Segment(start, end) for start, end in zip(bounds, bounds[1:], strict=False)]


def save_labels(labels: ClipLabels, path: str | Path) -> None:
    data = asdict(labels)
    for segment in data["segments"]:
        segment["boxes"] = {
            str(frame): boxes for frame, boxes in segment["boxes"].items()
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def load_labels(path: str | Path) -> ClipLabels:
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
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
        )
        for item in data["segments"]
    ]
    fields = {key: value for key, value in data.items() if key != "segments"}
    return ClipLabels(**fields, segments=segments)
