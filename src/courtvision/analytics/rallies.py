"""Split a camera shot into rallies from its ball events.

A rally runs from the serve until the ball is dead. Inside a labelled rally,
consecutive ball events are at most 2.5 s apart (95 % under 1.7 s), while
between points play stops for many seconds. So a silence longer than
``max_gap_seconds`` ends a rally. A rally starts at its first impact (events
before it, such as the server bouncing the ball, are not part of it) and ends
at its last event. Who won the point is not decided here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from courtvision.analytics.ball_events import BallEvent, Shot

# Labelled rallies have gaps up to 2.5 s; the classifier misses events, which
# makes its gaps longer, so twice that, still short of a pause between points.
DEFAULT_MAX_GAP_SECONDS = 5.0
# A serve and a return; a lone impact is more likely a false one.
DEFAULT_MIN_IMPACTS = 2


@dataclass(frozen=True)
class Rally:
    start_frame: int  # the first impact: the serve, when it is seen
    end_frame: int  # the last ball event before play stops
    shots: tuple[Shot, ...]


def segment_rallies(
    events: Iterable[BallEvent],
    fps: float,
    shots: Iterable[Shot] = (),
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    min_impacts: int = DEFAULT_MIN_IMPACTS,
) -> list[Rally]:
    """Rallies in frame order, each with the shots whose impact falls inside it."""
    events = sorted(events, key=lambda event: event.frame)
    shots = list(shots)
    max_gap = max_gap_seconds * fps
    groups: list[list[BallEvent]] = []
    for event in events:
        if groups and event.frame - groups[-1][-1].frame <= max_gap:
            groups[-1].append(event)
        else:
            groups.append([event])
    rallies = []
    for group in groups:
        impacts = [event for event in group if event.kind == "impact"]
        if len(impacts) < min_impacts:
            continue
        start, end = impacts[0].frame, group[-1].frame
        inside = tuple(shot for shot in shots if start <= shot.frame <= end)
        rallies.append(Rally(start, end, inside))
    return rallies
