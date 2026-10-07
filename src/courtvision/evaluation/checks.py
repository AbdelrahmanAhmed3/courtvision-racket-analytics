"""Find labels that break the rules of padel, so they are fixed before scoring.

The strongest rule is that a team never hits twice in a row: two impacts by the
same team mean an impact was missed or a player number is wrong. The checks
cannot prove labels right, only point at frames worth a second look.
"""

from __future__ import annotations

from dataclasses import dataclass

from courtvision.evaluation.labels import BallEvent, Segment, VideoLabels


@dataclass(frozen=True)
class Problem:
    frame: int
    text: str


def team(player: int) -> int:
    """0 for players 1 and 2 (near the camera), 1 for players 3 and 4."""
    return (player - 1) // 2


def describe(event: BallEvent) -> str:
    if event.kind == "impact":
        return f"impact by P{event.player}"
    return event.kind.replace("_", " ")


def find_problems(labels: VideoLabels, segment: Segment) -> list[Problem]:
    """Problems in one segment, in frame order."""
    if segment.status == "unreviewed":
        return [Problem(segment.start, "Segment not reviewed: label (l) or skip (o)")]
    if segment.status == "skip":
        return []
    start, end = segment.rally_start, segment.rally_end
    problems = []
    if start is None:
        problems.append(Problem(segment.start, "No rally start (r)"))
    if end is None:
        problems.append(Problem(segment.end - 1, "No rally end (e)"))
    if segment.court_points is None:
        problems.append(Problem(start or segment.start, "No court marked (k)"))
    events = sorted(segment.events, key=lambda event: event.frame)
    for event in events:
        before = start is not None and event.frame < start
        after = end is not None and event.frame > end
        if before or after:
            problems.append(
                Problem(event.frame, f"{describe(event)} is outside the rally")
            )
    impacts = [event for event in events if event.kind == "impact"]
    serve_expected = start is not None and start > segment.start
    if serve_expected and not any(impact.frame == start for impact in impacts):
        problems.append(
            Problem(start, "The rally starts mid-segment but not on an impact (serve)")
        )
    for first, second in zip(impacts, impacts[1:], strict=False):
        if team(first.player) == team(second.player):
            problems.append(
                Problem(
                    second.frame,
                    f"P{first.player} (frame {first.frame}) then P{second.player}: "
                    "a team hit twice. Missed impact or wrong player?",
                )
            )
        between = [e for e in events if first.frame < e.frame < second.frame]
        bounces = [event for event in between if event.kind == "bounce"]
        if serve_expected and first.frame == start and not bounces:
            problems.append(
                Problem(second.frame, "The serve was returned before it bounced")
            )
        if len(bounces) >= 2:
            problems.append(
                Problem(
                    bounces[1].frame,
                    f"Second bounce after frame {bounces[0].frame} ends the point, "
                    "but play goes on. Not a bounce?",
                )
            )
    for frame in labels.box_frames(segment):
        if frame not in segment.reviewed_box_frames:
            problems.append(Problem(frame, "Boxes not done"))
    for frame in segment.reviewed_box_frames:
        if any(box.player is None for box in segment.boxes.get(frame, [])):
            problems.append(
                Problem(frame, "A finished box frame has a box with no player")
            )
    return sorted(problems, key=lambda problem: problem.frame)


def find_all_problems(labels: VideoLabels) -> list[Problem]:
    return [
        problem
        for segment in labels.segments
        for problem in find_problems(labels, segment)
    ]
