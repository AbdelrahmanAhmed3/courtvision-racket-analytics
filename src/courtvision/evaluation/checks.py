"""Find labels that break the rules of padel, so they are fixed before scoring.

The strongest rule is that a team never hits twice in a row: two impacts by the
same team mean an impact was missed or a player number is wrong. A stroke must
also fit the bounces before it: a volley comes before the bounce, a forehand or
backhand after it, and a bounce on one side means that side's team hits next.
A video the coach feeds (coach_feeds) has no serves. The checks cannot prove
labels right, only point at frames worth a second look.
"""

from __future__ import annotations

from dataclasses import dataclass

from courtvision.evaluation.court import court_half
from courtvision.evaluation.labels import (
    BOUNCE_KINDS,
    STROKE_GROUPS,
    BallEvent,
    Segment,
    VideoLabels,
)

# Bounces closer to the net than this share of the court's length (about 1 m) are
# on no side: a click a few pixels off would move them across.
NET_MARGIN = 0.05


@dataclass(frozen=True)
class Problem:
    frame: int
    text: str


def team(player: int) -> int:
    """0 for players 1 and 2 (near the camera), 1 for players 3 and 4."""
    return (player - 1) // 2


def describe(event: BallEvent) -> str:
    if event.kind == "impact":
        stroke = f", {event.stroke.replace('_', ' ')}" if event.stroke else ""
        return f"impact by P{event.player}{stroke}"
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
    hit_frames = {impact.frame for impact in impacts}
    for event in events:
        if event.kind != "impact" and event.frame in hit_frames:
            problems.append(
                Problem(
                    event.frame,
                    f"A hit and a {describe(event)} on the same frame. Move one?",
                )
            )
    # A rally opens with a serve, unless a coach under the camera feeds the
    # first ball of every rally in this video.
    serve_expected = (
        start is not None and start > segment.start and not labels.coach_feeds
    )
    if labels.coach_feeds:
        problems.extend(
            Problem(impact.frame, "A serve in a video the coach feeds. Not a serve?")
            for impact in impacts
            if impact.stroke == "serve"
        )
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
        bounces = [e for e in between if e.kind in BOUNCE_KINDS]
        if serve_expected and first.frame == start and not bounces:
            problems.append(
                Problem(second.frame, "The serve was returned before it bounced")
            )
        problems.extend(_stroke_problems(labels, segment, first, second, between))
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


def _stroke_problems(
    labels: VideoLabels,
    segment: Segment,
    previous: BallEvent,
    hit: BallEvent,
    between: list[BallEvent],
) -> list[Problem]:
    """Problems with a hit that is not the segment's first, from the ball events
    since the hit before it."""
    problems = []
    bounces = [e for e in between if e.kind in BOUNCE_KINDS]
    group = STROKE_GROUPS.get(hit.stroke or "")
    # A hit before the rally start is reported as outside the rally, so it
    # doesn't make the serve after it a second one.
    in_rally = segment.rally_start is None or previous.frame >= segment.rally_start
    if group == "serve" and in_rally and not labels.coach_feeds:
        problems.append(Problem(hit.frame, "A serve that is not the rally's first hit"))
    # The near back glass is behind the camera, so only a far player's rebound
    # off their own glass can be seen and labelled.
    walls = [e for e in between if e.kind == "wall_rebound"]
    if hit.stroke in ("glass", "bajada") and team(hit.player) == 1 and not walls:
        problems.append(
            Problem(
                hit.frame,
                f"{hit.stroke.capitalize()} with no wall rebound before it. "
                "Missed rebound, or another stroke?",
            )
        )
    if group == "volley" and bounces:
        problems.append(
            Problem(
                hit.frame,
                f"Volley after the bounce at frame {bounces[0].frame}. "
                "Forehand or backhand, or not a bounce?",
            )
        )
    if group == "ground_stroke" and not bounces:
        problems.append(
            Problem(
                hit.frame,
                f"{hit.stroke.capitalize()} with no bounce before it. "
                "A volley, or a bounce out of sight (b twice) missing?",
            )
        )
    placed = [b for b in bounces if b.x is not None and b.y is not None]
    if segment.court_points is not None and placed and len(placed) == len(bounces):
        halves = {
            court_half(segment.court_points, b.x, b.y, NET_MARGIN) for b in placed
        }
        if None not in halves and team(hit.player) not in halves:
            side = "far" if team(hit.player) == 0 else "near"
            problems.append(
                Problem(
                    hit.frame,
                    f"P{hit.player} hits after the ball bounced on the {side} side. "
                    "Wrong player?",
                )
            )
    return problems


def find_all_problems(labels: VideoLabels) -> list[Problem]:
    return [
        problem
        for segment in labels.segments
        for problem in find_problems(labels, segment)
    ]
