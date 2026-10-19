"""Tracklets of the ball, and impacts where they start or end next to a player.

TrackNet gives one ball position per frame. In padel rallies it misses the ball
on about a third of frames, and some positions are on something else, such as
the ball's reflection in the glass (reports/ball_events.md). A tracklet is a run
of positions that one ball could really travel: it ends after a gap, or where
the next position is too far away for the frames between them.

At an impact the ball is often hidden by the player or racket for a frame or
two, so a tracklet ends just before the hit and the next starts just after it.
Tracklet ends next to a player are therefore impact candidates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import acos, degrees, dist, hypot

from courtvision.analytics.shots import nearest_player_to_ball
from courtvision.detectors.base import Detection
from courtvision.detectors.tracknet_adapter import BallPoint

# A hard smash (about 150 km/h) moves 1.7 m per frame at 25 fps: about 170 px
# where the court is largest in broadcast footage. 125 px drops most impossible
# jumps; tracklets are not sensitive to the exact value (reports/ball_events.md).
DEFAULT_MAX_PX_PER_FRAME = 125.0
DEFAULT_MAX_GAP_FRAMES = 2
# Shorter tracklets are mostly single wrong positions.
DEFAULT_MIN_TRACKLET_POINTS = 3
# Candidates this close together are one impact (the ends before and after it).
DEFAULT_MERGE_FRAMES = 3
# A gap between two tracklets this short is one event, hidden from TrackNet.
DEFAULT_MAX_HIDDEN_FRAMES = 10
# Bounces and wall rebounds often happen inside a tracklet, as a sharp turn.
DEFAULT_TURN_DEGREES = 30.0


@dataclass(frozen=True)
class Impact:
    frame: int
    track_id: int  # the player nearest the ball, not yet the hitter for certain


def split_into_tracklets(
    points: Iterable[BallPoint],
    max_px_per_frame: float = DEFAULT_MAX_PX_PER_FRAME,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
) -> list[list[BallPoint]]:
    """Runs of visible positions, in frame order, that one ball could travel."""
    tracklets: list[list[BallPoint]] = []
    for point in sorted(points, key=lambda p: p.frame):
        if not point.visible:
            continue
        if tracklets:
            last = tracklets[-1][-1]
            frames = point.frame - last.frame
            reachable = dist((point.x, point.y), (last.x, last.y)) <= (
                max_px_per_frame * frames
            )
            if frames <= max_gap_frames and reachable:
                tracklets[-1].append(point)
                continue
        tracklets.append([point])
    return tracklets


def detect_impacts(
    ball_points: Iterable[BallPoint],
    players_by_frame: Mapping[int, Sequence[tuple[int, Detection]]],
    min_tracklet_points: int = DEFAULT_MIN_TRACKLET_POINTS,
    merge_frames: int = DEFAULT_MERGE_FRAMES,
) -> list[Impact]:
    """Impacts where a tracklet starts or ends next to a player's box."""
    candidates = []
    for tracklet in split_into_tracklets(ball_points):
        if len(tracklet) < min_tracklet_points:
            continue
        for end in (tracklet[0], tracklet[-1]):
            track_id = nearest_player_to_ball(
                end, list(players_by_frame.get(end.frame, []))
            )
            if track_id is not None:
                candidates.append((end.frame, track_id))
    return _merge(sorted(set(candidates)), merge_frames)


def event_candidates(
    ball_points: Iterable[BallPoint],
    players_by_frame: Mapping[int, Sequence[tuple[int, Detection]]] | None = None,
    min_tracklet_points: int = DEFAULT_MIN_TRACKLET_POINTS,
    max_hidden_frames: int = DEFAULT_MAX_HIDDEN_FRAMES,
    turn_degrees: float = DEFAULT_TURN_DEGREES,
) -> list[int]:
    """Frames where an impact, bounce or wall rebound may have happened.

    Where one tracklet ends and the next starts at most ``max_hidden_frames``
    later, the event is one candidate where the two paths meet, which for a
    hit is where the racket is. A tracklet start with nothing just before it
    is a candidate too: TrackNet often finds the ball only a few frames after a
    hit, so with ``players_by_frame`` its path is followed back to the frame
    where it passes closest to a player (and an end with nothing after it,
    forward). Every turn of ``turn_degrees`` or more inside a tracklet is a
    candidate. Candidates at most 2 frames apart count once.
    """
    players_by_frame = players_by_frame or {}
    tracklets = [
        t for t in split_into_tracklets(ball_points) if len(t) >= min_tracklet_points
    ]
    frames = []
    for index, tracklet in enumerate(tracklets):
        before = tracklets[index - 1] if index else None
        after = tracklets[index + 1] if index + 1 < len(tracklets) else None
        if not before or tracklet[0].frame - before[-1].frame > max_hidden_frames:
            frames.append(_closest_to_player(tracklet[:4], -1, players_by_frame))
        if after and after[0].frame - tracklet[-1].frame <= max_hidden_frames:
            frames.append(meeting_frame(tracklet, after))
        else:
            frames.append(_closest_to_player(tracklet[-4:], 1, players_by_frame))
        for a, b, c in zip(tracklet, tracklet[1:], tracklet[2:], strict=False):
            incoming = (b.x - a.x, b.y - a.y)
            outgoing = (c.x - b.x, c.y - b.y)
            if turn_angle(incoming, outgoing) >= turn_degrees:
                frames.append(b.frame)
    candidates: list[int] = []
    for frame in sorted(set(frames)):
        if not candidates or frame - candidates[-1] > 2:
            candidates.append(frame)
    return candidates


def meeting_frame(before: Sequence[BallPoint], after: Sequence[BallPoint]) -> int:
    """The frame in the gap where the paths before and after come closest.

    Each path is extended at the speed of its last (or first) few positions;
    the event happened where the incoming ball meets the outgoing one.
    """
    end, start = before[-1], after[0]
    incoming = _velocity(before[-4:])
    outgoing = _velocity(after[:4])

    def apart(frame: int) -> float:
        coming = (
            end.x + incoming[0] * (frame - end.frame),
            end.y + incoming[1] * (frame - end.frame),
        )
        going = (
            start.x + outgoing[0] * (frame - start.frame),
            start.y + outgoing[1] * (frame - start.frame),
        )
        return dist(coming, going)

    return min(range(end.frame, start.frame + 1), key=apart)


def _closest_to_player(
    points: Sequence[BallPoint],
    direction: int,
    players_by_frame: Mapping[int, Sequence[tuple[int, Detection]]],
    max_frames: int = 6,
) -> int:
    """Follow a tracklet's path back (-1) or forward (+1) to the nearest player.

    Returns the frame where the extended path is closest to a player's box, if
    it comes within hitting distance; otherwise the tracklet's own first (or
    last) frame.
    """
    anchor = points[0] if direction < 0 else points[-1]
    velocity = _velocity(points)
    best_frame, best_distance = anchor.frame, None
    for step in range(max_frames + 1):
        frame = anchor.frame + direction * step
        x = anchor.x + velocity[0] * direction * step
        y = anchor.y + velocity[1] * direction * step
        players = list(players_by_frame.get(frame, []))
        if nearest_player_to_ball(BallPoint(frame, x, y), players) is None:
            continue
        distance = min(_distance_to_box(x, y, box) for _, box in players)
        if best_distance is None or distance < best_distance:
            best_frame, best_distance = frame, distance
    return best_frame


def _distance_to_box(x: float, y: float, box: Detection) -> float:
    return hypot(max(box.x1 - x, 0, x - box.x2), max(box.y1 - y, 0, y - box.y2))


def _velocity(points: Sequence[BallPoint]) -> tuple[float, float]:
    """Pixels per frame from the first to the last point; zero for one point."""
    first, last = points[0], points[-1]
    frames = last.frame - first.frame
    if frames == 0:
        return (0.0, 0.0)
    return ((last.x - first.x) / frames, (last.y - first.y) / frames)


def turn_angle(incoming: tuple[float, float], outgoing: tuple[float, float]) -> float:
    """Degrees between two directions; 0 when either has no length."""
    lengths = hypot(*incoming) * hypot(*outgoing)
    if lengths < 1e-9:
        return 0.0
    cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / lengths
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def _merge(candidates: list[tuple[int, int]], within: int) -> list[Impact]:
    """One impact per group of candidates at most ``within`` frames apart."""
    groups: list[list[tuple[int, int]]] = []
    for candidate in candidates:
        if groups and candidate[0] - groups[-1][-1][0] <= within:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    return [
        Impact(
            frame=round(sum(frame for frame, _ in group) / len(group)),
            track_id=Counter(track_id for _, track_id in group).most_common(1)[0][0],
        )
        for group in groups
    ]
