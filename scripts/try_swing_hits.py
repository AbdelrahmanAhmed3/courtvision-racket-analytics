"""Trial: do the near players' swings mark their hits (#60)?

From a phone on the back glass, RTMPose finds the two near players in every frame
(scripts/try_pose.py caches them in outputs/pose/). A hit is a swing, and in a
swing the hitting wrist moves fastest around contact, even when the ball tracker
has lost the ball. This script links each near player's poses over time with the
IoU tracker, measures their fastest wrist in body heights per second, finds the
peaks, and compares them with the labelled near-team impacts (P1 and P2):

- the hits with a peak within ±2 frames, including the hits where the ball
  tracker has no event candidate, where pose would add the most;
- the share of peaks during rallies that are at a hit, and how many peaks per
  second of play are not;
- whether the player whose wrist peaks is the labelled hitter, through the
  players boxed at each rally start;
- how far the nearest peak is from the labelled frame.

Needs the pose cache (try_pose.py) and the players and ball caches of
predict_shot_events.py and compare_ball_trackers.py.

Example:
    python scripts/try_swing_hits.py --labels evaluation/labels/IMG_1089.json \
        evaluation/labels/IMG_1086.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_event_classifier import tracked_players  # noqa: E402

from courtvision.analytics.ball_track import event_candidates  # noqa: E402
from courtvision.calibration.homography import project_image_points  # noqa: E402
from courtvision.detectors.base import Detection  # noqa: E402
from courtvision.detectors.tracknet_adapter import BallPoint  # noqa: E402
from courtvision.evaluation.court import court_estimate  # noqa: E402
from courtvision.evaluation.identity import (  # noqa: E402
    players_by_track,
    replay_tracker,
)
from courtvision.evaluation.labels import load_labels  # noqa: E402
from courtvision.tracking.simple_tracker import SimpleIouTracker  # noqa: E402

POSE_DIR = REPO_ROOT / "outputs" / "pose"
PREDICTIONS_DIR = REPO_ROOT / "outputs" / "predictions"
WRISTS = (9, 10)  # COCO left and right wrist
MIN_SCORE = 0.5  # a body point counts as found, as in reports/pose.md
TOLERANCE = 2  # frames, as the event scores
PEAK_GAP_SECONDS = 0.2  # a peak is the fastest within this either side
THRESHOLDS = (1.0, 2.0, 3.0)  # body heights per second
OFFSET_REACH = 8  # frames either side to look for the nearest peak
SWING_REACH = 3  # frames either side of a hit to look for the fastest wrist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", nargs="+", required=True, help="Labels JSON.")
    return parser.parse_args()


@dataclass
class Rally:
    """One labelled rally: its near hits, its near players' peaks, and who is who."""

    start: int
    end: int
    hits: list[tuple[int, int]]  # (frame, player) of near-team impacts
    no_candidate: set[int]  # hit frames with no ball candidate within the tolerance
    speeds: dict[int, dict[int, float]]  # track ID -> frame -> wrist speed
    player_of: dict[int, int]  # track ID -> player number, from the boxes
    peaks: dict[float, list[tuple[int, int]]] = field(default_factory=dict)
    # Who each rule names for each hit frame (None: no player named).
    named: dict[str, dict[int, int | None]] = field(default_factory=dict)


def load_pose(stem: str) -> dict[str, np.ndarray]:
    with np.load(POSE_DIR / f"{stem}--rtmpose-m.npz") as data:
        return {name: data[name] for name in data.files}


def near_player_tracks(pose, segment, fps: float) -> tuple[dict, dict[int, int]]:
    """The near players' pose boxes with track IDs, and each box's pose row."""
    rows = np.flatnonzero(
        (pose["frames"] >= segment.start) & (pose["frames"] < segment.end)
    )
    boxes = defaultdict(list)
    row_of: dict[int, int] = {}
    for row in rows:
        x1, y1, x2, y2, confidence = pose["boxes"][row]
        frame = int(pose["frames"][row])
        box = Detection(frame, "person", float(confidence), x1, y1, x2, y2, "rtmpose")
        boxes[frame].append(box)
        row_of[id(box)] = int(row)
    tracks = replay_tracker(
        SimpleIouTracker, boxes, range(segment.start, segment.end), fps
    )
    return tracks, row_of


def wrist_speeds(pose, tracks, row_of, fps: float) -> dict[int, dict[int, float]]:
    """Each track's fastest wrist per frame, in body heights per second.

    Measured between consecutive frames where the same wrist is found in both;
    the box height makes it the same for a player near or far from the camera.
    """
    speeds: dict[int, dict[int, float]] = defaultdict(dict)
    previous: dict[int, tuple[int, int]] = {}  # track ID -> (frame, row)
    for frame in sorted(tracks):
        for track_id, box in tracks[frame]:
            row = row_of[id(box)]
            last = previous.get(track_id)
            previous[track_id] = (frame, row)
            if last is None or last[0] != frame - 1:
                continue
            height = box.y2 - box.y1
            moves = [
                np.linalg.norm(pose["points"][row][w] - pose["points"][last[1]][w])
                for w in WRISTS
                if pose["scores"][row][w] >= MIN_SCORE
                and pose["scores"][last[1]][w] >= MIN_SCORE
            ]
            if moves and height > 0:
                speeds[track_id][frame] = max(moves) / height * fps
    return speeds


def peaks(speed: dict[int, float], threshold: float, gap: int) -> list[int]:
    """Frames where the speed reaches the threshold and is the fastest within gap."""
    return [
        frame
        for frame, value in speed.items()
        if value >= threshold
        and all(
            speed.get(other, 0.0) <= value
            for other in range(frame - gap, frame + gap + 1)
        )
    ]


def fastest_wrist(speeds, player_of: dict[int, int], frame: int) -> int | None:
    """The near player whose wrist moves fastest within SWING_REACH of the frame."""
    fastest = [
        (max(values), track_id)
        for track_id, speed in speeds.items()
        if (
            values := [
                speed[other]
                for other in range(frame - SWING_REACH, frame + SWING_REACH + 1)
                if other in speed
            ]
        )
    ]
    if not fastest:
        return None
    return player_of.get(max(fastest)[1])


def nearest_to_ball(segment, ball, players, frames: list[int]) -> dict:
    """Who the pipeline's rule names: the player whose box is nearest the ball.

    Once among everyone tracked, as the pipeline does, and once among the near
    half's players only, as pose is.
    """
    seen = {point.frame: point for point in ball if point.x is not None}
    player_of = players_by_track(segment, players)
    estimate = court_estimate(segment.court_points)
    named: dict[str, dict[int, int | None]] = {
        "ball: nearest player": {},
        "ball: nearest near player": {},
    }
    for frame in frames:
        spots = [
            other
            for other in range(frame - TOLERANCE, frame + TOLERANCE + 1)
            if other in seen and players.get(other)
        ]
        for rule in named:
            named[rule][frame] = None
        if not spots:
            continue
        spot = min(spots, key=lambda other: abs(other - frame))
        point = seen[spot]
        tracked = list(players[spot])
        feet = np.float32([((b.x1 + b.x2) / 2, b.y2) for _, b in tracked])
        near_half = project_image_points(feet, estimate)[:, 1] > 0.5
        for rule, choices in (
            ("ball: nearest player", tracked),
            (
                "ball: nearest near player",
                [t for t, near in zip(tracked, near_half, strict=True) if near],
            ),
        ):
            if choices:
                track_id, _ = min(choices, key=lambda t: _gap(point, t[1]))
                named[rule][frame] = player_of.get(track_id)
    return named


def _gap(point, box) -> float:
    """Pixels from the ball to the nearest edge of a box; zero inside it."""
    x = min(max(point.x, box.x1), box.x2)
    y = min(max(point.y, box.y1), box.y2)
    return float(np.hypot(point.x - x, point.y - y))


def load_rallies(path: str) -> tuple[list[Rally], float]:
    labels = load_labels(path)
    stem = Path(path).stem
    pose = load_pose(stem)
    cache = json.loads((PREDICTIONS_DIR / f"{stem}--tracks-wasb.json").read_text())
    gap = round(PEAK_GAP_SECONDS * labels.fps)
    rallies = []
    for segment in labels.segments:
        if segment.status != "label":
            continue
        cached = cache["segments"][str(segment.start)]
        ball = [
            BallPoint(f, x, y, 0.0 if x is None else 1.0) for f, x, y in cached["ball"]
        ]
        players = tracked_players(segment, cached, labels.fps)
        candidates = event_candidates(ball, players)
        hits = [
            (event.frame, event.player)
            for event in segment.events
            if event.kind == "impact" and event.player in (1, 2)
        ]
        no_candidate = {
            frame
            for frame, _ in hits
            if not any(abs(candidate - frame) <= TOLERANCE for candidate in candidates)
        }
        tracks, row_of = near_player_tracks(pose, segment, labels.fps)
        speeds = wrist_speeds(pose, tracks, row_of, labels.fps)
        rally = Rally(
            segment.rally_start,
            segment.rally_end,
            hits,
            no_candidate,
            speeds,
            players_by_track(segment, tracks),
        )
        rally.named = {
            "pose: fastest wrist": {
                frame: fastest_wrist(speeds, rally.player_of, frame)
                for frame, _ in hits
            },
            **nearest_to_ball(segment, ball, players, [frame for frame, _ in hits]),
        }
        for threshold in THRESHOLDS:
            rally.peaks[threshold] = sorted(
                (frame, track_id)
                for track_id, speed in speeds.items()
                for frame in peaks(speed, threshold, gap)
                if rally.start - TOLERANCE <= frame <= rally.end + TOLERANCE
            )
        rallies.append(rally)
    return rallies, labels.fps


def report(name: str, rallies: list[Rally], fps: float) -> None:
    hits = sum(len(r.hits) for r in rallies)
    no_candidate = sum(len(r.no_candidate) for r in rallies)
    seconds = sum(r.end - r.start + 1 for r in rallies) / fps
    print(
        f"\n{name}: {len(rallies)} rallies, {seconds:.0f} s, {hits} near-team hits, "
        f"{no_candidate} with no ball candidate within ±{TOLERANCE} frames"
    )
    print(
        f"{'peak at least':>16}{'hits with a peak':>18}{'no-candidate hits':>19}"
        f"{'peaks at a hit':>16}{'other peaks/s':>15}{'hitter right':>14}"
    )
    for threshold in THRESHOLDS:
        found = rescued = at_hit = other = right = named = 0
        for r in rallies:
            for frame, player in r.hits:
                near = [p for p in r.peaks[threshold] if abs(p[0] - frame) <= TOLERANCE]
                if not near:
                    continue
                found += 1
                rescued += frame in r.no_candidate
                _, track_id = min(near, key=lambda p: abs(p[0] - frame))
                if track_id in r.player_of:
                    named += 1
                    right += r.player_of[track_id] == player
            for frame, _ in r.peaks[threshold]:
                if any(abs(frame - hit) <= TOLERANCE for hit, _ in r.hits):
                    at_hit += 1
                else:
                    other += 1
        total = at_hit + other
        print(
            f"{threshold:>11.0f} bh/s{found:>12}/{hits:<5}"
            f"{rescued:>13}/{no_candidate:<5}{at_hit:>10}/{total:<5}"
            f"{other / seconds:>15.2f}{right:>9}/{named:<4}"
        )
    print(f"\nNaming the hitter of the {hits} labelled near-team hits:")
    for rule in rallies[0].named:
        named = [
            (r.named[rule][frame], player)
            for r in rallies
            for frame, player in r.hits
            if r.named[rule][frame] is not None
        ]
        right = sum(choice == player for choice, player in named)
        print(f"  {rule:28} names {len(named):>3}, right {right:>3}")
    offsets = []
    for r in rallies:
        for frame, _ in r.hits:
            near = [
                p[0] - frame for p in r.peaks[2.0] if abs(p[0] - frame) <= OFFSET_REACH
            ]
            if near:
                offsets.append(min(near, key=abs))
    if offsets:
        quartiles = np.percentile(offsets, [25, 50, 75])
        print(
            "nearest peak of 2 bh/s or more, minus the labelled frame, for "
            f"{len(offsets)} hits within ±{OFFSET_REACH}: "
            f"quartiles {quartiles.tolist()} frames"
        )


def main() -> None:
    args = parse_args()
    everything, fps_seen = [], set()
    for path in args.labels:
        rallies, fps = load_rallies(path)
        report(Path(path).stem, rallies, fps)
        everything += rallies
        fps_seen.add(round(fps))
    if len(args.labels) > 1 and len(fps_seen) == 1:
        report("All clips", everything, fps)


if __name__ == "__main__":
    main()
