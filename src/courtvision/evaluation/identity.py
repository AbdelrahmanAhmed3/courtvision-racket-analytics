"""Score a tracker's player identities against the labelled player numbers.

A labelled player keeps one number (P1 to P4) for a whole segment, on one frame
per second. A good tracker gives each player one track ID for as long as
possible. In each segment, every labelled player is paired with the track ID it
shares the most labelled frames with, one to one; frames where a player carries
another ID count against the tracker. That is IDF1, the standard identity score:
a player split across two IDs and two players merged into one ID both lower it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.detectors.base import Detection
from courtvision.evaluation.detection import DEFAULT_MIN_IOU, match_boxes
from courtvision.evaluation.labels import Segment, VideoLabels
from courtvision.evaluation.scoring import match_frames

TrackedBoxes = Mapping[int, Sequence[tuple[int, Detection]]]  # frame -> (id, box)


@dataclass(frozen=True)
class IdentityScore:
    id_hits: int  # labelled boxes whose track ID is their player's paired ID
    labelled: int  # labelled player boxes
    tracked: int  # tracked boxes on the labelled frames
    switches: int  # times a player's track ID changed between labelled frames
    ids_per_segment: tuple[int, ...]  # distinct IDs given to labelled players

    @property
    def idf1(self) -> float:
        total = self.labelled + self.tracked
        return 2 * self.id_hits / total if total else 0.0


def score_identities(
    labels: VideoLabels,
    tracks_by_segment: Mapping[int, TrackedBoxes],
    min_iou: float = DEFAULT_MIN_IOU,
) -> IdentityScore:
    """IDF1 and ID switches over every finished box frame of labelled segments.

    ``tracks_by_segment`` maps each labelled segment's start frame to its
    tracked boxes by frame.
    """
    id_hits = labelled = tracked = switches = 0
    ids_per_segment = []
    for segment in labels.segments:
        if segment.status != "label":
            continue
        tracks = tracks_by_segment.get(segment.start, {})
        shared: Counter[tuple[int, int]] = Counter()  # (player, track ID) frames
        ids_by_player: dict[int, list[int]] = defaultdict(list)
        for frame in sorted(segment.reviewed_box_frames):
            truth = segment.boxes.get(frame, [])
            found = list(tracks.get(frame, []))
            labelled += len(truth)
            tracked += len(found)
            boxes = [detection for _, detection in found]
            for row, column in match_boxes(truth, boxes, min_iou):
                player, track_id = truth[row].player, found[column][0]
                shared[player, track_id] += 1
                ids_by_player[player].append(track_id)
        id_hits += _paired_frames(shared)
        switches += sum(
            sum(a != b for a, b in zip(ids, ids[1:], strict=False))
            for ids in ids_by_player.values()
        )
        ids_per_segment.append(len({track_id for _, track_id in shared}))
    return IdentityScore(id_hits, labelled, tracked, switches, tuple(ids_per_segment))


def players_by_track(
    segment: Segment, tracks: TrackedBoxes, min_iou: float = DEFAULT_MIN_IOU
) -> dict[int, int]:
    """Which labelled player each track ID is, paired as in ``score_identities``."""
    shared: Counter[tuple[int, int]] = Counter()
    for frame in segment.reviewed_box_frames:
        truth = segment.boxes.get(frame, [])
        found = list(tracks.get(frame, []))
        for row, column in match_boxes(truth, [d for _, d in found], min_iou):
            shared[truth[row].player, found[column][0]] += 1
    return {track_id: player for player, track_id in _pairs(shared)}


def score_hitters(
    labels: VideoLabels,
    impacts_by_segment: Mapping[int, Sequence[tuple[int, int | None]]],
    tracks_by_segment: Mapping[int, TrackedBoxes],
    tolerance: int = 2,
) -> tuple[int, int]:
    """(right, scored): found impacts whose hitter is the labelled player.

    ``impacts_by_segment`` holds (frame, hitter track ID) per labelled segment.
    Only predicted impacts within ``tolerance`` frames of a labelled impact are
    scored; the track ID becomes a player number through ``players_by_track``.
    """
    right = scored = 0
    for segment in labels.segments:
        if segment.status != "label":
            continue
        player_of = players_by_track(segment, tracks_by_segment.get(segment.start, {}))
        predicted = dict(impacts_by_segment.get(segment.start, []))
        truth = {e.frame: e.player for e in segment.events if e.kind == "impact"}
        for label_frame, frame in match_frames(
            sorted(truth), sorted(predicted), tolerance
        ):
            scored += 1
            right += player_of.get(predicted[frame]) == truth[label_frame]
    return right, scored


def replay_tracker(
    make_tracker: Callable[[], object],
    boxes_by_frame: Mapping[int, Sequence[Detection]],
    frames: range,
    fps: float,
) -> dict[int, list[tuple[int, Detection]]]:
    """Track IDs by frame from a new tracker fed the boxes, frame by frame."""
    tracker = make_tracker()
    return {
        frame: [
            (t.track_id, t.detection)
            for t in tracker.update(
                list(boxes_by_frame.get(frame, [])), timestamp_seconds=frame / fps
            )
        ]
        for frame in frames
    }


def _paired_frames(shared: Counter[tuple[int, int]]) -> int:
    """Most frames covered when each player is paired with one track ID."""
    return sum(shared[pair] for pair in _pairs(shared))


def _pairs(shared: Counter[tuple[int, int]]) -> list[tuple[int, int]]:
    """(player, track ID) pairs, one to one, covering the most shared frames.

    The pairing maximises shared frames, not the number of pairs: a player who
    shares 10 frames with one ID keeps it, even if that leaves another player
    with no ID at all.
    """
    if not shared:
        return []
    players = sorted({player for player, _ in shared})
    track_ids = sorted({track_id for _, track_id in shared})
    counts = np.array([[shared[p, t] for t in track_ids] for p in players])
    rows, columns = linear_sum_assignment(counts, maximize=True)
    return [
        (players[row], track_ids[column])
        for row, column in zip(rows, columns, strict=True)
        if counts[row, column] > 0
    ]


def score_teams(
    labels: VideoLabels,
    tracks_by_segment: Mapping[int, TrackedBoxes],
    teams_by_segment: Mapping[int, Mapping[int, str]],
    min_iou: float = DEFAULT_MIN_IOU,
) -> tuple[int, int]:
    """(right, scored): labelled boxes whose track's team matches the label's.

    Labelled players 1 and 2 are the near team and 3 and 4 the far team
    (docs/evaluation.md). Only boxes matched to a tracked box are scored.
    """
    right = scored = 0
    for segment in labels.segments:
        if segment.status != "label":
            continue
        tracks = tracks_by_segment.get(segment.start, {})
        teams = teams_by_segment.get(segment.start, {})
        for frame in segment.reviewed_box_frames:
            truth = segment.boxes.get(frame, [])
            found = list(tracks.get(frame, []))
            for row, column in match_boxes(truth, [d for _, d in found], min_iou):
                expected = "near" if truth[row].player in (1, 2) else "far"
                scored += 1
                right += teams.get(found[column][0]) == expected
    return right, scored
