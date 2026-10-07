"""Score player detections against the hand-labelled player boxes.

A detection is a hit when it overlaps a labelled box by at least ``min_iou``
(intersection over union), one to one as for events. By default detections are
first treated the way the pipeline treats them: people standing on the court,
plus its margin, then the four most confident. Crowd, officials and ball kids
are then never false alarms, for any detector.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from courtvision.analytics.court_coordinates import filter_detections_on_court
from courtvision.detectors.base import Detection, expected_player_count
from courtvision.evaluation.court import court_estimate
from courtvision.evaluation.labels import PlayerBox, Segment, VideoLabels
from courtvision.evaluation.scoring import Score, pair_one_to_one
from courtvision.visualization.minimap import PADEL_COURT

DEFAULT_MIN_IOU = 0.5


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Intersection over union of every box in ``first`` with every box in ``second``.

    Boxes are rows of x1, y1, x2, y2.
    """
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    overlap = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    first_area = (first[:, 2:] - first[:, :2]).prod(axis=1)
    second_area = (second[:, 2:] - second[:, :2]).prod(axis=1)
    return overlap / (first_area[:, None] + second_area[None, :] - overlap)


def match_boxes(
    labelled: Sequence[PlayerBox],
    detected: Sequence[Detection],
    min_iou: float = DEFAULT_MIN_IOU,
) -> list[tuple[int, int]]:
    """Index pairs (labelled, detected) overlapping by ``min_iou``, one to one."""
    if not labelled or not detected:
        return []
    iou = box_iou(
        np.array([[b.x1, b.y1, b.x2, b.y2] for b in labelled], dtype=float),
        np.array([[d.x1, d.y1, d.x2, d.y2] for d in detected], dtype=float),
    )
    return pair_one_to_one(1 - iou, iou >= min_iou)


def players_as_the_pipeline_keeps_them(
    detections: Sequence[Detection], segment: Segment
) -> list[Detection]:
    """On the segment's court plus margin, then the four most confident."""
    on_court = filter_detections_on_court(
        list(detections), court_estimate(segment.court_points), PADEL_COURT
    )
    by_confidence = sorted(on_court, key=lambda d: d.confidence, reverse=True)
    return by_confidence[: expected_player_count("padel")]


def score_detections(
    labels: VideoLabels,
    detections: Sequence[Detection],
    min_iou: float = DEFAULT_MIN_IOU,
    as_the_pipeline: bool = True,
) -> Score:
    """Precision and recall over every finished box frame of labelled segments.

    Every labelled player counts towards recall, including players outside the
    court, whom the pipeline's court filter drops.
    """
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_frame[detection.frame].append(detection)
    hits = detected = labelled = 0
    for segment in labels.segments:
        if segment.status != "label":
            continue
        for frame in segment.reviewed_box_frames:
            truth = segment.boxes.get(frame, [])
            found = by_frame.get(frame, [])
            if as_the_pipeline:
                found = players_as_the_pipeline_keeps_them(found, segment)
            hits += len(match_boxes(truth, found, min_iou))
            detected += len(found)
            labelled += len(truth)
    return Score(hits, detected, labelled)


def box_frames(labels: VideoLabels) -> list[int]:
    """Every finished box frame of labelled segments, in order."""
    return sorted(
        frame
        for segment in labels.segments
        if segment.status == "label"
        for frame in segment.reviewed_box_frames
    )


def save_detections(path: str | Path, detections: Sequence[Detection]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"detections": [asdict(detection) for detection in detections]}
    path.write_text(json.dumps(data, indent=1) + "\n")


def load_detections(path: str | Path) -> list[Detection]:
    data = json.loads(Path(path).read_text())
    return [Detection(**item) for item in data["detections"]]
