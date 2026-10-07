"""Score predicted ball events against hand labels.

A prediction counts as a hit when it lands within ``tolerance`` frames of a
labelled event of the same kind, and each labelled event takes at most one
prediction: three predictions for one impact are one hit and two false alarms.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.analytics.shots import ShotEvent
from courtvision.evaluation.labels import EVENT_KINDS, VideoLabels

DEFAULT_TOLERANCE_FRAMES = 2


@dataclass(frozen=True)
class PredictedEvent:
    kind: str
    frame: int


@dataclass(frozen=True)
class Score:
    hits: int
    predicted: int
    labelled: int
    frame_errors: tuple[int, ...] = ()  # predicted minus labelled frame, per hit

    @property
    def precision(self) -> float:
        return self.hits / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.hits / self.labelled if self.labelled else 0.0


def match_frames(
    labelled: Sequence[int], predicted: Sequence[int], tolerance: int
) -> list[tuple[int, int]]:
    """Pair labelled and predicted frames at most ``tolerance`` apart, one to one.

    Finds the most pairs, then the smallest total frame error among those. Taking
    the nearest prediction for each label in turn can use up the only prediction
    a later label could have matched, so all pairs are chosen at once with an
    optimal assignment (the Hungarian algorithm).
    """
    if not labelled or not predicted:
        return []
    errors = np.abs(np.subtract.outer(labelled, predicted))
    pairs = pair_one_to_one(errors, errors <= tolerance)
    return [(labelled[row], predicted[column]) for row, column in pairs]


def pair_one_to_one(cost: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Row and column indexes of the most allowed pairs, then the lowest total cost."""
    if cost.size == 0:
        return []
    # One more pair always outweighs any total cost of the allowed pairs.
    forbidden_cost = cost[allowed].max(initial=0) * min(cost.shape) + 1
    rows, columns = linear_sum_assignment(np.where(allowed, cost, forbidden_cost))
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if allowed[row, column]
    ]


def score_events(
    labels: VideoLabels,
    predictions: Sequence[PredictedEvent],
    tolerance: int = DEFAULT_TOLERANCE_FRAMES,
) -> dict[str, Score]:
    """Precision and recall per event kind, over the labelled rallies only.

    Predictions outside labelled rallies are ignored: nothing there was
    labelled, so they can be neither right nor wrong. A rally window is widened
    by the tolerance, so a prediction just before the serve still counts.
    """
    scores = {}
    for kind in EVENT_KINDS:
        hits = predicted = labelled = 0
        frame_errors: list[int] = []
        for segment in labels.segments:
            if segment.status != "label":
                continue
            first = (
                segment.start if segment.rally_start is None else segment.rally_start
            )
            last = segment.end - 1 if segment.rally_end is None else segment.rally_end
            start = max(segment.start, first - tolerance)
            end = min(segment.end - 1, last + tolerance)
            truth = sorted(e.frame for e in segment.events if e.kind == kind)
            guesses = sorted(
                p.frame
                for p in predictions
                if p.kind == kind and start <= p.frame <= end
            )
            pairs = match_frames(truth, guesses, tolerance)
            hits += len(pairs)
            predicted += len(guesses)
            labelled += len(truth)
            frame_errors += [guess - label for label, guess in pairs]
        scores[kind] = Score(hits, predicted, labelled, tuple(frame_errors))
    return scores


def predictions_from_shots(shots: Sequence[ShotEvent]) -> list[PredictedEvent]:
    """Impacts and bounces that ``analytics.shots.detect_shots`` reports.

    Each shot holds the hitter's impact, the receiver's impact and the first
    bounce between them; consecutive shots share an impact, which counts once.
    The shot code reports no wall rebounds.
    """
    impacts = {shot.frame for shot in shots} | {shot.receive_frame for shot in shots}
    bounces = {shot.bounce_frame for shot in shots if shot.bounce_frame is not None}
    return [PredictedEvent("impact", frame) for frame in sorted(impacts)] + [
        PredictedEvent("bounce", frame) for frame in sorted(bounces)
    ]


def save_predictions(path: str | Path, predictions: Sequence[PredictedEvent]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [{"kind": event.kind, "frame": event.frame} for event in predictions]
    path.write_text(json.dumps({"events": events}, indent=1) + "\n")


def load_predictions(path: str | Path) -> list[PredictedEvent]:
    """Read ``{"events": [{"kind": "impact", "frame": 512}, ...]}``."""
    data = json.loads(Path(path).read_text())
    events = [
        PredictedEvent(item["kind"], int(item["frame"])) for item in data["events"]
    ]
    unknown = {event.kind for event in events} - set(EVENT_KINDS)
    if unknown:
        raise ValueError(f"Unknown event kinds in predictions: {sorted(unknown)}")
    return events
