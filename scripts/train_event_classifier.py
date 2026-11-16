"""Train a small classifier for impacts and bounces, and score it.

Example (after scripts/predict_shot_events.py has cached the players and ball):
    python scripts/train_event_classifier.py \
        --labels evaluation/labels/<video>.json

Candidates come from analytics.ball_track.event_candidates. Each gets a few
numbers computed from the ball track, the player boxes and the hand-marked
court, and the label of the nearest labelled impact or bounce within 2 frames
(or none). Candidates before the serve and after the point are "none" too, so
the models learn that the server's bounce before serving is not a hit. Wall
rebounds are labelled but not learned (#74). Impacts a few frames apart are
merged into one, as the pipeline does.
The rallies are split in two halves (or one rally at a time, with
--split rallies); a model trained on the others predicts each part, so every
rally is scored by a model that never saw it. Needs the train extra
(scikit-learn).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.analytics.ball_events import (  # noqa: E402
    FEATURES,
    BallEvent,
    EventModel,
    candidate_features,
    merge_close_impacts,
)
from courtvision.analytics.ball_track import detect_impacts  # noqa: E402
from courtvision.analytics.rallies import (  # noqa: E402
    DEFAULT_MAX_GAP_SECONDS,
    segment_rallies,
)
from courtvision.detectors.base import Detection  # noqa: E402
from courtvision.detectors.tracknet_adapter import BallPoint  # noqa: E402
from courtvision.evaluation.court import court_estimate  # noqa: E402
from courtvision.evaluation.identity import replay_tracker, score_hitters  # noqa: E402
from courtvision.evaluation.labels import load_labels  # noqa: E402
from courtvision.evaluation.scoring import (  # noqa: E402
    PredictedEvent,
    load_predictions,
    score_events,
    score_rallies,
)
from courtvision.tracking.fixed_count_tracker import FixedCountTracker  # noqa: E402

# Wall rebounds are labelled but not learned: from the image alone they look like
# bounces, and 4 in 5 of those reported were wrong. They come back with the ball's
# height (#73, #74).
CLASSES = ("impact", "bounce", "none")
# The model the pipeline runs; the hitter and rally scores below are for it.
# With close hits merged and trained on the moments outside the rallies too, the
# trees beat logistic regression on the Tapia rallies and tie with it on the
# Qatar clip (#85, reports/ball_tracker.md).
SHIPPED = "boosted trees"
PREDICTIONS_DIR = REPO_ROOT / "outputs" / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="Labels JSON.")
    parser.add_argument("--size", default="nano", help="RF-DETR size of the cache.")
    parser.add_argument(
        "--split",
        choices=("halves", "rallies"),
        default="halves",
        help="Score each half of the rallies with a model trained on the other "
        "half, or each rally with a model trained on the other nine.",
    )
    parser.add_argument(
        "--test-labels",
        help="Another labelled video, cached the same way: train on every rally of "
        "--labels and score on its rallies.",
    )
    parser.add_argument(
        "--save-model",
        help=f"Also train {SHIPPED} on every labelled rally and save them as JSON "
        "for analytics.ball_events (e.g. "
        "src/courtvision/analytics/ball_events_model.json).",
    )
    return parser.parse_args()


def load_rallies(path: str, size: str) -> tuple:
    """A labelled video, its cached players and ball, and its rallies' candidates."""
    labels = load_labels(path)
    cache = PREDICTIONS_DIR / f"{Path(path).stem}--tracks-{size}.json"
    tracks = json.loads(cache.read_text())
    segments = [s for s in labels.segments if s.status == "label"]
    players, rows = {}, {}
    for segment in segments:
        cached = tracks["segments"][str(segment.start)]
        players[segment.start] = tracked_players(segment, cached, labels.fps)
        rows[segment.start] = candidate_rows(segment, cached, players[segment.start])
    return labels, tracks, segments, players, rows


def tracked_players(segment, cached: dict, fps: float) -> dict:
    """The cached player boxes with IDs from the pipeline's four-ID tracker."""
    boxes = defaultdict(list)
    for frame, _, x1, y1, x2, y2, confidence in cached["players"]:
        boxes[frame].append(Detection(frame, "person", confidence, x1, y1, x2, y2, ""))
    frames = range(segment.start, segment.end)
    return replay_tracker(FixedCountTracker, boxes, frames, fps)


def candidate_rows(segment, cached: dict, players: dict) -> list[tuple]:
    """(frame, features, label, nearest track ID) per candidate in the segment.

    Outside the rally nothing is labelled, so every candidate there is "none".
    """
    ball = [BallPoint(f, x, y, 0.0 if x is None else 1.0) for f, x, y in cached["ball"]]
    rows = []
    estimate = court_estimate(segment.court_points)
    for candidate in candidate_features(ball, players, estimate):
        frame = candidate.frame
        near_events = [
            (abs(event.frame - frame), event.kind)
            for event in segment.events
            if abs(event.frame - frame) <= 2 and event.kind in CLASSES
        ]
        label = "none"
        if near_events and in_rally(segment, frame):
            label = min(near_events)[1]
        rows.append(
            (frame, list(candidate.features), label, candidate.nearest_track_id)
        )
    return rows


def found_events_of(model, segment_rows: list[tuple]) -> list[BallEvent]:
    """A fitted model's impacts and bounces, with how sure it is of each."""
    probabilities = model.predict_proba(np.array([row[1] for row in segment_rows]))
    events = []
    for (frame, _, _, track_id), row in zip(segment_rows, probabilities, strict=True):
        kind = str(model.classes_[row.argmax()])
        if kind != "none":
            track_id = track_id if kind == "impact" else None
            events.append(BallEvent(kind, frame, track_id, float(row.max())))
    return events


def in_rally(segment, frame: int) -> bool:
    """Within the labelled rally, give or take the 2 frames events are scored at."""
    return segment.rally_start - 2 <= frame <= segment.rally_end + 2


def outside_hits(segments, events_by_segment: dict) -> int:
    """Impacts reported before the serve or after the point: all of them false."""
    return sum(
        event.kind == "impact" and not in_rally(segment, event.frame)
        for segment in segments
        for event in events_by_segment.get(segment.start, [])
    )


def build_models() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ),
        "boosted trees": HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=150,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=0,
        ),
    }


def main() -> None:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import confusion_matrix

    args = parse_args()
    labels, tracks, segments, players, rows = load_rallies(args.labels, args.size)
    if args.split == "halves":
        folds = [[s.start for s in segments[0::2]], [s.start for s in segments[1::2]]]
    else:
        folds = [[s.start] for s in segments]

    predicted = defaultdict(list)  # close impacts merged, as the pipeline does
    unmerged = []  # the shipped model's events before merging close impacts
    hitters = defaultdict(list)  # segment -> (frame, track ID) of predicted impacts
    found_events = defaultdict(list)  # segment -> events of the shipped model
    outside = defaultdict(int)  # method -> impacts before the serve or after the point
    by_start = {segment.start: segment for segment in segments}
    truth, boosted, importances = [], [], defaultdict(list)
    for test in folds:
        train = [s.start for s in segments if s.start not in test]
        x_train = np.array([row[1] for s in train for row in rows[s]])
        y_train = [row[2] for s in train for row in rows[s]]
        x_test = np.array([row[1] for s in test for row in rows[s]])
        y_test = [row[2] for s in test for row in rows[s]]
        for name, model in build_models().items():
            model.fit(x_train, y_train)
            for segment in test:
                found = found_events_of(model, rows[segment])
                if name == SHIPPED:
                    unmerged += [PredictedEvent(e.kind, e.frame) for e in found]
                    outside[f"{SHIPPED}, hits not merged"] += outside_hits(
                        [by_start[segment]], {segment: found}
                    )
                found = merge_close_impacts(found, labels.fps)
                predicted[name] += [PredictedEvent(e.kind, e.frame) for e in found]
                outside[name] += outside_hits([by_start[segment]], {segment: found})
                if name == SHIPPED:
                    found_events[segment] = found
                    hitters[segment] = [
                        (e.frame, e.track_id) for e in found if e.kind == "impact"
                    ]
            if name == "boosted trees":
                truth += y_test
                boosted += list(model.predict(x_test))
                result = permutation_importance(
                    model,
                    x_test,
                    y_test,
                    n_repeats=20,
                    random_state=0,
                    scoring="balanced_accuracy",
                )
                for feature, value in zip(
                    FEATURES, result.importances_mean, strict=True
                ):
                    importances[feature].append(value)

    methods = {}
    shots = PREDICTIONS_DIR / f"{Path(args.labels).stem}--shots.json"
    if shots.exists():
        methods["today's shot code"] = load_predictions(shots)
    methods["rule: tracklet ends near a player"] = _rule_predictions(segments, tracks)
    methods[f"{SHIPPED}, hits not merged"] = unmerged
    methods.update(predicted)

    print(
        f"{labels.video}: ±2 frames; every rally scored by a model that never saw "
        f"it (split: {args.split})"
    )
    print(
        "Hits outside: impacts reported before the serve or after the point, "
        "where every one is false."
    )
    print(f"{'method':40}{'impact P/R':>13}{'bounce P/R':>13}{'hits outside':>14}")
    for name, events in methods.items():
        scores = score_events(labels, events, 2)
        print(
            f"{name:40}{_cell(scores['impact']):>13}{_cell(scores['bounce']):>13}"
            f"{outside.get(name, '-'):>14}"
        )

    print("\nBoosted trees on the candidates (rows: label, columns: answer)")
    print(f"{'':14}" + "".join(f"{name:>14}" for name in CLASSES))
    for name, row in zip(
        CLASSES, confusion_matrix(truth, boosted, labels=CLASSES), strict=True
    ):
        print(f"{name:14}" + "".join(f"{value:14}" for value in row))
    print("\nInputs that matter most (drop in balanced accuracy when shuffled)")
    ranked = sorted(importances.items(), key=lambda item: -np.mean(item[1]))
    for feature, values in ranked[:6]:
        print(f"  {feature:16}{np.mean(values):+.3f}")
    right, scored = score_hitters(labels, hitters, players)
    print(
        f"\nHitter of the impacts found ({SHIPPED}, four-ID tracker, "
        f"nearest player): {right}/{scored} = {right / scored:.2f}"
    )
    print(
        f"\nRallies from the events of {SHIPPED} (seconds; error is predicted "
        "minus labelled)"
    )
    for gap in (3.0, DEFAULT_MAX_GAP_SECONDS, 8.0):
        rallies = {
            s: segment_rallies(found_events[s], labels.fps, max_gap_seconds=gap)
            for s in found_events
        }
        score = score_rallies(labels, rallies)
        start, end = np.abs(score.start_errors), np.abs(score.end_errors)
        print(
            f"  silence {gap:.0f} s: found {score.found}/{score.labelled}, "
            f"extra {score.extra}; start error median {np.median(start):.2f} "
            f"(mean signed {np.mean(score.start_errors):+.2f}), end error median "
            f"{np.median(end):.2f} (mean signed {np.mean(score.end_errors):+.2f})"
        )
    if args.test_labels:
        score_other_video(args.test_labels, args.size, rows)
    if args.save_model:
        save_model(args.save_model, rows, labels)


def score_other_video(path: str, size: str, rows: dict) -> None:
    """Train each model on every rally above, then score it on another video."""
    other, _, segments, players, other_rows = load_rallies(path, size)
    x = np.array([row[1] for segment_rows in rows.values() for row in segment_rows])
    y = [row[2] for segment_rows in rows.values() for row in segment_rows]
    print(f"\n{other.video}: every model trained on all the rallies above, ±2 frames")
    print(
        f"{'method':40}{'impact P/R':>13}{'bounce P/R':>13}{'hitter':>9}"
        f"{'hits outside':>14}"
    )
    for name, model in build_models().items():
        model.fit(x, y)
        found = {s.start: found_events_of(model, other_rows[s.start]) for s in segments}
        versions = {f"{name}, hits not merged": found} if name == SHIPPED else {}
        versions[name] = {
            segment: merge_close_impacts(events, other.fps)
            for segment, events in found.items()
        }
        for label, by_segment in versions.items():
            events = [e for each in by_segment.values() for e in each]
            scores = score_events(
                other, [PredictedEvent(e.kind, e.frame) for e in events], 2
            )
            hitters = {
                segment: [(e.frame, e.track_id) for e in each if e.kind == "impact"]
                for segment, each in by_segment.items()
            }
            right, scored = score_hitters(other, hitters, players)
            print(
                f"{label:40}{_cell(scores['impact']):>13}{_cell(scores['bounce']):>13}"
                f"{f'{right}/{scored}':>9}{outside_hits(segments, by_segment):>14}"
            )


def _cell(score) -> str:
    return f"{score.precision:.2f}/{score.recall:.2f}" if score.predicted else "-"


def save_model(path: str, rows: dict, labels) -> None:
    """The shipped model trained on every candidate, exported for numpy."""
    model = build_models()[SHIPPED]
    x = np.array([row[1] for segment_rows in rows.values() for row in segment_rows])
    y = [row[2] for segment_rows in rows.values() for row in segment_rows]
    model.fit(x, y)
    exported = EventModel.from_scikit_learn(model)
    if exported.predict(x) != list(model.predict(x)) or not np.allclose(
        exported.probabilities(x), model.predict_proba(x)
    ):
        raise RuntimeError("The exported trees disagree with scikit-learn")
    exported.save(
        path,
        about=f"Boosted trees on {len(y)} candidates from the {len(rows)} "
        f"labelled segments of {labels.video}, before the serve and after the point "
        "included (scripts/train_event_classifier.py).",
    )
    print(f"\nSaved {SHIPPED} trained on all {len(rows)} rallies to {path}")


def _rule_predictions(segments, tracks: dict) -> list[PredictedEvent]:
    events = []
    for segment in segments:
        cached = tracks["segments"][str(segment.start)]
        ball = [
            BallPoint(f, x, y, 0.0 if x is None else 1.0) for f, x, y in cached["ball"]
        ]
        players = defaultdict(list)
        for frame, track_id, x1, y1, x2, y2, confidence in cached["players"]:
            detection = Detection(frame, "person", confidence, x1, y1, x2, y2, "cached")
            players[frame].append((track_id, detection))
        events += [
            PredictedEvent("impact", i.frame) for i in detect_impacts(ball, players)
        ]
    return events


if __name__ == "__main__":
    main()
