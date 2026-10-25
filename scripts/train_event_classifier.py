"""Train a small classifier for impacts, bounces and wall rebounds, and score it.

Example (after scripts/predict_shot_events.py has cached the players and ball):
    python scripts/train_event_classifier.py \
        --labels evaluation/labels/<video>.json

Candidates come from analytics.ball_track.event_candidates. Each gets a few
numbers computed from the ball track, the player boxes and the hand-marked
court, and the label of the nearest labelled event within 2 frames (or none).
The rallies are split in two halves; a model trained on one half predicts the
other, so every rally is scored by a model that never saw it. Needs the train
extra (scikit-learn).
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
    EventModel,
    candidate_features,
)
from courtvision.analytics.ball_track import detect_impacts  # noqa: E402
from courtvision.detectors.base import Detection  # noqa: E402
from courtvision.detectors.tracknet_adapter import BallPoint  # noqa: E402
from courtvision.evaluation.court import court_estimate  # noqa: E402
from courtvision.evaluation.identity import replay_tracker, score_hitters  # noqa: E402
from courtvision.evaluation.labels import load_labels  # noqa: E402
from courtvision.evaluation.scoring import (  # noqa: E402
    PredictedEvent,
    load_predictions,
    score_events,
)
from courtvision.tracking.fixed_count_tracker import FixedCountTracker  # noqa: E402

CLASSES = ("impact", "bounce", "wall_rebound", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="Labels JSON.")
    parser.add_argument("--size", default="nano", help="RF-DETR size of the cache.")
    parser.add_argument(
        "--save-model",
        help="Also train logistic regression on every labelled rally and save it "
        "as JSON for analytics.ball_events (e.g. "
        "src/courtvision/analytics/ball_events_model.json).",
    )
    return parser.parse_args()


def tracked_players(segment, cached: dict, fps: float) -> dict:
    """The cached player boxes with IDs from the pipeline's four-ID tracker."""
    boxes = defaultdict(list)
    for frame, _, x1, y1, x2, y2, confidence in cached["players"]:
        boxes[frame].append(Detection(frame, "person", confidence, x1, y1, x2, y2, ""))
    frames = range(segment.start, segment.end)
    return replay_tracker(FixedCountTracker, boxes, frames, fps)


def candidate_rows(segment, cached: dict, players: dict) -> list[tuple]:
    """(frame, features, label, nearest track ID) per candidate in the rally."""
    ball = [BallPoint(f, x, y, 0.0 if x is None else 1.0) for f, x, y in cached["ball"]]
    rows = []
    estimate = court_estimate(segment.court_points)
    for candidate in candidate_features(ball, players, estimate):
        frame = candidate.frame
        if not segment.rally_start - 2 <= frame <= segment.rally_end + 2:
            continue
        near_events = [
            (abs(event.frame - frame), event.kind)
            for event in segment.events
            if abs(event.frame - frame) <= 2
        ]
        label = min(near_events)[1] if near_events else "none"
        rows.append(
            (frame, list(candidate.features), label, candidate.nearest_track_id)
        )
    return rows


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
    labels = load_labels(args.labels)
    stem = Path(args.labels).stem
    predictions_dir = REPO_ROOT / "outputs" / "predictions"
    tracks = json.loads(
        (predictions_dir / f"{stem}--tracks-{args.size}.json").read_text()
    )
    segments = [s for s in labels.segments if s.status == "label"]
    players = {
        s.start: tracked_players(s, tracks["segments"][str(s.start)], labels.fps)
        for s in segments
    }
    rows = {
        s.start: candidate_rows(s, tracks["segments"][str(s.start)], players[s.start])
        for s in segments
    }
    halves = [[s.start for s in segments[0::2]], [s.start for s in segments[1::2]]]

    predicted = defaultdict(list)
    hitters = defaultdict(list)  # segment -> (frame, track ID) of predicted impacts
    truth, boosted, importances = [], [], defaultdict(list)
    for test, train in (halves, halves[::-1]):
        x_train = np.array([row[1] for s in train for row in rows[s]])
        y_train = [row[2] for s in train for row in rows[s]]
        x_test = np.array([row[1] for s in test for row in rows[s]])
        y_test = [row[2] for s in test for row in rows[s]]
        tested = [(s, row[0], row[3]) for s in test for row in rows[s]]
        for name, model in build_models().items():
            model.fit(x_train, y_train)
            answers = model.predict(x_test)
            predicted[name] += [
                PredictedEvent(kind, frame)
                for (_, frame, _), kind in zip(tested, answers, strict=True)
                if kind != "none"
            ]
            if name == "logistic regression":  # the model that ships
                for (segment, frame, track_id), kind in zip(
                    tested, answers, strict=True
                ):
                    if kind == "impact":
                        hitters[segment].append((frame, track_id))
            if name == "boosted trees":
                truth += y_test
                boosted += list(answers)
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
    shots = predictions_dir / f"{stem}--shots.json"
    if shots.exists():
        methods["today's shot code"] = load_predictions(shots)
    methods["rule: tracklet ends near a player"] = _rule_predictions(segments, tracks)
    methods.update(predicted)

    print(f"{labels.video}: ±2 frames; every rally scored by a model that never saw it")
    print(f"{'method':36}{'impact P/R':>13}{'bounce P/R':>13}{'wall P/R':>13}")
    for name, events in methods.items():
        scores = score_events(labels, events, 2)

        def cell(kind, scores=scores):
            score = scores[kind]
            return (
                f"{score.precision:.2f}/{score.recall:.2f}" if score.predicted else "-"
            )

        print(
            f"{name:36}{cell('impact'):>13}{cell('bounce'):>13}{cell('wall_rebound'):>13}"
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
        f"\nHitter of the impacts found (logistic regression, four-ID tracker, "
        f"nearest player): {right}/{scored} = {right / scored:.2f}"
    )
    if args.save_model:
        save_model(args.save_model, rows, labels)


def save_model(path: str, rows: dict, labels) -> None:
    """Logistic regression trained on every candidate, exported for numpy."""
    model = build_models()["logistic regression"]
    x = np.array([row[1] for segment_rows in rows.values() for row in segment_rows])
    y = [row[2] for segment_rows in rows.values() for row in segment_rows]
    model.fit(x, y)
    imputer, scaler, regression = (step for _, step in model.steps)
    EventModel(
        classes=tuple(str(c) for c in regression.classes_),
        fill=imputer.statistics_,
        mean=scaler.mean_,
        scale=scaler.scale_,
        coef=regression.coef_,
        intercept=regression.intercept_,
    ).save(
        path,
        about=f"Logistic regression on {len(y)} candidates from the {len(rows)} "
        f"labelled rallies of {labels.video} (scripts/train_event_classifier.py).",
    )
    print(f"\nSaved the model trained on all {len(rows)} rallies to {path}")


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
