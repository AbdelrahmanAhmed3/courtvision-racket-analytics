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
from math import hypot
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.analytics.ball_track import (  # noqa: E402
    detect_impacts,
    event_candidates,
    split_into_tracklets,
    turn_angle,
)
from courtvision.calibration.homography import project_image_points  # noqa: E402
from courtvision.detectors.base import Detection  # noqa: E402
from courtvision.detectors.tracknet_adapter import BallPoint  # noqa: E402
from courtvision.evaluation.court import court_estimate  # noqa: E402
from courtvision.evaluation.labels import load_labels  # noqa: E402
from courtvision.evaluation.scoring import (  # noqa: E402
    PredictedEvent,
    load_predictions,
    score_events,
)

FEATURES = (
    "speed_in",  # px per frame, over the last 4 positions up to 10 frames before
    "speed_out",  # px per frame, over the first 4 positions up to 10 frames after
    "turn_degrees",  # between the directions before and after
    "dy_in",  # vertical movement before, px per frame (positive: down the image)
    "dy_out",
    "hidden_frames",  # frames within 3 of the candidate without a ball
    "player_distance",  # ball to nearest player box, in box heights
    "box_y",  # height in that box: 0 at the head, 1 at the feet
    "box_x",  # across that box: 0 at its middle, ±0.5 at its sides
    "court_y",  # along the court, 0 at the far baseline, 1 at the near one
    "net_distance",  # |court_y - 0.5|
    "on_court",  # 1 if the ball projects inside the court (not beside it)
    "crosses_net",  # 1 if the ball is on the other half 3-20 frames later
    "court_y_change",  # court_y 3-20 frames later minus 3-15 frames before
)
CLASSES = ("impact", "bounce", "wall_rebound", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="Labels JSON.")
    parser.add_argument("--size", default="nano", help="RF-DETR size of the cache.")
    return parser.parse_args()


def velocity(points: list[BallPoint]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    first, last = points[0], points[-1]
    frames = last.frame - first.frame
    return ((last.x - first.x) / frames, (last.y - first.y) / frames)


def candidate_rows(segment, cached: dict) -> list[tuple[int, list[float], str]]:
    """(frame, features, label) for each candidate inside the segment's rally."""
    raw = [BallPoint(f, x, y, 0.0 if x is None else 1.0) for f, x, y in cached["ball"]]
    clean = {
        point.frame: point
        for tracklet in split_into_tracklets(raw)
        if len(tracklet) >= 3
        for point in tracklet
    }
    boxes, players = defaultdict(list), defaultdict(list)
    for frame, track_id, x1, y1, x2, y2, confidence in cached["players"]:
        boxes[frame].append((x1, y1, x2, y2))
        detection = Detection(frame, "person", confidence, x1, y1, x2, y2, "cached")
        players[frame].append((track_id, detection))
    estimate = court_estimate(segment.court_points)

    def court(point: BallPoint) -> tuple[float, float]:
        x, y = project_image_points(np.float32([[point.x, point.y]]), estimate)[0]
        return float(x), float(y)

    rows = []
    for frame in event_candidates(raw, players):
        if not segment.rally_start - 2 <= frame <= segment.rally_end + 2:
            continue
        values = dict.fromkeys(FEATURES, np.nan)
        # After a hidden gap the nearest positions can be several frames away.
        before = [clean[f] for f in range(frame - 10, frame + 1) if f in clean][-4:]
        after = [clean[f] for f in range(frame, frame + 11) if f in clean][:4]
        incoming, outgoing = velocity(before), velocity(after)
        if incoming:
            values["speed_in"], values["dy_in"] = hypot(*incoming), incoming[1]
        if outgoing:
            values["speed_out"], values["dy_out"] = hypot(*outgoing), outgoing[1]
        if incoming and outgoing:
            values["turn_degrees"] = turn_angle(incoming, outgoing)
        values["hidden_frames"] = sum(
            f not in clean for f in range(frame - 3, frame + 4)
        )
        nearby = [clean[f] for f in range(frame - 4, frame + 5) if f in clean]
        if nearby:
            ball = min(nearby, key=lambda point: abs(point.frame - frame))
            if boxes.get(ball.frame):
                _player_features(values, ball, boxes[ball.frame])
            court_x, court_y = court(ball)
            values["court_y"], values["net_distance"] = court_y, abs(court_y - 0.5)
            inside = -0.05 <= court_x <= 1.05 and -0.05 <= court_y <= 1.05
            values["on_court"] = float(inside)
            early = [
                court(clean[f])[1] for f in range(frame - 15, frame - 2) if f in clean
            ]
            late = [
                court(clean[f])[1] for f in range(frame + 3, frame + 21) if f in clean
            ]
            if early and late:
                early_y, late_y = float(np.median(early)), float(np.median(late))
                values["crosses_net"] = float((early_y - 0.5) * (late_y - 0.5) < 0)
                values["court_y_change"] = late_y - early_y
        near_events = [
            (abs(event.frame - frame), event.kind)
            for event in segment.events
            if abs(event.frame - frame) <= 2
        ]
        label = min(near_events)[1] if near_events else "none"
        rows.append((frame, [values[name] for name in FEATURES], label))
    return rows


def _player_features(values: dict, ball: BallPoint, boxes: list) -> None:
    def gap(box):
        x1, y1, x2, y2 = box
        return hypot(max(x1 - ball.x, 0, ball.x - x2), max(y1 - ball.y, 0, ball.y - y2))

    x1, y1, x2, y2 = min(boxes, key=gap)
    height, width = y2 - y1, x2 - x1
    values["player_distance"] = gap((x1, y1, x2, y2)) / height
    values["box_y"] = (ball.y - y1) / height
    values["box_x"] = (ball.x - (x1 + x2) / 2) / width


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
    rows = {
        s.start: candidate_rows(s, tracks["segments"][str(s.start)]) for s in segments
    }
    halves = [[s.start for s in segments[0::2]], [s.start for s in segments[1::2]]]

    predicted = defaultdict(list)
    truth, boosted, importances = [], [], defaultdict(list)
    for test, train in (halves, halves[::-1]):
        x_train = np.array([x for s in train for _, x, _ in rows[s]])
        y_train = [y for s in train for _, _, y in rows[s]]
        x_test = np.array([x for s in test for _, x, _ in rows[s]])
        y_test = [y for s in test for _, _, y in rows[s]]
        frames = [f for s in test for f, _, _ in rows[s]]
        for name, model in build_models().items():
            model.fit(x_train, y_train)
            answers = model.predict(x_test)
            predicted[name] += [
                PredictedEvent(kind, frame)
                for frame, kind in zip(frames, answers, strict=True)
                if kind != "none"
            ]
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
