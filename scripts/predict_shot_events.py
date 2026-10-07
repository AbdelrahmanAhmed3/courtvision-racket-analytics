"""Run today's shot code on every labelled rally and save its predicted events.

Example:
    python scripts/predict_shot_events.py \
        --labels evaluation/labels/<video>.json --video data/raw/<video>.mp4
    python scripts/score_events.py --labels evaluation/labels/<video>.json \
        --predictions outputs/predictions/<video>--shots.json

Each labelled segment goes through the pipeline's steps: RF-DETR players kept on
the segment's hand-marked court, the IoU tracker, the TrackNet ball, and
analytics.shots.detect_shots. Its impacts and bounces are the baseline that
shot and bounce detection (#8) must beat. Needs the local and tracknet extras
and the TrackNet weights (see README).

Players and ball are cached in outputs/predictions/ after the first run (about
4 frames per second), so changing the shot code and running again takes seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.analytics.court_coordinates import (  # noqa: E402
    filter_detections_on_court,
    project_ball_point,
)
from courtvision.analytics.shots import detect_shots  # noqa: E402
from courtvision.detectors.base import (  # noqa: E402
    Detection,
    expected_player_count,
    filter_player_detections,
)
from courtvision.detectors.rfdetr_detector import RFDetrDetector  # noqa: E402
from courtvision.detectors.tracknet_adapter import (  # noqa: E402
    BallPoint,
    get_device,
    infer_ball_point,
    load_tracknet_model,
)
from courtvision.evaluation.court import court_estimate  # noqa: E402
from courtvision.evaluation.labels import load_labels  # noqa: E402
from courtvision.evaluation.scoring import (  # noqa: E402
    predictions_from_shots,
    save_predictions,
)
from courtvision.tracking.simple_tracker import SimpleIouTracker  # noqa: E402
from courtvision.visualization.minimap import PADEL_COURT  # noqa: E402

TRACKNET_DIR = REPO_ROOT / "tracknet-model" / "TrackNet"
TRACKNET_WEIGHTS = REPO_ROOT / "tracknet-model" / "model_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="Labels JSON.")
    parser.add_argument("--video", required=True, help="The labelled video.")
    parser.add_argument("--output", help="Predictions JSON (default: outputs/).")
    parser.add_argument("--size", default="nano", help="RF-DETR size.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run_models(args: argparse.Namespace, labels) -> dict:
    """Players and ball on every frame of each labelled segment (the slow part)."""
    import torch

    detector = RFDetrDetector(size=args.size, device=args.device)
    device = get_device(args.device)
    ball_model = load_tracknet_model(TRACKNET_DIR, TRACKNET_WEIGHTS, device)
    max_players = expected_player_count("padel")
    capture = cv2.VideoCapture(args.video)
    tracks = {}
    segments = [s for s in labels.segments if s.status == "label"]
    for number, segment in enumerate(segments, start=1):
        estimate = court_estimate(segment.court_points)
        tracker = SimpleIouTracker()
        ball, players, history = [], [], []
        capture.set(cv2.CAP_PROP_POS_FRAMES, segment.start)
        frames = range(segment.start, segment.end)
        for frame_index in tqdm(frames, desc=f"Segment {number}/{len(segments)}"):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {frame_index}")
            detections = filter_detections_on_court(
                detector.predict_frame(frame, frame_index), estimate, PADEL_COURT
            )
            tracked = tracker.update(
                filter_player_detections(detections, max_players=max_players),
                timestamp_seconds=frame_index / labels.fps,
            )
            players += [
                [frame_index, t.track_id, *_box(t.detection), t.detection.confidence]
                for t in tracked
            ]
            history = [*history[-2:], frame]
            if len(history) == 3:
                with torch.no_grad():
                    point = infer_ball_point(frame_index, history, ball_model, device)
                ball.append([frame_index, point.x, point.y])
        tracks[str(segment.start)] = {"ball": ball, "players": players}
    capture.release()
    return {"segments": tracks}


def _box(detection) -> list[float]:
    return [detection.x1, detection.y1, detection.x2, detection.y2]


def predict_shots(labels, tracks: dict) -> list:
    """analytics.shots.detect_shots on each segment's cached players and ball."""
    shots = []
    for segment in labels.segments:
        if segment.status != "label":
            continue
        cached = tracks["segments"][str(segment.start)]
        estimate = court_estimate(segment.court_points)
        ball_points = {
            f: BallPoint(f, x, y, 0.0 if x is None else 1.0)
            for f, x, y in cached["ball"]
        }
        ball_coordinates = {}
        for frame, point in ball_points.items():
            coordinate = project_ball_point(point, estimate, PADEL_COURT)
            if coordinate is not None:
                ball_coordinates[frame] = coordinate
        players = defaultdict(list)
        for frame, track_id, x1, y1, x2, y2, confidence in cached["players"]:
            detection = Detection(frame, "person", confidence, x1, y1, x2, y2, "cached")
            players[frame].append((track_id, detection))
        shots += detect_shots(ball_points, players, ball_coordinates, labels.fps)
    return shots


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    stem = Path(args.labels).stem
    output = Path(
        args.output or REPO_ROOT / "outputs" / "predictions" / f"{stem}--shots.json"
    )
    cache = REPO_ROOT / "outputs" / "predictions" / f"{stem}--tracks-{args.size}.json"
    if cache.exists():
        tracks = json.loads(cache.read_text())
    else:
        tracks = run_models(args, labels)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(tracks) + "\n")
    shots = predict_shots(labels, tracks)
    predictions = predictions_from_shots(shots)
    save_predictions(output, predictions)
    print(f"{len(shots)} shots, {len(predictions)} predicted events: {output}")


if __name__ == "__main__":
    main()
