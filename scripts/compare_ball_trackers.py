"""Compare ball trackers on the hand-labelled rallies: TrackNet and WASB.

Runs WASB (nttcom/WASB-SBDT, MIT) on every labelled rally and caches its ball
next to the TrackNet cache of scripts/predict_shot_events.py, with the same
players. Then measures both trackers the same way:

- how often each finds the ball during the rallies;
- how close it is at the labelled bounces and wall rebounds, the events whose
  ball position was clicked;
- the candidate ceiling: labelled events with an event candidate within ±2
  frames, the most the event classifier can find.

Example (after predict_shot_events.py has cached TrackNet):
    python scripts/compare_ball_trackers.py \
        --labels evaluation/labels/<video>.json --video data/raw/<video>.mp4

Hits, bounces and wall rebounds from the new cache, scored the usual way:
    python scripts/train_event_classifier.py --labels ... --size wasb
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from math import hypot
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.analytics.ball_events import candidate_features  # noqa: E402
from courtvision.detectors.base import Detection  # noqa: E402
from courtvision.detectors.tracknet_adapter import BallPoint, get_device  # noqa: E402
from courtvision.detectors.wasb_adapter import (  # noqa: E402
    infer_ball_track_frames,
    load_wasb_model,
)
from courtvision.evaluation.court import court_estimate  # noqa: E402
from courtvision.evaluation.identity import replay_tracker  # noqa: E402
from courtvision.evaluation.labels import load_labels  # noqa: E402
from courtvision.tracking.fixed_count_tracker import FixedCountTracker  # noqa: E402

PREDICTIONS_DIR = REPO_ROOT / "outputs" / "predictions"
WASB_DIR = REPO_ROOT / "wasb-model" / "WASB-SBDT"
WASB_WEIGHTS = REPO_ROOT / "wasb-model" / "wasb_tennis_best.pth.tar"
# Two or three ball widths on the 720p labelled video.
NEAR_PX = 15.0
CANDIDATE_FRAMES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="Labels JSON.")
    parser.add_argument("--video", required=True, help="The labelled video.")
    parser.add_argument("--wasb-dir", default=str(WASB_DIR))
    parser.add_argument("--wasb-weights", default=str(WASB_WEIGHTS))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    stem = Path(args.labels).stem
    tracknet = json.loads((PREDICTIONS_DIR / f"{stem}--tracks-nano.json").read_text())
    wasb_cache = PREDICTIONS_DIR / f"{stem}--tracks-wasb.json"
    if wasb_cache.exists():
        wasb = json.loads(wasb_cache.read_text())
    else:
        wasb = run_wasb(args, labels, tracknet)
        wasb_cache.write_text(json.dumps(wasb) + "\n")

    results = {
        "TrackNet": measure(labels, tracknet),
        "WASB (tennis)": measure(labels, wasb),
    }
    print_table(labels, results)


def run_wasb(args: argparse.Namespace, labels, tracknet: dict) -> dict:
    """WASB's ball on every frame of each labelled rally, with TrackNet's players."""
    device = get_device(args.device)
    model = load_wasb_model(args.wasb_dir, args.wasb_weights, device)
    capture = cv2.VideoCapture(args.video)
    segments = {}
    labelled = [s for s in labels.segments if s.status == "label"]
    for number, segment in enumerate(labelled, start=1):
        frames = tqdm(
            read_frames(capture, segment.start, segment.end),
            total=segment.end - segment.start,
            desc=f"Rally {number}/{len(labelled)}",
        )
        points = infer_ball_track_frames(frames, segment.start, model, device)
        segments[str(segment.start)] = {
            "ball": [[p.frame, p.x, p.y] for p in points],
            "players": tracknet["segments"][str(segment.start)]["players"],
        }
    capture.release()
    return {"segments": segments}


def read_frames(capture, start: int, end: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for frame_index in range(start, end):
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_index}")
        yield frame


def measure(labels, tracks: dict) -> dict:
    """Ball found in rallies, distance at labelled events, candidate ceiling."""
    rally_frames = seen = 0
    distances = defaultdict(list)
    positioned, labelled, covered = Counter(), Counter(), Counter()
    for segment in (s for s in labels.segments if s.status == "label"):
        cached = tracks["segments"][str(segment.start)]
        ball = {f: (x, y) for f, x, y in cached["ball"] if x is not None}
        rally = range(segment.rally_start, segment.rally_end + 1)
        rally_frames += len(rally)
        seen += sum(frame in ball for frame in rally)

        points = [
            BallPoint(f, x, y, 0.0 if x is None else 1.0) for f, x, y in cached["ball"]
        ]
        players = tracked_players(segment, cached, labels.fps)
        candidates = [
            candidate.frame
            for candidate in candidate_features(
                points, players, court_estimate(segment.court_points)
            )
        ]
        for event in segment.events:
            labelled[event.kind] += 1
            if any(abs(c - event.frame) <= CANDIDATE_FRAMES for c in candidates):
                covered[event.kind] += 1
            if event.x is None:
                continue
            positioned[event.kind] += 1
            if event.frame in ball:
                x, y = ball[event.frame]
                distances[event.kind].append(hypot(x - event.x, y - event.y))
    return {
        "seen": seen / rally_frames,
        "distances": distances,
        "positioned": positioned,
        "labelled": labelled,
        "covered": covered,
    }


def tracked_players(segment, cached: dict, fps: float) -> dict:
    """The cached player boxes with IDs from the pipeline's four-ID tracker."""
    boxes = defaultdict(list)
    for frame, _, x1, y1, x2, y2, confidence in cached["players"]:
        boxes[frame].append(Detection(frame, "person", confidence, x1, y1, x2, y2, ""))
    frames = range(segment.start, segment.end)
    return replay_tracker(FixedCountTracker, boxes, frames, fps)


def print_table(labels, results: dict) -> None:
    names = list(results)
    first = results[names[0]]
    print(f"{labels.video}: {sum(first['labelled'].values())} labelled events")
    print(f"{'':42}" + "".join(f"{name:>16}" for name in names))
    print(
        f"{'ball found during rallies':42}"
        + "".join(f"{r['seen']:>16.0%}" for r in results.values())
    )
    for kind in sorted(first["positioned"]):
        total = first["positioned"][kind]
        print(
            f"{kind + f': ball within {NEAR_PX:.0f} px (of {total})':42}"
            + "".join(
                f"{sum(d <= NEAR_PX for d in r['distances'][kind]):>16}"
                for r in results.values()
            )
        )
        print(
            f"{'  median distance where found (px)':42}"
            + "".join(
                f"{np.median(r['distances'][kind]):>16.1f}"
                if r["distances"][kind]
                else f"{'-':>16}"
                for r in results.values()
            )
        )
    for kind in sorted(first["labelled"]):
        total = first["labelled"][kind]
        print(
            f"{kind + f': candidate within ±{CANDIDATE_FRAMES} (of {total})':42}"
            + "".join(f"{r['covered'][kind]:>16}" for r in results.values())
        )
    covered = {name: sum(r["covered"].values()) for name, r in results.items()}
    total = sum(first["labelled"].values())
    print(
        f"{'candidate ceiling, all events':42}"
        + "".join(f"{covered[name] / total:>16.2f}" for name in names)
    )


if __name__ == "__main__":
    main()
