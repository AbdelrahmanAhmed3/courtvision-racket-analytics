"""Score player trackers against the labelled player numbers (IDF1, ID switches).

Example (after scripts/predict_shot_events.py has cached the players):
    python scripts/score_trackers.py --labels evaluation/labels/<video>.json

Each tracker is replayed on the cached player boxes of every labelled segment:
RF-DETR detections on the hand-marked court, four at most per frame. Replaying
takes seconds, so trackers can be compared without detecting again.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.detectors.base import Detection  # noqa: E402
from courtvision.evaluation.identity import score_identities  # noqa: E402
from courtvision.evaluation.labels import load_labels  # noqa: E402
from courtvision.tracking.fixed_count_tracker import FixedCountTracker  # noqa: E402
from courtvision.tracking.kalman_iou_tracker import KalmanIouTracker  # noqa: E402
from courtvision.tracking.simple_tracker import SimpleIouTracker  # noqa: E402

TRACKERS = {
    "greedy IoU (pipeline default)": SimpleIouTracker,
    "Kalman + Hungarian": KalmanIouTracker,
    "4 IDs, greedy": FixedCountTracker,
    "4 IDs, Hungarian": lambda: FixedCountTracker(assignment="hungarian"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="Labels JSON.")
    parser.add_argument("--size", default="nano", help="RF-DETR size of the cache.")
    return parser.parse_args()


def replay(make_tracker, segment, cached: dict, fps: float) -> dict:
    """Track IDs by frame from a new tracker fed the cached boxes, frame by frame."""
    boxes = defaultdict(list)
    for frame, _, x1, y1, x2, y2, confidence in cached["players"]:
        boxes[frame].append(Detection(frame, "person", confidence, x1, y1, x2, y2, ""))
    tracker = make_tracker()
    tracks = {}
    for frame in range(segment.start, segment.end):
        tracked = tracker.update(boxes.get(frame, []), timestamp_seconds=frame / fps)
        tracks[frame] = [(t.track_id, t.detection) for t in tracked]
    return tracks


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    cache = (
        REPO_ROOT
        / "outputs"
        / "predictions"
        / f"{Path(args.labels).stem}--tracks-{args.size}.json"
    )
    cached = json.loads(cache.read_text())["segments"]
    segments = [s for s in labels.segments if s.status == "label"]
    print(f"{labels.video}: {len(segments)} labelled rallies, IoU >= 0.5")
    header = f"{'tracker':32}{'IDF1':>7}{'switches':>10}"
    print(f"{header}  IDs given to the players, per rally")
    for name, make_tracker in TRACKERS.items():
        tracks = {
            s.start: replay(make_tracker, s, cached[str(s.start)], labels.fps)
            for s in segments
        }
        score = score_identities(labels, tracks)
        ids = " ".join(str(n) for n in score.ids_per_segment)
        print(f"{name:32}{score.idf1:>7.2f}{score.switches:>10}  {ids}")


if __name__ == "__main__":
    main()
