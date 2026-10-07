"""Score a player detector against the hand-labelled player boxes.

Examples:
    python scripts/score_detectors.py --labels evaluation/labels/<video>.json \
        --video data/raw/<video>.mp4 --detector rfdetr --size nano
    python scripts/score_detectors.py --labels evaluation/labels/<video>.json \
        --video data/raw/<video>.mp4 --detector roboflow --model-id <project>/<version>

The detector runs once on every finished box frame; its detections are cached in
outputs/detections/, so scoring again (for example at another --min-iou) is free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.evaluation.detection import (  # noqa: E402
    DEFAULT_MIN_IOU,
    box_frames,
    load_detections,
    save_detections,
    score_detections,
)
from courtvision.evaluation.labels import load_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="Labels JSON.")
    parser.add_argument("--video", required=True, help="The labelled video.")
    parser.add_argument("--detector", choices=("rfdetr", "roboflow"), required=True)
    parser.add_argument("--size", default="nano", help="RF-DETR size.")
    parser.add_argument("--model-id", help="Roboflow model, as <project>/<version>.")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    parser.add_argument("--device", default="auto", help="RF-DETR device.")
    args = parser.parse_args()
    if args.detector == "roboflow" and not args.model_id:
        parser.error("--detector roboflow needs --model-id")
    return args


def build_detector(args: argparse.Namespace):
    if args.detector == "rfdetr":
        from courtvision.detectors.rfdetr_detector import RFDetrDetector

        return RFDetrDetector(
            size=args.size, confidence=args.confidence, device=args.device
        )
    from courtvision.detectors.roboflow_detector import RoboflowDetector

    return RoboflowDetector(model_id=args.model_id, confidence=args.confidence)


def detect(args: argparse.Namespace, frames: list[int]) -> list:
    detector = build_detector(args)
    capture = cv2.VideoCapture(args.video)
    detections = []
    for count, frame_index in enumerate(frames, start=1):
        # Seek the same way the labelling tool does, so frames match the labels.
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_index}")
        detections += detector.predict_frame(frame, frame_index)
        print(f"\r{count}/{len(frames)} frames", end="", flush=True)
    print()
    capture.release()
    return detections


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    model = f"rfdetr-{args.size}" if args.detector == "rfdetr" else args.model_id
    cache = (
        REPO_ROOT
        / "outputs"
        / "detections"
        / f"{Path(args.labels).stem}--{model.replace('/', '-')}--{args.confidence}.json"
    )
    if cache.exists():
        detections = load_detections(cache)
    else:
        detections = detect(args, box_frames(labels))
        save_detections(cache, detections)
    print(f"{model} at confidence {args.confidence}, IoU >= {args.min_iou}")
    print(f"{'detections':<34}{'labelled':>9}{'kept':>6}{'hits':>6}{'P':>7}{'R':>7}")
    rows = {
        "as the pipeline keeps them": True,
        "all people found": False,
    }
    for name, as_the_pipeline in rows.items():
        score = score_detections(labels, detections, args.min_iou, as_the_pipeline)
        print(
            f"{name:<34}{score.labelled:>9}{score.predicted:>6}{score.hits:>6}"
            f"{score.precision:>7.2f}{score.recall:>7.2f}"
        )


if __name__ == "__main__":
    main()
