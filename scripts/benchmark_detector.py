"""Measure local RF-DETR player detection speed and court filtering on a clip.

Example:
    python scripts/benchmark_detector.py --input data/raw/clip.mp4 \
        --calibration configs/calibrations/clip.json --frames 100
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.analytics.court_coordinates import (  # noqa: E402
    filter_detections_on_court,
)
from courtvision.calibration.homography import (  # noqa: E402
    estimate_template_homography,
)
from courtvision.calibration.io import load_calibration  # noqa: E402
from courtvision.detectors.rfdetr_detector import (  # noqa: E402
    RFDETR_SIZES,
    RFDetrDetector,
)
from courtvision.visualization.minimap import get_court_spec  # noqa: E402

WARMUP_FRAMES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Clip to read frames from.")
    parser.add_argument("--calibration", help="Calibration JSON for court filtering.")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--sizes", nargs="+", choices=RFDETR_SIZES, default=["nano"])
    parser.add_argument("--confidence", type=float, default=0.3)
    return parser.parse_args()


def read_frames(path: str, count: int) -> list:
    capture = cv2.VideoCapture(path)
    frames = []
    while len(frames) < count + WARMUP_FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) <= WARMUP_FRAMES:
        raise ValueError(f"Could not read enough frames from {path}")
    return frames


def main() -> None:
    args = parse_args()
    frames = read_frames(args.input, args.frames)
    court = None
    if args.calibration:
        calibration = load_calibration(args.calibration)
        court = (
            estimate_template_homography(calibration),
            get_court_spec(calibration.court_type),
        )

    print(f"Machine: {platform.platform()}, {platform.processor() or 'unknown CPU'}")
    print(f"Clip: {Path(args.input).name}, {len(frames) - WARMUP_FRAMES} frames")
    print()
    print("| Model | ms per frame | FPS | People per frame | On court per frame |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for size in args.sizes:
        detector = RFDetrDetector(size=size, confidence=args.confidence)
        for index, frame in enumerate(frames[:WARMUP_FRAMES]):
            detector.predict_frame(frame, index)
        people = on_court = 0
        started = time.perf_counter()
        for index, frame in enumerate(frames[WARMUP_FRAMES:]):
            detections = detector.predict_frame(frame, index)
            people += len(detections)
            if court is not None:
                on_court += len(filter_detections_on_court(detections, *court))
        elapsed = time.perf_counter() - started
        measured = len(frames) - WARMUP_FRAMES
        on_court_text = f"{on_court / measured:.2f}" if court else "n/a"
        print(
            f"| rf-detr-{size} | {elapsed / measured * 1000:.0f} "
            f"| {measured / elapsed:.1f} | {people / measured:.2f} "
            f"| {on_court_text} |"
        )


if __name__ == "__main__":
    main()
