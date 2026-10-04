"""Measure local RF-DETR player detection: speed, people kept on court, track stability.

The court comes from a calibration JSON or, for clips that cannot be calibrated,
an image-space court outline (configs/court_outlines/). Kept people are capped at
four, by confidence, and tracked with the default tracker.

Example:
    python scripts/benchmark_detector.py --input data/raw/clip.mp4 \
        --calibration configs/calibrations/clip.json --sizes nano small
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
from courtvision.detectors.base import Detection  # noqa: E402
from courtvision.detectors.rfdetr_detector import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    RFDETR_SIZES,
    RFDetrDetector,
    resolve_device,
)
from courtvision.geometry.homography import bottom_center  # noqa: E402
from courtvision.geometry.polygon import (  # noqa: E402
    load_polygon,
    point_inside_polygon,
)
from courtvision.tracking.simple_tracker import SimpleIouTracker  # noqa: E402
from courtvision.visualization.minimap import get_court_spec  # noqa: E402

WARMUP_FRAMES = 3
PLAYERS_ON_COURT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Clip to read frames from.")
    court = parser.add_mutually_exclusive_group()
    court.add_argument("--calibration", help="Calibration JSON for the court.")
    court.add_argument("--court-outline", help="Image-space court outline JSON.")
    parser.add_argument("--start", type=int, default=0, help="First frame.")
    parser.add_argument("--end", type=int, help="Frame to stop before.")
    parser.add_argument("--sizes", nargs="+", choices=RFDETR_SIZES, default=["nano"])
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu.")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    return parser.parse_args()


def make_court_filter(args: argparse.Namespace):
    if args.calibration:
        calibration = load_calibration(args.calibration)
        estimate = estimate_template_homography(calibration)
        spec = get_court_spec(calibration.court_type)
        return lambda detections: filter_detections_on_court(detections, estimate, spec)
    if args.court_outline:
        outline = load_polygon(args.court_outline)

        def inside_outline(detections: list[Detection]) -> list[Detection]:
            return [
                detection
                for detection in detections
                if point_inside_polygon(
                    bottom_center(
                        detection.x1, detection.y1, detection.x2, detection.y2
                    ),
                    outline,
                )
            ]

        return inside_outline
    return None


def benchmark(args, size: str, court_filter) -> str:
    capture = cv2.VideoCapture(args.input)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    end = args.end or int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    detector = RFDetrDetector(size=size, confidence=args.confidence, device=args.device)
    tracker = SimpleIouTracker()
    detect_seconds = 0.0
    frames = people = kept = frames_with_four = 0
    track_ids: set[int] = set()
    for frame_index in range(args.start, end):
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index < args.start + WARMUP_FRAMES:
            detector.predict_frame(frame, frame_index)
            continue
        started = time.perf_counter()
        detections = detector.predict_frame(frame, frame_index)
        detect_seconds += time.perf_counter() - started
        frames += 1
        people += len(detections)
        if court_filter is None:
            continue
        on_court = sorted(
            court_filter(detections), key=lambda item: item.confidence, reverse=True
        )[:PLAYERS_ON_COURT]
        kept += len(on_court)
        frames_with_four += len(on_court) == PLAYERS_ON_COURT
        tracked = tracker.update(on_court, frame_index / fps)
        track_ids.update(item.track_id for item in tracked)
    capture.release()
    if frames == 0:
        raise ValueError(f"No frames measured in {args.input}")
    four_percent = 100 * frames_with_four / frames
    court_columns = (
        f"{kept / frames:.2f} | {four_percent:.0f}% | {len(track_ids)}"
        if court_filter
        else "n/a | n/a | n/a"
    )
    return (
        f"| rf-detr-{size} | {detect_seconds / frames * 1000:.0f} "
        f"| {frames / detect_seconds:.1f} | {people / frames:.2f} | {court_columns} |"
    )


def main() -> None:
    args = parse_args()
    court_filter = make_court_filter(args)
    print(f"Machine: {platform.platform()}; device: {resolve_device(args.device)}")
    print(f"Clip: {Path(args.input).name}, frames {args.start} to {args.end or 'end'}")
    print()
    print(
        "| Model | ms per frame | FPS | People per frame | On court per frame "
        "| Frames with 4 | Track IDs |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for size in args.sizes:
        print(benchmark(args, size, court_filter))


if __name__ == "__main__":
    main()
