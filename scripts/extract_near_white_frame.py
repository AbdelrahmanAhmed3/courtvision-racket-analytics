#!/usr/bin/env python3
"""Save a video frame showing only white or near-white pixels."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.calibration.pixel_masks import near_white_pixel_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one frame and black out every non-near-white pixel."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the binary near-white mask.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Timestamp to extract, in seconds (default: 0).",
    )
    parser.add_argument(
        "--minimum-value",
        type=int,
        default=170,
        help="Minimum HSV brightness from 0 to 255 (default: 170).",
    )
    parser.add_argument(
        "--maximum-saturation",
        type=int,
        default=80,
        help="Maximum HSV saturation from 0 to 255 (default: 80).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Video does not exist: {args.input}")
    if args.seconds < 0:
        raise ValueError("--seconds must be zero or greater")

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = round(args.seconds * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {args.input}")

    mask = near_white_pixel_mask(
        frame,
        minimum_value=args.minimum_value,
        maximum_saturation=args.maximum_saturation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mask):
        raise ValueError(f"Could not write image: {args.output}")
    selected = int(cv2.countNonZero(mask))
    total = mask.size
    print(
        f"Saved {args.output} from {args.seconds:.2f}s "
        f"({selected / total:.1%} near-white pixels)."
    )


if __name__ == "__main__":
    main()
