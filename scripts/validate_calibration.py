from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.calibration.homography import (  # noqa: E402
    estimate_template_homography,
)
from courtvision.calibration.io import load_calibration  # noqa: E402
from courtvision.calibration.validation import validate_homography  # noqa: E402
from courtvision.visualization.court_overlay import (  # noqa: E402
    draw_calibration_overlay,
)


def load_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a named-landmark calibration and render an overlay."
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--calibration", required=True, help="Calibration JSON path.")
    parser.add_argument("--output", required=True, help="Overlay image path.")
    parser.add_argument("--report", help="Optional validation report JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    estimate = estimate_template_homography(calibration)
    validation = validate_homography(estimate)
    frame = load_frame(Path(args.input), calibration.frame_index)
    overlay = draw_calibration_overlay(frame, calibration, estimate, validation)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), overlay)
    print(f"Wrote validation overlay to: {output}")
    print(
        f"status={validation.status} "
        f"mean_error={validation.mean_reprojection_error_px:.2f}px "
        f"max_error={validation.max_reprojection_error_px:.2f}px"
    )
    if args.report:
        report = {
            "status": validation.status,
            "mean_reprojection_error_px": validation.mean_reprojection_error_px,
            "max_reprojection_error_px": validation.max_reprojection_error_px,
            "per_landmark_error_px": validation.per_landmark_error_px,
            "inlier_count": validation.inlier_count,
            "landmark_count": validation.landmark_count,
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote validation report to: {report_path}")


if __name__ == "__main__":
    main()
