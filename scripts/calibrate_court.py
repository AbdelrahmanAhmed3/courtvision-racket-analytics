from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.calibration.io import (  # noqa: E402
    CalibrationRecord,
    LandmarkObservation,
    save_calibration,
)
from courtvision.calibration.landmarks import (  # noqa: E402
    default_landmark_names,
    landmark_by_name,
)
from courtvision.calibration.manual import collect_manual_landmarks  # noqa: E402


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
        description="Click named court landmarks and save a calibration JSON."
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--court-type", choices=["tennis", "padel"], required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", required=True, help="Calibration JSON path.")
    parser.add_argument(
        "--landmarks",
        help="Comma-separated named landmark order. Defaults to all court landmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_frame(Path(args.input), args.frame)
    landmark_names = (
        [name.strip() for name in args.landmarks.split(",") if name.strip()]
        if args.landmarks
        else default_landmark_names(args.court_type)
    )
    definitions = [landmark_by_name(args.court_type, name) for name in landmark_names]
    selected = collect_manual_landmarks(frame, definitions)
    observations = {
        definition.name: LandmarkObservation(
            image=selected[definition.name],
            template=definition.template,
            visible=True,
            confidence=1.0,
            source="manual",
        )
        for definition in definitions
        if definition.name in selected
    }
    height, width = frame.shape[:2]
    calibration = CalibrationRecord(
        video_id=Path(args.input).name,
        frame_index=args.frame,
        frame_width=width,
        frame_height=height,
        court_type=args.court_type,
        landmarks=observations,
    )
    save_calibration(args.output, calibration)
    print(f"Saved {len(observations)} landmarks to: {args.output}")


if __name__ == "__main__":
    main()
