from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.calibration.homography import (  # noqa: E402
    estimate_template_homography,
    project_image_points,
)
from courtvision.calibration.io import load_calibration  # noqa: E402
from courtvision.calibration.validation import validate_homography  # noqa: E402
from courtvision.geometry.homography import bottom_center  # noqa: E402
from courtvision.visualization.court_overlay import (  # noqa: E402
    draw_calibration_overlay,
)
from courtvision.visualization.minimap import (  # noqa: E402
    draw_court_map,
    draw_projected_point,
    get_court_spec,
)


def load_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def load_player_rows(detections_path: Path, frame_index: int) -> list[dict[str, str]]:
    with detections_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return [
        row
        for row in rows
        if int(row["frame"]) == frame_index
        and row["class_name"].strip().lower() in {"player", "person"}
    ]


def draw_player_foot_points(
    image: np.ndarray,
    rows: list[dict[str, str]],
) -> np.ndarray:
    for row in rows:
        x1, y1, x2, y2 = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        foot_x, foot_y = bottom_center(x1, y1, x2, y2)
        track_id = row.get("track_id") or "?"
        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 220, 255),
            2,
        )
        cv2.circle(image, (int(foot_x), int(foot_y)), 6, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"player #{track_id}",
            (int(x1), max(16, int(y1) - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project one frame of tracked players onto a validated 2D court map."
        )
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument(
        "--detections",
        required=True,
        help="Tracked detection CSV path.",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="Named calibration JSON path.",
    )
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output preview image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    frame_index = calibration.frame_index if args.frame is None else args.frame
    estimate = estimate_template_homography(calibration)
    validation = validate_homography(estimate)

    frame = load_frame(Path(args.input), frame_index)
    player_rows = load_player_rows(Path(args.detections), frame_index)
    source = draw_calibration_overlay(frame, calibration, estimate, validation)
    source = draw_player_foot_points(source, player_rows)

    map_image = draw_court_map(calibration.court_type)
    spec = get_court_spec(calibration.court_type)
    for row in player_rows:
        x1, y1, x2, y2 = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        foot = bottom_center(x1, y1, x2, y2)
        normalized = project_image_points(np.asarray([foot]), estimate)[0]
        track_id = row.get("track_id") or "?"
        in_bounds = 0 <= normalized[0] <= 1 and 0 <= normalized[1] <= 1
        color = (0, 220, 255) if in_bounds else (0, 0, 255)
        draw_projected_point(
            map_image,
            (float(normalized[0] * spec.width_m), float(normalized[1] * spec.length_m)),
            calibration.court_type,
            f"P{track_id}",
            color,
        )
        print(
            f"frame={frame_index} track_id={track_id} "
            f"template_x={normalized[0]:.3f} template_y={normalized[1]:.3f}"
        )

    source_height, _ = source.shape[:2]
    map_width = int(map_image.shape[1] * source_height / map_image.shape[0])
    map_image = cv2.resize(map_image, (map_width, source_height))
    preview = np.hstack((source, map_image))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), preview)
    print(f"Wrote homography preview to: {output}")


if __name__ == "__main__":
    main()
