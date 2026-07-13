from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.analytics.court_coordinates import (  # noqa: E402
    CourtCoordinate,
    project_player_detection,
)
from courtvision.calibration.homography import (  # noqa: E402
    estimate_template_homography,
)
from courtvision.calibration.io import load_calibration  # noqa: E402
from courtvision.calibration.validation import validate_homography  # noqa: E402
from courtvision.detectors.base import Detection, filter_player_detections  # noqa: E402
from courtvision.detectors.roboflow_detector import RoboflowDetector  # noqa: E402
from courtvision.tracking.simple_tracker import (  # noqa: E402
    DEFAULT_IOU_THRESHOLD,
    SimpleIouTracker,
    TrackedDetection,
)
from courtvision.visualization.court_overlay import (  # noqa: E402
    draw_calibration_overlay,
)
from courtvision.visualization.minimap import (  # noqa: E402
    draw_court_map,
    draw_projected_point,
    get_court_spec,
)

DEFAULT_MODEL_ID = "tennis-v4d0h/2"


@dataclass(frozen=True)
class FrameTrack:
    detection: Detection
    track_id: int


def draw_player(frame, track: FrameTrack):
    detection = track.detection
    x1 = int(round(detection.x1))
    y1 = int(round(detection.y1))
    x2 = int(round(detection.x2))
    y2 = int(round(detection.y2))
    label = f"player #{track.track_id} {detection.confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.circle(frame, ((x1 + x2) // 2, y2), 5, (0, 0, 255), -1)
    cv2.putText(
        frame,
        label,
        (x1, max(16, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def tracks_from_detections(
    detections: list[Detection],
    tracker: SimpleIouTracker,
) -> list[FrameTrack]:
    tracked = tracker.update(filter_player_detections(detections))
    return [frame_track_from_tracked(item) for item in tracked]


def frame_track_from_tracked(tracked: TrackedDetection) -> FrameTrack:
    return FrameTrack(detection=tracked.detection, track_id=tracked.track_id)


def load_tracks_csv(path: Path) -> dict[int, list[FrameTrack]]:
    tracks_by_frame: dict[int, list[FrameTrack]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["class_name"].strip().lower() not in {"player", "person"}:
                continue
            if not row.get("track_id"):
                continue
            frame = int(row["frame"])
            detection = Detection(
                frame=frame,
                class_name=row["class_name"],
                confidence=float(row["confidence"]),
                x1=float(row["x1"]),
                y1=float(row["y1"]),
                x2=float(row["x2"]),
                y2=float(row["y2"]),
                model_name=row.get("model_name", "csv"),
            )
            tracks_by_frame.setdefault(frame, []).append(
                FrameTrack(detection=detection, track_id=int(row["track_id"]))
            )
    return tracks_by_frame


def write_detections_csv(
    path: Path,
    tracks_by_frame: dict[int, list[FrameTrack]],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frame",
                "track_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "model_name",
            ]
        )
        for frame_index, tracks in tracks_by_frame.items():
            for track in tracks:
                detection = track.detection
                writer.writerow(
                    [
                        frame_index,
                        track.track_id,
                        detection.class_name,
                        f"{detection.confidence:.6f}",
                        f"{detection.x1:.2f}",
                        f"{detection.y1:.2f}",
                        f"{detection.x2:.2f}",
                        f"{detection.y2:.2f}",
                        detection.model_name,
                    ]
                )


def write_court_coordinates_csv(path: Path, points: list[CourtCoordinate]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frame",
                "track_id",
                "object_type",
                "image_x",
                "image_y",
                "template_x",
                "template_y",
                "court_x_m",
                "court_y_m",
                "in_bounds",
                "confidence",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.frame,
                    point.track_id,
                    point.object_type,
                    f"{point.image_x:.2f}",
                    f"{point.image_y:.2f}",
                    f"{point.template_x:.5f}",
                    f"{point.template_y:.5f}",
                    f"{point.court_x_m:.3f}",
                    f"{point.court_y_m:.3f}",
                    int(point.in_bounds),
                    f"{point.confidence:.6f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run player tracking with optional validated court mapping."
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--detections",
        help="Existing tracked detections CSV. Skips Roboflow inference when set.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--max-missing-frames", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument(
        "--calibration",
        help="Named calibration JSON. Enables homography mapping and coordinate CSV.",
    )
    parser.add_argument(
        "--draw-court-map",
        action="store_true",
        help=(
            "Write a side-by-side source video and court-map video. "
            "Requires --calibration."
        ),
    )
    parser.add_argument(
        "--draw-calibration-overlay",
        action="store_true",
        help="Draw validated court lines and named landmarks on the source video.",
    )
    parser.add_argument(
        "--allow-calibration-warning",
        action="store_true",
        help="Allow a four-point warning calibration. Bad calibrations always fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.draw_court_map and not args.calibration:
        raise ValueError("--draw-court-map requires --calibration")
    if args.draw_calibration_overlay and not args.calibration:
        raise ValueError("--draw-calibration-overlay requires --calibration")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = None
    estimate = None
    spec = None
    validation = None
    if args.calibration:
        calibration = load_calibration(args.calibration)
        estimate = estimate_template_homography(calibration)
        validation = validate_homography(estimate)
        if validation.status == "bad":
            raise ValueError(
                "Calibration status is bad. Fix the landmark clicks before "
                "mapping tracks."
            )
        if validation.status == "warning" and not args.allow_calibration_warning:
            raise ValueError(
                "Calibration has only four points. Add landmarks or pass "
                "--allow-calibration-warning explicitly."
            )
        spec = get_court_spec(calibration.court_type)
        print(
            f"Calibration {validation.status}: "
            f"{validation.inlier_count}/{validation.landmark_count} inliers, "
            f"mean error {validation.mean_reprojection_error_px:.2f}px"
        )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if calibration and (
        width != calibration.frame_width or height != calibration.frame_height
    ):
        raise ValueError(
            "Input dimensions do not match the calibration frame. Recalibrate at this "
            "resolution before mapping."
        )

    source_writer = cv2.VideoWriter(
        str(output_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    map_writer = None
    map_width = 0
    if args.draw_court_map:
        map_preview = draw_court_map(calibration.court_type)
        map_width = int(map_preview.shape[1] * height / map_preview.shape[0])
        map_writer = cv2.VideoWriter(
            str(output_dir / "court_map_annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width + map_width, height),
        )

    tracks_from_csv = (
        load_tracks_csv(Path(args.detections)) if args.detections else None
    )
    detector = None
    tracker = None
    if tracks_from_csv is None:
        detector = RoboflowDetector(
            model_id=args.model_id,
            confidence=args.confidence,
            overlap=args.overlap,
        )
        tracker = SimpleIouTracker(
            iou_threshold=args.iou_threshold,
            max_missing_frames=args.max_missing_frames,
        )

    tracks_by_frame: dict[int, list[FrameTrack]] = {}
    court_points: list[CourtCoordinate] = []
    last_tracks: list[FrameTrack] = []
    frame_index = 0
    progress = tqdm(total=frame_count, desc="Full pipeline")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if tracks_from_csv is not None:
            current_tracks = tracks_from_csv.get(frame_index, [])
        elif frame_index % args.frame_stride == 0:
            assert detector is not None and tracker is not None
            current_tracks = tracks_from_detections(
                detector.predict_frame(frame, frame_index),
                tracker,
            )
        else:
            current_tracks = last_tracks

        last_tracks = current_tracks
        tracks_by_frame[frame_index] = current_tracks
        annotated = frame.copy()
        if args.draw_calibration_overlay:
            assert (
                calibration is not None
                and estimate is not None
                and validation is not None
            )
            annotated = draw_calibration_overlay(
                annotated,
                calibration,
                estimate,
                validation,
            )
        for track in current_tracks:
            annotated = draw_player(annotated, track)
        source_writer.write(annotated)

        if estimate is not None and spec is not None:
            frame_points = [
                replace(
                    project_player_detection(
                        track.detection,
                        track.track_id,
                        estimate,
                        spec,
                    ),
                    frame=frame_index,
                )
                for track in current_tracks
            ]
            court_points.extend(frame_points)
            if map_writer is not None:
                map_image = draw_court_map(calibration.court_type)
                for point in frame_points:
                    color = (0, 220, 255) if point.in_bounds else (0, 0, 255)
                    draw_projected_point(
                        map_image,
                        (point.court_x_m, point.court_y_m),
                        calibration.court_type,
                        f"P{point.track_id}",
                        color,
                    )
                map_image = cv2.resize(map_image, (map_width, height))
                map_writer.write(cv2.hconcat((annotated, map_image)))

        frame_index += 1
        progress.update(1)

    progress.close()
    cap.release()
    source_writer.release()
    if map_writer is not None:
        map_writer.release()

    write_detections_csv(output_dir / "detections.csv", tracks_by_frame)
    if court_points:
        write_court_coordinates_csv(
            output_dir / "tracks_with_court_coords.csv", court_points
        )
    print(f"Wrote detections: {output_dir / 'detections.csv'}")
    print(f"Wrote annotated video: {output_dir / 'annotated.mp4'}")
    if court_points:
        print(f"Wrote court coordinates: {output_dir / 'tracks_with_court_coords.csv'}")
    if map_writer is not None:
        print(f"Wrote court-map video: {output_dir / 'court_map_annotated.mp4'}")


if __name__ == "__main__":
    main()
