from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.detectors.base import Detection, filter_player_detections  # noqa: E402
from courtvision.detectors.roboflow_detector import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    RoboflowDetector,
)
from courtvision.tracking.kalman_iou_tracker import KalmanIouTracker  # noqa: E402
from courtvision.tracking.simple_tracker import (  # noqa: E402
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_MISSING_SECONDS,
    DEFAULT_REASSOCIATION_DISTANCE_PX,
    SimpleIouTracker,
    TrackedDetection,
)

DEFAULT_MODEL_ID = "tennis-v4d0h/2"
DEFAULT_OUTPUT_DIR = "outputs/tracker_comparison"


@dataclass
class TrackerMetrics:
    visible_detections: int = 0
    initial_tracks: int = 0
    created_tracks: int = 0
    potential_id_switches: int = 0
    recovered_tracks: int = 0


def load_detections_csv(
    path: Path,
    minimum_confidence: float,
) -> dict[int, list[Detection]]:
    """Load cached detector output so tracker comparisons do not call the API."""
    detections_by_frame: dict[int, list[Detection]] = {}
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            confidence = float(row["confidence"])
            if confidence < minimum_confidence:
                continue
            frame_index = int(row["frame"])
            detections_by_frame.setdefault(frame_index, []).append(
                Detection(
                    frame=frame_index,
                    class_name=row["class_name"],
                    confidence=confidence,
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                    model_name=row.get("model_name", "cached-detections"),
                )
            )
    return detections_by_frame


def draw_tracks(
    frame,
    tracks: list[TrackedDetection],
    title: str,
    color: tuple[int, int, int],
):
    canvas = frame.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(canvas, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    for track in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"#{track.track_id}  {track.detection.confidence:.2f}"
        cv2.rectangle(canvas, (x1, max(38, y1 - 22)), (x1 + 104, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, max(54, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_track_rows(
    writer: csv.writer,
    frame_index: int,
    tracker_name: str,
    tracks: list[TrackedDetection],
) -> None:
    for track in tracks:
        detection = track.detection
        writer.writerow(
            [
                frame_index,
                tracker_name,
                track.track_id,
                f"{detection.confidence:.6f}",
                f"{detection.x1:.2f}",
                f"{detection.y1:.2f}",
                f"{detection.x2:.2f}",
                f"{detection.y2:.2f}",
            ]
        )


def write_tracker_events(
    writer: csv.writer,
    frame_index: int,
    tracker_name: str,
    tracks: list[TrackedDetection],
    new_tracks: int,
    recovered_tracks: int,
) -> None:
    if new_tracks:
        new_ids = [track.track_id for track in tracks[-new_tracks:]]
        writer.writerow(
            [frame_index, tracker_name, "new_track", ";".join(map(str, new_ids))]
        )
    if recovered_tracks:
        writer.writerow(
            [frame_index, tracker_name, "recovered_track", recovered_tracks]
        )


def run_comparison(args: argparse.Namespace) -> dict[str, TrackerMetrics]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_by_frame = None
    detector = None
    if args.detections_csv:
        cache_path = Path(args.detections_csv)
        if not cache_path.is_file():
            raise ValueError(f"Could not find detections CSV: {cache_path}")
        detections_by_frame = load_detections_csv(cache_path, args.confidence)
    else:
        detector = RoboflowDetector(
            model_id=args.model_id, confidence=args.confidence, overlap=args.overlap
        )
    simple = SimpleIouTracker(
        args.iou_threshold, args.max_missing_seconds, args.reassociation_distance_px
    )
    kalman = KalmanIouTracker(
        args.iou_threshold, args.max_missing_seconds, args.reassociation_distance_px
    )
    metrics = {"simple_iou": TrackerMetrics(), "kalman_hungarian": TrackerMetrics()}

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = frame_count
    if args.max_seconds is not None:
        max_frames = min(frame_count, int(round(args.max_seconds * fps)))
    video_path = output_dir / "simple_vs_kalman.mp4"
    video_writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )

    csv_path = output_dir / "track_comparison.csv"
    event_path = output_dir / "tracker_events.csv"
    with csv_path.open("w", newline="") as output_file, event_path.open(
        "w", newline=""
    ) as event_file:
        csv_writer = csv.writer(output_file)
        event_writer = csv.writer(event_file)
        csv_writer.writerow(
            ["frame", "tracker", "track_id", "confidence", "x1", "y1", "x2", "y2"]
        )
        event_writer.writerow(["frame", "tracker", "event", "value"])
        progress = tqdm(total=max_frames, desc="Compare trackers")
        frame_index = 0
        simple_tracks: list[TrackedDetection] = []
        kalman_tracks: list[TrackedDetection] = []
        while frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.frame_stride == 0:
                raw_detections = (
                    detections_by_frame.get(frame_index, [])
                    if detections_by_frame is not None
                    else detector.predict_frame(frame, frame_index)
                )
                detections = filter_player_detections(raw_detections)
                timestamp_seconds = frame_index / fps
                simple_tracks = simple.update(detections, timestamp_seconds)
                kalman_tracks = kalman.update(detections, timestamp_seconds)
                tracker_results = (
                    ("simple_iou", simple_tracks),
                    ("kalman_hungarian", kalman_tracks),
                )
                for name, tracks in tracker_results:
                    metrics[name].visible_detections += len(tracks)
                    write_track_rows(csv_writer, frame_index, name, tracks)
                simple_metrics = metrics["simple_iou"]
                simple_stats = simple.last_update_stats
                simple_metrics.created_tracks += simple_stats.new_tracks
                if simple_metrics.initial_tracks == 0:
                    simple_metrics.initial_tracks = simple_stats.new_tracks
                else:
                    simple_metrics.potential_id_switches += simple_stats.new_tracks
                simple_metrics.recovered_tracks += (
                    simple_stats.dormant_recoveries
                )
                write_tracker_events(
                    event_writer,
                    frame_index,
                    "simple_iou",
                    simple_tracks,
                    simple_stats.new_tracks,
                    simple_stats.dormant_recoveries,
                )
                kalman_metrics = metrics["kalman_hungarian"]
                kalman_stats = kalman.last_update_stats
                kalman_metrics.created_tracks += kalman_stats.new_tracks
                if kalman_metrics.initial_tracks == 0:
                    kalman_metrics.initial_tracks = kalman_stats.new_tracks
                else:
                    kalman_metrics.potential_id_switches += kalman_stats.new_tracks
                kalman_metrics.recovered_tracks += (
                    kalman_stats.dormant_recoveries
                )
                write_tracker_events(
                    event_writer,
                    frame_index,
                    "kalman_hungarian",
                    kalman_tracks,
                    kalman_stats.new_tracks,
                    kalman_stats.dormant_recoveries,
                )
            left = draw_tracks(
                frame,
                simple_tracks if frame_index % args.frame_stride == 0 else [],
                "Simple IoU (greedy)",
                (0, 220, 255),
            )
            right = draw_tracks(
                frame,
                kalman_tracks if frame_index % args.frame_stride == 0 else [],
                "Kalman + Hungarian",
                (100, 255, 120),
            )
            cv2.line(left, (width - 1, 0), (width - 1, height), (255, 255, 255), 2)
            video_writer.write(cv2.hconcat([left, right]))
            frame_index += 1
            progress.update(1)
        progress.close()
    capture.release()
    video_writer.release()
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps({name: asdict(value) for name, value in metrics.items()}, indent=2)
        + "\n"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare greedy IoU and Kalman/Hungarian player trackers side by side."
        )
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--detections-csv",
        help="Cached detector CSV. When provided, no Roboflow API calls are made.",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Optional duration to compare from the start of the video.",
    )
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument(
        "--max-missing-seconds",
        type=float,
        default=DEFAULT_MAX_MISSING_SECONDS,
    )
    parser.add_argument(
        "--reassociation-distance-px",
        type=float,
        default=DEFAULT_REASSOCIATION_DISTANCE_PX,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("frame_stride must be greater than zero")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("max_seconds must be greater than zero")
    metrics = run_comparison(args)
    print(
        json.dumps({name: asdict(value) for name, value in metrics.items()}, indent=2)
    )
    print(
        "False reassociations require visual review of simple_vs_kalman.mp4 "
        "or labeled ground truth."
    )


if __name__ == "__main__":
    main()
