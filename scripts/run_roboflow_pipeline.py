from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from courtvision.detectors.base import (
    Detection,
    filter_player_detections,
)
from courtvision.detectors.roboflow_detector import RoboflowDetector
from courtvision.tracking.simple_tracker import (
    DEFAULT_IOU_THRESHOLD,
    SimpleIouTracker,
    TrackedDetection,
)

DEFAULT_MODEL_ID = "tennis-v4d0h/2"
DEFAULT_OUTPUT_DIR = "outputs/roboflow_demo"


@dataclass(frozen=True)
class PipelineDetection:
    detection: Detection
    track_id: int | None = None


def draw_pipeline_detection(frame, item: PipelineDetection):
    detection = item.detection
    x1 = int(round(detection.x1))
    y1 = int(round(detection.y1))
    x2 = int(round(detection.x2))
    y2 = int(round(detection.y2))
    color = (0, 220, 255)

    label_parts = [detection.class_name]
    if item.track_id is not None:
        label_parts.append(f"#{item.track_id}")
    label_parts.append(f"{detection.confidence:.2f}")
    label = " ".join(label_parts)

    label_width = max(120, min(220, 10 * len(label)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + label_width, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, max(14, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return frame


def tracked_items_from_detections(
    detections: list[Detection],
    tracker: SimpleIouTracker,
) -> list[PipelineDetection]:
    player_detections = filter_player_detections(detections)
    tracked_players = tracker.update(player_detections)
    return [
        *[
            pipeline_detection_from_tracked(tracked_player)
            for tracked_player in tracked_players
        ],
    ]


def pipeline_detection_from_tracked(tracked: TrackedDetection) -> PipelineDetection:
    return PipelineDetection(detection=tracked.detection, track_id=tracked.track_id)


def write_detections_csv(path: Path, detections: list[PipelineDetection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        for item in detections:
            detection = item.detection
            writer.writerow(
                [
                    detection.frame,
                    "" if item.track_id is None else item.track_id,
                    detection.class_name,
                    f"{detection.confidence:.6f}",
                    f"{detection.x1:.2f}",
                    f"{detection.y1:.2f}",
                    f"{detection.x2:.2f}",
                    f"{detection.y2:.2f}",
                    detection.model_name,
                ]
            )


def run_video(
    input_path: Path,
    output_dir: Path,
    detector: RoboflowDetector,
    frame_stride: int,
    tracker: SimpleIouTracker,
) -> list[PipelineDetection]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    all_detections: list[PipelineDetection] = []
    last_detections: list[PipelineDetection] = []
    progress = tqdm(total=frame_count, desc="Roboflow")
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_stride == 0:
            raw_detections = detector.predict_frame(frame, frame_index)
            last_detections = tracked_items_from_detections(raw_detections, tracker)
            all_detections.extend(last_detections)

        annotated = frame.copy()
        for item in last_detections:
            annotated = draw_pipeline_detection(annotated, item)
        writer.write(annotated)

        frame_index += 1
        progress.update(1)

    progress.close()
    cap.release()
    writer.release()
    return all_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Roboflow detection pipeline.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for outputs.",
    )
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--max-missing-frames", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    detector = RoboflowDetector(
        model_id=args.model_id,
        confidence=args.confidence,
        overlap=args.overlap,
    )
    tracker = SimpleIouTracker(
        iou_threshold=args.iou_threshold,
        max_missing_frames=args.max_missing_frames,
    )
    detections = run_video(
        input_path=Path(args.input),
        output_dir=output_dir,
        detector=detector,
        frame_stride=args.frame_stride,
        tracker=tracker,
    )
    write_detections_csv(output_dir / "detections.csv", detections)
    print(f"Wrote {len(detections)} detections to {output_dir / 'detections.csv'}")
    print(f"Wrote annotated video to {output_dir / 'annotated.mp4'}")


if __name__ == "__main__":
    main()
