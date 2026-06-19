from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.detectors.base import Detection, filter_player_detections  # noqa: E402
from courtvision.detectors.roboflow_detector import RoboflowDetector  # noqa: E402
from courtvision.tracking.simple_tracker import (  # noqa: E402
    DEFAULT_IOU_THRESHOLD,
    SimpleIouTracker,
    TrackedDetection,
)

DEFAULT_ROBOFLOW_MODEL_ID = "tennis-v4d0h/2"
DEFAULT_TRACKNET_REPO_URL = "https://github.com/yastrebksv/TrackNet.git"
DEFAULT_TRACKNET_DIR = "/kaggle/working/TrackNet"
DEFAULT_OUTPUT_DIR = "outputs/player_ball_pipeline"
TRACKNET_RUNTIME_DEPS = ["numpy", "opencv-python", "scipy", "tqdm"]


@dataclass(frozen=True)
class PlayerTrack:
    detection: Detection
    track_id: int

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            self.detection.x1,
            self.detection.y1,
            self.detection.x2,
            self.detection.y2,
        )


@dataclass(frozen=True)
class BallPoint:
    frame: int
    x: float | None
    y: float | None
    interpolated: bool = False

    @property
    def visible(self) -> bool:
        return self.x is not None and self.y is not None


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def ensure_tracknet_repo(tracknet_dir: Path, repo_url: str) -> None:
    if (tracknet_dir / "model.py").exists():
        print(f"Using existing TrackNet repo: {tracknet_dir}")
        return
    tracknet_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", repo_url, str(tracknet_dir)])


def maybe_install_tracknet_runtime(python_executable: str, install: bool) -> None:
    if install:
        run_command([python_executable, "-m", "pip", "install", *TRACKNET_RUNTIME_DEPS])


def get_device(preferred: str) -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tracknet_model(
    tracknet_dir: Path,
    model_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"TrackNet weights not found: {model_path}")
    if not (tracknet_dir / "model.py").exists():
        raise FileNotFoundError(f"TrackNet model.py not found in: {tracknet_dir}")

    sys.path.insert(0, str(tracknet_dir))
    try:
        from model import BallTrackerNet
    finally:
        sys.path.pop(0)

    model = BallTrackerNet()
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"Video has no readable frames: {path}")
    return frames, fps


def postprocess_tracknet_heatmap(
    output: np.ndarray,
) -> tuple[float | None, float | None]:
    heatmap = (output.reshape((360, 640)) * 255).astype(np.uint8)
    _, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )
    if circles is None or len(circles) != 1:
        return None, None
    return float(circles[0][0][0]), float(circles[0][0][1])


def infer_ball_track(
    frames: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[list[BallPoint], list[float]]:
    input_width = 640
    input_height = 360
    original_height, original_width = frames[0].shape[:2]
    scale_x = original_width / input_width
    scale_y = original_height / input_height

    ball_track = [BallPoint(0, None, None), BallPoint(1, None, None)]
    dists = [-1.0, -1.0]

    with torch.no_grad():
        for frame_index in tqdm(range(2, len(frames)), desc="TrackNet ball"):
            current = cv2.resize(frames[frame_index], (input_width, input_height))
            previous = cv2.resize(frames[frame_index - 1], (input_width, input_height))
            pre_previous = cv2.resize(
                frames[frame_index - 2],
                (input_width, input_height),
            )
            stacked = np.concatenate((current, previous, pre_previous), axis=2)
            stacked = stacked.astype(np.float32) / 255.0
            stacked = np.rollaxis(stacked, 2, 0)
            tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)

            output = model(tensor).argmax(dim=1).detach().cpu().numpy()[0]
            x_pred, y_pred = postprocess_tracknet_heatmap(output)
            point = BallPoint(
                frame=frame_index,
                x=None if x_pred is None else x_pred * scale_x,
                y=None if y_pred is None else y_pred * scale_y,
            )
            ball_track.append(point)

            previous_point = ball_track[-2]
            if point.visible and previous_point.visible:
                dist = float(
                    np.hypot(point.x - previous_point.x, point.y - previous_point.y)
                )
            else:
                dist = -1.0
            dists.append(dist)

    return ball_track, dists


def remove_ball_outliers(
    ball_track: list[BallPoint],
    dists: list[float],
    max_dist: float = 100,
) -> list[BallPoint]:
    cleaned = list(ball_track)
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for index in outliers:
        next_dist = dists[index + 1] if index + 1 < len(dists) else -1
        prev_dist = dists[index - 1] if index - 1 >= 0 else -1
        if next_dist > max_dist or next_dist == -1:
            cleaned[index] = BallPoint(index, None, None)
        elif prev_dist == -1 and index - 1 >= 0:
            cleaned[index - 1] = BallPoint(index - 1, None, None)
    return cleaned


def split_ball_track(
    ball_track: list[BallPoint],
    max_gap: int = 4,
    max_dist_gap: float = 80,
    min_track: int = 5,
) -> list[tuple[int, int]]:
    detected_flags = [0 if point.visible else 1 for point in ball_track]
    groups = [(key, sum(1 for _ in group)) for key, group in groupby(detected_flags)]
    cursor = 0
    min_value = 0
    result = []

    for index, (key, length) in enumerate(groups):
        if key == 1 and 0 < index < len(groups) - 1:
            prev_point = ball_track[cursor - 1]
            next_point = ball_track[cursor + length]
            if prev_point.visible and next_point.visible:
                dist = float(
                    np.hypot(
                        prev_point.x - next_point.x,
                        prev_point.y - next_point.y,
                    )
                )
            else:
                dist = float("inf")
            if length >= max_gap or dist / length > max_dist_gap:
                if cursor - min_value > min_track:
                    result.append((min_value, cursor))
                min_value = cursor + length - 1
        cursor += length

    if len(detected_flags) - min_value > min_track:
        result.append((min_value, len(detected_flags)))
    return result


def interpolate_ball_points(points: list[BallPoint]) -> list[BallPoint]:
    xs = np.array([point.x if point.x is not None else np.nan for point in points])
    ys = np.array([point.y if point.y is not None else np.nan for point in points])
    indexes = np.arange(len(points))

    for values in (xs, ys):
        missing = np.isnan(values)
        if missing.any() and (~missing).sum() >= 2:
            values[missing] = np.interp(
                indexes[missing],
                indexes[~missing],
                values[~missing],
            )

    return [
        BallPoint(
            frame=point.frame,
            x=float(x) if not np.isnan(x) else None,
            y=float(y) if not np.isnan(y) else None,
            interpolated=not point.visible and not np.isnan(x) and not np.isnan(y),
        )
        for point, x, y in zip(points, xs, ys, strict=True)
    ]


def interpolate_ball_track(ball_track: list[BallPoint]) -> list[BallPoint]:
    result = list(ball_track)
    for start, end in split_ball_track(result):
        result[start:end] = interpolate_ball_points(result[start:end])
    return result


def run_player_tracking(
    frames: list[np.ndarray],
    detector: RoboflowDetector,
    frame_stride: int,
    tracker: SimpleIouTracker,
) -> dict[int, list[PlayerTrack]]:
    tracks_by_frame: dict[int, list[PlayerTrack]] = {}
    last_tracks: list[PlayerTrack] = []

    for frame_index, frame in enumerate(tqdm(frames, desc="Roboflow players")):
        if frame_index % frame_stride == 0:
            detections = filter_player_detections(
                detector.predict_frame(frame, frame_index)
            )
            tracked = tracker.update(detections)
            last_tracks = [player_track_from_tracked(item) for item in tracked]
            tracks_by_frame[frame_index] = last_tracks
        else:
            tracks_by_frame[frame_index] = last_tracks

    return tracks_by_frame


def player_track_from_tracked(tracked: TrackedDetection) -> PlayerTrack:
    return PlayerTrack(detection=tracked.detection, track_id=tracked.track_id)


def draw_player(frame: np.ndarray, player: PlayerTrack) -> np.ndarray:
    detection = player.detection
    x1, y1, x2, y2 = [int(round(value)) for value in player.bbox]
    label = f"player #{player.track_id} {detection.confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 150, y1), (0, 220, 255), -1)
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


def draw_ball_trace(
    frame: np.ndarray,
    ball_track: list[BallPoint],
    frame_index: int,
    trace: int,
) -> np.ndarray:
    for offset in range(trace):
        index = frame_index - offset
        if index < 0:
            break
        point = ball_track[index]
        if not point.visible:
            continue
        radius = max(2, 8 - offset)
        cv2.circle(
            frame,
            (int(round(point.x)), int(round(point.y))),
            radius=radius,
            color=(0, 0, 255),
            thickness=-1,
        )
    return frame


def write_combined_video(
    frames: list[np.ndarray],
    player_tracks: dict[int, list[PlayerTrack]],
    ball_track: list[BallPoint],
    output_path: Path,
    fps: float,
    ball_trace: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame_index, frame in enumerate(tqdm(frames, desc="Writing video")):
        annotated = frame.copy()
        for player in player_tracks.get(frame_index, []):
            annotated = draw_player(annotated, player)
        annotated = draw_ball_trace(annotated, ball_track, frame_index, ball_trace)
        writer.write(annotated)
    writer.release()


def write_player_csv(path: Path, player_tracks: dict[int, list[PlayerTrack]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "track_id", "confidence", "x1", "y1", "x2", "y2"])
        for frame_index, tracks in player_tracks.items():
            for player in tracks:
                detection = player.detection
                writer.writerow(
                    [
                        frame_index,
                        player.track_id,
                        f"{detection.confidence:.6f}",
                        f"{detection.x1:.2f}",
                        f"{detection.y1:.2f}",
                        f"{detection.x2:.2f}",
                        f"{detection.y2:.2f}",
                    ]
                )


def write_ball_csv(path: Path, ball_track: list[BallPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "visible", "x", "y", "interpolated"])
        for point in ball_track:
            writer.writerow(
                [
                    point.frame,
                    int(point.visible),
                    "" if point.x is None else f"{point.x:.2f}",
                    "" if point.y is None else f"{point.y:.2f}",
                    int(point.interpolated),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Roboflow player tracking plus TrackNet ball tracking."
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roboflow-model-id", default=DEFAULT_ROBOFLOW_MODEL_ID)
    parser.add_argument("--roboflow-frame-stride", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--max-missing-frames", type=int, default=10)
    parser.add_argument("--tracknet-model-path", required=True)
    parser.add_argument("--tracknet-dir", default=DEFAULT_TRACKNET_DIR)
    parser.add_argument("--tracknet-repo-url", default=DEFAULT_TRACKNET_REPO_URL)
    parser.add_argument("--tracknet-device", default="auto")
    parser.add_argument("--skip-tracknet-clone", action="store_true")
    parser.add_argument("--install-tracknet-runtime", action="store_true")
    parser.add_argument("--extrapolate-ball", action="store_true")
    parser.add_argument("--ball-trace", type=int, default=7)
    parser.add_argument("--python", default="python")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracknet_dir = Path(args.tracknet_dir)
    if not args.skip_tracknet_clone:
        ensure_tracknet_repo(tracknet_dir, args.tracknet_repo_url)
    maybe_install_tracknet_runtime(args.python, args.install_tracknet_runtime)

    frames, fps = read_video(input_path)
    device = get_device(args.tracknet_device)
    print(f"Using TrackNet device: {device}")
    tracknet_model = load_tracknet_model(
        tracknet_dir=tracknet_dir,
        model_path=Path(args.tracknet_model_path),
        device=device,
    )

    ball_track, dists = infer_ball_track(frames, tracknet_model, device)
    ball_track = remove_ball_outliers(ball_track, dists)
    if args.extrapolate_ball:
        ball_track = interpolate_ball_track(ball_track)

    roboflow = RoboflowDetector(
        model_id=args.roboflow_model_id,
        confidence=args.confidence,
        overlap=args.overlap,
    )
    player_tracker = SimpleIouTracker(
        iou_threshold=args.iou_threshold,
        max_missing_frames=args.max_missing_frames,
    )
    player_tracks = run_player_tracking(
        frames=frames,
        detector=roboflow,
        frame_stride=args.roboflow_frame_stride,
        tracker=player_tracker,
    )

    write_player_csv(output_dir / "player_tracks.csv", player_tracks)
    write_ball_csv(output_dir / "ball_track.csv", ball_track)
    write_combined_video(
        frames=frames,
        player_tracks=player_tracks,
        ball_track=ball_track,
        output_path=output_dir / "combined_annotated.mp4",
        fps=fps,
        ball_trace=args.ball_trace,
    )

    print(f"Wrote player tracks: {output_dir / 'player_tracks.csv'}")
    print(f"Wrote ball track: {output_dir / 'ball_track.csv'}")
    print(f"Wrote combined video: {output_dir / 'combined_annotated.mp4'}")


if __name__ == "__main__":
    main()
