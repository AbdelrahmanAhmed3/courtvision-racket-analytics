"""Optional adapter for yastrebksv/TrackNet ball detection."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TRACKNET_INPUT_SIZE = (640, 360)
DEFAULT_TRACKNET_REPO_URL = "https://github.com/yastrebksv/TrackNet.git"


@dataclass(frozen=True)
class BallPoint:
    frame: int
    x: float | None
    y: float | None
    confidence: float = 1.0

    @property
    def visible(self) -> bool:
        return self.x is not None and self.y is not None


def get_device(preferred: str = "auto"):
    torch = _get_torch()
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tracknet_model(tracknet_dir: str | Path, model_path: str | Path, device):
    torch = _get_torch()
    tracknet_dir = Path(tracknet_dir)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"TrackNet weights not found: {model_path}")
    if not (tracknet_dir / "model.py").exists():
        raise FileNotFoundError(
            "TrackNet model.py not found. Clone yastrebksv/TrackNet and pass its "
            f"directory with --tracknet-dir. Missing: {tracknet_dir / 'model.py'}"
        )

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


def ensure_tracknet_repo(
    tracknet_dir: str | Path,
    repo_url: str = DEFAULT_TRACKNET_REPO_URL,
) -> Path:
    tracknet_dir = Path(tracknet_dir)
    if (tracknet_dir / "model.py").exists():
        return tracknet_dir
    tracknet_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo_url, str(tracknet_dir)], check=True)
    return tracknet_dir


def infer_ball_track(
    video_path: str | Path,
    model,
    device,
    max_frames: int | None = None,
) -> list[BallPoint]:
    """Run TrackNet on a video, returning source-resolution ball centers."""
    torch = _get_torch()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open input video: {video_path}")

    history: list[np.ndarray] = []
    ball_track: list[BallPoint] = []
    frame_index = 0
    try:
        with torch.no_grad():
            while max_frames is None or frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                history.append(frame)
                if len(history) < 3:
                    ball_track.append(BallPoint(frame_index, None, None, 0.0))
                else:
                    ball_track.append(
                        infer_ball_point(
                            frame_index,
                            history[-3:],
                            model,
                            device,
                        )
                    )
                    history.pop(0)
                frame_index += 1
    finally:
        capture.release()
    return ball_track


def infer_ball_point(
    frame_index: int,
    frames: list[np.ndarray],
    model,
    device,
) -> BallPoint:
    input_width, input_height = TRACKNET_INPUT_SIZE
    original_height, original_width = frames[-1].shape[:2]
    resized = [
        cv2.resize(frame, (input_width, input_height)) for frame in reversed(frames)
    ]
    stacked = np.concatenate(resized, axis=2).astype(np.float32) / 255.0
    tensor = np.rollaxis(stacked, 2, 0)
    torch = _get_torch()
    output = model(torch.from_numpy(tensor).unsqueeze(0).float().to(device))
    heatmap = output.argmax(dim=1).detach().cpu().numpy()[0]
    x, y = postprocess_heatmap(heatmap)
    if x is None or y is None:
        return BallPoint(frame_index, None, None, 0.0)
    return BallPoint(
        frame=frame_index,
        x=x * original_width / input_width,
        y=y * original_height / input_height,
        confidence=1.0,
    )


def postprocess_heatmap(output: np.ndarray) -> tuple[float | None, float | None]:
    input_width, input_height = TRACKNET_INPUT_SIZE
    heatmap = (output.reshape((input_height, input_width)) * 255).astype(np.uint8)
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
    if circles is None:
        return None, None
    return float(circles[0][0][0]), float(circles[0][0][1])


def _get_torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "TrackNet ball tracking requires PyTorch. Install torch in the "
            "environment that runs the pipeline."
        ) from error
    return torch
