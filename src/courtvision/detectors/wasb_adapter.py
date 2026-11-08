"""Optional adapter for the WASB ball tracker (nttcom/WASB-SBDT, MIT licence).

WASB, like TrackNet, reads three frames in a row and paints a heat map of where
the ball is, but it paints one for each of the three frames. Its model code is
loaded from a local copy of the WASB repository; the weights are downloaded
separately (MODEL_ZOO.md there).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from courtvision.detectors.tracknet_adapter import BallPoint, _get_torch

WASB_INPUT_SIZE = (512, 288)
WASB_FRAMES = 3
WASB_SCORE_THRESHOLD = 0.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# src/configs/model/wasb.yaml in the WASB repository.
WASB_HRNET_CONFIG = {
    "name": "hrnet",
    "frames_in": 3,
    "frames_out": 3,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1],
                "NUM_CHANNELS": [32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2],
                "NUM_CHANNELS": [16, 32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64, 128],
                "FUSE_METHOD": "SUM",
            },
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}


class _Config(dict):
    """A dict whose keys can also be read as attributes, as WASB's HRNet does."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as error:
            raise AttributeError(name) from error
        return _Config(value) if isinstance(value, dict) else value


def load_wasb_model(wasb_dir: str | Path, model_path: str | Path, device):
    """WASB's HRNet from a local copy of its repository, with trained weights."""
    torch = _get_torch()
    hrnet_path = Path(wasb_dir) / "src" / "models" / "hrnet.py"
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"WASB weights not found: {model_path}")
    if not hrnet_path.exists():
        raise FileNotFoundError(
            "WASB's hrnet.py not found. Download nttcom/WASB-SBDT and pass its "
            f"directory. Missing: {hrnet_path}"
        )
    # Load the one file, so WASB's models package and its other models stay out.
    spec = importlib.util.spec_from_file_location("wasb_hrnet", hrnet_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.HRNet(_Config(WASB_HRNET_CONFIG))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()
    return model


def infer_ball_track_frames(
    frames: Iterable[np.ndarray], first_frame: int, model, device
) -> list[BallPoint]:
    """The ball on each of a run of frames, three at a time.

    A last group shorter than three is padded with copies of its last frame.
    """
    points: list[BallPoint] = []
    group: list[np.ndarray] = []
    for frame in frames:
        group.append(frame)
        if len(group) == WASB_FRAMES:
            points += infer_ball_points(first_frame + len(points), group, model, device)
            group = []
    if group:
        padded = group + [group[-1]] * (WASB_FRAMES - len(group))
        found = infer_ball_points(first_frame + len(points), padded, model, device)
        points += found[: len(group)]
    return points


def infer_ball_points(
    first_frame: int, frames: list[np.ndarray], model, device
) -> list[BallPoint]:
    """The ball on each of three frames in a row, in source-resolution pixels."""
    torch = _get_torch()
    width, height = WASB_INPUT_SIZE
    source_height, source_width = frames[0].shape[:2]
    images = []
    for frame in frames:
        rgb = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
        images.append((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
    stacked = np.concatenate(images, axis=2).transpose(2, 0, 1)
    with torch.no_grad():
        logits = model(torch.from_numpy(stacked).unsqueeze(0).to(device))[0]
    heatmaps = logits.sigmoid()[0].cpu().numpy()

    points = []
    for offset, heatmap in enumerate(heatmaps):
        frame_index = first_frame + offset
        peak = heatmap_peak(heatmap)
        if peak is None:
            points.append(BallPoint(frame_index, None, None, 0.0))
            continue
        x, y, confidence = peak
        points.append(
            BallPoint(
                frame_index,
                x * source_width / width,
                y * source_height / height,
                confidence,
            )
        )
    return points


def heatmap_peak(
    heatmap: np.ndarray, threshold: float = WASB_SCORE_THRESHOLD
) -> tuple[float, float, float] | None:
    """(x, y, peak value) of the strongest blob above the threshold, or None.

    As in WASB: blobs are connected pixels above the threshold, the strongest is
    the one with the largest sum, and its centre is weighted by the heat map.
    """
    if heatmap.max() <= threshold:
        return None
    count, blobs = cv2.connectedComponents((heatmap > threshold).astype(np.uint8))
    best = None
    for blob in range(1, count):
        ys, xs = np.nonzero(blobs == blob)
        weights = heatmap[ys, xs]
        total = float(weights.sum())
        if best is None or total > best[0]:
            x = float((xs * weights).sum() / total)
            y = float((ys * weights).sum() / total)
            best = (total, x, y, float(weights.max()))
    return best[1], best[2], best[3]
