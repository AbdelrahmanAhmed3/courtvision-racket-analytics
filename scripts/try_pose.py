"""Trial: how much of the near player can a pose model see in a phone clip (#87).

Finds people with RF-DETR (feet inside a court outline traced by eye), drops still
objects such as jackets on the glass, and keeps the two most confident people on
the camera's side of the net: the near team. Runs RTMPose on their boxes and reports
how often each body part is found, for the near left and near right player. The
near player, filming themselves, is who the report is for.

People are cached first (--people-only), so pose can run later. Pose needs rtmlib
and onnxruntime, installed without rtmlib's extra OpenCV wheel (reports/pose.md):

    pip install onnxruntime && pip install --no-deps rtmlib

Example:
    python scripts/try_pose.py --input ~/Downloads/IMG_1089.MOV \
        --court-outline configs/court_outlines/phone_IMG_1089.json --end 10600 \
        --overlay 3000 3900 --sample 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.detectors.rfdetr_detector import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_SIZE,
    RFDetrDetector,
)
from courtvision.geometry.homography import bottom_center  # noqa: E402
from courtvision.geometry.polygon import point_inside_polygon  # noqa: E402

PREDICTIONS_DIR = REPO_ROOT / "outputs" / "predictions"
POSE_DIR = REPO_ROOT / "outputs" / "pose"
PLAYERS_ON_COURT = 4

# RTMPose-m, trained on 7 public body datasets, 256x192 input (OpenMMLab, Apache-2.0).
POSE_MODEL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
POSE_INPUT_SIZE = (192, 256)  # width, height

# COCO body points, as RTMPose returns them.
POINT_NAMES = (
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
)
BODY_PARTS = {
    "shoulders": (5, 6),
    "elbows": (7, 8),
    "wrists": (9, 10),
    "hips": (11, 12),
    "knees": (13, 14),
    "ankles": (15, 16),
}
SKELETON = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
EDGE_PIXELS = 3
# Something found in the same place for most of a minute is a jacket, a bag or a
# chair, not a player (a jacket on the left glass in IMG_1084): checked once a
# second, this far either side.
STILL_WINDOW_SECONDS = 30.0
STILL_SHARE = 0.5
# Same place: the centre moved less than this share of the box height, and the
# height changed by less than SAME_SIZE. Small boxes jitter by a few pixels.
SAME_CENTRE = 0.1
SAME_SIZE = 1.25
# The near team: at most two people, each found with at least this confidence.
# Jackets and bags by the glass score 0.30-0.51; players a median of 0.88.
NEAR_PLAYERS = 2
NEAR_MIN_CONFIDENCE = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Clip to read frames from.")
    parser.add_argument(
        "--court-outline", required=True, help="Image-space court outline JSON."
    )
    parser.add_argument("--start", type=int, default=0, help="First frame.")
    parser.add_argument("--end", type=int, help="Frame to stop before.")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="RF-DETR size.")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda or cpu.")
    parser.add_argument(
        "--people-only", action="store_true", help="Find and cache the people only."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence from which a body point counts as found.",
    )
    parser.add_argument(
        "--overlay",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Write a video of these frames with the skeletons drawn.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Save this many random near player crops to check by eye.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Outline:
    polygon: np.ndarray
    net_base_y: float

    @classmethod
    def load(cls, path: str | Path) -> Outline:
        data = json.loads(Path(path).read_text())
        return cls(np.asarray(data["polygon"], np.float32), float(data["net_base_y"]))

    def side(self, box) -> str:
        """Far if the player's feet are beyond the net's base in the image."""
        return "far" if box[3] < self.net_base_y else "near"


def half(box, width: int) -> str:
    """Which half of the picture the player stands in, seen from the camera."""
    return "left" if (box[0] + box[2]) / 2 < width / 2 else "right"


def touches_edge(box, width: int, height: int, margin: int = EDGE_PIXELS) -> bool:
    """True when the box is cut by the frame, so part of the player is out of view."""
    x1, y1, x2, y2 = box[:4]
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def same_place(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """For every pair of boxes, whether they have about the same centre and size."""
    height = (first[:, 3] - first[:, 1])[:, None]
    ratio = (second[:, 3] - second[:, 1])[None, :] / height
    centre_moved = np.hypot(
        (
            first[:, None, 0]
            + first[:, None, 2]
            - second[None, :, 0]
            - second[None, :, 2]
        )
        / 2,
        (
            first[:, None, 1]
            + first[:, None, 3]
            - second[None, :, 1]
            - second[None, :, 3]
        )
        / 2,
    )
    return (
        (centre_moved <= SAME_CENTRE * height)
        & (ratio >= 1 / SAME_SIZE)
        & (ratio <= SAME_SIZE)
    )


def drop_still_boxes(
    people: dict[int, list[list[float]]],
    step: int,
    reach: int,
    min_share: float = STILL_SHARE,
) -> dict[int, list[list[float]]]:
    """Drop boxes with something in the same place in most frames around them.

    The frames checked are every ``step``-th one within ``reach`` frames either
    side. A box is dropped when the share of them holding a box in the same place
    reaches ``min_share``.
    """
    boxes = {
        frame: np.array([box[:4] for box in found], np.float32).reshape(-1, 4)
        for frame, found in people.items()
    }
    kept = {}
    for frame, found in people.items():
        if not found:
            kept[frame] = []
            continue
        checked = [
            other
            for other in range(frame - reach, frame + reach + 1, step)
            if other in boxes and other != frame
        ]
        hits = np.zeros(len(found))
        for other in checked:
            if len(boxes[other]):
                hits += same_place(boxes[frame], boxes[other]).any(axis=1)
        still = hits >= min_share * len(checked) if checked else np.zeros(len(found))
        kept[frame] = [box for box, drop in zip(found, still, strict=True) if not drop]
    return kept


def people_cache(args: argparse.Namespace) -> Path:
    return PREDICTIONS_DIR / f"{Path(args.input).stem}--people-{args.size}.json"


def find_people(args: argparse.Namespace) -> dict[int, list[list[float]]]:
    """Everyone standing inside the court outline, most confident first, per frame."""
    cache = people_cache(args)
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["outline"] == Path(args.court_outline).name and data["range"] == [
            args.start,
            args.end,
        ]:
            return {int(frame): boxes for frame, boxes in data["people"].items()}
    outline = Outline.load(args.court_outline)
    capture = cv2.VideoCapture(str(Path(args.input).expanduser()))
    end = args.end or int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    detector = RFDetrDetector(
        size=args.size, confidence=DEFAULT_CONFIDENCE, device=args.device
    )
    people = {}
    for frame_index in tqdm(range(args.start, end), desc="people"):
        ok, frame = capture.read()
        if not ok:
            break
        on_court = [
            detection
            for detection in detector.predict_frame(frame, frame_index)
            if point_inside_polygon(
                bottom_center(detection.x1, detection.y1, detection.x2, detection.y2),
                outline.polygon,
            )
        ]
        on_court.sort(key=lambda detection: detection.confidence, reverse=True)
        people[frame_index] = [
            [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)]
            + [round(d.confidence, 3)]
            for d in on_court
        ]
    capture.release()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "clip": Path(args.input).name,
                "outline": Path(args.court_outline).name,
                "range": [args.start, args.end],
                "people": people,
            }
        )
    )
    return people


@dataclass
class Poses:
    frames: np.ndarray  # (n,)
    boxes: np.ndarray  # (n, 5): x1, y1, x2, y2, detector confidence
    points: np.ndarray  # (n, 17, 2) in image pixels
    scores: np.ndarray  # (n, 17)
    seconds_per_player: float


def poses_cache(args: argparse.Namespace) -> Path:
    return POSE_DIR / f"{Path(args.input).stem}--rtmpose-m.npz"


def estimate_poses(args, players: dict[int, list[list[float]]], outline) -> Poses:
    """RTMPose on every near player's box."""
    cache = poses_cache(args)
    if cache.exists():
        data = np.load(cache)
        return Poses(
            data["frames"],
            data["boxes"],
            data["points"],
            data["scores"],
            float(data["seconds_per_player"]),
        )
    from rtmlib import RTMPose

    model = RTMPose(POSE_MODEL, model_input_size=POSE_INPUT_SIZE, device="cpu")
    capture = cv2.VideoCapture(str(Path(args.input).expanduser()))
    end = args.end or int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frames, boxes, points, scores = [], [], [], []
    seconds = 0.0
    for frame_index in tqdm(range(args.start, end), desc="pose"):
        ok, frame = capture.read()
        if not ok:
            break
        found = [
            box for box in players.get(frame_index, []) if outline.side(box) == "near"
        ]
        if not found:
            continue
        started = time.perf_counter()
        keypoints, confidences = model(frame, bboxes=[box[:4] for box in found])
        seconds += time.perf_counter() - started
        frames += [frame_index] * len(found)
        boxes += found
        points.append(keypoints)
        scores.append(confidences)
    capture.release()
    poses = Poses(
        np.array(frames),
        np.array(boxes, np.float32),
        np.concatenate(points).astype(np.float32),
        np.concatenate(scores).astype(np.float32),
        seconds / max(len(frames), 1),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        frames=poses.frames,
        boxes=poses.boxes,
        points=poses.points,
        scores=poses.scores,
        seconds_per_player=poses.seconds_per_player,
    )
    return poses


def near_team(poses: Poses) -> Poses:
    """Per frame, the two most confident near boxes found with enough confidence."""
    keep = np.zeros(len(poses.frames), bool)
    for frame in np.unique(poses.frames):
        rows = np.flatnonzero(poses.frames == frame)
        rows = rows[poses.boxes[rows, 4] >= NEAR_MIN_CONFIDENCE]
        rows = rows[np.argsort(-poses.boxes[rows, 4])][:NEAR_PLAYERS]
        keep[rows] = True
    return Poses(
        poses.frames[keep],
        poses.boxes[keep],
        poses.points[keep],
        poses.scores[keep],
        poses.seconds_per_player,
    )


def found_rates(scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Share of player-frames in which each body part (either side) is found."""
    if len(scores) == 0:
        return {part: float("nan") for part in BODY_PARTS}
    return {
        part: float((scores[:, list(indices)] >= threshold).mean())
        for part, indices in BODY_PARTS.items()
    }


def summary(args, poses: Poses, size: tuple[int, int], frames: int) -> str:
    """One row per near player: how often found, cut by the frame, body parts seen."""
    width, height = size
    halves = np.array([half(box, width) for box in poses.boxes])
    cut = np.array([touches_edge(box, width, height) for box in poses.boxes])
    heights = poses.boxes[:, 3] - poses.boxes[:, 1]
    lines = [
        f"{Path(args.input).name}: {frames} frames, "
        f"{poses.seconds_per_player * 1000:.0f} ms per player on the CPU, "
        f"body point found from confidence {args.threshold}",
        "",
        "| near player | found in frames | median height (px) | cut by the edge | "
        + " | ".join(BODY_PARTS)
        + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in BODY_PARTS) + " |",
    ]
    for name, mask in (
        ("left", halves == "left"),
        ("right", halves == "right"),
        ("all near", np.ones(len(halves), bool)),
    ):
        if not mask.any():
            lines.append(f"| {name} | 0 % | | | " + " | ".join("" for _ in BODY_PARTS))
            continue
        found_in = len(set(poses.frames[mask].tolist())) / frames
        rates = found_rates(poses.scores[mask], args.threshold)
        lines.append(
            f"| {name} | {found_in:.0%} | {np.median(heights[mask]):.0f} | "
            f"{cut[mask].mean():.0%} | "
            + " | ".join(f"{rates[part]:.0%}" for part in BODY_PARTS)
            + " |"
        )
    return "\n".join(lines)


# The player's left side green, right side magenta (BGR), so swaps show.
LEFT_COLOUR = (80, 220, 0)
RIGHT_COLOUR = (220, 0, 220)
LEFT_POINTS = frozenset({1, 3, 5, 7, 9, 11, 13, 15})
RIGHT_POINTS = frozenset({2, 4, 6, 8, 10, 12, 14, 16})


def point_colour(index: int):
    if index in LEFT_POINTS:
        return LEFT_COLOUR
    if index in RIGHT_POINTS:
        return RIGHT_COLOUR
    return (255, 255, 255)


def draw_pose(frame, box, points, scores, threshold: float, color) -> None:
    """The box in ``color``; the found points, left green and right magenta."""
    x1, y1, x2, y2 = (int(value) for value in box[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    thickness = max(1, int((y2 - y1) / 80))
    for a, b in SKELETON:
        if scores[a] >= threshold and scores[b] >= threshold:
            same_side = point_colour(a) == point_colour(b)
            cv2.line(
                frame,
                tuple(int(v) for v in points[a]),
                tuple(int(v) for v in points[b]),
                point_colour(a) if same_side else (255, 255, 255),
                thickness,
            )
    for index, (point, score) in enumerate(zip(points, scores, strict=True)):
        if score < threshold:
            continue
        centre = tuple(int(v) for v in point)
        if index in BODY_PARTS["wrists"]:
            cv2.circle(frame, centre, thickness * 4 + 1, (0, 0, 0), -1)
            cv2.circle(frame, centre, thickness * 3, point_colour(index), -1)
        else:
            cv2.circle(frame, centre, thickness * 2, point_colour(index), -1)


def write_overlay(args, poses: Poses, start: int, end: int) -> Path:
    path = POSE_DIR / f"{Path(args.input).stem}--overlay-{start}-{end}.mp4"
    capture = cv2.VideoCapture(str(Path(args.input).expanduser()))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = None
    for frame_index in range(start, end):
        ok, frame = capture.read()
        if not ok:
            break
        for row in np.flatnonzero(poses.frames == frame_index):
            box = poses.boxes[row]
            color = (
                (0, 200, 255) if half(box, frame.shape[1]) == "left" else (255, 200, 0)
            )
            draw_pose(
                frame, box, poses.points[row], poses.scores[row], args.threshold, color
            )
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
            )
        writer.write(frame)
    capture.release()
    if writer is not None:
        writer.release()
    return path


def write_samples(args, poses: Poses, count: int) -> list[Path]:
    """Random near player crops, points drawn and numbered, to check by eye."""
    rng = random.Random(87)
    capture = cv2.VideoCapture(str(Path(args.input).expanduser()))
    paths = []
    for side in ("near",):
        rows = list(range(len(poses.boxes)))
        picked = sorted(rng.sample(rows, min(count, len(rows))))
        tiles, index = [], []
        for number, row in enumerate(picked):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(poses.frames[row]))
            ok, frame = capture.read()
            if not ok:
                continue
            box = poses.boxes[row]
            draw_pose(
                frame,
                box,
                poses.points[row],
                poses.scores[row],
                args.threshold,
                (255, 200, 0),
            )
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            pad = int(0.15 * (y2 - y1))
            crop = frame[
                max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad
            ].copy()
            scale = 240 / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 240))
            tile = np.zeros((260, 200, 3), np.uint8)
            visible = crop[:, : min(200, crop.shape[1])]
            left = (200 - visible.shape[1]) // 2
            tile[20:, left : left + visible.shape[1]] = visible
            cv2.putText(
                tile, str(number), (4, 16), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
            tiles.append(tile)
            index.append(
                {
                    "number": number,
                    "frame": int(poses.frames[row]),
                    "box": [round(float(v), 1) for v in box[:4]],
                    "wrist scores": [
                        round(float(poses.scores[row][i]), 2)
                        for i in BODY_PARTS["wrists"]
                    ],
                }
            )
        while len(tiles) % 6:
            tiles.append(np.zeros((260, 200, 3), np.uint8))
        rows_of_tiles = [np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]
        path = POSE_DIR / f"{Path(args.input).stem}--sample-{side}.jpg"
        cv2.imwrite(str(path), np.vstack(rows_of_tiles))
        path.with_suffix(".json").write_text(json.dumps(index, indent=1))
        paths.append(path)
    capture.release()
    return paths


def find_players(args) -> dict[int, list[list[float]]]:
    """Per frame, the four most confident people on court that are not still objects."""
    people = find_people(args)
    capture = cv2.VideoCapture(str(Path(args.input).expanduser()))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    capture.release()
    moving = drop_still_boxes(people, round(fps), round(STILL_WINDOW_SECONDS * fps))
    dropped = sum(map(len, people.values())) - sum(map(len, moving.values()))
    print(f"{len(people)} frames, {dropped} still boxes dropped")
    return {frame: boxes[:PLAYERS_ON_COURT] for frame, boxes in moving.items()}


def main() -> None:
    args = parse_args()
    if args.people_only:
        find_people(args)
        return
    players = find_players(args)
    outline = Outline.load(args.court_outline)
    poses = near_team(estimate_poses(args, players, outline))
    capture = cv2.VideoCapture(str(Path(args.input).expanduser()))
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    print(summary(args, poses, size, len(players)))
    if args.overlay:
        print(write_overlay(args, poses, *args.overlay))
    if args.sample:
        for path in write_samples(args, poses, args.sample):
            print(path)


if __name__ == "__main__":
    main()
