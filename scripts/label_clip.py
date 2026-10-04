"""Label rallies, impacts, bounces and player boxes on a clip for evaluation.

Example:
    python scripts/label_clip.py --input data/raw/clip.mp4 \
        --source-url "https://www.youtube.com/watch?v=..."

Labels autosave to evaluation/labels/<clip name>.json after every change.
The keys are listed in the side panel; see docs/evaluation.md for what to label.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.evaluation.cuts import detect_cuts  # noqa: E402
from courtvision.evaluation.labels import (  # noqa: E402
    VideoLabels,
    load_labels,
    save_labels,
    segments_from_cuts,
)
from courtvision.evaluation.session import KEY_HELP, LabelSession  # noqa: E402

WINDOW = "CourtVision labelling"
PANEL_WIDTH = 400
DRAG_THRESHOLD_PX = 5
PLAYER_COLOURS = {
    1: (60, 200, 255),
    2: (60, 255, 120),
    3: (255, 120, 60),
    4: (230, 80, 230),
}
STATUS_COLOURS = {
    "label": (80, 200, 80),
    "skip": (90, 90, 90),
    "unreviewed": (40, 160, 230),
}
KEY_CODES = {27: "esc", 32: "space"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Clip to label.")
    parser.add_argument("--labels", help="Labels JSON (default: evaluation/labels/).")
    parser.add_argument(
        "--source-url",
        default="",
        help="Where the video came from (quote URLs in zsh).",
    )
    parser.add_argument("--box-interval", type=int, default=25)
    parser.add_argument("--max-width", type=int, default=1400, help="Window width.")
    parser.add_argument("--rfdetr-size", default="nano")
    parser.add_argument(
        "--no-prefill", action="store_true", help="Do not propose boxes with RF-DETR."
    )
    return parser.parse_args()


class FrameReader:
    """Random access to frames, reading sequentially when stepping forward."""

    def __init__(self, path: str) -> None:
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open {path}")
        self.position = -1
        self.frame: np.ndarray | None = None

    def read(self, index: int) -> np.ndarray:
        if index == self.position and self.frame is not None:
            return self.frame
        if index != self.position + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {index}")
        self.position, self.frame = index, frame
        return frame


def create_labels(args: argparse.Namespace) -> VideoLabels:
    capture = cv2.VideoCapture(args.input)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0

    def frames():
        nonlocal frame_count
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            frame_count += 1
            yield frame

    print("Finding camera cuts (one pass over the clip)...")
    cuts = detect_cuts(frames())
    capture.release()
    print(f"{frame_count} frames, {len(cuts)} cuts")
    return VideoLabels(
        video=Path(args.input).name,
        source_url=args.source_url,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        box_interval=args.box_interval,
        segments=segments_from_cuts(cuts, frame_count),
    )


class BoxProposer:
    """Lazily loads RF-DETR and proposes person boxes for a frame."""

    def __init__(self, size: str) -> None:
        self.size = size
        self.detector = None

    def __call__(self, frame: np.ndarray, index: int) -> list[tuple]:
        if self.detector is None:
            from courtvision.detectors.rfdetr_detector import RFDetrDetector

            print("Loading RF-DETR to propose boxes...")
            self.detector = RFDetrDetector(size=self.size)
        return [
            (d.x1, d.y1, d.x2, d.y2)
            for d in self.detector.predict_frame(frame, index)
            if d.y2 - d.y1 >= 15
        ]


def render(session: LabelSession, frame: np.ndarray, scale: float, playing: bool):
    view = cv2.resize(frame, None, fx=scale, fy=scale)
    for index, box in enumerate(session.boxes()):
        colour = PLAYER_COLOURS.get(box.player, (180, 180, 180))
        thickness = 4 if index == session.selected else 2
        top_left = (int(box.x1 * scale), int(box.y1 * scale))
        bottom_right = (int(box.x2 * scale), int(box.y2 * scale))
        cv2.rectangle(view, top_left, bottom_right, colour, thickness)
        label = f"P{box.player}" if box.player else "?"
        cv2.putText(view, label, (top_left[0], top_left[1] - 6), 0, 0.7, colour, 2)
    for event in session.segment.events:
        if event.frame == session.frame and event.x is not None:
            centre = (int(event.x * scale), int(event.y * scale))
            colour = (0, 255, 255) if event.kind == "bounce" else (255, 0, 255)
            cv2.circle(view, centre, 9, colour, 2)

    panel = np.full((view.shape[0], PANEL_WIDTH, 3), 30, np.uint8)
    header = f"Frame {session.frame}/{session.labels.frame_count - 1}"
    y = _text(panel, header + ("   PLAYING" if playing else ""), 26, 0.7)
    y += 6
    colours = {True: (120, 220, 120), False: (60, 170, 255), None: (220, 220, 220)}
    for item in session.checklist():
        mark = {True: "[x]", False: "[ ]", None: " - "}[item.done]
        for line in textwrap.wrap(f"{mark} {item.text}", 42, subsequent_indent="    "):
            y = _text(panel, line, y, 0.45, colours[item.done])
    for line in textwrap.wrap(session.message, 42):
        y = _text(panel, line, y + 4, 0.45, (0, 220, 255))
    y = _timeline(panel, session, y + 12)
    for key, meaning in KEY_HELP:
        y = _text(panel, f"{key:>9}  {meaning}", y, 0.36, (170, 170, 170))
    return np.hstack([view, panel])


def _text(panel, text, y, size, colour=(235, 235, 235)) -> int:
    cv2.putText(panel, text, (12, y), 0, size, colour, 1, cv2.LINE_AA)
    return y + int(32 * size + 5)


def _timeline(panel, session: LabelSession, y: int) -> int:
    width = PANEL_WIDTH - 24
    total = session.labels.frame_count
    for segment in session.labels.segments:
        x1 = 12 + int(width * segment.start / total)
        x2 = 12 + max(int(width * segment.end / total), x1 + 1)
        cv2.rectangle(
            panel, (x1, y), (x2 - 1, y + 14), STATUS_COLOURS[segment.status], -1
        )
    marker = 12 + int(width * session.frame / total)
    cv2.line(panel, (marker, y - 4), (marker, y + 18), (255, 255, 255), 2)
    return y + 34


def main() -> None:
    args = parse_args()
    labels_path = Path(
        args.labels
        or REPO_ROOT / "evaluation" / "labels" / f"{Path(args.input).stem}.json"
    )
    if labels_path.exists():
        labels = load_labels(labels_path)
    else:
        labels = create_labels(args)
        save_labels(labels, labels_path)
    print(f"Labels: {labels_path}")

    reader = FrameReader(args.input)
    proposer = None if args.no_prefill else BoxProposer(args.rfdetr_size)
    session = LabelSession(labels)
    scale = min(1.0, (args.max_width - PANEL_WIDTH) / labels.width)
    mouse = {"down": None, "events": []}

    def on_mouse(event, x, y, _flags, _param):
        point = (x / scale, y / scale)
        if x >= labels.width * scale:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = point
        elif event == cv2.EVENT_LBUTTONUP and mouse["down"] is not None:
            mouse["events"].append((mouse["down"], point))
            mouse["down"] = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    playing = False
    while True:
        frame = reader.read(session.frame)
        if proposer is not None and session.needs_prefill():
            session.prefill(proposer(frame, session.frame))
        for start, end in mouse["events"]:
            moved = abs(end[0] - start[0]) + abs(end[1] - start[1])
            if moved * scale > DRAG_THRESHOLD_PX:
                session.drag(*start, *end)
            else:
                session.click(*start)
        mouse["events"].clear()
        if session.dirty:
            save_labels(session.labels, labels_path)
            session.dirty = False
        cv2.imshow(WINDOW, render(session, frame, scale, playing))
        delay = max(1, int(1000 / labels.fps)) if playing else 30
        code = cv2.waitKeyEx(delay)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if code == -1:
            if playing:
                if session.frame + 1 >= labels.frame_count:
                    playing = False
                else:
                    session.go_to(session.frame + 1)
            continue
        # Only plain ASCII keys; arrows and other special keys are ignored.
        key = KEY_CODES.get(code, chr(code).lower() if 0 <= code < 128 else "")
        if key == "space":
            playing = not playing
        elif key == "q":
            break
        else:
            playing = False
            session.key(key)
    save_labels(session.labels, labels_path)
    cv2.destroyAllWindows()
    print(f"Saved {labels_path}")


if __name__ == "__main__":
    main()
