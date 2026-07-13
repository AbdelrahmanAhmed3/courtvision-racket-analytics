"""Native macOS launcher for manual calibration and a short local pipeline run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "tennis-v4d0h/2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a local video picker, calibrate the court, then run CourtVision."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional output directory. "
            "The default is a timestamped local run folder."
        ),
    )
    return parser.parse_args()


def prompt_configuration(root: Tk) -> tuple[Path, str, float, float] | None:
    video_path = filedialog.askopenfilename(
        parent=root,
        title="Choose a racket-sport video",
        filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
    )
    if not video_path:
        return None

    court_type = simpledialog.askstring(
        "Court type",
        "Enter court type: tennis or padel",
        initialvalue="tennis",
        parent=root,
    )
    if court_type is None:
        return None
    court_type = court_type.strip().lower()
    if court_type not in {"tennis", "padel"}:
        messagebox.showerror("CourtVision", "Court type must be tennis or padel.")
        return None

    duration = simpledialog.askfloat(
        "Run duration",
        "How many seconds should CourtVision process from the start?",
        initialvalue=15.0,
        minvalue=0.01,
        parent=root,
    )
    if duration is None:
        return None

    calibration_time = simpledialog.askfloat(
        "Calibration frame",
        "At what video timestamp (seconds) should you click the court landmarks?",
        initialvalue=0.0,
        minvalue=0.0,
        parent=root,
    )
    if calibration_time is None:
        return None
    return Path(video_path), court_type, duration, calibration_time


def frame_index_at_seconds(video_path: Path, seconds: float) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    frame_index = int(round(seconds * fps))
    if frame_index >= frame_count:
        raise ValueError(
            f"Calibration timestamp {seconds:g}s is beyond the video duration."
        )
    return frame_index


def run(command: list[str]) -> None:
    print("\nRunning:\n", " ".join(command), "\n", sep="")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    root = Tk()
    root.withdraw()
    root.update()
    try:
        configuration = prompt_configuration(root)
        if configuration is None:
            return
        video_path, court_type, duration, calibration_time = configuration
        frame_index = frame_index_at_seconds(video_path, calibration_time)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) if args.output_dir else (
            REPO_ROOT / "outputs" / "local_runs" / f"{video_path.stem}_{timestamp}"
        )
        calibration_path = output_dir / "calibration.json"

        messagebox.showinfo(
            "CourtVision",
            "The point-click window will open next. Click all eight prompted "
            "landmarks, then press Enter to save.",
        )
        run(
            [
                sys.executable,
                "scripts/calibrate_court.py",
                "--input",
                str(video_path),
                "--court-type",
                court_type,
                "--frame",
                str(frame_index),
                "--output",
                str(calibration_path),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/run_full_pipeline.py",
                "--input",
                str(video_path),
                "--output-dir",
                str(output_dir),
                "--model-id",
                args.model_id,
                "--max-seconds",
                str(duration),
                "--calibration",
                str(calibration_path),
                "--draw-court-map",
                "--draw-calibration-overlay",
            ]
        )
        messagebox.showinfo("CourtVision", f"Finished. Outputs are in:\n{output_dir}")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        messagebox.showerror("CourtVision", str(error))
        raise
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
