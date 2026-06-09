from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/yastrebksv/TrackNet.git"
DEFAULT_TRACKNET_DIR = "/kaggle/working/TrackNet"
DEFAULT_INPUT = "data/raw/youtube_-jtV7IJP8NU_15s.mp4"
DEFAULT_OUTPUT_DIR = "outputs/tracknet_ball"


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def ensure_tracknet_repo(tracknet_dir: Path, repo_url: str) -> None:
    if (tracknet_dir / "infer_on_video.py").exists():
        print(f"Using existing TrackNet repo: {tracknet_dir}")
        return

    tracknet_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", repo_url, str(tracknet_dir)])


def maybe_install_requirements(
    python_executable: str,
    tracknet_dir: Path,
    install_requirements: bool,
) -> None:
    requirements_path = tracknet_dir / "requirements.txt"
    if not install_requirements:
        return
    if not requirements_path.exists():
        raise FileNotFoundError(f"TrackNet requirements not found: {requirements_path}")

    run_command(
        [python_executable, "-m", "pip", "install", "-r", str(requirements_path)]
    )


def infer_tracknet(
    python_executable: str,
    tracknet_dir: Path,
    model_path: Path,
    input_video: Path,
    output_video: Path,
    extrapolation: bool,
) -> None:
    script_path = tracknet_dir / "infer_on_video.py"
    if not script_path.exists():
        raise FileNotFoundError(f"TrackNet inference script not found: {script_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            "TrackNet weights not found. On Kaggle, add the weights as a Dataset "
            f"and pass --model-path. Missing path: {model_path}"
        )
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(script_path),
        "--model_path",
        str(model_path),
        "--video_path",
        str(input_video),
        "--video_out_path",
        str(output_video),
    ]
    if extrapolation:
        command.append("--extrapolation")

    run_command(command, cwd=tracknet_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run yastrebksv/TrackNet ball tracking on a video."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "Input video path. Default points to the 15s clip downloaded "
            "in this repo."
        ),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help=(
            "Path to TrackNet .pt/.pth weights. On Kaggle, add the weights as a "
            "Dataset and pass the file path under /kaggle/input/..."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for TrackNet outputs.",
    )
    parser.add_argument(
        "--output-name",
        default="tracknet_ball_annotated.avi",
        help="Output video filename. Upstream TrackNet writes AVI/DIVX by default.",
    )
    parser.add_argument(
        "--tracknet-dir",
        default=DEFAULT_TRACKNET_DIR,
        help="Where the TrackNet repo exists or should be cloned.",
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable to use for the upstream TrackNet script.",
    )
    parser.add_argument(
        "--install-requirements",
        action="store_true",
        help="Install TrackNet requirements before inference.",
    )
    parser.add_argument(
        "--extrapolation",
        action="store_true",
        help="Use TrackNet's interpolation/extrapolation post-processing.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Assume --tracknet-dir already exists and do not clone it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_video = Path(args.input)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    tracknet_dir = Path(args.tracknet_dir)
    output_video = output_dir / args.output_name

    if not args.skip_clone:
        ensure_tracknet_repo(tracknet_dir, args.repo_url)
    maybe_install_requirements(args.python, tracknet_dir, args.install_requirements)
    infer_tracknet(
        python_executable=args.python,
        tracknet_dir=tracknet_dir,
        model_path=model_path,
        input_video=input_video,
        output_video=output_video,
        extrapolation=args.extrapolation,
    )
    print(f"TrackNet output written to: {output_video}")


if __name__ == "__main__":
    main()
