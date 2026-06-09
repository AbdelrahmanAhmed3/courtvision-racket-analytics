from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a short resized video clip.")
    parser.add_argument("--input", help="Local source video path.")
    parser.add_argument("--youtube-url", help="Optional YouTube URL for Kaggle runs.")
    parser.add_argument("--output", required=True, help="Output video path.")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"Clip creation is not implemented yet: {args}")


if __name__ == "__main__":
    main()
