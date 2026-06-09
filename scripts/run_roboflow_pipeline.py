from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Roboflow detection pipeline.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--polygon", required=True, help="Polygon JSON path.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--frame-stride", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"Roboflow pipeline is not implemented yet: {args}")


if __name__ == "__main__":
    main()
