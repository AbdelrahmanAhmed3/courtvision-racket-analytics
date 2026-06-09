from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a court polygon preview.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--polygon", required=True, help="Polygon JSON path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"Polygon preview is not implemented yet: {args}")


if __name__ == "__main__":
    main()
