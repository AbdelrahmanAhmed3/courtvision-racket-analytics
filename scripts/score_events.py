"""Score predicted impacts, bounces and wall rebounds against hand labels.

Example:
    python scripts/score_events.py \
        --labels evaluation/labels/<video>.json --predictions predictions.json

Predictions are JSON: {"events": [{"kind": "impact", "frame": 512}, ...]}, with
kinds impact, bounce or wall_rebound. See docs/evaluation.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.evaluation.labels import load_labels  # noqa: E402
from courtvision.evaluation.scoring import (  # noqa: E402
    DEFAULT_TOLERANCE_FRAMES,
    load_predictions,
    score_events,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", required=True, help="Labels JSON.")
    parser.add_argument("--predictions", required=True, help="Predictions JSON.")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE_FRAMES,
        help="Frames a prediction may be off by (default: %(default)s).",
    )
    args = parser.parse_args()
    labels = load_labels(args.labels)
    scores = score_events(labels, load_predictions(args.predictions), args.tolerance)

    seconds = args.tolerance / labels.fps
    print(f"{labels.video}: tolerance ±{args.tolerance} frames ({seconds:.2f} s)")
    print(
        f"{'event':<14}{'labelled':>9}{'predicted':>10}{'hits':>6}"
        f"{'precision':>11}{'recall':>8}{'mean error':>12}"
    )
    for kind, score in scores.items():
        errors = score.frame_errors
        mean = f"{sum(errors) / len(errors):+.1f}" if errors else "-"
        print(
            f"{kind:<14}{score.labelled:>9}{score.predicted:>10}{score.hits:>6}"
            f"{score.precision:>11.2f}{score.recall:>8.2f}{mean:>12}"
        )


if __name__ == "__main__":
    main()
