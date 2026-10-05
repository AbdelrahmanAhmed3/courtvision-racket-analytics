"""Check hand labels against the rules of padel and print what to fix.

Example:
    python scripts/check_labels.py evaluation/labels/*.json

Exits with status 1 if any file has problems. Press n in label_clip.py to step
through the same problems.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.evaluation.checks import find_problems  # noqa: E402
from courtvision.evaluation.labels import load_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="+", help="Labels JSON files.")
    args = parser.parse_args()
    total = 0
    for path in args.labels:
        labels = load_labels(path)
        statuses = Counter(segment.status for segment in labels.segments)
        kinds = Counter(
            event.kind for segment in labels.segments for event in segment.events
        )
        box_frames = sum(len(s.reviewed_box_frames) for s in labels.segments)
        print(f"{path}")
        print(f"  segments: {dict(statuses)}")
        print(f"  events: {dict(kinds)}, finished box frames: {box_frames}")
        for number, segment in enumerate(labels.segments, start=1):
            problems = find_problems(labels, segment)
            total += len(problems)
            if segment.status == "unreviewed":
                continue  # counted above; one line each would bury the rest
            for problem in problems:
                print(f"  segment {number}, frame {problem.frame}: {problem.text}")
        if statuses["unreviewed"]:
            print(f"  {statuses['unreviewed']} segments not reviewed (l or o)")
    print(f"{total} problems")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
