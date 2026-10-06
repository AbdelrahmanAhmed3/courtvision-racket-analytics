"""Committed hand labels must pass the checks, so CI rejects a broken labels file."""

from pathlib import Path

import pytest

from courtvision.evaluation.checks import find_all_problems
from courtvision.evaluation.labels import load_labels

LABEL_FILES = sorted(
    (Path(__file__).resolve().parents[1] / "evaluation" / "labels").glob("*.json")
)


@pytest.mark.parametrize("path", LABEL_FILES, ids=lambda path: path.stem)
def test_committed_labels_pass_the_checks(path: Path) -> None:
    assert find_all_problems(load_labels(path)) == []
