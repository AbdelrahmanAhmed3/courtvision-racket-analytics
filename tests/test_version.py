import re
from pathlib import Path

import courtvision


def test_the_package_version_matches_pyproject() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)

    assert courtvision.__version__ == version
