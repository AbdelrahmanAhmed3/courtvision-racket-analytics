# CourtVision Racket Analytics

Production-style computer vision pipeline for tennis/racket-sport video analytics.

The project is designed for a hybrid workflow:

- Local development for package code, preprocessing, polygon logic, visualization, tests, and docs.
- Kaggle GPU notebooks for heavier inference, training, and longer video runs.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Project Layout

```text
src/courtvision/      Reusable Python package
scripts/              CLI entrypoints for local and Kaggle runs
configs/              Demo configs and polygon files
tests/                Small local tests
reports/              Model comparison and failure analysis notes
outputs/              Generated local outputs, ignored by Git
data/                 Local raw/processed data, ignored by Git
```

## Secrets

Copy `.env.example` to `.env` locally and set your own values. Never commit `.env`.
