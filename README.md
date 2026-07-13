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

## Court Calibration and Mapping

CourtVision v1 uses named manual landmarks for reliable court calibration. The
calibration is validated with RANSAC inliers and reprojection error before player
tracks are projected into normalized and meter-space court coordinates.

Create a calibration locally with the OpenCV click tool:

```bash
python scripts/calibrate_court.py \
  --input data/raw/your_video.mp4 \
  --court-type tennis \
  --frame 90 \
  --output configs/calibrations/your_video_frame_90.json
```

For Kaggle, create the calibration directly from the notebook's input video.
Run these cells before the pipeline command; `%run` is essential because it
keeps the interactive Matplotlib canvas in the notebook kernel:

```python
%pip install -q ipympl
%matplotlib widget
```

```python
%run scripts/run_full_pipeline.py \
  --input /kaggle/input/your-video/video.mp4 \
  --output-dir /kaggle/working/outputs/courtvision_v1 \
  --create-calibration \
  --court-type tennis \
  --calibration-frame 90 \
  --calibration-backend matplotlib \
  --draw-court-map \
  --draw-calibration-overlay
```

Click the eight prompted landmarks in order. The pipeline saves
`/kaggle/working/outputs/courtvision_v1/calibration.json`, validates it, then
continues with tracking and projection in the same run.

Validate the saved landmarks and render a court overlay:

```bash
python scripts/validate_calibration.py \
  --input data/raw/your_video.mp4 \
  --calibration configs/calibrations/your_video_frame_90.json \
  --output outputs/calibration/your_video_frame_90_overlay.jpg
```

Render existing tracked detections with a court map. This mode is Kaggle-safe and
does not require a Roboflow API call:

```bash
python scripts/run_full_pipeline.py \
  --input /kaggle/input/your-video/video.mp4 \
  --detections /kaggle/input/your-tracks/detections.csv \
  --calibration /kaggle/input/your-calibration/calibration.json \
  --draw-court-map \
  --draw-calibration-overlay \
  --output-dir /kaggle/working/outputs/courtvision_v1
```

Omit `--detections` to run Roboflow player detection and IoU tracking in the same
pipeline. In that mode, set `ROBOFLOW_API_KEY` through `.env` locally or a Kaggle
Secret.
