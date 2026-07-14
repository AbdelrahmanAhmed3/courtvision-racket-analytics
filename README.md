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

## Local Web App

Install the UI and Roboflow extras, then start the local app:

```bash
pip install -e ".[ui,roboflow]"
streamlit run app.py
```

Open the localhost URL printed by Streamlit. Choose a video from your computer,
select its court type, click the eight prompted court landmarks in the browser,
then run player tracking and court mapping. The app uses the local `.env` for
`ROBOFLOW_API_KEY` and writes generated artifacts under `outputs/ui_runs/`.
Completed runs show the tracked broadcast and the standalone court map side by
side; their play, pause, and seek controls stay synchronized.

When `tracknet-model/model_best.pt` exists, the app automatically enables the
unified player-plus-ball projection option. It can clone the matching TrackNet
source into `tracknet-model/TrackNet/` on the first run when the setup checkbox
is enabled.

To include TrackNet ball tracking and project the ball to the court map, pass
the TrackNet clone and weights to the full pipeline:

```bash
python scripts/run_full_pipeline.py \
  --input data/raw/your_video.mp4 \
  --output-dir outputs/your_run \
  --calibration configs/calibrations/your_video.json \
  --draw-court-map \
  --tracknet-dir /path/to/TrackNet \
  --tracknet-model-path /path/to/tracknet_weights.pth
```

TrackNet runs only when its weights are supplied. Ball points are recorded as
`object_type=ball` alongside player rows in `tracks_with_court_coords.csv`.
Install PyTorch for TrackNet runs with:

```bash
pip install -e ".[tracknet]"
```

The ball center is projected onto the court plane. During high airborne shots,
that is an approximation because a planar homography cannot represent height.

## Moving Cameras

Enable **Adapt calibration to camera movement** in the local web app, or pass
`--track-calibration` to the full pipeline. Starting from the manually
calibrated frame, CourtVision tracks the named court landmarks with sparse
optical flow and uses RANSAC to estimate a fresh homography per frame. Frames
before the selected calibration timestamp, or frames that fail validation after
a cut or landmark drift, are deliberately left unmapped rather than projected
with stale geometry.

## Versioning

CourtVision uses feature releases such as `v1.1`, `v1.2`, and `v1.3` for
backwards-compatible additions. Changes to calibration/homography logic, model
or tracking behavior, ball projection, analytics definitions, or published
output schemas start a new major release (`v2.0`, `v3.0`, and so on). See
[CHANGELOG.md](CHANGELOG.md) for the release history and full policy.

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

For an interactive local macOS run, open a native video picker, choose the
court type and duration, then click the eight prompted landmarks in the OpenCV
window:

```bash
python scripts/run_local_interactive.py
```

It writes a timestamped folder under `outputs/local_runs/` containing the saved
calibration JSON, annotated video, court-map video, and CSV files. The launcher
uses `ROBOFLOW_API_KEY` from the local `.env` file.

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
