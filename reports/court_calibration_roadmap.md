# Court Calibration Roadmap

## Objective

Court calibration is a core CourtVision capability: map player foot positions and ball positions from video pixels into a top-down tennis or padel court coordinate system. The implementation should make the geometry, validation, and failure handling visible to portfolio reviewers.

## Phase 1 - Manual Calibration MVP

Status: complete

Deliverables:

- named landmark schema shared by tennis and padel
- OpenCV click tool for selecting visible court landmarks on a chosen video frame
- calibration JSON with image point, normalized template point, confidence, and source metadata
- template-to-image homography and inverse image-to-template homography
- RANSAC inlier report, per-landmark reprojection error, and good/warning/bad status
- validation overlay that renders the projected court template back onto the source frame

Success condition:

```text
A reviewer can open a calibration JSON and overlay image, identify every landmark,
and see that the projected court lines align with the real court.
```

Current note:

The earlier numeric point experiment is not a Phase 1 calibration artifact. Its manually guessed service-line points were not reliable enough to validate geometry. Phase 1 replaces that approach with named, intentionally clicked landmarks.

Validated evidence:

```text
Tennis, US Open frame 90:
- 8/8 RANSAC inliers
- 1.31px mean reprojection error
- 1.70px max reprojection error

Padel, Qatar Major frame 90:
- 8/8 RANSAC inliers
- 2.60px mean reprojection error
- 3.43px max reprojection error
```

## Phase 2 - Project Tracks Into Court Coordinates

Status: complete with optional TrackNet ball projection

Deliverables:

- player bottom-center projection into normalized court coordinates
- TrackNet ball-center projection into the same coordinate system
- `tracks_with_court_coords.csv` with image and court coordinates
- out-of-court flag for players standing behind a baseline or outside an alley

Success condition:

```text
Player and ball positions appear in the expected regions of a top-down court map.
```

Current v1 evidence:

```text
Tracked tennis player foot positions are projected into normalized and meter-space
coordinates. tracks_with_court_coords.csv records in-bounds status for each frame.
```

Current v2 addition:

```text
When TrackNet weights are configured, visible ball centers are projected with the
same validated homography and emitted as object_type=ball rows. This treats the
ball image center as its court contact estimate; airborne ball positions can be
geometrically offset because a planar homography cannot recover height.
```

## Phase 3 - Minimap and Trajectory Visualization

Status: complete for player and optional TrackNet ball rendering

Deliverables:

- tennis and padel top-down court renderers
- player IDs, ball point, and trajectory trails
- side-by-side source video and minimap render
- saved preview images for README/report use

Success condition:

```text
The court map communicates player and ball movement more clearly than bounding boxes alone.
```

Current v1 evidence:

```text
run_full_pipeline.py can render a side-by-side source video and player minimap with
validated calibration lines, tracked player IDs, and out-of-court markers.
```

## Phase 4 - Model-Assisted Calibration

Deliverables:

- `ModelLandmarkProvider` that proposes named court landmarks
- confidence threshold and minimum-visible-landmarks checks
- RANSAC inlier count, reprojection error, court-shape sanity, and temporal-stability checks
- manual correction fallback using the Phase 1 tool

Success condition:

```text
Automation speeds up calibration, but the system rejects or corrects unreliable model output.
```

### Camera motion and RANSAC

RANSAC is not a court-point detector and it does not by itself make a one-time
calibration follow a moving camera. Given several proposed image-to-court point
pairs, it repeatedly estimates a homography from a small random subset, counts
how many of the remaining pairs agree within a pixel-error threshold, and keeps
the best-supported transform. It protects the calibration from one or two bad
clicks or bad model landmarks.

To handle pan, zoom, or broadcast cuts, Phase 4 must produce landmarks on each
usable frame (or track the last reliable landmarks with optical flow), estimate
a fresh RANSAC homography, validate it, and temporally smooth only compatible
transforms. A cut or failed validation must invalidate mapping until a new
calibration is obtained. Manual correction remains the fallback.

## Phase 5 - Ball Projection and Analytics

Deliverables:

- a common ball-track interface backed initially by TrackNet
- ball center projection through the calibrated image-to-court homography
- ball coordinates and confidence in the same per-frame CSV as player tracks
- minimap ball marker and trajectory trail for tennis and padel
- analytics built only from validated court-space tracks: player speed,
  distance, court coverage, ball speed, rally segments, and shot events

Success condition:

```text
Every metric records its source tracks and confidence, and clips with unstable
calibration or missing ball observations are visibly marked rather than silently
reported as precise measurements.
```

## Architecture

```text
LandmarkProvider
|- ManualLandmarkProvider       Phase 1
|- JsonLandmarkProvider         Phase 1
|- ModelLandmarkProvider        Phase 4
`- HybridLandmarkProvider       Phase 4

Named landmarks
-> validation / RANSAC homography
-> image-to-court projection
-> minimap / analytics
```

Every provider produces the same named-landmark calibration format. This keeps the geometry and validation layers independent of whether landmarks came from a human click or a model prediction.
