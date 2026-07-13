# Changelog

## v1.1 - Local Web Interface

- added the Streamlit localhost app for browser-based video selection,
  landmark clicking, calibration validation, and pipeline execution
- added configurable processing duration through `--max-seconds`
- documented the camera-motion, ball-projection, and analytics roadmap

## v1.0 - Calibration and Player Mapping

- named manual court landmarks for tennis and padel
- RANSAC-validated homography and calibration overlay
- persistent player tracking and court-coordinate projection
- player minimap video and coordinate CSV output

## Versioning Policy

- `v1.1`, `v1.2`, `v1.3`, etc.: additive UI, visualization, workflow, export,
  or other backwards-compatible features that do not change core CV meaning.
- `v2.0`, `v3.0`, etc.: a change to homography/calibration logic, detector or
  tracking model behavior, ball projection behavior, analytics definitions, or
  a published output schema. These can change the interpretation or
  comparability of prior results.
- Patch versions such as `v1.1.1`: bug fixes that preserve existing behavior
  and output meaning.
