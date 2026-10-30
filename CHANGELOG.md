# Changelog

## Unreleased (planned for v4.0)

- the app opens on **Analyse a clip**: drop a clip, click six court points (with a guide), press Analyse, and get the tracked video beside the court map, the report, and a PDF; the old page is now **Advanced**; clicks land correctly when the frame is shown smaller than its size
- the pipeline writes the match report as a PDF too (`report.pdf`, with reportlab, BSD-3-Clause); the Markdown and the PDF share their wording; `report.json` says whether the ball was tracked (`ball_tracked`), so a clip without ball tracking no longer reports "0 rallies"
- README rewritten around padel: a demo of the tracked video beside the court map, what you get from a clip, the measured results with links to `reports/`, a quick start, and known limitations; footage is credited, and the README may show one short excerpt of the pipeline's output (ADR 0006)
- the court map no longer labels a shot "received" by the hitter's own team, which meant one of its two impacts was false, and the label reads `P4 -> P1  ~99 km/h`
- a match report per clip, as Markdown and JSON (`report.md`, `report.json`) with player heatmaps (`heatmaps.png`), built from the clip's stats with templates and shown in the app under the videos; each rally counts the places where the same team hit twice in a row, which the rules do not allow; every JSON key is described in `report.schema.json`, which the tests check reports against; the court map's running distance counter is removed in its favour
- movement stats per player from smoothed court positions: distance, average and top speed, time at the net, in the middle and at the back (`movement.csv`), and a heatmap of seconds per square metre (`heatmaps.json`); tracking jumps such as ID swaps cut the track instead of counting as running (`reports/player_movement.md`); `player_distances.csv` is replaced by `movement.csv`
- split a clip into rallies from its ball events (a rally starts at its first impact and ends at its last event; 5 s of silence ends it); the full pipeline writes `rallies.csv`; all 10 labelled rallies found, start and end within 0.2 s at the median (`reports/ball_events.md`)
- detect impacts, bounces and wall rebounds with a small trained model (`--events classifier`, the default; `--events legacy` for the older tennis-tuned shot code); the full pipeline writes `ball_events.csv`, shots with hitter, receiver and bounce in `shots.csv` (speed column `approx_court_speed_kmh`: the ball's height is not modelled), and marks each event on `annotated.mp4`; impact F1 0.41, hitter right for 65% of impacts found (`reports/ball_events.md`)
- assign each tracked player to the near or far team by the half of the court they play in; the full pipeline writes `teams.csv` (all labelled players on the right team, `reports/player_identity.md`)
- track players with at most as many IDs as the court has players (4 for padel, 2 for tennis): a player who is hidden or whose box changes shape gets their ID back instead of a new one (`--tracker fixed`, the default; `--tracker simple` for the old greedy tracker); IDF1 0.96 → 0.97 and ID switches 8 → 5 on the labelled rallies (`reports/player_identity.md`)
- `scripts/label_clip.py`: label rallies, impacts, bounces, wall rebounds and player boxes for evaluation, with RF-DETR box proposals; format and guide in `docs/evaluation.md`
- detect players locally with RF-DETR (Apache-2.0 COCO weights, Nano by default) instead of the hosted API: `--detector rfdetr|roboflow`, `--rfdetr-size`, and a detector choice in the app; install with the `local` extra
- drop people standing off the court before tracking (0.5 m margin for padel; 4 m sides and 7 m ends for tennis)
- `scripts/benchmark_detector.py` (calibration or image-space court outline, speed, players kept, track IDs) and results on four clips in `reports/model_comparison.md`
- remove the unused Ultralytics adapter and the `gpu` extra (AGPL, ADR 0002)
- set Roboflow player detection confidence explicitly to `0.30`
- replace frame-count track expiry with a three-second missing-player window
- reassociate returning dormant tracks by a conservative center-distance gate
- expose confidence, recovery time, and reassociation distance in the local app
- retry temporary hosted-inference failures with backoff; send confidence and overlap to the hosted API
- optionally keep only the most confident players (2 for tennis, 4 for padel)
- add an experimental Kalman IoU tracker and `scripts/compare_trackers.py`; add `scipy` dependency
- remove `--max-missing-frames` in favour of the time-based missing-player window
- per-player distance in court metres; experimental shots, bounces and court-plane ball speed (tuned on tennis)
- experimental automatic calibration: RANSAC court lines, padel surface colour, and a hosted court-keypoint model with line refinement
- `net_center` and service-centre landmarks for padel and tennis
- full pipeline writes `player_distances.csv` and `shots.csv`; the app gains an automatic calibration lab

## v3.0 - Temporal Court Calibration

- track named manual court landmarks with sparse Lucas-Kanade optical flow
- use forward-backward tracking consistency and per-frame RANSAC validation
- estimate a fresh homography for every validated frame after calibration
- leave frames unmapped after validation failure instead of applying stale geometry
- added a default-on moving-camera option to the local web app

## v2.2 - Projection Visibility

- added an extended out-of-court region to the minimap so players behind a
  baseline remain visible
- show player and ball projection counts in the local app result view
- restored TrackNet's upstream Hough-circle candidate behavior

## v2.1 - Unified Local Ball Projection

- auto-detect `tracknet-model/model_best.pt` in the local web app
- prefill local TrackNet source and weights paths for a unified player-plus-ball run
- optionally clone the required TrackNet source automatically when it is missing

## v2.0 - TrackNet Ball Projection

- added optional yastrebksv/TrackNet inference to the full pipeline
- projected visible ball centers through the selected tennis or padel homography
- added `object_type=ball` rows to `tracks_with_court_coords.csv`
- rendered a ball marker and trajectory on both the broadcast and court map
- added local web-app controls for TrackNet repository, weights, and device

## v1.2 - Browser-Safe Video Results

- transcode app results to browser-compatible H.264/yuv420p MP4 files
- write a standalone court-map video alongside the combined render
- display tracked video and court map as independently visible, synchronized
  players in the local web app

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
