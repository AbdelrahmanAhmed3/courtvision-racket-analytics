# Roadmap

CourtVision is open-source padel analytics: drop in a match clip, get player movement, shots and rallies on a court map, on a laptop. Terms used here are defined in [CONTEXT.md](../CONTEXT.md); the decisions behind this plan are in [docs/adr](adr/).

Priorities, in order: learning, portfolio, open-source adoption, product.

## v4.0: Showcase edition

Tracked in the [v4.0 milestone](https://github.com/AbdelrahmanAhmed3/courtvision-racket-analytics/milestone/1). Padel only, on high-angle clips with no cuts.

| Milestone | Goal | Issues |
| --- | --- | --- |
| M0 Cleanup | Land the local work, licence the repo, make tests and CI trustworthy | #1, #2, #3 |
| M1 Players | Run locally with no API key; four players with stable identities and teams | #4, #5 |
| M2 Ball events | Clean ball track, shots with hitter and receiver, bounces, rallies, all measured on labelled clips | #6, #7, #8, #9 |
| M3 Movement stats and report | Per-player movement stats, a match report, a documented output format | #10, #11, #12 |
| M4 Launch | README with demo GIF, one-command install, Colab, hosted viewer, release and launch post | #13, #14, #15, #16, #17 |

## v4.1: Trained models

- **Automatic calibration**: train and publish a padel court-keypoint model (RF-DETR keypoints, Apache-2.0) with a model card; the court-colour method is the baseline and manual calibration stays as the fallback.
- **Padel ball model**: fine-tune the ball tracker on PadelTracker100 (CC BY 4.0) and publish it.
- **Shot type**, from easiest to hardest, each compared on the same labelled shots:
  1. Vision-language model with no training (baseline).
  2. Pose estimation plus rules on joint angles.
  3. A classifier trained on pose keypoint sequences.
  4. Padel-specific shot types (bandeja, víbora, bajada) labelled by the author.

## v5.0: Personal edition

- Fixed camera behind the back glass; analytics for near-side players.
- Point outcomes signalled by a hand gesture to the camera (one finger: won, two: lost).
- LLM coaching report built from the match report and stats, with claims that cite moments in the clip, and an evaluation harness for its accuracy and cost.
- A web frontend (API plus React) instead of Streamlit.
- Tennis ball events and movement stats.

## Not planned yet

- Point outcomes and forced/unforced errors on broadcast clips.
- Wall rebounds and 3D ball height.
