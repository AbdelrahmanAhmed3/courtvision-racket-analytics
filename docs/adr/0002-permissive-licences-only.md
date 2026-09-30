# Permissive dependencies only: no AGPL, non-commercial or unlicensed components

_Amended by [ADR 0004](0004-agpl-with-commercial-licence.md): CourtVision itself is now AGPL-3.0 with commercial licences available, not Apache-2.0. The dependency rule below is unchanged._

CourtVision must be usable both under its open-source licence and under a commercial licence sold by its author, so that a product can grow from it later. Every model, weight file, dataset and library it ships or downloads must therefore be under a permissive licence (Apache-2.0, MIT, BSD, CC BY). This rules out the most common choices in open-source sports-analytics projects, which is why this is written down.

## Considered Options

- **Ultralytics YOLO (detection, pose)**: rejected. AGPL-3.0 covers the code and the weights, including fine-tunes, so CourtVision could not be offered under a commercial licence without a paid Ultralytics licence. We use RF-DETR (Apache-2.0) for detection and rtmlib (Apache-2.0) for pose instead.
- **Roboflow hosted inference**: kept only as an optional backend. It needs an API key and a network connection, so it cannot be the default.
- **Existing tennis/padel repos and weights** (TrackNet by yastrebksv, TennisCourtDetector, padel_analytics): not reused. They are unlicensed (all rights reserved) or CC BY-NC-SA. Their published ideas may be reimplemented; their code, weights and data may not be copied.

## Consequences

- There are no public padel ball or padel court-keypoint weights under a usable licence, so CourtVision trains and publishes its own (planned for v4.1).
- Clips downloaded from YouTube are never committed or re-hosted; the README credits the source channel and links to it.
