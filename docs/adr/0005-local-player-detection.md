# Players are detected only locally, with RF-DETR; the hosted tennis model is dropped

CourtVision offers two player detectors: RF-DETR with COCO weights on the user's computer (the default), and Roboflow's hosted `tennis-v4d0h/2`, kept as an optional backend that needs an API key (ADR 0002). Measured against 651 hand-labelled players from 10 padel rallies on 5 courts (#6, [the report](../../reports/model_comparison.md)), RF-DETR Nano keeps only players once the court filter runs (precision 1.00) and finds 97 % of them. The hosted model finds 83 %, and only 40 % on a court that looks unlike the tennis broadcasts it was trained on. We detect players with RF-DETR only, and remove the hosted player model from the app and the pipeline.

## Considered Options

- **Keep the hosted model as an option**: rejected. An option that is measurably worse on padel invites wrong results, and it needs an API key, a network connection and a model whose licence we have not verified.
- **Look for a hosted padel player model**: not pursued. None is known under a licence ADR 0002 allows, and there is little left to gain: several of RF-DETR's remaining misses are players who have run out of the court, which the court filter drops (#38), not players the detector cannot see.
- **RF-DETR Small instead of Nano**: rejected. It finds 3 more players out of 651 and is about 1.6× slower.
- **Fine-tune RF-DETR on padel players**: deferred until the misses on the court, not beside it, are what limits recall.

## Consequences

- Player detection needs no API key or network. `ROBOFLOW_API_KEY` remains only for the optional court-keypoint models, whose future is decided in #26.
- Removing the hosted option from `app.py`, `scripts/run_full_pipeline.py` and the older scripts is follow-up work. `RoboflowDetector` stays in the library, so another hosted model can still be measured with `scripts/score_detectors.py`.
- The evidence is one compilation, and its boxes started as Nano's proposals, which flatters Nano's overlap. Measure again when more footage is labelled.
