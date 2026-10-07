# Model Comparison

## Player detection: local RF-DETR (2026-10-04)

RF-DETR with COCO-pretrained weights (Apache-2.0), keeping only `person` boxes,
confidence 0.3, on an Apple M4 (16 GB) using its GPU (MPS) unless noted. For each
frame, people whose feet are on the court are kept, capped at four by confidence,
and tracked with the default tracker. "ms per frame" times detection only, after
three warm-up frames.

The court comes from a calibration where one exists (0.5 m margin for padel). The
other clips cannot be calibrated today, so their court floor was traced by eye as an
image-space outline in [`configs/court_outlines/`](../configs/court_outlines/).

| Clip | Camera | Model | ms per frame | FPS | People per frame | On court per frame | Frames with 4 | Track IDs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qatar broadcast, 15 s, 720p | high, fixed | Nano | 40 | 25.0 | 7.57 | 3.98 | 98% | 5 |
| | | Small | 60 | 16.5 | 10.37 | 3.99 | 99% | 6 |
| | | Medium | 75 | 13.4 | 11.04 | 4.00 | 100% | 5 |
| | | Small, CPU | 114 | 8.8 | 10.37 | 3.99 | 99% | 6 |
| WPT "best point", 77 s, 480p | high, fixed | Nano | 48 | 20.7 | 3.99 | 3.92 | 95% | 9 |
| | | Small | 76 | 13.1 | 4.05 | 3.93 | 95% | 9 |
| Brussels, 10 s, 970p, large crowd | high, fixed | Nano | 48 | 20.6 | 11.43 | 4.00 | 100% | 5 |
| | | Small | 79 | 12.7 | 12.89 | 4.00 | 100% | 5 |
| Club close-up behind the back glass, 49 s, 720p | low, fixed | Nano | 49 | 20.3 | 5.26 | 3.64 | 66% | 11 |
| | | Small | 79 | 12.7 | 5.27 | 3.65 | 67% | 13 |

Reproduce a row with `scripts/benchmark_detector.py`, for example:

```bash
python scripts/benchmark_detector.py --input data/raw/youtube_gFl3ADnFRtc_15s.mp4 \
    --calibration configs/calibrations/qatar_padel_frame_90.json --sizes nano small medium
python scripts/benchmark_detector.py --input "data/raw/<close-up clip>.mp4" \
    --court-outline configs/court_outlines/close_up_back_glass.json --sizes nano small
```

### Findings

- **Nano is the default.** It keeps four players on court as often as Small and
  Medium on every clip, and is about 1.6× faster. Small and Medium detect more people
  in total, but the extra ones are off the court (crowd, officials) and get filtered.
- **The court filter does the heavy lifting.** On the Brussels clip RF-DETR finds 11–13
  people per frame and the filter keeps exactly the four players in every frame.
- **The low close-up camera is the hard case**, and not because of detection: when a
  player is missing, they are hidden behind the near player or their partner. Track
  IDs jump to 11–13 for four players because hidden players return with a new ID
  (#5). This clip also cannot be calibrated, because the near baseline is out of frame (#32).
- An earlier version of this report recommended Small from the first 100 frames of
  one clip without the four-player cap. Measuring whole clips changed the conclusion.

### Not measured yet

- Comparison with the hosted Roboflow model (#6).
- Whether reflections in the padel glass are filtered: none were seen in these clips.

## Player detection against hand labels (2026-10-07)

651 hand-labelled player boxes on 165 frames: one frame per second of the 10
labelled rallies in the Agustin Tapia compilation, 5 tournaments (#34, #36). A
detection is a hit when it overlaps a labelled box by IoU ≥ 0.5, one to one. "As the
pipeline keeps them": people whose feet are on the hand-marked court plus 0.5 m, the
four most confident. `scripts/score_detectors.py`, confidence 0.3.

| Model | Precision (pipeline) | Recall (pipeline) | Recall (all people found) | People found per frame |
| --- | --- | --- | --- | --- |
| RF-DETR Nano | 1.00 (633/633) | 0.97 (633/651) | 0.98 (639/651) | 8.9 |
| RF-DETR Small | 1.00 (636/639) | 0.98 (636/651) | 0.98 (639/651) | 11.9 |

### Findings

- **What the pipeline keeps is almost always a player.** Without the court filter,
  precision is 0.43 (Nano) and 0.33 (Small): most people found are crowd and
  officials, and the filter removes them all.
- **The court filter drops the out-of-court shots this video is about.** Of Nano's 18
  missed players, 6 were found but dropped by the filter. Five of them stood 3–4 m
  outside a side wall, having run out through the door to play the ball (for
  example frame 3250). The sixth projects 0.9 m behind the far baseline, where a
  player at the glass cannot be: feet in the air or a slightly wrong court. A 0.5 m
  margin assumes padel players stay inside the walls; these points break that.
- **12 players were never found.** Those checked by eye (frames 150 and 175) are a
  player running outside the court, blurred, behind the glass and mesh.
- **Small is not clearly better.** It finds 3 more players out of 651, while running
  about 1.6× slower (see above). Nano stays the default.
- **The labels favour Nano.** Boxes started as Nano's proposals and were corrected by
  hand where they did not fit, so Nano's IoU is flattered. The comparison with a
  model whose boxes were never proposed (Roboflow) must keep this in mind.
