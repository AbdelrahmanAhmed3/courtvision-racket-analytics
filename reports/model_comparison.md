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

- Precision and recall against labelled players, and comparison with the hosted
  Roboflow model (#6).
- Whether reflections in the padel glass are filtered: none were seen in these clips.
