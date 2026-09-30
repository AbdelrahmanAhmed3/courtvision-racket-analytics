# Model Comparison

## Player detection: local RF-DETR (2026-09-30)

RF-DETR with COCO-pretrained weights (Apache-2.0), keeping only `person` boxes, on
100 frames of `youtube_gFl3ADnFRtc_15s.mp4` (padel broadcast, 1280×720), confidence
0.3, Apple M4, 16 GB, on Apple's GPU (MPS). The same model on the CPU takes about
100 ms per frame instead of 61 ms. "On court" counts people whose feet project
inside the court plus a 0.5 m margin, using `configs/calibrations/qatar_padel_frame_90.json`.

| Model | ms per frame | FPS | People per frame | On court per frame |
| --- | ---: | ---: | ---: | ---: |
| rf-detr-nano | 41 | 24.3 | 7.84 | 4.47 |
| rf-detr-small | 61 | 16.5 | 10.92 | 4.03 |
| rf-detr-medium | 75 | 13.3 | 11.11 | 4.15 |

Reproduce with:

```bash
python scripts/benchmark_detector.py --input data/raw/youtube_gFl3ADnFRtc_15s.mp4 \
    --calibration configs/calibrations/qatar_padel_frame_90.json \
    --frames 100 --sizes nano small medium
```

**Findings**

- The court filter matters more than model size: every size sees 8–11 people per
  frame (spectators, officials, people on nearby courts), and filtering leaves about 4.
- **Small is the default.** It is the closest to four players on court; Nano is 1.5×
  faster but keeps more extra people on court (4.47), which the four-player cap then
  has to remove by confidence.
- A full 15 s run with Small, court filter and four-player cap produced exactly four
  players in 447 of 450 frames, with 6 track IDs (some ID switches, see #5).

**Not measured yet:** recall and precision against labelled players, and comparison
with the hosted Roboflow model. Both need the evaluation clips from #6.
