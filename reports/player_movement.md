# Player Movement

## Movement stats per player (2026-10-27)

`analytics/movement.py`, #10. Each player's position is their feet projected onto the
hand-checked court. The full pipeline writes `movement.csv` (distance, average and top
speed, time at the net, in the middle and at the back) and `heatmaps.json` (seconds per
square metre).

### Why positions are smoothed

Feet positions wobble by a few pixels from frame to frame; far from the camera a pixel
covers more court, so the wobble is larger. On the 15-second Qatar Major clip (30 fps),
summing raw frame-to-frame steps gives:

| Player | Raw distance | Smoothed over 0.17 / 0.43 / 0.83 s |
| --- | --- | --- |
| 1 (near) | 22.9 m | 18.1 / 15.6 / 13.3 m |
| 2 (far) | 58.4 m | 41.2 / 30.0 / 22.8 m |
| 3 (near) | 27.6 m | 23.6 / 20.7 / 17.9 m |
| 4 (far) | 73.0 m | 44.2 / 33.8 / 27.7 m |

The far players' raw distances are two to three times too high. Positions are smoothed
with a centred moving average over 0.5 s, short enough to keep a sprint or a change of
direction. Top speed is measured over 0.5 s spans of smoothed positions, not single
frames.

### Tracking jumps are cut, not smoothed

The first version still gave the far players top speeds of 29 km/h. Tracing them showed
an **ID swap between the two far players, 7 m apart**: at frame 405 one far player's box
jumped to a false detection at the back fence, and by frame 410 the two far players had
exchanged IDs. Smoothing spread the 7 m jump over half a second, which read as a sprint;
a running median before the average did not help (28–29 km/h), because the jump lasts
longer than a spike. So a step faster than 30 km/h, faster than any padel player runs,
now cuts the track like a gap: no distance, smoothing or top speed runs across it.

### Results on the Qatar clip

| Player | Team | Distance | Average | Top speed | At the net | Middle | At the back |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | near | 13.3 m | 3.2 km/h | 8.6 km/h | 12.5 s | 1.4 s | 1.1 s |
| 2 | far | 15.3 m | 3.7 km/h | 5.9 km/h | 0.8 s | 3.4 s | 10.7 s |
| 3 | near | 16.9 m | 4.1 km/h | 11.4 km/h | 3.8 s | 9.7 s | 1.5 s |
| 4 | far | 16.1 m | 3.9 km/h | 9.6 km/h | 1.3 s | 8.5 s | 5.0 s |

"At the net" is within 4 m of it, where volleys are played; "at the back" is behind the
service line (6.95 m from the net); the middle is in between.

### Limits

- **No ground truth for distance or speed.** The labels give player positions once per
  second, too sparse to measure running. The stats are checked by synthetic tracks with
  known answers (`tests/test_movement.py`) and by being plausible on real clips.
- **The ID swap above is a second failure mode for identity**, after crossings (#47): a
  false detection near one player can hand their ID to their partner.
- Positions outside the court count in the edge squares of the heatmap.
- The court-map video's running distance counter still uses the older, unsmoothed
  tracker; it moves to these stats with the match report (#11).
