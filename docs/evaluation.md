# Evaluation

CourtVision's accuracy is measured against hand labels. Terms (Source video, Segment,
Clip, Rally, Impact, Bounce, Wall rebound, Player) are defined in [CONTEXT.md](../CONTEXT.md).

## Source videos

Footage is never committed (ADR 0002). Labels are committed in `evaluation/labels/`,
one JSON file per source video, named after it. A whole compilation is labelled, so
no timestamps are needed: segments record their own frame ranges.

| Video file | Source | Notes |
| --- | --- | --- |
| `BEST-OF-AGUSTIN-TAPIA-RED-BULL-OUT-THE-COURT.mp4` | [Premier Padel on YouTube](https://www.youtube.com/watch?v=ZSneXJfDqjI), 720p | 7.5 min compilation: many courts, lighting and shirt colours, points played outside the court |

Download a video into `data/raw/` (ignored by git), for example:

```bash
pip install -e ".[media]"
yt-dlp -f "bv*[height=720][ext=mp4]" -o "data/raw/BEST-OF-AGUSTIN-TAPIA-RED-BULL-OUT-THE-COURT.%(ext)s" \
    "https://www.youtube.com/watch?v=ZSneXJfDqjI"
```

## Labelling a source video

```bash
pip install -e ".[local]"   # RF-DETR proposes player boxes
python scripts/label_clip.py --input data/raw/<video>.mp4 --source-url "<link>"
```

The first run finds camera cuts and splits the video into **segments**. Labels save
after every change; run the same command again to continue. The side panel shows a
checklist for the current segment and frame, and every key.

Label each rally in **two passes**:

- **Events pass, frame by frame.** Play with `space`, pause near each shot, then step
  with `a` / `d` to the exact frame of each impact, bounce and wall rebound. Never
  place events while jumping between box frames with `g`: box frames are a second
  apart, and an impact even a few frames off cannot be scored.
- **Boxes pass.** `g` to the next box frame, `p` to copy the players, check, `y`.

For each segment:

1. **Label or skip** it (`l` / `o`). Label segments that show one rally from the
   main camera. Skip replays, close-ups and crowd views. If a cut was missed, press
   `c` on the first frame after it; if a segment was split where there is no cut,
   press `m` to merge it with the previous one. A segment holds one rally: if it
   shows two points, press `c` between them.
2. **Rally start** (`r`): the frame of the serve's impact. **Rally end** (`e`): the
   frame the ball is dead (second bounce, net, out, or play visibly stops). The
   server's bounce before serving is not labelled: it comes before the rally starts.
   On the serve's frame press `r`, then the server's number. If the segment starts
   after the serve, press `r` on its first frame. After a fault, the rally starts at
   the second serve; do not label the first. To move a start or end that is already
   set, press `r` or `e` twice: one press never moves it, so a slip cannot.
3. **Impacts** (`1`-`4`): on the frame where the racket meets the ball, press the
   hitter's number. Use the closest frame if the contact falls between two.
4. **Bounces** (`b`, then click the ball): the frame the ball touches the floor,
   including the serve's bounce in the service box.
   **Wall rebounds** (`w`, then click): the frame it touches glass or mesh, whether
   play goes on or the point ends there. Label what happens, not whether it was
   legal. The net, a player's body and a ball leaving over the walls are not wall
   rebounds. Click where the ball is in the image. Positions are image pixels; court
   positions come later, once a segment can be calibrated.
5. **Player boxes** on every 25th frame of the rally (one per second at 25 fps; `g`
   jumps to the next one that is not done): RF-DETR proposes boxes. Click a player's
   box and press their number. If a box is loose or cut off, keep it selected and
   drag a tighter one to resize it; drag with nothing selected to draw a missing box.
   On later box frames, `p` copies the players from the previous finished box frame
   onto the new proposals (and copies a box as-is if the detector missed a player);
   check them, fix any mistake, then press `y`.
   Press `y` when every player you can see is assigned: boxes left without a player
   (crowd, officials, ball kids) are removed. Box a player even when they are outside
   the court. Do not box a player you cannot see.

   The boxes start as RF-DETR proposals, so accepting them unchanged favours RF-DETR
   when comparing detectors (#6). Tighten any box that does not fit the player.
6. **Court** (`k`, then 4 clicks), on any frame of the segment: click where each
   service line meets the side wall, in order: far left, far right, near right, near
   left. The tool then draws the court from those points. Check the drawing: the far
   baseline on the bottom of the far glass, the net line where the net meets the
   floor, the centre line on the painted one. If they are off, press `k` and click
   again; `esc` keeps the old points. Service line ends are used because they are
   painted and sharp, while boards and glass often hide the near floor corners. The
   drawn near baseline is the least reliable part: a camera lens bends straight
   lines, which four points cannot model.

### Checking labels

The rules of padel catch many labelling mistakes. Press `n` to jump to the next
frame that breaks one; the side panel says what is wrong there. To list them all:

```bash
python scripts/check_labels.py evaluation/labels/*.json
```

- **A team hits twice in a row.** Never legal, so an impact was missed or a player
  number is wrong.
- **Two bounces between impacts.** The second bounce ends the point, so if play goes
  on, one of them is not a bounce.
- **A serve returned before it bounced**, which is never legal.
- **An event outside the rally**, or a rally that starts mid-segment but not on the
  serve's impact.
- A rally with no start or end, box frames not done, a segment not reviewed.

The checks point at frames worth a second look; they cannot prove labels right.

### Player numbers

Numbers are per segment. At the rally start, the team **nearer the camera** is 1
and 2, the far team is 3 and 4; within each team, the player on the **left of the
image** gets the lower number. Then follow the **same person** for the whole
segment, even if partners swap sides.

## Scoring predictions

```bash
python scripts/score_events.py --labels evaluation/labels/<video>.json \
    --predictions predictions.json
```

Predictions are JSON, `{"events": [{"kind": "impact", "frame": 512}, ...]}`, with
kinds `impact`, `bounce` or `wall_rebound`. To produce them from today's shot code
(`analytics/shots.py`) on every labelled rally, run
`scripts/predict_shot_events.py` with the same `--labels` and `--video`; results are
in `reports/ball_events.md`. The scoring script prints, per kind:

- **Precision**: the share of predictions that match a labelled event.
- **Recall**: the share of labelled events that a prediction matches.
- **Mean error**: predicted minus labelled frame over the matches; above zero means
  late.

A prediction matches a labelled event of the same kind at most `--tolerance` frames
away (default 2, 0.08 s at 25 fps). Each labelled event takes **at most one**
prediction, so three predictions for one impact are one hit and two false alarms.
Pairs are chosen all at once (the Hungarian algorithm), because taking the nearest
prediction for each label in turn can use up the only one a later label could have
matched. Only labelled rallies count, widened by the tolerance; predictions anywhere
else are ignored.

The pairing gives predictions the benefit of the doubt: a false alarm next to a
missed event counts as a hit, because frames alone cannot say which event a
prediction was meant for. This can only happen when two labelled events of one kind
are at most twice the tolerance apart. In the Tapia labels at ±2 frames, that is 3
pairs of wall rebounds (the ball touching two walls in a corner) and never two
impacts or two bounces, which are at least 8 and 14 frames apart.

## Scoring player detection

```bash
python scripts/score_detectors.py --labels evaluation/labels/<video>.json \
    --video data/raw/<video>.mp4 --detector rfdetr --size nano
python scripts/score_detectors.py --labels evaluation/labels/<video>.json \
    --video data/raw/<video>.mp4 --detector roboflow --model-id <project>/<version>
```

The detector runs on every finished box frame, and its detections are cached in
`outputs/detections/`. A detection is a hit when it overlaps a labelled box by
`--min-iou` (default 0.5), one labelled box per detection. Two rows are printed:

- **As the pipeline keeps them**: people whose feet are on the segment's court plus
  0.5 m, the four most confident. Crowd and officials are never false alarms, so
  detectors that find everyone are not punished for it.
- **All people found**: every detection, to show what the filter removes.

Every labelled player counts towards recall, including players outside the court.

## File format

Schema version 2, produced by `courtvision.evaluation.labels` (field order in saved
files may differ; coordinates are floats). Version 1 files, without court points,
still load and are saved as version 2.

```jsonc
{
  "schema_version": 2,
  "video": "BEST-OF-AGUSTIN-TAPIA-RED-BULL-OUT-THE-COURT.mp4",
  "source_url": "https://www.youtube.com/watch?v=ZSneXJfDqjI",
  "fps": 25.0, "width": 1280, "height": 720, "frame_count": 11244,
  "box_interval": 25,                       // box frames are multiples of this
  "segments": [
    {
      "start": 462, "end": 915,             // frames; end is exclusive
      "status": "label",                    // "label", "skip" or "unreviewed"
      "rally_start": 500, "rally_end": 880,
      "events": [
        {"kind": "impact", "frame": 512, "player": 3, "x": null, "y": null},
        {"kind": "bounce", "frame": 530, "player": null, "x": 640.0, "y": 410.0},
        {"kind": "wall_rebound", "frame": 545, "player": null, "x": 980.0, "y": 300.0}
      ],
      "boxes": {"525": [{"x1": 430.0, "y1": 245.0, "x2": 485.0, "y2": 358.0, "player": 1}]},
      "reviewed_box_frames": [525],         // box frames whose boxes are final
      // where the service lines meet the side walls: far left, far right,
      // near right, near left
      "court_points": [[378.0, 230.0], [897.0, 230.0], [1140.0, 500.0], [135.0, 500.0]]
    }
  ]
}
```

Coordinates are pixels in the original video. Only reviewed box frames count as
labels; other frames in `boxes` hold unreviewed model proposals.
