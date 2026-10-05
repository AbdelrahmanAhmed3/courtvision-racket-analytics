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

### Checking labels

The rules of padel catch many labelling mistakes. Press `n` to jump to the next
frame that breaks one; the side panel says what is wrong there. To list them all:

```bash
python scripts/check_labels.py evaluation/labels/*.json
```

- **A team hits twice in a row.** Never legal, so an impact was missed or a player
  number is wrong.
- **An event outside the rally**, or a rally that starts mid-segment but not on the
  serve's impact.
- A rally with no start or end, box frames not done, a segment not reviewed.

The checks point at frames worth a second look; they cannot prove labels right.

### Player numbers

Numbers are per segment. At the rally start, the team **nearer the camera** is 1
and 2, the far team is 3 and 4; within each team, the player on the **left of the
image** gets the lower number. Then follow the **same person** for the whole
segment, even if partners swap sides.

## File format

Schema version 1, produced by `courtvision.evaluation.labels` (field order in saved
files may differ; coordinates are floats):

```jsonc
{
  "schema_version": 1,
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
      "reviewed_box_frames": [525]          // box frames whose boxes are final
    }
  ]
}
```

Coordinates are pixels in the original video. Only reviewed box frames count as
labels; other frames in `boxes` hold unreviewed model proposals.
