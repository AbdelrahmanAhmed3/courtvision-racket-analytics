# Evaluation

CourtVision's accuracy is measured against hand labels. Terms (Clip, Rally, Shot,
Impact, Bounce, Wall rebound, Player) are defined in [CONTEXT.md](../CONTEXT.md).

## Clips

Footage is never committed (ADR 0002). Labels are committed in `evaluation/labels/`,
one JSON file per clip, named after the clip.

| Clip file | Source | Notes |
| --- | --- | --- |
| `BEST-OF-AGUSTIN-TAPIA-RED-BULL-OUT-THE-COURT.mp4` | [Premier Padel on YouTube](https://www.youtube.com/watch?v=ZSneXJfDqjI), 720p | 7.5 min compilation: many courts, lighting and shirt colours, points played outside the court |

Download a clip into `data/raw/` (ignored by git), for example:

```bash
pip install -e ".[media]"
yt-dlp -f "bv*[height=720][ext=mp4]" -o "data/raw/BEST-OF-AGUSTIN-TAPIA-RED-BULL-OUT-THE-COURT.%(ext)s" \
    "https://www.youtube.com/watch?v=ZSneXJfDqjI"
```

## Labelling a clip

```bash
pip install -e ".[local]"   # RF-DETR proposes player boxes
python scripts/label_clip.py --input data/raw/<clip>.mp4 --source-url <link>
```

The first run finds camera cuts and splits the clip into **segments**. Labels save
after every change; run the same command again to continue. The side panel shows a
checklist for the current segment and frame, and every key.

For each segment:

1. **Label or skip** it (`l` / `o`). Label segments that show one rally from the
   main camera. Skip replays, close-ups, crowd shots and anything after a cut you
   cannot follow. If a cut was missed, press `c` on the first frame of the new shot.
2. **Rally start** (`r`): the frame of the serve's impact. **Rally end** (`e`): the
   frame the ball is dead (second bounce, net, out, or play visibly stops).
3. **Impacts** (`1`-`4`): on the frame where the racket meets the ball, press the
   hitter's number. Use the closest frame if the contact falls between two.
4. **Bounces** (`b`, then click the ball): the frame the ball touches the floor.
   **Wall rebounds** (`w`, then click): the frame it touches glass or mesh. Click
   where the ball is in the image.
5. **Player boxes** (`g` jumps to the next frame that needs them, one per second):
   RF-DETR proposes boxes. Click a player's box and press their number; drag to draw
   a box it missed. Press `y` when the four players are assigned: boxes left without
   a player (crowd, officials, ball kids) are removed. Box a player even when they
   are outside the court. Do not box a player you cannot see.

### Player numbers

Numbers are per segment. At the rally start, the team **nearer the camera** is 1
and 2, the far team is 3 and 4; within each team, the player on the **left of the
image** gets the lower number. Then follow the **same person** for the whole
segment, even if partners swap sides.

## File format

Schema version 1, produced by `courtvision.evaluation.labels`:

```jsonc
{
  "schema_version": 1,
  "clip": "BEST-OF-AGUSTIN-TAPIA-RED-BULL-OUT-THE-COURT.mp4",
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
      "boxes": {"525": [{"x1": 430, "y1": 245, "x2": 485, "y2": 358, "player": 1}]},
      "reviewed_box_frames": [525]          // box frames whose boxes are final
    }
  ]
}
```

Coordinates are pixels in the original clip. Only reviewed box frames count as
labels; other frames in `boxes` hold unreviewed model proposals.
