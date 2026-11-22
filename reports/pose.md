# Pose on the phone clips

## Can a pose model see the near player? (2026-11-18, #87)

The feature to sell: a player sticks their phone to the outside of their own back
glass, films themselves playing, and gets a report on their own movement and
shots. So this trial asks one question: from that angle, how much of the **near
player** (the camera's side of the net) does a pose model see? The far players are
out of scope. Whether a swing marks a hit is the next question (#60).

### Setup

- **Clips:** the author's own phone recordings (not published), one club at night
  under floodlights, 1080p at 30 fps with the 0.5× (ultra wide) lens, the phone
  leaning on the near back glass at about waist height. Five clips: four landscape
  and one portrait. IMG_1089 is cut
  at frame 10,600 (5 min 53 s), where the phone tipped over.
- **Players:** RF-DETR Nano finds people. People whose feet are inside a court
  outline traced by eye are kept (`configs/court_outlines/phone_IMG_*.json`); these
  clips can't be calibrated yet (#32). Anything found in the same place in half of
  the seconds within 30 s either side is dropped as a still object. **Near** means
  feet below the row where the net meets the floor. The **near team** is the two
  most confident near people, each found with confidence 0.55 or more.
- **Pose:** RTMPose-m (256×192 input, trained on seven public body datasets) through
  rtmlib on onnxruntime, on the CPU of an Apple M4. A body point counts as found from
  confidence 0.5. A part such as "wrists" is the share of left and right points found.
- **Left and right:** which half of the picture the player stands in.

### Results

Every near player in every frame. "Near player found" is the share of frames with
at least one; "cut by the edge" is the share of near boxes touching the frame.

| Clip | Length | Near player found | Median height (px) | Cut by the edge | Shoulders | Elbows | Wrists | Hips | Knees | Ankles | ms per player |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMG_1089 | 353 s | 98% | 310 | 6% | 98% | 94% | 71% | 97% | 96% | 91% | 33 |
| IMG_1086 | 46 s | 100% | 297 | 5% | 99% | 97% | 76% | 99% | 99% | 99% | 27 |
| IMG_1085 | 21 s | 100% | 321 | 7% | 97% | 93% | 70% | 96% | 96% | 93% | 31 |
| IMG_1084 | 12 s | 80% | 587 | 65% | 97% | 78% | 60% | 85% | 80% | 65% | 25 |
| IMG_1083, portrait | 70 s | 77% | 537 | 39% | 96% | 93% | 67% | 93% | 92% | 91% | 37 |

Which edge cuts them: almost always a side of the picture, almost never the top or
bottom. (A first version of this report blamed a low phone; these numbers say
otherwise.)

| Clip | Cut by the edge | Left | Right | Top | Bottom |
| --- | ---: | ---: | ---: | ---: | ---: |
| IMG_1089 | 6% | 5% | 1% | 0% | 0% |
| IMG_1086 | 5% | 5% | 0% | 0% | 1% |
| IMG_1085 | 7% | 3% | 3% | 0% | 0% |
| IMG_1084 | 65% | 14% | 51% | 0% | 0% |
| IMG_1083, portrait | 39% | 7% | 32% | 0% | 0% |

The two worst single players:

| Clip | Near player | Found in frames | Median height (px) | Cut by the edge | Wrists | Ankles |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| IMG_1084 | right, close to the glass | 78% | 703 | 100% | 47% | 37% |
| IMG_1083, portrait | left | 15% | 288 | 31% | 70% | 91% |

### Checked by eye

Pose confidence says how sure the model is, not whether it is right. So 30 random
near-player crops per clip (150 in all) were checked by eye: Claude looked at the
found wrists, with the player's left side drawn green and right side magenta. The
crops are 200×260 px; the sheets are written to `outputs/pose/` for a second look.

| Clip | Not a near player | No wrist found | Wrists on the right hands | Can't tell | Clearly wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| IMG_1089 | 1 | 1 | 19 | 9 | 0 |
| IMG_1086 | 0 | 3 | 20 | 7 | 0 |
| IMG_1085 | 5 | 1 | 13 | 11 | 0 |
| IMG_1084 | 1 | 3 | 15 | 11 | 0 |
| IMG_1083 | 1 | 2 | 14 | 13 | 0 |
| All | 8 | 10 | 81 | 51 | 0 |

- **No wrist was clearly on the wrong hand or side**, including players walking
  toward the camera, whose right side is on the left of the picture.
- **In 51 of the 132 crops with a wrist found, the picture can't tell where the
  hands are:** they are in front of the body, seen from behind; the player is cut
  to a sliver at the frame edge; or the racket is on the backhand side, so it is
  unclear which hand holds it. The model still places a wrist there, often
  confidently, so a swing measured from these wrists will be noisy.
- **Not a near player (8):** a jacket on the glass and a far player in IMG_1085,
  where the phone slid on the glass during the clip, so the jacket moved and a far
  player's feet fell below the net's row; a spectator's arm beside the phone; a
  finger on the lens; a far player close to the net.

### What goes wrong from this angle

1. **The near player is cut off by the sides of the picture when close to the
   glass and off to the side,** which is where padel defence happens: from the
   middle of the back glass, even the 0.5× lens doesn't reach that far sideways.
   In IMG_1084 the right player stands close to the glass for the whole clip and
   touches the frame edge in every frame: ankles found 37 % of the time, wrists
   47 %. The top and bottom almost never cut a near player: a flat phone sees about
   as far above its centre as below, so at waist height a standing player is
   centred head to feet.
2. **Seen from behind, the hands are often hidden by the body.** Wrists are the
   weakest part in every clip (60–76 %), and a third of the wrists found can't be
   checked by eye.
3. **Things near the camera are taken for players:** jackets and bags on the
   glass, people beside the phone, fingers on the lens. The still-object rule and
   the 0.55 confidence remove most of them (jackets and bags score 0.30–0.51,
   players a median of 0.88). They fail when the phone moves.
4. **Portrait loses a player.** IMG_1083 shows half the court width, so the left
   near player is in the picture 15 % of the time.
5. **Speed:** 25–37 ms per player on the CPU. With two near players at 30 fps that
   is about twice as slow as real time, before player detection. onnxruntime can
   also use Core ML on this Mac; that is untested.

### Recommendation

- **Pose is good enough for a near-player report when the whole player is in the
  picture.** On the two clips where the near players were rarely cut (IMG_1086 and
  IMG_1089): shoulders, hips and knees 96–99 %, ankles 91–99 %, wrists 71–76 %. Of
  their 60 crops checked by eye, 39 had the wrists on the right hands, none on the
  wrong ones, and 16 couldn't be told. Position, footwork and body direction can
  build on it now.
- **The racket arm is the weak part:** hidden hands from behind make wrist speed
  noisy. Whether swings still mark hits is #60, which should be measured on
  hand-labelled hits in phone clips.
- **The problems are the camera's, not the model's.** The cuts come from where the
  phone is, not from its height or tilt, since they are at the sides. Hidden hands
  come from filming from behind, which is the product's choice. A flat phone at
  hip height needs no mount and keeps the near player in the picture top to
  bottom. So the next step is a recording test from one session, measured with
  this script: the phone stuck flat to the glass at hip height, landscape, 0.5×,
  60 fps, once in the middle of the back glass and once behind the recording
  player's own side. A higher, tilted phone would only help to see where the ball
  lands on the far side; try it only if the landing spots from hip height turn out
  too rough, checked against labelled bounces. Keep landscape: portrait loses a
  player.
- **Before pose goes into the pipeline:**
  - rtmlib requires `opencv-contrib-python` as well as `opencv-python`; two OpenCV
    packages in one environment break each other, so it is installed with
    `--no-deps`.
  - Licences: rtmlib is Apache-2.0, onnxruntime MIT. The RTMPose weights are
    published by OpenMMLab under Apache-2.0, but some of the seven datasets they
    were trained on are released for research use. Whether that matters for the
    commercial licence (ADR 0002, ADR 0004) has to be checked before pose ships.

### Reproducing

Pose needs onnxruntime and rtmlib, the latter without its own OpenCV:

```bash
pip install onnxruntime && pip install --no-deps rtmlib
```

On first use rtmlib downloads the RTMPose-m model (51 MB, from
download.openmmlab.com). If Python fails the SSL check there (the python.org build
on macOS), fetch the zip with `curl` and put its `end2end.onnx` in
`~/.cache/rtmlib/hub/checkpoints/` under the zip's name with `.onnx`.

```bash
python scripts/try_pose.py --input ~/Downloads/IMG_1089.MOV \
    --court-outline configs/court_outlines/phone_IMG_1089.json --end 10600 \
    --overlay 3000 3900 --sample 30
python scripts/try_pose.py --input ~/Downloads/IMG_1086.MOV \
    --court-outline configs/court_outlines/phone_IMG_1086.json --overlay 0 1372 --sample 30
```

The people found are cached in `outputs/predictions/`, the poses in `outputs/pose/`
with the skeleton video and the sample sheets. None of it is committed.

## Hits from the swing (2026-11-22, #60)

A hit is a swing, and the hitting wrist moves fastest around contact, even when the
ball tracker has lost the ball. So does a peak in the near player's wrist speed mark
their hits? The trial runs on the phone clips' near players, the players the
product is for: IMG_1089 and IMG_1086, 19 rallies, 154 s of play, **53 near-team
hits** (P1 and P2). The far players, 80 to 95 px tall, are left out, as in the
section above.

### How

- **Poses:** the cache above, RTMPose-m on the two near players in every frame.
- **Players over time:** the pose boxes are linked with the IoU tracker. Who is who
  comes from the players boxed at each rally start (`--rally-start-boxes`, #93).
- **Wrist speed:** the fastest found wrist between consecutive frames, in **body
  heights per second** (box heights), so a player far from the camera counts the
  same as one near it.
- **Peaks:** the fastest within 0.2 s either side, at or above a threshold. Three
  thresholds were tried, and all three are reported.
- A peak **marks a hit** within ±2 frames, as the event scores.

### Peaks as hits

| Peak at least | Hits with a peak | Hits the ball tracker has no candidate for, with a peak | Peaks that are at a hit | Other peaks per second of play |
| --- | --- | --- | --- | --- |
| 1 body height/s | 23 of 53 | 9 of 20 | 23 of 246 | 1.45 |
| 2 body heights/s | 18 of 53 | 7 of 20 | 18 of 98 | 0.52 |
| 3 body heights/s | 10 of 53 | 5 of 20 | 10 of 38 | 0.18 |

Where a peak is near a hit (2 body heights/s, within ±8 frames), it falls from 3
frames before to 2 after the labelled frame, quartiles: a swing is wider than a
contact.

**Peaks don't find hits.** At most a third of the hits have one, and four in five
peaks aren't hits. Possible reasons, not checked one by one: arms move fast when
players run, feint or catch the ball, and seen from behind a third of the wrists
can't be placed by eye (section above).

### Who swung

But when a peak is at a hit, the player who swung is the labelled hitter 11 times
out of 11 (2 body heights/s). So the second question: given a hit, can pose say
which near player made it? Three rules, on the same 53 hits:

| Rule | Names a player | Right | Right, of all 53 |
| --- | --- | --- | --- |
| **Pose: the near player with the fastest wrist within ±3 frames** | 40 | **28 (70 %)** | **53 %** |
| Ball: the nearest player among everyone tracked (the pipeline's rule) | 25 | 10 (40 %) | 19 % |
| Ball: the nearest player on the near half | 31 | 16 (52 %) | 30 % |

- **From behind, the nearest player to the ball is often the wrong one.** The
  likely reason, not checked frame by frame: the near players' boxes are large and
  cover much of the court behind them in the picture, so a ball near one of them in
  the picture can be metres away.
- **Pose doesn't need the ball,** so it names a hitter even where the ball is lost.
- **It names no one 13 times,** each time because the fastest player's track
  started after the rally start (the tracker lost and found them), so the boxes
  there can't say who it is. Linking the same person across tracker breaks could
  recover these.

### Recommendation

- **Don't find hits from the swing.** It finds a third of the near players' hits,
  and most peaks are false.
- **Do name the near hitter from the swing.** For hits found from the ball, the
  near player whose wrist moves fastest around the hit is right 70 % of the times
  it names someone, against 52 % for the nearest near player, from the same clips.
  This is the next issue: a pipeline step, when pose runs, that names the near
  hitter of each near-team hit.
- **Swing as an input to the hit model is untested.** 7 of the 20 near hits the ball
  tracker has no candidate for have a peak, but each comes with about five false
  ones. Worth trying only together with the phone event model (#94).
- **How far to trust this:** 53 hits from one court on one evening, two players
  near the camera, and identity from one box per rally. The three thresholds were
  not chosen on separate clips, so the best row flatters itself a little; the "who
  swung" rule has no threshold.

### Reproducing

Needs the pose cache above, and the players and ball caches of
`scripts/predict_shot_events.py` and `scripts/compare_ball_trackers.py` (see
`reports/ball_tracker.md`):

```bash
python scripts/try_swing_hits.py \
    --labels evaluation/labels/IMG_1089.json evaluation/labels/IMG_1086.json
```
