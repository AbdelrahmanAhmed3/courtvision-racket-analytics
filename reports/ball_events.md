# Ball Events

## Baseline: today's shot code (2026-10-07)

`analytics/shots.py`, unchanged, on the 10 labelled rallies of the Agustin Tapia
compilation (163 s; 138 impacts, 67 bounces, 47 wall rebounds; #6). Each segment runs
the pipeline's steps: RF-DETR Nano players on the hand-marked court, the IoU tracker,
the TrackNet ball, then `detect_shots`. Produced by `scripts/predict_shot_events.py`
and scored by `scripts/score_events.py`. This is the number #8 must beat.

| Event | Labelled | Predicted | Precision (±2 frames) | Recall (±2 frames) | Recall (±5 frames) |
| --- | --- | --- | --- | --- | --- |
| Impact | 138 | 74 | 0.14 | **0.07** | 0.17 |
| Bounce | 67 | 8 | 0.12 | **0.01** | 0.06 |
| Wall rebound | 47 | 0 | – | 0.00 | 0.00 |

40 shots were reported. ±2 frames is 0.08 s at 25 fps.

### Where the impacts are lost

Each labelled impact, followed through the steps (±2 frames):

| Step | Impacts lost | Share |
| --- | --- | --- |
| TrackNet does not see the ball | 20 | 14 % |
| The ball is seen, but not near a tracked player | 16 | 12 % |
| The ball is near a player, but the contact is timed elsewhere | 50 | 36 % |
| A contact at the right time, but the shot filters drop it | 42 | 30 % |
| **Reported** | **10** | **7 %** |

- **Timing is the biggest loss.** A contact is the first frame the ball comes near
  a player, not the hit. 134 of the 138 impacts have a contact within ±15 frames,
  but only 52 within ±2. An impact should be where the ball's path turns near the
  player.
- **Contacts are also far too many.** 382 contacts for 138 impacts: the ball passes
  close to players who do not hit it.
- **The filters then drop most of the rest.** A shot needs a flight of at least
  5 m that crosses the net, with both ends inside the court; the last shot of a
  rally has no receiver and is never reported. Out-of-court shots (#38) cannot pass.
- **TrackNet sees the ball on 66 % of rally frames**, and within ±2 frames of 61 of
  the 67 bounces. Bounces fail not because the ball is unseen, but because bounce
  detection only runs between reported shots and keeps only the first direction
  change of 45° or more, which also counts glass rebounds.

### Reproducing

The players and ball are cached in `outputs/predictions/` after the first run (about
18 minutes on an Apple M4); after that, changing the shot code and scoring again takes
under a second. The step counts above came from the cached tracks and the shot code's
contact step (`_find_player_contact_events`).

## Impacts where ball tracklets start or end (2026-10-19)

`analytics/ball_track.py`, part of #8. The TrackNet positions are split into
**tracklets**, runs of positions one ball could really travel: a tracklet ends
after a gap of more than 2 frames, or where the next position is more than 125 px
per frame away. At an impact the ball is often hidden by the player for a frame or
two, so a tracklet start or end next to a player is an impact candidate.
Candidates at most 3 frames apart are one impact. Same cached players and ball as
the baseline; `scripts/predict_shot_events.py --method tracklets`.

| Method | Predicted | Precision (±2) | Recall (±2) | Precision (±5) | Recall (±5) |
| --- | --- | --- | --- | --- | --- |
| Today's shot code | 74 | 0.14 | 0.07 | 0.31 | 0.17 |
| Tracklet ends near a player | 247 | 0.25 | **0.44** | 0.38 | **0.70** |

The settings come from physics and were not tuned on the labels: 125 px is a hard
smash near the camera plus margin. Tuning them on half the rallies did not help on
the other half: the best settings on rallies 1, 3, 5, 7, 9 (F1 0.34) scored F1 0.28
on the others, below the untuned settings (0.30). With 10 rallies, tuning mostly
fits noise.

### Where the wrong impacts come from

Of the 271 predictions in the labelled segments (24 fall outside the rallies and are
not scored above), 210 match no impact within ±2 frames:

| Cause | Wrong |
| --- | --- |
| 2–8 frames from an impact: the tracklet ends before and starts after a hit more than 3 frames apart, or the timing is off | 79 |
| At or near a bounce (often just before the hit, at the player's feet) | 47 |
| At or near a wall rebound | 29 |
| Nothing labelled within 8 frames: wrong positions, reflections, the ball passing a player | 55 |

Each cause is a next step for #8: one impact per hit, telling a hit from a bounce
next to a player, and dropping reflections and single wrong positions.

## A small classifier on the candidates (2026-10-19)

`scripts/train_event_classifier.py`, part of #8. Instead of hand-written rules, a
small model decides what happened at each **candidate**: an impact, a bounce, a
wall rebound or nothing. Candidates are every tracklet end, the middle of every gap
of up to 10 frames between two tracklets, and every turn of 30° or more inside a
tracklet (`analytics/ball_track.event_candidates`). They cover 91 of 138 impacts,
47 of 67 bounces and 34 of 47 wall rebounds within ±2 frames, which caps recall at
about 0.66.

Each candidate gets 14 numbers from the ball track, the player boxes and the court:
speed and vertical movement before and after, the angle it turns, frames hidden,
distance to the nearest player and where in their box, position along the court,
and whether the ball reaches the other half soon after. The label is the nearest
labelled event within 2 frames, or none. A model trained on rallies 1, 3, 5, 7, 9
predicts 2, 4, 6, 8, 10 and the other way round, so no rally is scored by a model
that saw it.

| Method | Impact P / R | Bounce P / R | Wall rebound P / R |
| --- | --- | --- | --- |
| Today's shot code | 0.14 / 0.07 | 0.12 / 0.01 | – |
| Rule: tracklet ends near a player | 0.25 / **0.44** | – | – |
| Logistic regression | 0.33 / 0.28 | 0.28 / **0.34** | 0.19 / **0.47** |
| Boosted trees | **0.47** / 0.27 | **0.38** / 0.22 | 0.14 / 0.13 |

All at ±2 frames.

- **More precise, but it finds fewer impacts.** Boosted trees are right on 47 % of
  their impacts (the rule: 25 %) but find 27 % (the rule: 44 %). For impacts alone,
  neither is clearly better (F1 0.34 vs 0.32).
- **The first real bounce and wall-rebound detection.** Today's shot code finds 1 %
  of bounces and no wall rebounds.
- **Its main mistake is answering "nothing".** Of the 100 candidates that are
  impacts, boosted trees call 45 nothing and only 6 a bounce: with 5 rallies to
  learn from, the most common answer is the safest. Hits and bounces are rarely
  confused.
- **What it relies on:** the ball's vertical movement before and after, the
  distance to the nearest player and where in their box the ball is, then the
  position along the court. Whether the ball crosses the net soon after barely
  helps, probably because a ball in the air projects to the wrong place on the
  floor.
- **More labels are the next lever.** Each half trains on about 250 candidates, 50 of
  them impacts. The candidates' 0.66 ceiling is the other.
