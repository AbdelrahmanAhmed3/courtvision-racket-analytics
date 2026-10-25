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

### Better candidates and inputs

Two problems in the first version, and their fixes:

- **Candidates after a hit were placed late.** TrackNet often finds the ball only a
  few frames after it leaves the racket, with no tracklet just before. Such a
  tracklet start is now followed back along its path to the frame where it passes
  closest to a player (and a tracklet end with nothing after it, forward). A gap
  between two tracklets is placed where their two paths meet instead of in the
  middle. The candidates now cover **103 of 138 impacts (75 %, from 66 %)**, 48 of
  67 bounces and 36 of 47 wall rebounds.
- **After a hidden gap the inputs were empty.** Speed, vertical movement and the
  turn angle came from positions within 4 frames of the candidate, so after a gap
  of 7 frames they were missing. They now come from the last and first 4
  positions within 10 frames.

| Method | Impact P / R | Impact F1 | Bounce P / R | Wall rebound P / R |
| --- | --- | --- | --- | --- |
| Rule: tracklet ends near a player | 0.25 / 0.44 | 0.32 | – | – |
| **Logistic regression** | **0.44 / 0.38** | **0.41** | 0.30 / 0.33 | 0.24 / **0.51** |
| Boosted trees | 0.42 / 0.36 | 0.39 | **0.62** / 0.22 | 0.22 / 0.17 |

The classifiers now find impacts better than the rule. The turn angle became the
most useful input, followed by the vertical movement before and the height in the
player's box. Boosted trees still answer "nothing" for 40 of the 110 impact
candidates.

## Shots and hitters (2026-10-25)

`analytics/ball_events.py`, part of #8. The classifier ships as a JSON model applied
with numpy (`detect_ball_events`), and `shots_from_events` turns events into
**shots**: each impact is paired with the next impact within 3 s (a deep lob takes
2–2.5 s), which gives the receiver; the first bounce in between is the shot's bounce;
the last shot of a point has no receiver. Court-plane distance and speed run between
the ball's floor projections at the impact and the receive (or the bounce): the
ball's height is not modelled, so they are an **approximation**.

**Hitter accuracy**: the hitter is the player nearest the ball at the impact, with
IDs from the four-ID tracker. Over impacts found by the shipped model (logistic
regression, cross-validated by rally), matched to a labelled impact within ±2 frames:

| Hitter rule | Right |
| --- | --- |
| **Nearest player** | **34 / 52 = 0.65** |
| Nearest player on the half of the court the ball is in | 30 / 52 = 0.58 |
| Nearest player on the other team from the previous hitter | 23 / 52 = 0.44 |

Of the 18 wrong hitters with the nearest-player rule, 10 are an opponent, 6 have no
tracked player near the ball, and 2 are the partner. Both rules that looked like
fixes made it worse:

- **The ball's half** is unreliable for the same reason "crosses the net" barely
  helps the classifier: a ball in the air projects to the wrong place on the floor.
- **"A team never hits twice"** needs reliable events. More than half of the impacts
  found are false (precision 0.44), and each false one flips the expected team for
  the next real impact, so errors cascade.

More precise impacts come first; then sequence rules can help the hitter.

## Rallies (2026-10-25)

`analytics/rallies.py`, #9. A rally starts at its first impact (the serve, when it is
seen) and ends at its last ball event; a silence longer than 5 s ends it. In the
labels, consecutive events inside a rally are at most 2.5 s apart (95 % under
1.7 s), the rally starts on its first impact in 9 of 10 rallies (the tenth begins at
a camera cut), and ends 0–0.8 s after its last event. 5 s doubles the longest
labelled gap, because the classifier misses events, and stays short of the pause
between points. Events before the first impact, such as the server bouncing the ball,
are left out of the rally.

Scored with the shipped model's events, cross-validated by rally:

| Measure | Result |
| --- | --- |
| Labelled rallies found | 10 / 10, none extra |
| Start error, median (mean signed) | 0.20 s (+0.86 s) |
| End error, median (mean signed) | 0.14 s (−0.56 s) |

- Three rallies start 2.3–2.7 s late, where the serve and the return were both
  missed; two end 2.1–2.4 s early, one of them the out-of-court point, where the
  ball leaves the view.
- A 3, 5 or 8 s silence gives the same result: no rally is split. Each labelled
  segment holds one point, so these labels cannot test two points inside one
  continuous camera shot; the 5 s limit is untested there.
- On the 15-second Qatar clip the pipeline finds one rally of 13.9 s with 12 shots,
  starting at the serve; the false wall rebounds from the server's bounces before it
  fall outside the rally.
