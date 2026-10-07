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
