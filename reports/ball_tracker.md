# Ball Tracker

## WASB against TrackNet on the labelled rallies (2026-11-08)

Part of #61. [WASB](https://github.com/nttcom/WASB-SBDT), by NTT, works like
TrackNet: it reads three frames in a row and paints a heat map of where the ball
is. It paints one for each of the three frames, with a small HRNet (1.5 million
weights). Its code and the weights linked from it are MIT-licensed; TrackNet has no
licence. It ran as downloaded, with the weights trained on tennis and no padel
training.

Same 10 labelled rallies of the Agustin Tapia compilation (138 impacts, 67 bounces,
47 wall rebounds), same players and same event code: only the ball changes.
Produced by `scripts/compare_ball_trackers.py`, then
`scripts/train_event_classifier.py --size wasb` for the events.

### The ball

| | TrackNet | WASB (tennis) |
| --- | --- | --- |
| Ball found during rallies | 66 % | 70 % |
| Bounces: ball within 15 px of the labelled position | 34 of 67 | 44 of 67 |
| Wall rebounds: ball within 15 px | 35 of 47 | 35 of 47 |
| Median distance where found, bounces / wall rebounds | 2.7 / 3.2 px | 3.9 / 4.6 px |
| Events with a candidate within ±2 frames | 188 of 252 (**0.75**) | 201 of 252 (**0.80**) |
| ... impacts / bounces / wall rebounds | 103 / 48 / 37 | 114 / 53 / 34 |

WASB runs at about 40 frames a second on an Apple M4.

### Events, scored as before

±2 frames; each rally is scored by a model trained on the other half of the rallies.

| Method | Ball | Impact P / R | Bounce P / R | Wall rebound P / R |
| --- | --- | --- | --- | --- |
| Logistic regression (ships) | TrackNet | 0.44 / 0.38 | 0.30 / 0.33 | **0.24 / 0.51** |
| Logistic regression | WASB | 0.50 / 0.43 | 0.34 / **0.46** | 0.19 / 0.21 |
| Boosted trees | TrackNet | 0.42 / 0.36 | **0.62** / 0.22 | 0.22 / 0.17 |
| Boosted trees | WASB | **0.63 / 0.62** | 0.54 / 0.42 | 0.26 / 0.13 |

| | TrackNet | WASB |
| --- | --- | --- |
| Hitter of the impacts found (logistic regression) | 34 of 52 (0.65) | **51 of 59 (0.86)** |
| Rallies found | 10 of 10 | 10 of 10 |
| Rally start error, median | 0.20 s | **0.02 s** |

- **A better ball lifts every step after it.** With no change to the event code,
  more impacts are found and more of them are right, and the hitter is right 86 % of
  the time instead of 65 %.
- **Boosted trees gain the most:** 62 % of impacts found, 63 % of them real. The
  candidates are also cleaner: 151 of them are not an event, against 281 with
  TrackNet, so the classifier has less noise to sort out.
- **Wall rebounds get worse:** recall falls from 0.51 to 0.21 with the model that
  ships. The candidates still cover nearly as many (34 against 37), so the drop is
  in telling them apart. Tennis has no walls; padel training should show whether
  that is the reason.
- **Slightly less precise where it finds the ball:** 3.9 px from the labelled
  bounce against 2.7 px, about half a ball's width. WASB reads frames at 512 × 288,
  TrackNet at 640 × 360.
- **How far to trust this:** 10 rallies from one video, and the event features were
  designed on TrackNet's tracks, which if anything favours TrackNet. A few points
  either way are noise; the gains in impacts and hitters are not.

### Reproducing

TrackNet's ball comes from the cache of `scripts/predict_shot_events.py`. WASB's
weights download to `~/.cache/courtvision/` on first use.

## The pipeline uses WASB (2026-11-08)

Part of #61. The app and `scripts/run_full_pipeline.py` now track the ball with WASB,
on by default (`--no-ball` skips it). The network is copied into
`src/courtvision/detectors/wasb_hrnet.py` with its MIT notices; the tennis weights
(6 MB) download on first use and are checked against their SHA-256. On the first
rally, the copy gives the same ball as WASB's own file on all 280 frames.

The event model that ships (`analytics/ball_events_model.json`) is retrained on WASB's
candidates from all 10 rallies, so the scores above for logistic regression with
WASB are the ones the pipeline gets: impacts 0.50 / 0.43, the hitter right 86 % of
the time.

On the Qatar Major clip, with the same six court points:

| | TrackNet | WASB |
| --- | --- | --- |
| Shots | 12 | 11 |
| Same team twice in a row | 4 | 4 |
| Bounces / impacts / wall rebounds marked | 16 / 12 / 8 | 16 / 11 / **19** |

- **Wall rebounds are now over-reported** on this clip: 19 in a 12-second rally. On the
  labelled rallies only 1 in 5 reported wall rebounds is real (precision 0.19), so most
  of these are false. Shots don't use them, but a rally ends at its last ball event of
  any kind, so a false one can stretch a rally's end; the rally scores above include
  that.

### Next

- **Steady the ball between frames.** WASB's full method adds a tracking step that
  picks, on each frame, the spot that fits the ball's motion; the adapter takes the
  strongest spot. Its one-frame jumps looked like the source of the false wall
  rebounds; they turned out not to be (next section).
- **Fine-tune WASB on padel,** which needs padel frames with the ball labelled on every
  frame. The PadelTracker100 labels refer to clips that are not public, so the first
  padel labels will be our own.

## Wall rebounds switched off (2026-11-11)

#74. The event model no longer answers "wall rebound". Wall rebounds stay in the labels, but candidates take the label of the nearest impact or bounce only.

### Why

On the labelled rallies, with the logistic regression that shipped:

| | TrackNet | WASB |
| --- | --- | --- |
| Wall rebounds reported | 99 | 54 |
| ... really wall rebounds | 25 | 10 |
| ... really hits | 22 | 20 |
| ... really nothing | 45 | 21 |
| One-frame jumps of the ball during the rallies | 92 | 20 |

- **TrackNet wasn't better at walls.** It found more of them (51 % against 21 %) only
  because it answered "wall" almost twice as often. Either way, 3 or 4 in 5 of the
  wall rebounds reported were wrong.
- **Not WASB's jumps,** the guess above: WASB jumps a quarter as often as TrackNet, and
  only 4 of its 21 false wall rebounds are within 3 frames of a jump.
- **Not fixable with a confidence bar:** real and false wall rebounds get the same
  confidence (median 0.54 against 0.56), so precision stays near 0.18 at any bar.
- **The likely reasons.** From a high camera, a ball turning off the back glass and one
  bouncing on the floor next to it make the same shape on screen; what tells them
  apart is the ball's height, which isn't measured yet. And players often hit the ball
  just after it leaves the glass, so a wall rebound and a hit are a few frames apart.
  There are also only 47 labelled wall rebounds to learn from.

### Without wall rebounds

Each rally scored by a model trained on the other half, ±2 frames:

| Logistic regression (ships) | Impact P / R | Bounce P / R | Hitter right |
| --- | --- | --- | --- |
| With wall rebounds | 0.50 / 0.43 | 0.34 / 0.46 | 86 % |
| Without | 0.49 / **0.50** | 0.34 / **0.51** | **88 %** |

Rallies are unchanged: 10 of 10, start within 0.02 s and end within 0.12 s (median).
Boosted trees without wall rebounds: impacts 0.62 / 0.63, bounces 0.56 / 0.37 (#75).

On the Qatar Major clip, same six court points:

| | TrackNet | WASB, with walls | WASB, without |
| --- | --- | --- | --- |
| Shots | 12 | 11 | 12 |
| Same team twice in a row | 4 | 4 | 4 |
| Bounces / impacts / wall rebounds | 16 / 12 / 8 | 16 / 11 / 19 | 19 / 12 / – |

Some of the candidates that were called wall rebounds are now called bounces (16 → 19).
Bounce precision is 0.34, so most extra bounces are probably false too. A shot uses
only the first bounce after its hit: it is recorded as the shot's bounce, and ends the
speed measurement when no next hit is found.

Wall rebounds come back with the ball's height (#72, #73).
