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

## Boosted trees: better on the labelled rallies, not on Qatar (2026-11-15)

#75, part of #52. On the labelled rallies, boosted trees found more impacts than
logistic regression, and more of the ones they reported were real. On the Qatar clip,
the one other event we checked, they reported far too many. Logistic regression keeps
shipping.

### On the labelled rallies

Each rally scored by a model that never saw it, ±2 frames:

| Split | Model | Impact P / R | Bounce P / R | Hitter right |
| --- | --- | --- | --- | --- |
| Halves | Logistic regression | 0.49 / 0.50 | 0.34 / 0.51 | 61 / 69 = 88 % |
| Halves | Boosted trees | **0.62 / 0.63** | **0.56** / 0.37 | 77 / 87 = 89 % |
| One rally at a time | Logistic regression | 0.49 / 0.51 | 0.32 / 0.52 | – |
| One rally at a time | Boosted trees | **0.65 / 0.62** | **0.64** / 0.43 | 75 / 85 = 88 % |

- **Every split agreed.** The trees also won on six more splits tried by hand (first
  five rallies against the last five, and five random halvings), with impact F1 0.56 to
  0.64 against 0.41 to 0.53.
- **Rallies ended early.** With the trees they ended within 0.50 s instead of 0.12 s
  (median), and early on average (−1.1 s). A rally ends at its last event, often the
  ball's last bounce, and the trees missed more of those (recall 0.37).

### On the Qatar Major clip

The author counted 12 shots in its one rally:

| | Logistic regression | Boosted trees |
| --- | --- | --- |
| Shots (12 counted by hand) | 12 | 17 |
| Same team twice in a row | 4 of 11 pairs | 7 of 16 pairs |
| Rally | 14.1 s, from frame 21 | 14.6 s, from frame 6 |

Looking at the frames where the two models disagree:

- **False hits from the trees.** They call the server's bounce before the serve a hit
  (frame 6, which also starts the rally too early). They call another while the ball
  is over the net (frame 69).
- **Mistakes both models make.** Both count the serve twice, three frames apart. The
  second time the ball has passed the server's partner at the net, so the hitter, the
  player nearest the ball, is the partner. Both also count one volley twice.

So logistic regression's 12 aren't 12 right hits either: it reports some false ones
and misses about as many real ones.

### Why the labelled score didn't carry over

- **One video.** All 10 labelled rallies come from one video. Halves, or one rally at
  a time, test the model on rallies it never saw, not on a video it never saw. The
  trees, 450 small trees against one weight per input for each event, have more room
  to learn details of that video that don't hold elsewhere. That is likely, not
  measured.
- **Before the serve.** Training only uses candidates inside the labelled rallies, but
  the pipeline asks about every moment of the clip. Neither model has seen a bounce
  before a serve.

### Kept from this work

- **`--split rallies`** scores one rally at a time.
- **CI installs the train extra,** so the test that the NumPy model answers exactly as
  scikit-learn now runs on every pull request. It was skipped before.
- **The trees' export to JSON and NumPy** gives exactly scikit-learn's answers, takes
  178 KB, and runs 1,000 candidates in 17 ms. It is in commit 3c8f8e4 (PR #82), to
  bring back if a second labelled clip shows the trees winning there too.

### Reproducing

```bash
L=evaluation/labels/BEST-OF-AGUSTIN-TAPIA-RED-BULL-OUT-THE-COURT.json
python scripts/train_event_classifier.py --labels $L --size wasb
python scripts/train_event_classifier.py --labels $L --size wasb --split rallies
```

### Next

- **Label the Qatar clip's 12 hits** with `scripts/label_clip.py`: a test from another
  event, and more to train on.
- **Train on candidates outside the rallies too,** before the serve and after the
  point, as "none".
- **#52: two hits a few frames apart are one hit.** Nobody hits twice in a tenth of a
  second, so keeping only the first would remove the doubled serve and volley.

## One hit, found twice, counts once (2026-11-16)

#83, part of #52. The event model often reports one hit twice, a few frames apart:
on the Qatar clip it counted the serve twice and gave the copy to the server's
partner. Real hits are never that close. On the labelled rallies they are at least
0.32 s apart, and the ball cannot go from one racket to another faster. So impacts
within 0.2 s of the one before now count as one, in the pipeline and in the training
script's scores. Bounces are left alone: merging them lifted their F1 only from 0.40
to 0.41 and lost some real ones.

The Qatar clip is now hand-labelled too (12 impacts, `evaluation/labels/`), so it
tests the model on another event, scored like the labelled rallies.

### Which copy to keep

Impact precision / recall at ±2 frames. On the Tapia rallies each rally is scored by
a model that never saw it. On Qatar, the shipped model, trained on all ten Tapia
rallies, is scored on its rally:

| Keep | Tapia, halves | Tapia, one rally at a time | Qatar |
| --- | --- | --- | --- |
| Both (before) | 0.49 / 0.50 | 0.49 / 0.51 | 0.50 / 0.50 |
| The first | 0.50 / 0.46 | 0.50 / 0.46 | 0.60 / 0.50 |
| The last | 0.54 / 0.49 | 0.56 / 0.51 | 0.40 / 0.33 |
| **The one the model is surest of** | **0.54 / 0.49** | **0.54 / 0.49** | **0.60 / 0.50** |

The first and the last each win on one video and lose on the other. On Tapia the
extra copy tends to come a few frames before the hit, and on Qatar a few frames
after it. Keeping the
copy the model is surest of is close to the best on both, so that is what ships.
When it is equally sure of both, the first is kept.

Hitters stay as good: 88 % of the found impacts on Tapia (59 of 67), and all 6 on
Qatar, where the serve now goes to the server again. Rallies barely move: one start
on Tapia moves 0.12 s later. On the Qatar clip the pipeline now reports 10 shots
instead of 12, and the same team hitting twice drops from 4 of 11 pairs to 2 of 8.

### The trees were better on Qatar after all

The hand labels show that matching the count of 12 was not the same as being right:

| Qatar, scored against the labels | Real hits found | False hits | Precision / recall | Hitter |
| --- | --- | --- | --- | --- |
| Logistic regression (ships) | 6 of 12 | 6 | 0.50 / 0.50 | 6 of 6 |
| Logistic regression, close hits merged | 6 of 12 | 4 | 0.60 / 0.50 | 6 of 6 |
| Boosted trees, close hits merged | 10 of 12 | 5 | 0.67 / 0.83 | 10 of 10 |

Logistic regression's 12 shots included 6 false ones: the copies of the serve and
of a volley, a wall rebound and a bounce called hits, and two real hits placed 3 and
4 frames off. It missed 6 real ones. The trees found 19 hits on the rally, and the
merge removed 4 of them as copies. With the merge, the trees also lead
on the Tapia rallies: 0.67 / 0.61 (halves) and 0.72 / 0.61 (one rally at a time).

These scores only count the labelled rally, from the serve on. The trees' other
mistake on Qatar, calling the server's bounce before the serve a hit (frame 6), is
not scored, and it still starts the rally too early. Switching to the trees needs
that fixed or measured first.

### Reproducing

```bash
T=evaluation/labels/BEST-OF-AGUSTIN-TAPIA-RED-BULL-OUT-THE-COURT.json
Q=evaluation/labels/youtube_gFl3ADnFRtc_15s.json
python scripts/predict_shot_events.py --labels $Q --video data/raw/youtube_gFl3ADnFRtc_15s.mp4
python scripts/compare_ball_trackers.py --labels $Q --video data/raw/youtube_gFl3ADnFRtc_15s.mp4
python scripts/train_event_classifier.py --labels $T --size wasb --test-labels $Q
python scripts/train_event_classifier.py --labels $T --size wasb --split rallies
```

### Next

- **Bring the boosted trees back,** with the merge, and measure the rally start on
  Qatar: the bounce before the serve must not become a hit.
- **Train on candidates outside the rallies too,** before the serve and after the
  point, as "none". That is the likely fix for the pre-serve hit.
