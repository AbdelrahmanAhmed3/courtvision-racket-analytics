# Player Identity

## Baseline: today's trackers (2026-10-23)

Identities scored against the 651 labelled player boxes of the 10 labelled rallies
in the Agustin Tapia compilation (#6): one frame per second, each player numbered
P1–P4 for the whole rally. In each rally every labelled player is paired with the
track ID it shares the most labelled frames with, one to one; **IDF1** is the share
of boxes on their player's paired ID (1.0 is perfect), and a **switch** is a
player's ID changing between labelled frames. Both trackers are replayed on the same
cached boxes (RF-DETR Nano on the hand-marked court, at most four per frame) with
`scripts/score_trackers.py`.

| Tracker | IDF1 | Switches | IDs given to the 4 players, per rally |
| --- | --- | --- | --- |
| Greedy IoU (`SimpleIouTracker`, the pipeline's) | **0.96** | **8** | 5 4 4 4 5 4 5 4 5 4 |
| Kalman + Hungarian (`KalmanIouTracker`) | 0.65 | 97 | 6 8 13 11 7 7 7 6 11 6 |

### The greedy tracker's 8 switches

| Kind | Switches | What happens |
| --- | --- | --- |
| Swap | 5 | Two players cross and their IDs are exchanged; both IDs stay alive (rally 1: three, during the out-of-court point; rally 23: P3 and P4 crossing at the net). |
| Split | 3 | A player's ID ends and a new one starts on the very next frame: the box changed too much in one frame to match, and a player just lost cannot yet be matched by distance. |

The Kalman tracker's prediction keeps moving a hidden player at their last velocity,
past the distance it may reconnect from (noted on #5), so it splits players far more
often.
