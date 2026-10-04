# CourtVision

CourtVision turns a racket-sport match video into player movement and ball events on a real-world court map. Padel is the primary sport; tennis is supported for calibration and projection.

## Editions and footage

**Edition**:
A packaging of CourtVision for one kind of footage, sharing the same core. There are two: the Showcase edition and the Personal edition.
_Avoid_: version (reserved for releases such as v4.0), app

**Showcase edition**:
The edition for high-angle or broadcast-style footage where all four players and the whole court are visible.

**Personal edition**:
The edition for a fixed, low camera behind one back glass, where only the near-side players are reliably visible.

**Clip**:
A continuous piece of match video with no cuts, the unit CourtVision analyses in one run.
_Avoid_: video (when meaning the analysed unit), footage

**Source video**:
A match recording as filmed or downloaded, which may contain camera cuts, replays and close-ups.
_Avoid_: clip (when it has cuts)

**Segment**:
A stretch of a source video between two camera cuts. A segment that shows one rally from the main camera is a clip.
_Avoid_: shot (a Shot is a strike of the ball), scene

## Court and calibration

**Court type**:
The sport's court layout (padel or tennis) that fixes the court dimensions and landmark set.
_Avoid_: sport mode

**Landmark**:
A named point on the court floor with known real-world coordinates, such as a corner or where the service line meets a side line.
_Avoid_: keypoint (reserved for model outputs), corner

**Calibration**:
The mapping from image pixels to court-floor coordinates for a clip, estimated from landmarks and accepted only if it passes validation.
_Avoid_: homography (the mathematical object, not the domain concept), registration

**Court map**:
The top-down view of the court on which players and the ball are drawn in metres.
_Avoid_: minimap, radar, template

**Near side** / **Far side**:
The half of the court closer to / further from the camera.

## Players

**Player**:
One of the (up to four) people playing the point, tracked with a stable identity for the whole clip.
_Avoid_: person, detection, track

**Team**:
The two players on the same side of the net.
_Avoid_: pair, side

## Ball events

**Shot**:
One strike of the ball by a player.
_Avoid_: hit, stroke, contact

**Impact**:
The moment within a shot when the racket meets the ball.
_Avoid_: contact frame

**Hitter** / **Receiver**:
The player who plays a shot / the player who plays the next shot.

**Bounce**:
The ball touching the court floor.
_Avoid_: rebound (reserved for Wall rebound)

**Wall rebound**:
The ball touching a glass or mesh wall (padel only).
_Avoid_: bounce

**Rally**:
The sequence of shots from a serve until the ball goes dead.
_Avoid_: point (the scoring unit, which also includes the outcome)

**Point**:
A rally together with its outcome: which team won it.
_Avoid_: rally

**Shot type**:
The kind of shot by technique, such as forehand, backhand, volley, smash, bandeja, or víbora.
_Avoid_: stroke, shot class

## Outputs

**Movement stats**:
Per-player measures of how they moved during a clip: distance covered, speed, positions over time, and time at the net versus the back of the court.

**Match report**:
A short plain-language summary of a clip built from movement stats and ball events, such as who played the most shots or covered the most distance.
_Avoid_: coaching report (reserved for the LLM-written report planned for the Personal edition)

**Coaching report**:
Advice written by an LLM from the match report and underlying stats, where every claim cites the moments in the clip it is based on.
