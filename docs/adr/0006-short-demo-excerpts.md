# The README and launch posts may show a short, credited excerpt of the pipeline's output on broadcast footage

ADR 0002 says clips downloaded from YouTube are never committed or re-hosted. The README's demo has to show CourtVision working on real padel video, and we have no footage of our own yet. A court map alone does not show that the players and the ball were found in a real broadcast. So the README, and the posts announcing a release, may show **one short excerpt of the pipeline's output**: at most about 10 seconds, at reduced size, with the tracking drawn over the broadcast and the court map beside it. The source video and its channel are credited and linked next to it. In the repository the excerpt is hosted as a GitHub attachment, not committed, so it can be taken down on request without rewriting the repository's history.

## Considered Options

- **Court map only**: rejected for now. It stays within ADR 0002, but it shows nothing a visitor can check against the video.
- **Our own match footage**: preferred once it exists. We would own it, so none of this applies. It needs a recording and a calibration, and the pipeline is untested from a camera behind the glass at player height.
- **Commit the excerpt to `docs/assets/`**: rejected. A takedown would leave it in the git history unless public history were rewritten.

## Consequences

- Raw clips are still never committed or re-hosted; datasets and labels never include frames.
- The excerpt is removed promptly if the rights holder asks, from the README and from the posts, and the README falls back to the court map.
- Replace the excerpt with our own footage when we have it.
