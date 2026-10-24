"""Which team each tracked player is on, from the half of the court they play in.

Partners share one half of the court for a whole rally: they swap sides within
it, never halves. So a player's team is the half where their feet are most
often. The "near" team plays the half closer to the camera, the "far" team the
other; positions use court template coordinates, where y runs from 0 at the
far baseline to 1 at the near one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

NEAR, FAR = "near", "far"


def assign_teams(positions: Iterable[tuple[int, float]]) -> dict[int, str]:
    """Team by track ID, from (track ID, court template y) feet positions.

    A player seen as often on each half is placed on the near team.
    """
    halves: dict[int, Counter[str]] = defaultdict(Counter)
    for track_id, court_y in positions:
        halves[track_id][NEAR if court_y >= 0.5 else FAR] += 1
    return {
        track_id: NEAR if counts[NEAR] >= counts[FAR] else FAR
        for track_id, counts in halves.items()
    }
