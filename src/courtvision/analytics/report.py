"""A short match report for a clip, as JSON and Markdown, from the clip's stats.

Built from templates, not a language model: who played the most shots, covered
the most distance and reached the top speed, and how many rallies there were
and how long. The JSON is also the structured input for the coaching report
planned for the Personal edition. Accuracy notes come from the measured results
in reports/, so a reader knows how far to trust each number, and each rally is
checked against the rule that the teams take turns. Every JSON key is described
in report.schema.json.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from courtvision.analytics.ball_events import Shot, has_credible_speed
from courtvision.analytics.movement import MovementStats
from courtvision.analytics.rallies import Rally

REPORT_SCHEMA_VERSION = 1
ACCURACY_NOTES = (
    "Shots are counted from detected impacts: about 4 in 10 real impacts are "
    "found, and fewer than half of the detected ones are real, so a count can be "
    "off either way (reports/ball_events.md).",
    "The hitter is the player nearest the ball at the impact, right about 2 times "
    "in 3.",
    "Shot speeds are measured on the court floor; the ball's height is not "
    "modelled, so they are approximate.",
    "Distances and speeds come from smoothed positions of each player's feet "
    "(reports/player_movement.md).",
)


def build_report(
    video: str,
    fps: float,
    frames: int,
    teams: Mapping[int, str],
    movement: Mapping[int, MovementStats],
    shots: Sequence[Shot] = (),
    rallies: Sequence[Rally] = (),
    ball_tracked: bool = True,
) -> dict:
    """The report as plain data (JSON-ready).

    ``ball_tracked=False`` marks a clip run without ball tracking, whose shots
    and rallies were not measured rather than zero.
    """
    shot_counts = Counter(s.hitter_track_id for s in shots if s.hitter_track_id)
    players = [
        {
            "track_id": track_id,
            "team": teams.get(track_id),
            "shots": shot_counts.get(track_id, 0),
            "distance_m": round(stats.distance_m, 1),
            "average_speed_kmh": round(stats.average_speed_kmh, 1),
            "top_speed_kmh": round(stats.top_speed_kmh, 1),
            "seconds_at_net": round(stats.seconds_at_net, 1),
            "seconds_in_middle": round(stats.seconds_in_middle, 1),
            "seconds_at_back": round(stats.seconds_at_back, 1),
        }
        for track_id, stats in sorted(movement.items())
    ]
    rally_list = [
        {
            "start_frame": rally.start_frame,
            "end_frame": rally.end_frame,
            "seconds": round((rally.end_frame - rally.start_frame) / fps, 1),
            "shots": len(rally.shots),
            "same_team_twice": _same_team_twice(rally.shots, teams),
        }
        for rally in rallies
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "video": video,
        "seconds_analyzed": round(frames / fps, 1),
        "ball_tracked": ball_tracked,
        "players": players,
        "rallies": rally_list,
        "highlights": _highlights(players, rally_list, shots),
        "accuracy_notes": list(ACCURACY_NOTES),
    }


def _same_team_twice(shots: Sequence[Shot], teams: Mapping[int, str]) -> int:
    """Places where one team hit twice in a row, which the rules do not allow.

    The ball crosses the net after every shot, so the teams take turns. Each
    repeat means a missed shot in between, a false one, or a wrong hitter. A
    check of the order, not the totals: near, near, far, far balances 2 to 2 but
    has two repeats. Shots without a known team are not judged.
    """
    sides = [
        teams.get(shot.hitter_track_id)
        for shot in sorted(shots, key=lambda shot: shot.frame)
    ]
    return sum(1 for a, b in zip(sides, sides[1:], strict=False) if a and a == b)


def _highlights(players: list[dict], rallies: list[dict], shots: Sequence[Shot]):
    def leader(key: str) -> dict | None:
        best = max(players, key=lambda player: player[key], default=None)
        if best is None or not best[key]:
            return None
        return {"track_id": best["track_id"], "team": best["team"], key: best[key]}

    teams = {player["track_id"]: player["team"] for player in players}
    fastest = max(
        (shot for shot in shots if has_credible_speed(shot, teams)),
        key=lambda shot: shot.court_speed_kmh,
        default=None,
    )
    seconds = [rally["seconds"] for rally in rallies]
    return {
        "most_shots": leader("shots"),
        "most_distance": leader("distance_m"),
        "top_speed": leader("top_speed_kmh"),
        "rally_count": len(rallies),
        "average_rally_seconds": round(sum(seconds) / len(seconds), 1)
        if seconds
        else None,
        "longest_rally_seconds": max(seconds, default=None),
        "average_shots_per_rally": (
            round(sum(rally["shots"] for rally in rallies) / len(rallies), 1)
            if rallies
            else None
        ),
        "same_team_twice": sum(rally["same_team_twice"] for rally in rallies),
        "fastest_shot": (
            {
                "hitter_track_id": fastest.hitter_track_id,
                "receiver_track_id": fastest.receiver_track_id,
                "approx_court_speed_kmh": round(fastest.court_speed_kmh, 1),
            }
            if fastest
            else None
        ),
    }


PLAYER_COLUMNS = (
    "Player",
    "Team",
    "Shots",
    "Distance",
    "Average speed",
    "Top speed",
    "At the net",
    "Middle",
    "At the back",
)
RALLY_COLUMNS = ("Rally", "Start frame", "Length", "Shots", "Same team twice")


def summary_line(report: dict) -> str:
    if not report.get("ball_tracked", True):
        return (
            f"{report['seconds_analyzed']} s analyzed, {len(report['players'])} "
            "players. The ball was not tracked, so there are no shots or rallies."
        )
    rallies = report["highlights"]["rally_count"]
    return (
        f"{report['seconds_analyzed']} s analyzed, {len(report['players'])} players, "
        f"{rallies} {'rally' if rallies == 1 else 'rallies'}."
    )


def highlight_items(report: dict) -> list[tuple[str, str]]:
    """The highlights as (what, sentence) pairs, shared by every report format."""
    highlights = report["highlights"]
    items = []
    for key, unit, what in (
        ("most_shots", "shots", "Most shots"),
        ("most_distance", "distance_m", "Most distance"),
        ("top_speed", "top_speed_kmh", "Top speed"),
    ):
        best = highlights[key]
        if best:
            value = {
                "shots": "{} shots",
                "distance_m": "{} m",
                "top_speed_kmh": "{} km/h",
            }
            items.append((what, f"{_name(best)}, {value[unit].format(best[unit])}."))
    if highlights["rally_count"] == 1:
        only = report["rallies"][0]
        items.append(("Rally", f"{only['seconds']} s long, {only['shots']} shots."))
    elif highlights["rally_count"]:
        items.append(
            (
                "Rallies",
                f"{highlights['rally_count']}, "
                f"{highlights['average_rally_seconds']} s on average, longest "
                f"{highlights['longest_rally_seconds']} s, "
                f"{highlights['average_shots_per_rally']} shots each on average.",
            )
        )
    fastest = highlights["fastest_shot"]
    if fastest:
        route = f"Player {fastest['hitter_track_id']}"
        if fastest["receiver_track_id"]:
            route += f" to Player {fastest['receiver_track_id']}"
        items.append(
            (
                "Fastest shot (approximate)",
                f"{route}, {fastest['approx_court_speed_kmh']} km/h along the court.",
            )
        )
    return items


def player_rows(report: dict) -> list[list[str]]:
    return [
        [
            str(player["track_id"]),
            player["team"] or "–",
            str(player["shots"]),
            f"{player['distance_m']} m",
            f"{player['average_speed_kmh']} km/h",
            f"{player['top_speed_kmh']} km/h",
            f"{player['seconds_at_net']} s",
            f"{player['seconds_in_middle']} s",
            f"{player['seconds_at_back']} s",
        ]
        for player in report["players"]
    ]


def rally_rows(report: dict) -> list[list[str]]:
    return [
        [
            str(number),
            str(rally["start_frame"]),
            f"{rally['seconds']} s",
            str(rally["shots"]),
            str(rally["same_team_twice"]),
        ]
        for number, rally in enumerate(report["rallies"], start=1)
    ]


def trust_notes(report: dict) -> list[str]:
    """What this clip's own checks found, then the measured accuracy notes."""
    notes = []
    repeats = report["highlights"]["same_team_twice"]
    if repeats:
        places = "1 place" if repeats == 1 else f"{repeats} places"
        notes.append(
            f"In {places}, the same team hit twice in a row, which the rules do not "
            "allow: at each, a shot was missed, a false one was counted, or the "
            "hitter is wrong."
        )
    return notes + list(report["accuracy_notes"])


def report_markdown(report: dict, heatmap_image: str | None = None) -> str:
    """The report as Markdown, for people."""
    lines = [f"# Match report: {report['video']}", "", summary_line(report)]
    lines += ["", "## Highlights", ""]
    lines += [f"- **{what}:** {text}" for what, text in highlight_items(report)]
    lines += ["", "## Players", "", *_table(PLAYER_COLUMNS, player_rows(report))]
    if heatmap_image:
        lines += ["", "## Where each player stood", "", f"![Heatmaps]({heatmap_image})"]
    if report["rallies"]:
        lines += ["", "## Rallies", "", *_table(RALLY_COLUMNS, rally_rows(report))]
    lines += ["", "## How far to trust these numbers", ""]
    lines += [f"- {note}" for note in trust_notes(report)]
    return "\n".join(lines) + "\n"


def _table(columns: Sequence[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _name(player: dict) -> str:
    team = f" ({player['team']} team)" if player.get("team") else ""
    return f"Player {player['track_id']}{team}"
