"""A short match report for a clip, as JSON and Markdown, from the clip's stats.

Built from templates, not a language model: who played the most shots, covered
the most distance and reached the top speed, and how many rallies there were
and how long. The JSON is also the structured input for the coaching report
planned for the Personal edition. Accuracy notes come from the measured results
in reports/, so a reader knows how far to trust each number.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from courtvision.analytics.ball_events import Shot
from courtvision.analytics.movement import MovementStats
from courtvision.analytics.rallies import Rally

REPORT_SCHEMA_VERSION = 1
ACCURACY_NOTES = (
    "Shots are counted from detected impacts, and about 4 in 10 impacts are found, "
    "so shot counts are low (reports/ball_events.md).",
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
) -> dict:
    """The report as plain data (JSON-ready)."""
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
        }
        for rally in rallies
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "video": video,
        "seconds_analysed": round(frames / fps, 1),
        "players": players,
        "rallies": rally_list,
        "highlights": _highlights(players, rally_list, shots),
        "accuracy_notes": list(ACCURACY_NOTES),
    }


def _highlights(players: list[dict], rallies: list[dict], shots: Sequence[Shot]):
    def leader(key: str) -> dict | None:
        best = max(players, key=lambda player: player[key], default=None)
        if best is None or not best[key]:
            return None
        return {"track_id": best["track_id"], "team": best["team"], key: best[key]}

    fastest = max(
        (shot for shot in shots if shot.court_speed_kmh is not None),
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


def report_markdown(report: dict, heatmap_image: str | None = None) -> str:
    """The report as Markdown, for people."""
    highlights = report["highlights"]
    lines = [f"# Match report: {report['video']}", ""]
    rallies = highlights["rally_count"]
    lines.append(
        f"{report['seconds_analysed']} s analysed, {len(report['players'])} players, "
        f"{rallies} {'rally' if rallies == 1 else 'rallies'}."
    )
    lines += ["", "## Highlights", ""]
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
            lines.append(
                f"- **{what}:** {_name(best)}, {value[unit].format(best[unit])}."
            )
    if highlights["rally_count"] == 1:
        only = report["rallies"][0]
        lines.append(f"- **Rally:** {only['seconds']} s long, {only['shots']} shots.")
    elif highlights["rally_count"]:
        lines.append(
            f"- **Rallies:** {highlights['rally_count']}, "
            f"{highlights['average_rally_seconds']} s on average, longest "
            f"{highlights['longest_rally_seconds']} s, "
            f"{highlights['average_shots_per_rally']} shots each on average."
        )
    fastest = highlights["fastest_shot"]
    if fastest:
        route = f"Player {fastest['hitter_track_id']}"
        if fastest["receiver_track_id"]:
            route += f" to Player {fastest['receiver_track_id']}"
        lines.append(
            f"- **Fastest shot (approximate):** {route}, "
            f"{fastest['approx_court_speed_kmh']} km/h along the court."
        )
    lines += [
        "",
        "## Players",
        "",
        "| Player | Team | Shots | Distance | Average speed | Top speed "
        "| At the net | Middle | At the back |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for player in report["players"]:
        lines.append(
            f"| {player['track_id']} | {player['team'] or '–'} | {player['shots']} "
            f"| {player['distance_m']} m | {player['average_speed_kmh']} km/h "
            f"| {player['top_speed_kmh']} km/h | {player['seconds_at_net']} s "
            f"| {player['seconds_in_middle']} s | {player['seconds_at_back']} s |"
        )
    if heatmap_image:
        lines += ["", "## Where each player stood", "", f"![Heatmaps]({heatmap_image})"]
    if report["rallies"]:
        lines += ["", "## Rallies", "", "| Rally | Start frame | Length | Shots |"]
        lines.append("| --- | --- | --- | --- |")
        for number, rally in enumerate(report["rallies"], start=1):
            lines.append(
                f"| {number} | {rally['start_frame']} | {rally['seconds']} s "
                f"| {rally['shots']} |"
            )
    lines += ["", "## How far to trust these numbers", ""]
    lines += [f"- {note}" for note in report["accuracy_notes"]]
    return "\n".join(lines) + "\n"


def _name(player: dict) -> str:
    team = f" ({player['team']} team)" if player.get("team") else ""
    return f"Player {player['track_id']}{team}"
