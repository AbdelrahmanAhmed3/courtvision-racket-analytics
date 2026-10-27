import json

import numpy as np

from courtvision.analytics.ball_events import Shot
from courtvision.analytics.movement import MovementStats
from courtvision.analytics.rallies import Rally
from courtvision.analytics.report import build_report, report_markdown
from courtvision.visualization.heatmaps import draw_heatmaps

FPS = 25.0


def stats(distance, top_speed, at_net=0.0) -> MovementStats:
    heatmap = np.zeros((20, 10))
    heatmap[15, 5] = 10.0
    return MovementStats(
        10.0, distance, distance / 10 * 3.6, top_speed, at_net, 0.0, 0.0, heatmap
    )


def shot(frame, hitter, receiver=None, speed=None) -> Shot:
    return Shot(frame, hitter, receiver, None, None, None, None, speed)


def clip_report() -> dict:
    movement = {
        1: stats(20.0, 12.0, at_net=6.0),
        2: stats(25.0, 9.0),
        3: stats(18.0, 14.5),
    }
    shots = [shot(10, 1, 3, 40.0), shot(40, 3, 1, 65.0), shot(70, 1), shot(400, 2)]
    rallies = [Rally(10, 110, tuple(shots[:3])), Rally(400, 450, (shots[3],))]
    teams = {1: "near", 2: "near", 3: "far"}
    return build_report("clip.mp4", FPS, 500, teams, movement, shots, rallies)


def test_the_highlights_name_the_leaders() -> None:
    highlights = clip_report()["highlights"]

    assert highlights["most_shots"] == {"track_id": 1, "team": "near", "shots": 2}
    assert highlights["most_distance"]["track_id"] == 2
    assert highlights["top_speed"] == {
        "track_id": 3,
        "team": "far",
        "top_speed_kmh": 14.5,
    }
    assert highlights["rally_count"] == 2
    assert highlights["average_rally_seconds"] == 3.0  # 4 s and 2 s
    assert highlights["longest_rally_seconds"] == 4.0
    assert highlights["average_shots_per_rally"] == 2.0
    assert highlights["fastest_shot"] == {
        "hitter_track_id": 3,
        "receiver_track_id": 1,
        "approx_court_speed_kmh": 65.0,
    }


def test_the_report_is_json_and_markdown() -> None:
    report = clip_report()

    assert json.loads(json.dumps(report)) == report
    text = report_markdown(report, heatmap_image="heatmaps.png")
    assert "- **Most shots:** Player 1 (near team), 2 shots." in text
    assert "- **Top speed:** Player 3 (far team), 14.5 km/h." in text
    assert "approximate" in text
    assert "![Heatmaps](heatmaps.png)" in text
    assert "| 1 | near | 2 | 20.0 m |" in text
    assert "- **Rallies:** 2, 3.0 s on average" in text


def test_one_rally_is_described_on_its_own() -> None:
    rally = Rally(10, 110, (shot(10, 1), shot(40, 2)))
    report = build_report("clip.mp4", FPS, 250, {}, {1: stats(5.0, 7.0)}, (), [rally])

    text = report_markdown(report)
    assert "1 rally." in text
    assert "- **Rally:** 4.0 s long, 2 shots." in text


def test_a_clip_without_ball_tracking_still_gets_a_report() -> None:
    report = build_report("clip.mp4", FPS, 250, {}, {1: stats(5.0, 7.0)})

    assert report["highlights"]["most_shots"] is None
    assert report["highlights"]["rally_count"] == 0
    text = report_markdown(report)
    assert "Most shots" not in text and "## Rallies" not in text
    assert "- **Most distance:** Player 1, 5.0 m." in text


def test_heatmaps_draw_one_court_per_player() -> None:
    one = stats(1.0, 1.0).heatmap_seconds

    image = draw_heatmaps({1: one, 2: one}, {1: "near", 2: "far"})

    court_width = 10 * 16
    assert image.shape[1] == 2 * court_width + 12
    assert image.dtype == np.uint8
