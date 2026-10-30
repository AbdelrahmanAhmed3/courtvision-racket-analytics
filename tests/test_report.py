import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from courtvision.analytics import report as report_module
from courtvision.analytics.ball_events import Shot
from courtvision.analytics.movement import MovementStats
from courtvision.analytics.rallies import Rally
from courtvision.analytics.report import (
    REPORT_SCHEMA_VERSION,
    build_report,
    report_markdown,
)
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
    report = build_report(
        "clip.mp4", FPS, 250, {}, {1: stats(5.0, 7.0)}, ball_tracked=False
    )

    assert report["highlights"]["most_shots"] is None
    assert report["highlights"]["rally_count"] == 0
    text = report_markdown(report)
    assert "Most shots" not in text and "## Rallies" not in text
    assert "The ball was not tracked, so there are no shots or rallies." in text
    assert "- **Most distance:** Player 1, 5.0 m." in text


def test_a_team_hitting_twice_in_a_row_is_counted() -> None:
    teams = {1: "near", 2: "far", 3: "near", 4: "far"}
    # near, near, far, far: the totals balance 2 to 2, the order does not.
    order = (shot(10, 1), shot(40, 3), shot(70, 2), shot(100, 4))
    unknown = (shot(10, 1), shot(40, None), shot(70, 3))  # not judged
    rallies = [Rally(10, 100, order), Rally(200, 300, unknown)]

    movement = {1: stats(5.0, 7.0)}
    report = build_report("clip.mp4", FPS, 500, teams, movement, (), rallies)

    assert [rally["same_team_twice"] for rally in report["rallies"]] == [2, 0]
    assert report["highlights"]["same_team_twice"] == 2
    text = report_markdown(report)
    assert "- In 2 places, the same team hit twice in a row" in text
    assert "| 1 | 10 | 3.6 s | 4 | 2 |" in text
    assert "hit twice in a row" not in report_markdown(clip_report())


def test_every_report_matches_its_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        Path(report_module.__file__).with_name("report.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["properties"]["schema_version"]["const"] == REPORT_SCHEMA_VERSION

    one_rally = Rally(10, 110, (shot(10, 1, None, 30.0),))
    for report in (
        clip_report(),
        build_report(
            "clip.mp4", FPS, 250, {}, {1: stats(5.0, 7.0)}, ball_tracked=False
        ),
        build_report("clip.mp4", FPS, 250, {}, {1: stats(5.0, 7.0)}, (), [one_rally]),
    ):
        jsonschema.validate(json.loads(json.dumps(report)), schema)


def test_heatmaps_draw_one_court_per_player() -> None:
    one = stats(1.0, 1.0).heatmap_seconds

    image = draw_heatmaps({1: one, 2: one}, {1: "near", 2: "far"})

    court_width = 10 * 16
    assert image.shape[1] == 2 * court_width + 12
    assert image.dtype == np.uint8


def test_the_pdf_report_has_every_section(tmp_path) -> None:
    pytest.importorskip("reportlab")
    from courtvision.visualization.report_pdf import report_pdf

    heatmaps = tmp_path / "heatmaps.png"
    one = stats(1.0, 1.0).heatmap_seconds
    cv2.imwrite(str(heatmaps), draw_heatmaps({1: one, 2: one}))

    pdf = report_pdf(clip_report(), heatmaps, compress=False)

    assert pdf.startswith(b"%PDF-")
    for section in (
        b"Highlights",
        b"Players",
        b"Where each player stood",
        b"Rallies",
        b"How far to trust these numbers",
    ):
        assert section in pdf
    assert b"/Subtype /Image" in pdf
    assert report_pdf(clip_report()).startswith(b"%PDF-")  # no heatmaps is fine
