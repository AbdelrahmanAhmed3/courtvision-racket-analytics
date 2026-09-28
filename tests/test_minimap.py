from __future__ import annotations

import numpy as np

from courtvision.visualization.minimap import (
    draw_shot_speed,
    get_court_spec,
    world_to_canvas,
)


def test_player_behind_baseline_stays_inside_map_canvas() -> None:
    spec = get_court_spec("tennis")
    canvas_size = (500, 900)
    margin = 45

    x, y = world_to_canvas(
        (spec.width_m / 2, spec.length_m + 2.0),
        spec,
        canvas_size,
        margin,
    )

    assert margin < x < canvas_size[0] - margin
    assert margin < y < canvas_size[1] - margin


def test_court_baseline_is_inset_from_extended_map_boundary() -> None:
    spec = get_court_spec("padel")
    canvas_size = (500, 900)
    margin = 45

    _, far_baseline_y = world_to_canvas((0, 0), spec, canvas_size, margin)
    _, near_baseline_y = world_to_canvas(
        (0, spec.length_m),
        spec,
        canvas_size,
        margin,
    )

    assert far_baseline_y > margin
    assert near_baseline_y < canvas_size[1] - margin


def test_shot_speed_label_draws_on_map() -> None:
    image = np.zeros((900, 500, 3), dtype=np.uint8)

    draw_shot_speed(
        image,
        hitter_id=2,
        receiver_id=3,
        speed_kmh=84.5,
    )

    assert image.any()
