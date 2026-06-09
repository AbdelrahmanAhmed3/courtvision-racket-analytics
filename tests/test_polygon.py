import numpy as np

from courtvision.geometry.polygon import point_inside_polygon


def test_point_inside_polygon() -> None:
    polygon = np.array(
        [
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100],
        ],
        dtype=np.float32,
    )

    assert point_inside_polygon((50, 50), polygon) is True
    assert point_inside_polygon((150, 50), polygon) is False
