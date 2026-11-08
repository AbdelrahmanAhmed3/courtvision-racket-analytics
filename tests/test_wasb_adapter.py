import numpy as np

from courtvision.detectors.wasb_adapter import heatmap_peak


def blob(heatmap, x, y, value):
    heatmap[y - 1 : y + 2, x - 1 : x + 2] = value


def test_no_ball_when_nothing_reaches_the_threshold():
    heatmap = np.full((288, 512), 0.4, dtype=np.float32)
    assert heatmap_peak(heatmap) is None


def test_the_strongest_blob_wins_not_the_highest_pixel():
    heatmap = np.zeros((288, 512), dtype=np.float32)
    heatmap[50, 60] = 0.99  # one bright pixel
    blob(heatmap, 200, 100, 0.8)  # nine pixels: a larger sum
    x, y, peak = heatmap_peak(heatmap)
    assert (round(x), round(y)) == (200, 100)
    assert peak == np.float32(0.8)


def test_the_centre_is_weighted_by_the_heat_map():
    heatmap = np.zeros((288, 512), dtype=np.float32)
    heatmap[100, 200] = 0.9
    heatmap[100, 201] = 0.6
    x, y, _ = heatmap_peak(heatmap)
    assert x == np.float32((200 * 0.9 + 201 * 0.6) / 1.5)
    assert y == 100
