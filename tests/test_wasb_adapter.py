import hashlib
import io

import numpy as np
import pytest

from courtvision.detectors import wasb_adapter
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


def fake_download(monkeypatch, data: bytes):
    monkeypatch.setattr(
        wasb_adapter.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(data)
    )


def test_downloaded_weights_are_kept_when_the_checksum_matches(tmp_path, monkeypatch):
    fake_download(monkeypatch, b"weights")
    monkeypatch.setattr(
        wasb_adapter, "WASB_WEIGHTS_SHA256", hashlib.sha256(b"weights").hexdigest()
    )
    path = wasb_adapter.ensure_wasb_weights(tmp_path / "wasb.pth.tar")
    assert path.read_bytes() == b"weights"
    assert list(tmp_path.iterdir()) == [path]


def test_a_download_that_does_not_match_is_thrown_away(tmp_path, monkeypatch):
    fake_download(monkeypatch, b"an error page")
    with pytest.raises(ValueError, match="don't match"):
        wasb_adapter.ensure_wasb_weights(tmp_path / "wasb.pth.tar")
    assert list(tmp_path.iterdir()) == []


def test_the_network_paints_one_heat_map_per_frame():
    torch = pytest.importorskip("torch")
    from courtvision.detectors.wasb_hrnet import HRNet

    model = HRNet(wasb_adapter.WASB_HRNET_CONFIG).eval()
    width, height = wasb_adapter.WASB_INPUT_SIZE
    with torch.no_grad():
        heatmaps = model(torch.zeros(1, 9, height, width))[0]
    assert tuple(heatmaps.shape) == (1, 3, height, width)
