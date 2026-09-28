import numpy as np

from courtvision.calibration.pixel_masks import near_white_pixel_mask


def test_near_white_pixel_mask_keeps_bright_low_saturation_pixels() -> None:
    frame = np.asarray([[[245, 245, 245], [20, 180, 20], [35, 35, 35]]], dtype=np.uint8)

    mask = near_white_pixel_mask(frame, minimum_value=170, maximum_saturation=80)

    assert mask.tolist() == [[255, 0, 0]]
