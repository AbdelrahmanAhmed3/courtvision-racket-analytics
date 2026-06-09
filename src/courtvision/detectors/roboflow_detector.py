from __future__ import annotations


def roboflow_box_to_xyxy(prediction: dict) -> tuple[float, float, float, float]:
    """Convert Roboflow center-width-height boxes to xyxy coordinates."""
    x = float(prediction["x"])
    y = float(prediction["y"])
    width = float(prediction["width"])
    height = float(prediction["height"])

    return (
        x - width / 2,
        y - height / 2,
        x + width / 2,
        y + height / 2,
    )
