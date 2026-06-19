from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from courtvision.detectors.base import Detection

DEFAULT_API_URL = "https://detect.roboflow.com"
DEFAULT_API_KEY_ENV = "ROBOFLOW_API_KEY"


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


class RoboflowDetector:
    """Roboflow hosted/serverless object-detection adapter."""

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        api_url: str = DEFAULT_API_URL,
        model_name: str | None = None,
        confidence: float | None = None,
        overlap: float | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.api_key = api_key or get_api_key(api_key_env)
        self.api_url = api_url
        self.confidence = confidence
        self.overlap = overlap
        self.client = build_inference_client(api_url=api_url, api_key=self.api_key)

    def predict_frame(self, frame, frame_index: int) -> list[Detection]:
        response = self.client.infer(
            frame,
            model_id=self.model_id,
            **self._inference_kwargs(),
        )
        return detections_from_response(
            response,
            frame_index=frame_index,
            model_name=self.model_name,
        )

    def predict_image(
        self,
        image_path: str | Path,
        frame_index: int = 0,
    ) -> list[Detection]:
        response = self.client.infer(
            str(image_path),
            model_id=self.model_id,
            **self._inference_kwargs(),
        )
        return detections_from_response(
            response,
            frame_index=frame_index,
            model_name=self.model_name,
        )

    def _inference_kwargs(self) -> dict[str, float]:
        kwargs = {}
        if self.confidence is not None:
            kwargs["confidence"] = self.confidence
        if self.overlap is not None:
            kwargs["overlap"] = self.overlap
        return kwargs


def get_api_key(api_key_env: str = DEFAULT_API_KEY_ENV) -> str:
    load_dotenv()
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"{api_key_env} is missing. Add it to .env locally or Kaggle Secrets."
        )
    return api_key


def build_inference_client(api_url: str, api_key: str) -> Any:
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError as exc:
        raise ImportError(
            "Roboflow inference-sdk is not installed. Install it with "
            "`pip install -e '.[roboflow]'` or `pip install inference-sdk`."
        ) from exc

    return InferenceHTTPClient(api_url=api_url, api_key=api_key)


def detections_from_response(
    response: dict[str, Any],
    frame_index: int,
    model_name: str,
) -> list[Detection]:
    predictions = response.get("predictions", [])
    return [
        detection_from_prediction(
            prediction,
            frame_index=frame_index,
            model_name=model_name,
        )
        for prediction in predictions
    ]


def detection_from_prediction(
    prediction: dict[str, Any],
    frame_index: int,
    model_name: str,
) -> Detection:
    x1, y1, x2, y2 = roboflow_box_to_xyxy(prediction)
    return Detection(
        frame=frame_index,
        class_name=str(prediction.get("class", prediction.get("class_name", ""))),
        confidence=float(prediction.get("confidence", 0.0)),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        model_name=model_name,
    )
