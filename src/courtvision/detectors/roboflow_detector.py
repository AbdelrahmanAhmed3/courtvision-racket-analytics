from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from courtvision.detectors.base import Detection

DEFAULT_API_URL = "https://detect.roboflow.com"
DEFAULT_API_KEY_ENV = "ROBOFLOW_API_KEY"
DEFAULT_CONFIDENCE = 0.3
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 524})


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
        confidence: float | None = DEFAULT_CONFIDENCE,
        overlap: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.api_key = api_key or get_api_key(api_key_env)
        self.api_url = api_url
        self.confidence = confidence
        self.overlap = overlap
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.client = build_inference_client(api_url=api_url, api_key=self.api_key)

    def predict_frame(self, frame, frame_index: int) -> list[Detection]:
        response = infer_with_retries(
            self.client,
            frame,
            self.model_id,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )
        return filter_by_confidence(
            detections_from_response(
                response,
                frame_index=frame_index,
                model_name=self.model_name,
            ),
            self.confidence,
        )

    def predict_image(
        self,
        image_path: str | Path,
        frame_index: int = 0,
    ) -> list[Detection]:
        response = infer_with_retries(
            self.client,
            str(image_path),
            self.model_id,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )
        return filter_by_confidence(
            detections_from_response(
                response,
                frame_index=frame_index,
                model_name=self.model_name,
            ),
            self.confidence,
        )


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


def infer_with_retries(
    client: Any,
    source: Any,
    model_id: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Retry temporary hosted-inference failures without hiding real API errors."""
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be zero or greater")

    for attempt in range(max_retries + 1):
        try:
            return client.infer(source, model_id=model_id)
        except Exception as error:
            if not is_retryable_error(error) or attempt == max_retries:
                raise
            time.sleep(retry_backoff_seconds * (2**attempt))
    raise RuntimeError("Hosted inference retry loop ended unexpectedly")


def is_retryable_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return error.__class__.__name__ in {
        "ConnectionError",
        "ConnectTimeout",
        "ReadTimeout",
    }


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


def filter_by_confidence(
    detections: list[Detection],
    minimum_confidence: float | None,
) -> list[Detection]:
    if minimum_confidence is None:
        return detections
    return [
        detection
        for detection in detections
        if detection.confidence >= minimum_confidence
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
