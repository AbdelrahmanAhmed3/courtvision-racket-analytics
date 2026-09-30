#!/usr/bin/env python3
"""Compare a Roboflow court-keypoint model across controlled frame sizes.

The model may resize an image internally, but its returned coordinates should
still match the image that was submitted. This tool makes that contract visible:
it saves both the raw-model overlay at each submitted size and the same points
scaled back to the original video frame.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from courtvision.calibration.io import load_calibration  # noqa: E402
from courtvision.calibration.keypoints import (  # noqa: E402
    default_keypoint_model_id,
    keypoint_proposal_from_response,
)
from courtvision.detectors.roboflow_detector import (  # noqa: E402
    DEFAULT_API_URL,
    build_inference_client,
    get_api_key,
    infer_with_retries,
)


def parse_size(value: str) -> tuple[int, int] | None:
    """Parse ``WIDTHxHEIGHT`` or the special ``original`` size."""
    if value.lower() == "original":
        return None
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected WIDTHxHEIGHT or original, got {value!r}."
        ) from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Image dimensions must be positive.")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save Roboflow keypoint-model diagnostics for several frame sizes."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video.")
    parser.add_argument(
        "--seconds", type=float, default=0.0, help="Frame timestamp (default: 0)."
    )
    parser.add_argument(
        "--court-type", choices=("tennis", "padel"), required=True, help="Court type."
    )
    parser.add_argument(
        "--model-id", help="Roboflow model ID; defaults to the court-type model."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_size,
        default=[None, (640, 360), (1280, 720), (1920, 1080)],
        help="Submitted sizes, e.g. original 640x360 1280x720.",
    )
    parser.add_argument(
        "--include-roboflow-stretch",
        action="store_true",
        help=(
            "Also submit a 640x640 stretched frame, matching a Roboflow "
            "Stretch-to-640x640 training preprocessing step."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for images and JSON."
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.15,
        help="Confidence used when listing canonical landmarks (default: 0.15).",
    )
    parser.add_argument(
        "--reference-calibration",
        type=Path,
        help=(
            "Optional manual calibration JSON for a frame at this timestamp. "
            "Writes a green-reference versus magenta-model comparison overlay."
        ),
    )
    return parser.parse_args()


def read_frame(video_path: Path, seconds: float) -> tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = round(seconds * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame, fps, frame_index


def raw_keypoints(response: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = response.get("predictions", [])
    candidates = [
        prediction
        for prediction in predictions
        if isinstance(prediction, dict)
        and isinstance(prediction.get("keypoints"), list)
    ]
    if not candidates:
        return []
    best = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
    return [point for point in best["keypoints"] if isinstance(point, dict)]


def response_image_size(response: dict[str, Any]) -> tuple[int, int] | None:
    """Read common hosted-inference image-size fields without assuming one SDK shape."""
    for container in (response, response.get("image", {})):
        if not isinstance(container, dict):
            continue
        width = container.get("width")
        height = container.get("height")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return int(width), int(height)
    return None


def draw_points(
    image: np.ndarray,
    keypoints: list[dict[str, Any]],
    x_scale: float,
    y_scale: float,
    title: str,
) -> np.ndarray:
    rendered = image.copy()
    for point in keypoints:
        if "x" not in point or "y" not in point:
            continue
        x = round(float(point["x"]) * x_scale)
        y = round(float(point["y"]) * y_scale)
        label = str(point.get("class", point.get("class_id", "?")))
        confidence = float(point.get("confidence", 0.0))
        cv2.circle(rendered, (x, y), 6, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            rendered,
            f"{label} {confidence:.2f}",
            (x + 8, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        rendered,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write image: {path}")


def draw_reference_comparison(
    image: np.ndarray,
    reference_points: dict[str, tuple[float, float]],
    model_points: dict[str, tuple[float, float]],
    x_scale: float,
    y_scale: float,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Draw manual points in green and model points in magenta."""
    rendered = image.copy()
    errors: dict[str, dict[str, float]] = {}
    for name, reference in reference_points.items():
        if name not in model_points:
            continue
        model_x = model_points[name][0] * x_scale
        model_y = model_points[name][1] * y_scale
        ref_x, ref_y = reference
        delta_x = model_x - ref_x
        delta_y = model_y - ref_y
        distance = float(np.hypot(delta_x, delta_y))
        errors[name] = {
            "dx_px": delta_x,
            "dy_px": delta_y,
            "distance_px": distance,
        }
        reference_pixel = (round(ref_x), round(ref_y))
        model_pixel = (round(model_x), round(model_y))
        cv2.line(rendered, reference_pixel, model_pixel, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(
            rendered,
            reference_pixel,
            (0, 220, 0),
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(rendered, model_pixel, 6, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            rendered,
            f"{name}: {distance:.0f}px",
            (model_pixel[0] + 7, max(25, model_pixel[1] - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        rendered,
        "Green cross: manual reference | Magenta dot: model",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered, errors


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Video does not exist: {args.input}")
    if args.seconds < 0:
        raise ValueError("--seconds must be zero or greater")
    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise ValueError("--minimum-confidence must be between zero and one")

    original, fps, frame_index = read_frame(args.input, args.seconds)
    original_height, original_width = original.shape[:2]
    model_id = args.model_id or default_keypoint_model_id(args.court_type)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_image(output_dir / "frame_original.png", original)
    reference_points: dict[str, tuple[float, float]] | None = None
    if args.reference_calibration is not None:
        reference = load_calibration(args.reference_calibration)
        if reference.court_type.lower() != args.court_type:
            raise ValueError(
                "Reference calibration court type does not match --court-type"
            )
        if (reference.frame_width, reference.frame_height) != (
            original_width,
            original_height,
        ):
            raise ValueError(
                "Reference calibration dimensions do not match the extracted "
                "video frame"
            )
        reference_points = {
            name: observation.image
            for name, observation in reference.landmarks.items()
            if observation.visible
        }

    api_url = DEFAULT_API_URL.replace("detect", "serverless")
    client = build_inference_client(api_url, get_api_key())
    report: dict[str, Any] = {
        "input": str(args.input),
        "seconds": args.seconds,
        "frame_index": frame_index,
        "fps": fps,
        "original_size": {"width": original_width, "height": original_height},
        "court_type": args.court_type,
        "model_id": model_id,
        "runs": [],
    }

    run_sizes = list(args.sizes)
    if args.include_roboflow_stretch and (640, 640) not in run_sizes:
        run_sizes.append((640, 640))

    for size in run_sizes:
        submitted = original if size is None else cv2.resize(original, size)
        submitted_height, submitted_width = submitted.shape[:2]
        label = f"{submitted_width}x{submitted_height}"
        print(f"Running {model_id} with submitted size {label}...")
        response = infer_with_retries(client, submitted, model_id)
        keypoints = raw_keypoints(response)
        proposal = keypoint_proposal_from_response(
            response,
            args.court_type,
            model_id,
            args.minimum_confidence,
        )
        raw_path = output_dir / f"response_{label}.json"
        raw_path.write_text(json.dumps(response, indent=2) + "\n")
        write_image(
            output_dir / f"overlay_{label}_submitted.png",
            draw_points(
                submitted,
                keypoints,
                1.0,
                1.0,
                f"Returned on submitted {label}",
            ),
        )
        write_image(
            output_dir / f"overlay_{label}_original.png",
            draw_points(
                original,
                keypoints,
                original_width / submitted_width,
                original_height / submitted_height,
                "Returned from "
                f"{label}, scaled to original {original_width}x{original_height}",
            ),
        )
        reference_errors = None
        if reference_points is not None and proposal is not None:
            comparison, reference_errors = draw_reference_comparison(
                original,
                reference_points,
                proposal.landmarks,
                original_width / submitted_width,
                original_height / submitted_height,
            )
            write_image(output_dir / f"comparison_{label}_reference.png", comparison)
        report["runs"].append(
            {
                "submitted_size": {
                    "width": submitted_width,
                    "height": submitted_height,
                },
                "response_image_size": response_image_size(response),
                "raw_keypoint_count": len(keypoints),
                "canonical_landmarks": None if proposal is None else proposal.landmarks,
                "canonical_confidences": None
                if proposal is None
                else proposal.landmark_confidences,
                "reference_errors_px": reference_errors,
                "response_path": raw_path.name,
            }
        )

    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved diagnostic images and report to: {output_dir}")
    print("Compare *_submitted.png with *_original.png.")


if __name__ == "__main__":
    main()
