"""Roboflow court-keypoint proposals mapped into CourtVision landmarks.

Experimental: calls a hosted Roboflow model and needs ROBOFLOW_API_KEY; an
optional backend under ADR 0002. See docs/roadmap.md (v4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from courtvision.calibration.court_surface import (
    CourtSurfaceSettings,
    SurfaceBoundaryRefinement,
    SurfaceBoundarySettings,
    analyze_court_surface_color,
    refine_surface_boundaries,
)
from courtvision.calibration.homography import BASELINE_CORNER_NAMES
from courtvision.calibration.keypoint_refinement import (
    KeypointRefinementResult,
    KeypointRefinementSettings,
    refine_keypoint_landmarks,
)
from courtvision.detectors.roboflow_detector import (
    DEFAULT_API_URL,
    build_inference_client,
    get_api_key,
    infer_with_retries,
)

DEFAULT_TENNIS_MODEL_ID = "tennis-court-detection-onesd/10"
DEFAULT_PADEL_MODEL_ID = "padel-court-fmfv8/15"
DEFAULT_KEYPOINT_CONFIDENCE = 0.15
# Bump this when a published model's point-order mapping changes. The UI uses
# it to avoid reusing a proposal parsed under an older mapping.
KEYPOINT_MAPPING_REVISION = "2026-08-19-padel-front-back-v2"

# Verified from each model's public hosted-inference response. The tennis model
# has fourteen numbered vertices; the padel model publishes descriptive labels.
KEYPOINT_LANDMARK_NAMES = {
    "tennis": {
        "1": "near_left_baseline_corner",
        "4": "near_right_baseline_corner",
        "5": "far_right_baseline_corner",
        "8": "far_left_baseline_corner",
        "9": "near_left_service_intersection",
        "10": "far_left_service_intersection",
        "11": "near_right_service_intersection",
        "12": "far_right_service_intersection",
        "13": "near_service_center",
        "14": "far_service_center",
    },
    "padel": {
        "FCL": "near_left_baseline_corner",
        "FCR": "near_right_baseline_corner",
        "BCR": "far_right_baseline_corner",
        "BCL": "far_left_baseline_corner",
        "FSL": "near_left_service_intersection",
        "FSR": "near_right_service_intersection",
        "BSR": "far_right_service_intersection",
        "BSL": "far_left_service_intersection",
        "NL": "left_net_point",
        "NR": "right_net_point",
        "NC": "net_center",
        "FSM": "near_service_center",
        "BSM": "far_service_center",
    },
}


@dataclass(frozen=True)
class KeypointCalibrationProposal:
    """A confidence-scored calibration proposal from a court keypoint model."""

    landmarks: dict[str, tuple[float, float]]
    landmark_confidences: dict[str, float]
    confidence: float
    model_id: str
    refinement: KeypointRefinementResult | None = None
    surface_refinement: SurfaceBoundaryRefinement | None = None

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        return tuple(self.landmarks[name] for name in BASELINE_CORNER_NAMES)

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks)

    @property
    def refined_count(self) -> int:
        return 0 if self.refinement is None else self.refinement.refined_count

    @property
    def auto_ready(self) -> bool:
        return self.refinement is not None and self.refinement.auto_ready

    @property
    def mean_line_support(self) -> float:
        return 0.0 if self.refinement is None else self.refinement.mean_line_support


@dataclass(frozen=True)
class LandmarkAuditEntry:
    """Final landmark coordinate and the evidence responsible for it."""

    name: str
    source: str
    original: tuple[float, float]
    final: tuple[float, float]
    shift_px: float
    confirmed: bool


def landmark_audit_entries(
    proposal: KeypointCalibrationProposal,
) -> tuple[LandmarkAuditEntry, ...]:
    """Explain which stage supplied each final keypoint coordinate."""
    if proposal.surface_refinement is not None:
        original = proposal.surface_refinement.original_landmarks
        surface_names = set(proposal.surface_refinement.refined_landmark_names)
    elif proposal.refinement is not None:
        original = proposal.refinement.original_landmarks
        surface_names = set()
    else:
        original = proposal.landmarks
        surface_names = set()
    line_names = (
        set(proposal.refinement.refined_landmark_names)
        if proposal.refinement is not None
        else set()
    )
    entries = []
    for name, final in proposal.landmarks.items():
        initial = original.get(name, final)
        if name in surface_names:
            source = "Court-color outer boundary"
        elif name in line_names and "service_intersection" in name:
            source = "Near-white service line"
        elif name in line_names:
            source = "Near-white RANSAC intersection"
        else:
            source = "Keypoint model only"
        confirmed = source != "Keypoint model only"
        entries.append(
            LandmarkAuditEntry(
                name=name,
                source=source,
                original=initial,
                final=final,
                shift_px=float(np.linalg.norm(np.asarray(final) - np.asarray(initial))),
                confirmed=confirmed,
            )
        )
    return tuple(entries)


def draw_landmark_audit(
    frame: np.ndarray,
    proposal: KeypointCalibrationProposal,
) -> np.ndarray:
    """Render all final landmarks using colors keyed to their evidence source."""
    rendered = frame.copy()
    colors = {
        "Court-color outer boundary": (0, 220, 0),
        "Near-white service line": (0, 220, 255),
        "Near-white RANSAC intersection": (255, 210, 0),
        "Keypoint model only": (255, 0, 255),
    }
    for index, entry in enumerate(landmark_audit_entries(proposal), start=1):
        point = tuple(round(value) for value in entry.final)
        color = colors[entry.source]
        cv2.drawMarker(
            rendered,
            point,
            color,
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            str(index),
            (point[0] + 5, point[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return rendered


def default_keypoint_model_id(court_type: str) -> str:
    normalized = court_type.strip().lower()
    if normalized == "tennis":
        return DEFAULT_TENNIS_MODEL_ID
    if normalized in {"padel", "paddle"}:
        return DEFAULT_PADEL_MODEL_ID
    raise ValueError(f"Unsupported court type: {court_type}")


def detect_keypoint_calibration(
    frame: np.ndarray,
    court_type: str,
    model_id: str | None = None,
    minimum_keypoint_confidence: float = DEFAULT_KEYPOINT_CONFIDENCE,
    refine_with_court_lines: bool = True,
    refinement_settings: KeypointRefinementSettings | None = None,
    refine_padel_surface: bool = True,
    court_surface_settings: CourtSurfaceSettings | None = None,
    surface_boundary_settings: SurfaceBoundarySettings | None = None,
) -> KeypointCalibrationProposal | None:
    """Run hosted inference and map the best court prediction to landmarks."""
    if not 0.0 <= minimum_keypoint_confidence <= 1.0:
        raise ValueError("minimum_keypoint_confidence must be between zero and one")
    model_id = model_id or default_keypoint_model_id(court_type)
    api_url = DEFAULT_API_URL.replace("detect", "serverless")
    client = build_inference_client(api_url, get_api_key())
    response = infer_with_retries(client, frame, model_id)
    proposal = keypoint_proposal_from_response(
        response,
        court_type,
        model_id,
        minimum_keypoint_confidence,
    )
    if proposal is None:
        return proposal
    normalized_court_type = court_type.strip().lower()
    if normalized_court_type == "paddle":
        normalized_court_type = "padel"
    surface_refinement = None
    refinement_seed = proposal.landmarks
    if normalized_court_type == "padel" and refine_padel_surface:
        try:
            surface_analysis = analyze_court_surface_color(
                frame,
                proposal.landmarks,
                court_surface_settings,
            )
            surface_refinement = refine_surface_boundaries(
                surface_analysis,
                proposal.landmarks,
                surface_boundary_settings,
            )
            refinement_seed = surface_refinement.landmarks
        except ValueError:
            surface_refinement = None

    refinement = None
    final_landmarks = dict(refinement_seed)
    if refine_with_court_lines:
        refinement = refine_keypoint_landmarks(
            frame,
            refinement_seed,
            court_type,
            refinement_settings,
        )
        final_landmarks = dict(refinement.landmarks)
    if surface_refinement is not None:
        for name in BASELINE_CORNER_NAMES:
            final_landmarks[name] = surface_refinement.landmarks[name]
        if refinement is not None:
            refined_names = tuple(
                dict.fromkeys(
                    (*refinement.refined_landmark_names, *BASELINE_CORNER_NAMES)
                )
            )
            refinement = replace(
                refinement,
                landmarks=final_landmarks,
                original_landmarks=dict(proposal.landmarks),
                refined_landmark_names=refined_names,
                refinement_distances_px={
                    name: float(
                        np.linalg.norm(
                            np.asarray(final_landmarks[name])
                            - np.asarray(proposal.landmarks[name])
                        )
                    )
                    for name in refined_names
                    if name in proposal.landmarks and name in final_landmarks
                },
            )
    return KeypointCalibrationProposal(
        landmarks=final_landmarks,
        landmark_confidences=proposal.landmark_confidences,
        confidence=proposal.confidence,
        model_id=proposal.model_id,
        refinement=refinement,
        surface_refinement=surface_refinement,
    )


def keypoint_proposal_from_response(
    response: dict[str, Any],
    court_type: str,
    model_id: str,
    minimum_keypoint_confidence: float = DEFAULT_KEYPOINT_CONFIDENCE,
) -> KeypointCalibrationProposal | None:
    """Parse a hosted keypoint response without coupling to a model SDK type."""
    normalized = court_type.strip().lower()
    if normalized == "paddle":
        normalized = "padel"
    mapping = KEYPOINT_LANDMARK_NAMES.get(normalized)
    if mapping is None:
        raise ValueError(f"Unsupported court type: {court_type}")
    predictions = response.get("predictions", [])
    if not isinstance(predictions, list):
        return None
    candidates = [
        prediction
        for prediction in predictions
        if isinstance(prediction, dict)
        and isinstance(prediction.get("keypoints"), list)
    ]
    if not candidates:
        return None
    prediction = max(candidates, key=lambda item: float(item.get("confidence", 0.0)))
    landmarks: dict[str, tuple[float, float]] = {}
    confidences: dict[str, float] = {}
    for keypoint in prediction["keypoints"]:
        if not isinstance(keypoint, dict):
            continue
        key = str(keypoint.get("class", keypoint.get("class_name", ""))).upper()
        if normalized == "tennis":
            key = keypoint.get("class", keypoint.get("class_id", ""))
            key = str(key)
        landmark_name = mapping.get(key)
        confidence = float(keypoint.get("confidence", 0.0))
        if landmark_name is None or confidence < minimum_keypoint_confidence:
            continue
        landmarks[landmark_name] = (float(keypoint["x"]), float(keypoint["y"]))
        confidences[landmark_name] = confidence
    if not all(name in landmarks for name in BASELINE_CORNER_NAMES):
        return None
    return KeypointCalibrationProposal(
        landmarks=landmarks,
        landmark_confidences=confidences,
        confidence=float(prediction.get("confidence", 0.0)),
        model_id=model_id,
    )
