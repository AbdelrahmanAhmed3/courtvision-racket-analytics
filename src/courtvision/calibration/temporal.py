"""Sparse optical-flow court calibration with per-frame RANSAC validation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from courtvision.calibration.homography import (
    HomographyEstimate,
    estimate_template_homography,
    project_template_points,
)
from courtvision.calibration.io import CalibrationRecord, LandmarkObservation
from courtvision.calibration.validation import (
    CalibrationValidation,
    validate_homography,
)


@dataclass(frozen=True)
class TemporalCalibrationResult:
    calibration: CalibrationRecord | None
    estimate: HomographyEstimate | None
    validation: CalibrationValidation | None
    tracked_landmark_count: int
    valid: bool
    reason: str | None = None


class TemporalCourtCalibrator:
    """Track manual court landmarks and re-estimate a homography per frame."""

    def __init__(
        self,
        calibration: CalibrationRecord,
        initial_frame: np.ndarray,
        min_inliers: int = 6,
        ransac_reprojection_threshold_px: float = 6.0,
        max_forward_backward_error_px: float = 2.0,
    ) -> None:
        if min_inliers < 4:
            raise ValueError("min_inliers must be at least four")
        self._initial = calibration
        self._min_inliers = min_inliers
        self._ransac_reprojection_threshold_px = ransac_reprojection_threshold_px
        self._max_forward_backward_error_px = max_forward_backward_error_px
        self._previous_gray = _gray(initial_frame)
        self._points = {
            name: np.asarray(observation.image, dtype=np.float32)
            for name, observation in calibration.landmarks.items()
            if observation.visible
        }
        self._templates = {
            name: observation.template
            for name, observation in calibration.landmarks.items()
        }

    def initial_result(self) -> TemporalCalibrationResult:
        estimate = estimate_template_homography(self._initial)
        validation = validate_homography(estimate)
        return TemporalCalibrationResult(
            calibration=self._initial,
            estimate=estimate,
            validation=validation,
            tracked_landmark_count=len(self._points),
            valid=validation.status in {"good", "acceptable"},
        )

    def update(self, frame: np.ndarray, frame_index: int) -> TemporalCalibrationResult:
        if len(self._points) < 4:
            self._previous_gray = _gray(frame)
            return TemporalCalibrationResult(
                calibration=None,
                estimate=None,
                validation=None,
                tracked_landmark_count=len(self._points),
                valid=False,
                reason="Fewer than four landmarks remain available for optical flow.",
            )

        current_gray = _gray(frame)
        names = tuple(self._points)
        previous_points = np.asarray(
            [self._points[name] for name in names],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        current_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or forward_status is None:
            self._previous_gray = current_gray
            return TemporalCalibrationResult(
                calibration=None,
                estimate=None,
                validation=None,
                tracked_landmark_count=0,
                valid=False,
                reason="Optical flow could not track the court landmarks.",
            )

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self._previous_gray,
            current_points,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward_points is None or backward_status is None:
            backward_status = np.zeros_like(forward_status)
            backward_points = previous_points

        forward_ok = forward_status.reshape(-1).astype(bool)
        backward_ok = backward_status.reshape(-1).astype(bool)
        forward_error = forward_error.reshape(-1) if forward_error is not None else None
        backward_error = np.linalg.norm(
            backward_points.reshape(-1, 2) - previous_points.reshape(-1, 2),
            axis=1,
        )
        accepted = forward_ok & backward_ok & (
            backward_error <= self._max_forward_backward_error_px
        )
        if forward_error is not None:
            accepted &= forward_error <= 25.0

        tracked = {
            name: tuple(float(value) for value in current_points[index, 0])
            for index, name in enumerate(names)
            if accepted[index]
        }
        self._previous_gray = current_gray
        self._points = {
            name: np.asarray(point, dtype=np.float32) for name, point in tracked.items()
        }
        if len(tracked) < 4:
            return TemporalCalibrationResult(
                calibration=None,
                estimate=None,
                validation=None,
                tracked_landmark_count=len(tracked),
                valid=False,
                reason="Too few optical-flow landmarks passed the consistency check.",
            )

        calibration = self._record_from_tracked_points(tracked, frame_index, frame)
        try:
            estimate = estimate_template_homography(
                calibration,
                ransac_reprojection_threshold_px=self._ransac_reprojection_threshold_px,
                preserve_manual_baseline_corners=False,
            )
        except ValueError as error:
            return TemporalCalibrationResult(
                calibration=calibration,
                estimate=None,
                validation=None,
                tracked_landmark_count=len(tracked),
                valid=False,
                reason=str(error),
            )
        validation = validate_homography(estimate)
        valid = (
            validation.status in {"good", "acceptable"}
            and validation.inlier_count >= self._min_inliers
        )
        if not valid:
            return TemporalCalibrationResult(
                calibration=calibration,
                estimate=estimate,
                validation=validation,
                tracked_landmark_count=len(tracked),
                valid=False,
                reason=(
                    "RANSAC calibration validation failed: "
                    f"{validation.inlier_count}/{validation.landmark_count} inliers."
                ),
            )

        self._reseed_from_inliers(calibration, estimate)
        return TemporalCalibrationResult(
            calibration=calibration,
            estimate=estimate,
            validation=validation,
            tracked_landmark_count=len(tracked),
            valid=True,
        )

    def _record_from_tracked_points(
        self,
        tracked: dict[str, tuple[float, float]],
        frame_index: int,
        frame: np.ndarray,
    ) -> CalibrationRecord:
        height, width = frame.shape[:2]
        landmarks = {
            name: LandmarkObservation(
                image=tracked.get(name, observation.image),
                template=observation.template,
                visible=name in tracked,
                confidence=1.0 if name in tracked else 0.0,
                source="optical-flow" if name in tracked else "optical-flow-lost",
            )
            for name, observation in self._initial.landmarks.items()
        }
        return CalibrationRecord(
            video_id=self._initial.video_id,
            frame_index=frame_index,
            frame_width=width,
            frame_height=height,
            court_type=self._initial.court_type,
            landmarks=landmarks,
        )

    def _reseed_from_inliers(
        self,
        calibration: CalibrationRecord,
        estimate: HomographyEstimate,
    ) -> None:
        projected = project_template_points(estimate.template_points, estimate)
        self._points = {
            name: projected[index].astype(np.float32)
            for index, name in enumerate(estimate.landmark_names)
            if name in estimate.inlier_names
        }


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
