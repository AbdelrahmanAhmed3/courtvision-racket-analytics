"""Shots, bounces and ball speed from the ball track and tracked players.

Experimental. Tuned on tennis broadcast clips; not yet adapted to padel.
Known limits (see issue #8): thresholds are in pixels and frames, the ball is
projected onto the floor even when it is in the air, glass rebounds count as
bounces, the last shot of a rally has no receiver and is not reported, and
exchanges shorter than 5 m (volleys at the net) are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot, isfinite

from courtvision.analytics.court_coordinates import CourtCoordinate
from courtvision.detectors.base import Detection
from courtvision.detectors.tracknet_adapter import BallPoint

DEFAULT_DIRECTION_WINDOW_FRAMES = 2
DEFAULT_MIN_BOUNCE_CHANGE_DEGREES = 45.0
# A 170 km/h serve can cross a 17 m court in roughly 0.36 seconds.  Keep this
# guard below that so fast, valid serve contacts are not discarded.
DEFAULT_MIN_SHOT_INTERVAL_SECONDS = 0.12
DEFAULT_MAX_SPEED_KMH = 300.0
DEFAULT_CONTACT_DISTANCE_PX = 90.0
DEFAULT_CONTACT_GAP_FRAMES = 3
DEFAULT_MIN_FLIGHT_DISTANCE_M = 5.0


@dataclass(frozen=True)
class ShotEvent:
    event_type: str
    frame: int
    hitter_track_id: int | None
    receiver_track_id: int | None
    receive_frame: int
    bounce_frame: int | None
    court_plane_speed_kmh: float
    segment_distance_m: float
    segment_duration_seconds: float


def detect_shots(
    ball_points_by_frame: dict[int, BallPoint],
    player_detections_by_frame: dict[int, list[tuple[int, Detection]]],
    ball_coordinates_by_frame: dict[int, CourtCoordinate],
    fps: float,
    direction_window_frames: int = DEFAULT_DIRECTION_WINDOW_FRAMES,
) -> list[ShotEvent]:
    """Estimate court-plane ball flights between consecutive player contacts."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if direction_window_frames < 1:
        raise ValueError("direction_window_frames must be at least one")

    contacts = _find_player_contact_events(
        ball_points_by_frame,
        player_detections_by_frame,
    )
    events: list[ShotEvent] = []
    min_interval_frames = round(DEFAULT_MIN_SHOT_INTERVAL_SECONDS * fps)
    for (frame, hitter_track_id), (receive_frame, receiver_track_id) in zip(
        contacts,
        contacts[1:],
        strict=False,
    ):
        if receive_frame - frame < min_interval_frames:
            continue
        start = ball_coordinates_by_frame.get(frame)
        end = ball_coordinates_by_frame.get(receive_frame)
        if not _is_cross_court_flight(start, end):
            continue
        bounce_frame = _find_first_bounce(
            ball_points_by_frame,
            player_detections_by_frame,
            frame,
            receive_frame,
            direction_window_frames,
            fps,
        )
        event = _build_shot_event(
            "transit",
            frame,
            hitter_track_id,
            receiver_track_id,
            receive_frame,
            bounce_frame,
            ball_coordinates_by_frame,
            fps,
        )
        if event is not None:
            events.append(event)
    return sorted(events, key=lambda event: event.frame)


def _find_player_contact_events(
    ball_points_by_frame: dict[int, BallPoint],
    player_detections_by_frame: dict[int, list[tuple[int, Detection]]],
) -> list[tuple[int, int]]:
    """Group consecutive near-player ball positions into player-contact events."""
    contacts: list[tuple[int, int]] = []
    current_track_id: int | None = None
    current_frame: int | None = None
    for frame in sorted(ball_points_by_frame):
        track_id = _nearest_player_at_impact(
            ball_points_by_frame[frame],
            player_detections_by_frame.get(frame, []),
            minimum_distance_px=DEFAULT_CONTACT_DISTANCE_PX,
        )
        if track_id is None:
            continue
        if (
            current_track_id == track_id
            and current_frame is not None
            and frame - current_frame <= DEFAULT_CONTACT_GAP_FRAMES
        ):
            current_frame = frame
            continue
        contacts.append((frame, track_id))
        current_track_id = track_id
        current_frame = frame
    return contacts


def _is_cross_court_flight(
    start: CourtCoordinate | None,
    end: CourtCoordinate | None,
) -> bool:
    if start is None or end is None or not start.in_bounds or not end.in_bounds:
        return False
    distance_m = hypot(
        end.court_x_m - start.court_x_m,
        end.court_y_m - start.court_y_m,
    )
    crosses_net = (start.template_y - 0.5) * (end.template_y - 0.5) < 0
    return distance_m >= DEFAULT_MIN_FLIGHT_DISTANCE_M and crosses_net


def _nearest_player_at_impact(
    ball_point: BallPoint,
    players: list[tuple[int, Detection]],
    minimum_distance_px: float = DEFAULT_CONTACT_DISTANCE_PX,
) -> int | None:
    if not ball_point.visible or ball_point.x is None or ball_point.y is None:
        return None

    nearest: tuple[float, int] | None = None
    for track_id, detection in players:
        distance = _point_to_bbox_distance(ball_point.x, ball_point.y, detection)
        box_diagonal = hypot(detection.x2 - detection.x1, detection.y2 - detection.y1)
        max_distance = max(minimum_distance_px, box_diagonal * 0.45)
        if distance <= max_distance and (nearest is None or distance < nearest[0]):
            nearest = (distance, track_id)
    return nearest[1] if nearest else None


def _point_to_bbox_distance(x: float, y: float, box: Detection) -> float:
    closest_x = min(max(x, box.x1), box.x2)
    closest_y = min(max(y, box.y1), box.y2)
    return hypot(x - closest_x, y - closest_y)


def _has_direction_change(
    ball_points_by_frame: dict[int, BallPoint],
    frame: int,
    window: int,
    minimum_angle_degrees: float,
) -> bool:
    before = ball_points_by_frame.get(frame - window)
    current = ball_points_by_frame.get(frame)
    after = ball_points_by_frame.get(frame + window)
    if not all(point and point.visible for point in (before, current, after)):
        return False
    assert before and current and after
    assert before.x is not None and before.y is not None
    assert current.x is not None and current.y is not None
    assert after.x is not None and after.y is not None
    incoming = (current.x - before.x, current.y - before.y)
    outgoing = (after.x - current.x, after.y - current.y)
    return _vector_angle_degrees(incoming, outgoing) >= minimum_angle_degrees


def _vector_angle_degrees(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    first_length = hypot(*first)
    second_length = hypot(*second)
    if first_length < 1e-6 or second_length < 1e-6:
        return 0.0
    cosine = (first[0] * second[0] + first[1] * second[1]) / (
        first_length * second_length
    )
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def _find_first_bounce(
    ball_points_by_frame: dict[int, BallPoint],
    player_detections_by_frame: dict[int, list[tuple[int, Detection]]],
    impact_frame: int,
    receive_frame: int,
    window: int,
    fps: float,
) -> int | None:
    """First direction change away from players between the hit and the receive."""
    latest_frame = max(ball_points_by_frame, default=impact_frame)
    min_delay_frames = max(window + 1, round(0.12 * fps))
    max_search_frames = round(2.5 * fps)
    for frame in range(
        impact_frame + min_delay_frames,
        min(
            latest_frame - window,
            impact_frame + max_search_frames,
            receive_frame - 1,
        )
        + 1,
    ):
        if (
            _nearest_player_at_impact(
                ball_points_by_frame.get(frame, BallPoint(frame, None, None)),
                player_detections_by_frame.get(frame, []),
            )
            is not None
        ):
            continue
        if _has_direction_change(
            ball_points_by_frame,
            frame,
            window,
            DEFAULT_MIN_BOUNCE_CHANGE_DEGREES,
        ):
            return frame
    return None


def _build_shot_event(
    event_type: str,
    impact_frame: int,
    hitter_track_id: int | None,
    receiver_track_id: int | None,
    receive_frame: int,
    bounce_frame: int | None,
    ball_coordinates_by_frame: dict[int, CourtCoordinate],
    fps: float,
) -> ShotEvent | None:
    if receive_frame <= impact_frame:
        return None
    start = ball_coordinates_by_frame.get(impact_frame)
    end = ball_coordinates_by_frame.get(receive_frame)
    if start is None or end is None or not start.in_bounds or not end.in_bounds:
        return None
    values = (start.court_x_m, start.court_y_m, end.court_x_m, end.court_y_m)
    if not all(isfinite(value) for value in values):
        return None
    distance_m = hypot(end.court_x_m - start.court_x_m, end.court_y_m - start.court_y_m)
    duration_seconds = (receive_frame - impact_frame) / fps
    speed_kmh = distance_m / duration_seconds * 3.6
    if speed_kmh <= 0 or speed_kmh > DEFAULT_MAX_SPEED_KMH:
        return None
    return ShotEvent(
        event_type=event_type,
        frame=impact_frame,
        hitter_track_id=hitter_track_id,
        receiver_track_id=receiver_track_id,
        receive_frame=receive_frame,
        bounce_frame=bounce_frame,
        court_plane_speed_kmh=speed_kmh,
        segment_distance_m=distance_m,
        segment_duration_seconds=duration_seconds,
    )
