from dataclasses import replace

from courtvision.analytics.court_coordinates import CourtCoordinate
from courtvision.analytics.shots import detect_shots
from courtvision.detectors.base import Detection
from courtvision.detectors.tracknet_adapter import BallPoint


def ball_coordinate(frame: int, x: float, y: float) -> CourtCoordinate:
    return CourtCoordinate(
        frame=frame,
        track_id=0,
        object_type="ball",
        image_x=x,
        image_y=y,
        template_x=x / 100,
        template_y=y / 100,
        court_x_m=x / 10,
        court_y_m=y / 10,
        in_bounds=True,
        confidence=1.0,
    )


def test_detects_player_impact_and_post_impact_speed() -> None:
    ball_points = {
        frame: BallPoint(frame, 100.0, y)
        for frame, y in enumerate([30, 40, 50, 60, 70, 60, 50, 40, 20, 10, 0])
    }
    hitter = Detection(4, "player", 0.9, 90, 65, 110, 80, "test")
    receiver = Detection(8, "player", 0.9, 90, 15, 110, 30, "test")
    player_detections = {4: [(7, hitter)], 8: [(8, receiver)]}
    coordinates = {
        frame: ball_coordinate(frame, point.x or 0.0, point.y or 0.0)
        for frame, point in ball_points.items()
    }

    shots = detect_shots(ball_points, player_detections, coordinates, fps=10.0)

    assert len(shots) == 1
    assert shots[0].event_type == "transit"
    assert shots[0].frame == 4
    assert shots[0].hitter_track_id == 7
    assert shots[0].receiver_track_id == 8
    assert shots[0].receive_frame == 8
    assert shots[0].bounce_frame is None
    assert shots[0].court_plane_speed_kmh == 45.0


def test_detects_player_flight_anywhere_in_the_video() -> None:
    ball_points = {
        frame: BallPoint(frame, 100.0, float((frame - 40) * 10))
        for frame in range(40, 51)
    }
    server = Detection(40, "player", 0.9, 90, -10, 110, 10, "test")
    receiver = Detection(50, "player", 0.9, 90, 90, 110, 110, "test")
    coordinates = {
        frame: ball_coordinate(frame, point.x or 0.0, point.y or 0.0)
        for frame, point in ball_points.items()
    }

    shots = detect_shots(
        ball_points,
        {40: [(4, server)], 50: [(7, receiver)]},
        coordinates,
        fps=10.0,
    )

    assert len(shots) == 1
    assert shots[0].event_type == "transit"
    assert shots[0].hitter_track_id == 4
    assert shots[0].receiver_track_id == 7
    assert shots[0].frame == 40
    assert shots[0].receive_frame == 50
    assert shots[0].court_plane_speed_kmh == 36.0


def test_keeps_a_fast_serve_with_contacts_nine_frames_apart() -> None:
    """A 30 fps serve can cross the court in about 0.3 seconds."""
    ball_points = {
        0: BallPoint(0, 100.0, 0.0),
        9: BallPoint(9, 100.0, 140.0),
    }
    server = Detection(0, "player", 0.9, 90, -10, 110, 10, "test")
    receiver = Detection(9, "player", 0.9, 90, 130, 110, 150, "test")
    coordinates = {
        frame: ball_coordinate(frame, point.x or 0.0, point.y or 0.0)
        for frame, point in ball_points.items()
    }

    shots = detect_shots(
        ball_points,
        {0: [(4, server)], 9: [(7, receiver)]},
        coordinates,
        fps=30.0,
    )

    assert len(shots) == 1
    assert shots[0].frame == 0
    assert shots[0].receive_frame == 9
    assert round(shots[0].court_plane_speed_kmh, 2) == 168.0


def test_rejects_speed_when_ball_is_outside_the_court() -> None:
    ball_points = {
        frame: BallPoint(frame, 100.0, y)
        for frame, y in enumerate([30, 40, 50, 60, 70, 60, 50, 40, 20, 10, 0])
    }
    hitter = Detection(4, "player", 0.9, 90, 65, 110, 80, "test")
    coordinates = {
        frame: ball_coordinate(frame, point.x or 0.0, point.y or 0.0)
        for frame, point in ball_points.items()
    }
    coordinates[4] = replace(coordinates[4], in_bounds=False)

    shots = detect_shots(ball_points, {4: [(7, hitter)]}, coordinates, fps=10.0)

    assert shots == []


def test_skips_transit_when_player_contacts_are_missing() -> None:
    ball_points = {
        frame: BallPoint(frame, 100.0, y)
        for frame, y in enumerate([30, 40, 50, 60, 70, 60, 50, 40, 20, 10, 0])
    }
    coordinates = {
        frame: ball_coordinate(frame, point.x or 0.0, point.y or 0.0)
        for frame, point in ball_points.items()
    }

    shots = detect_shots(ball_points, {}, coordinates, fps=10.0)

    assert shots == []


def test_bounce_is_searched_only_before_the_receivers_contact() -> None:
    track = [
        (100, 0),
        (100, 15),
        (100, 30),
        (100, 45),
        (100, 60),
        (100, 75),
        (100, 90),
        (100, 105),
        (120, 105),
        (140, 105),
        (160, 105),
    ]
    ball_points = {frame: BallPoint(frame, x, y) for frame, (x, y) in enumerate(track)}
    hitter = Detection(0, "player", 0.9, 90, -10, 110, 10, "test")
    receiver = Detection(5, "player", 0.9, 90, 70, 110, 80, "test")
    player_detections = {0: [(1, hitter)], 5: [(2, receiver)]}
    coordinates = {
        frame: ball_coordinate(frame, point.x or 0.0, point.y or 0.0)
        for frame, point in ball_points.items()
    }

    shots = detect_shots(ball_points, player_detections, coordinates, fps=10.0)

    assert len(shots) == 1
    assert shots[0].receive_frame == 5
    assert shots[0].bounce_frame is None
