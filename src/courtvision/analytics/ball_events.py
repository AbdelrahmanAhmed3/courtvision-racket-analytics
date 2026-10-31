"""Impacts, bounces and wall rebounds from the ball track, decided by a small model.

Candidates (``ball_track.event_candidates``) are moments where something may
have happened. Each gets the numbers in ``FEATURES`` from the ball track, the
player boxes and the court; a logistic regression trained on the labelled
rallies (``scripts/train_event_classifier.py``) says which event it was, if
any. The model ships as JSON (weights, feature means and scales, fill-in values
for missing numbers) and is applied with numpy, so running it needs no
scikit-learn. Results: reports/ball_events.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from pathlib import Path

import numpy as np

from courtvision.analytics.ball_track import (
    event_candidates,
    split_into_tracklets,
    turn_angle,
)
from courtvision.calibration.homography import HomographyEstimate, project_image_points
from courtvision.detectors.base import Detection
from courtvision.detectors.tracknet_adapter import BallPoint

PlayersByFrame = Mapping[int, Sequence[tuple[int, Detection]]]

FEATURES = (
    "speed_in",  # px per frame, over the last 4 positions up to 10 frames before
    "speed_out",  # px per frame, over the first 4 positions up to 10 frames after
    "turn_degrees",  # between the directions before and after
    "dy_in",  # vertical movement before, px per frame (positive: down the image)
    "dy_out",
    "hidden_frames",  # frames within 3 of the candidate without a ball
    "player_distance",  # ball to nearest player box, in box heights
    "box_y",  # height in that box: 0 at the head, 1 at the feet
    "box_x",  # across that box: 0 at its middle, ±0.5 at its sides
    "court_y",  # along the court, 0 at the far baseline, 1 at the near one
    "net_distance",  # |court_y - 0.5|
    "on_court",  # 1 if the ball projects inside the court (not beside it)
    "crosses_net",  # 1 if the ball is on the other half 3-20 frames later
    "court_y_change",  # court_y 3-20 frames later minus 3-15 frames before
)
NONE = "none"
DEFAULT_MODEL_PATH = Path(__file__).with_name("ball_events_model.json")


@dataclass(frozen=True)
class Candidate:
    frame: int
    features: tuple[float, ...]  # in the order of FEATURES; nan when unknown
    nearest_track_id: int | None  # the player nearest the ball, if any is seen


@dataclass(frozen=True)
class BallEvent:
    kind: str  # "impact", "bounce" or "wall_rebound"
    frame: int
    track_id: int | None  # for an impact, the player nearest the ball


@dataclass(frozen=True)
class EventModel:
    """Logistic regression over FEATURES, as exported by the training script."""

    classes: tuple[str, ...]
    fill: np.ndarray  # replaces a missing number (the training median)
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray  # one row of weights per class
    intercept: np.ndarray

    def predict(self, rows: np.ndarray) -> list[str]:
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        rows = np.where(np.isnan(rows), self.fill, rows)
        scores = ((rows - self.mean) / self.scale) @ self.coef.T + self.intercept
        return [self.classes[i] for i in scores.argmax(axis=1)]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> EventModel:
        data = json.loads(Path(path).read_text())
        if tuple(data["features"]) != FEATURES:
            raise ValueError("The model was trained on different features")
        return cls(
            classes=tuple(data["classes"]),
            **{key: np.array(data[key], dtype=float) for key in _ARRAYS},
        )

    def save(self, path: str | Path, about: str = "") -> None:
        data = {
            "about": about,
            "features": list(FEATURES),
            "classes": list(self.classes),
        }
        data |= {key: getattr(self, key).tolist() for key in _ARRAYS}
        Path(path).write_text(json.dumps(data, indent=1) + "\n")


_ARRAYS = ("fill", "mean", "scale", "coef", "intercept")


def detect_ball_events(
    ball_points: Iterable[BallPoint],
    players_by_frame: PlayersByFrame,
    estimate: HomographyEstimate,
    model: EventModel | None = None,
) -> list[BallEvent]:
    """Every candidate the model calls an impact, bounce or wall rebound."""
    candidates = candidate_features(ball_points, players_by_frame, estimate)
    if not candidates:
        return []
    model = model or EventModel.load()
    kinds = model.predict(np.array([c.features for c in candidates]))
    return [
        BallEvent(kind, c.frame, c.nearest_track_id if kind == "impact" else None)
        for c, kind in zip(candidates, kinds, strict=True)
        if kind != NONE
    ]


def candidate_features(
    ball_points: Iterable[BallPoint],
    players_by_frame: PlayersByFrame,
    estimate: HomographyEstimate,
) -> list[Candidate]:
    """FEATURES for every event candidate in a camera shot."""
    points = list(ball_points)
    clean = {
        point.frame: point
        for tracklet in split_into_tracklets(points)
        if len(tracklet) >= 3
        for point in tracklet
    }

    def court(point: BallPoint) -> tuple[float, float]:
        x, y = project_image_points(np.float32([[point.x, point.y]]), estimate)[0]
        return float(x), float(y)

    candidates = []
    for frame in event_candidates(points, players_by_frame):
        values = dict.fromkeys(FEATURES, np.nan)
        # After a hidden gap the nearest positions can be several frames away.
        before = [clean[f] for f in range(frame - 10, frame + 1) if f in clean][-4:]
        after = [clean[f] for f in range(frame, frame + 11) if f in clean][:4]
        incoming, outgoing = _velocity(before), _velocity(after)
        if incoming:
            values["speed_in"], values["dy_in"] = hypot(*incoming), incoming[1]
        if outgoing:
            values["speed_out"], values["dy_out"] = hypot(*outgoing), outgoing[1]
        if incoming and outgoing:
            values["turn_degrees"] = turn_angle(incoming, outgoing)
        values["hidden_frames"] = sum(
            f not in clean for f in range(frame - 3, frame + 4)
        )
        track_id = None
        nearby = [clean[f] for f in range(frame - 4, frame + 5) if f in clean]
        if nearby:
            ball = min(nearby, key=lambda point: abs(point.frame - frame))
            if players_by_frame.get(ball.frame):
                track_id = _player_features(values, ball, players_by_frame[ball.frame])
            court_x, court_y = court(ball)
            values["court_y"], values["net_distance"] = court_y, abs(court_y - 0.5)
            inside = -0.05 <= court_x <= 1.05 and -0.05 <= court_y <= 1.05
            values["on_court"] = float(inside)
            early = [
                court(clean[f])[1] for f in range(frame - 15, frame - 2) if f in clean
            ]
            late = [
                court(clean[f])[1] for f in range(frame + 3, frame + 21) if f in clean
            ]
            if early and late:
                early_y, late_y = float(np.median(early)), float(np.median(late))
                values["crosses_net"] = float((early_y - 0.5) * (late_y - 0.5) < 0)
                values["court_y_change"] = late_y - early_y
        features = tuple(float(values[name]) for name in FEATURES)
        candidates.append(Candidate(frame, features, track_id))
    return candidates


def _velocity(points: list[BallPoint]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    first, last = points[0], points[-1]
    frames = last.frame - first.frame
    return ((last.x - first.x) / frames, (last.y - first.y) / frames)


def _player_features(
    values: dict, ball: BallPoint, players: Sequence[tuple[int, Detection]]
) -> int:
    """Fill the player features from the box nearest the ball; return its ID."""

    def gap(box: Detection) -> float:
        return hypot(
            max(box.x1 - ball.x, 0, ball.x - box.x2),
            max(box.y1 - ball.y, 0, ball.y - box.y2),
        )

    track_id, box = min(players, key=lambda player: gap(player[1]))
    height, width = box.y2 - box.y1, box.x2 - box.x1
    values["player_distance"] = gap(box) / height
    values["box_y"] = (ball.y - box.y1) / height
    values["box_x"] = (ball.x - (box.x1 + box.x2) / 2) / width
    return track_id


# The same exchange: a deep lob takes about 2 to 2.5 s to reach the other team.
MAX_EXCHANGE_SECONDS = 3.0


@dataclass(frozen=True)
class Shot:
    """One impact and what followed it, up to the next impact if any.

    Court distance and speed are measured on the floor plane between the ball's
    projected positions at the impact and at the receive (or the bounce when
    nobody receives). The ball's height is not modelled, so they are an
    approximation, not the ball's true speed.
    """

    frame: int  # the impact
    hitter_track_id: int | None
    receiver_track_id: int | None  # who hits it next, if anyone does
    receive_frame: int | None
    bounce_frame: int | None  # the first bounce after the impact, if any
    distance_m: float | None
    duration_seconds: float | None
    court_speed_kmh: float | None  # approximate: floor-plane distance over time


# The longest a ball can travel along a padel court floor: corner to corner.
COURT_DIAGONAL_M = hypot(10.0, 20.0)


def has_credible_speed(shot: Shot, teams: Mapping[int, str] | None = None) -> bool:
    """Whether a shot's speed is worth showing, or comes from a wrong pair.

    Partners never hit in turn, so a shot "received" by the hitter's own team
    pairs two impacts of which one is false; so does a receiving impact with no
    player near it. A floor distance longer than the court's diagonal means the
    ball was in the air and projected far off the court, or the impacts are
    mismatched. Either way the speed means nothing. A shot timed to its bounce,
    with nobody receiving it, keeps its speed.
    """
    if shot.court_speed_kmh is None:
        return False
    if shot.receive_frame is not None and shot.receiver_track_id is None:
        return False
    if shot.distance_m is not None and shot.distance_m > COURT_DIAGONAL_M:
        return False
    teams = teams or {}
    hitter = teams.get(shot.hitter_track_id)
    return hitter is None or hitter != teams.get(shot.receiver_track_id)


def shots_from_events(
    events: Iterable[BallEvent],
    ball_points: Iterable[BallPoint],
    estimate: HomographyEstimate,
    fps: float,
    court_width_m: float = 10.0,
    court_length_m: float = 20.0,
) -> list[Shot]:
    """A shot for every impact, paired with the next impact of the exchange."""
    events = sorted(events, key=lambda event: event.frame)
    impacts = [event for event in events if event.kind == "impact"]
    bounces = [event.frame for event in events if event.kind == "bounce"]
    visible = {point.frame: point for point in ball_points if point.visible}
    longest = round(MAX_EXCHANGE_SECONDS * fps)

    def court_position(frame: int) -> tuple[float, float] | None:
        nearby = [visible[f] for f in range(frame - 4, frame + 5) if f in visible]
        if not nearby:
            return None
        ball = min(nearby, key=lambda point: abs(point.frame - frame))
        x, y = project_image_points(np.float32([[ball.x, ball.y]]), estimate)[0]
        return float(x) * court_width_m, float(y) * court_length_m

    shots = []
    for index, impact in enumerate(impacts):
        after = impacts[index + 1] if index + 1 < len(impacts) else None
        if after is not None and after.frame - impact.frame > longest:
            after = None  # a new point, not this exchange
        end = after.frame if after else impact.frame + longest
        bounce = next((f for f in bounces if impact.frame < f < end), None)
        target = after.frame if after else bounce
        distance = duration = speed = None
        start_position = court_position(impact.frame)
        end_position = court_position(target) if target is not None else None
        if start_position and end_position and target > impact.frame:
            distance = hypot(
                end_position[0] - start_position[0],
                end_position[1] - start_position[1],
            )
            duration = (target - impact.frame) / fps
            speed = distance / duration * 3.6
        shots.append(
            Shot(
                frame=impact.frame,
                hitter_track_id=impact.track_id,
                receiver_track_id=after.track_id if after else None,
                receive_frame=after.frame if after else None,
                bounce_frame=bounce,
                distance_m=distance,
                duration_seconds=duration,
                court_speed_kmh=speed,
            )
        )
    return shots
