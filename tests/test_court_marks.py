"""Marking the court from any four or more landmarks the camera sees (#90)."""

import json

import cv2
import numpy as np
import pytest

from courtvision.calibration.homography import project_image_points
from courtvision.calibration.landmarks import landmark_by_name
from courtvision.evaluation.court import (
    LANDMARK_NAMES,
    court_estimate,
    court_lines,
    court_residuals,
)
from courtvision.evaluation.labels import Segment, load_labels, save_labels
from courtvision.evaluation.session import LabelSession
from tests.test_labelling import COURT_POINTS, labelled_session, make_labels

# What a phone on the back glass sees: the far half, the net's ends and the near
# T, but not the near service line's ends.
BACK_GLASS = (
    "far_left_service_intersection",
    "far_right_service_intersection",
    "far_service_center",
    "near_service_center",
    "left_net_point",
    "right_net_point",
)
CAMERA = court_estimate(COURT_POINTS).template_to_image  # template to image


def seen(names, camera=CAMERA) -> dict[str, list[float]]:
    """Where the camera sees each landmark, in image pixels."""
    template = np.float32([landmark_by_name("padel", name).template for name in names])
    image = cv2.perspectiveTransform(template.reshape(-1, 1, 2), camera)
    return {
        name: point.tolist()
        for name, point in zip(names, image.reshape(-1, 2), strict=True)
    }


def test_the_court_fits_from_named_landmarks_without_the_near_service_line() -> None:
    estimate = court_estimate(seen(BACK_GLASS))

    near_corner = seen(["near_left_baseline_corner"])["near_left_baseline_corner"]
    on_court = project_image_points(np.float32([near_corner]), estimate)
    assert on_court[0] == pytest.approx([0, 1], abs=1e-4)
    assert max(court_residuals(seen(BACK_GLASS)).values()) < 0.01


def test_with_five_points_a_bad_click_stands_out() -> None:
    points = seen(BACK_GLASS)
    points["far_service_center"][0] += 15

    residuals = court_residuals(points)

    assert max(residuals, key=residuals.get) == "far_service_center"
    assert residuals["far_service_center"] > 5


def test_three_points_on_one_line_and_one_more_cannot_place_the_court() -> None:
    on_the_far_service_line = BACK_GLASS[:4]

    with pytest.raises(ValueError, match="three on one line"):
        court_estimate(seen(on_the_far_service_line))
    with pytest.raises(ValueError):
        Segment(0, 10, court_points=seen(on_the_far_service_line))


def test_unknown_landmarks_and_too_few_points_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        court_estimate(seen(BACK_GLASS) | {"service_box_corner": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least 4"):
        court_estimate(seen(BACK_GLASS[:3]))


def test_named_court_points_save_and_load(tmp_path) -> None:
    labels = make_labels()
    labels.segments[0].court_points = seen(BACK_GLASS)
    path = tmp_path / "labels.json"

    save_labels(labels, path)

    assert load_labels(path) == labels


def test_version_2_labels_load_with_the_service_line_ends(tmp_path) -> None:
    labels = make_labels()
    labels.segments[0].court_points = COURT_POINTS
    path = tmp_path / "labels.json"
    save_labels(labels, path)
    data = json.loads(path.read_text())
    data["schema_version"] = 2
    path.write_text(json.dumps(data))

    loaded = load_labels(path)

    assert loaded.segments[0].court_points == COURT_POINTS
    assert court_estimate(COURT_POINTS).landmark_names == LANDMARK_NAMES


def mark_back_glass_court(session: LabelSession) -> None:
    points = seen(BACK_GLASS)
    session.key("k")
    session.click(*points["far_left_service_intersection"])
    session.click(*points["far_right_service_intersection"])
    session.key("x")  # near service line, right end: out of the picture
    session.key("x")  # and its left end
    session.click(*points["far_service_center"])
    session.click(*points["near_service_center"])
    session.click(*points["left_net_point"])
    session.click(*points["right_net_point"])
    session.key("k")


def test_the_labeller_skips_landmarks_out_of_the_picture() -> None:
    session = labelled_session(frame=10)

    mark_back_glass_court(session)

    assert session.segment.court_points == seen(BACK_GLASS)
    assert "6 points" in session.message
    assert session.segment.events == []  # x skipped landmarks, deleted nothing


def test_saving_with_fewer_than_four_points_keeps_marking() -> None:
    session = labelled_session(frame=10)
    session.key("k")
    session.click(400, 250)
    session.click(880, 250)

    session.key("k")

    assert session.segment.court_points is None
    assert session.court_clicks is not None
    assert "at least 4" in session.message


def test_the_court_is_copied_from_the_nearest_earlier_segment() -> None:
    session = LabelSession(make_labels(cuts=(50, 100, 150)), frame=10)
    session.key("l")
    mark_back_glass_court(session)
    session.go_to(120)
    session.key("l")

    session.key("k")
    session.key("p")

    assert session.segment.court_points == session.labels.segments[0].court_points
    assert session.segment.court_points is not session.labels.segments[0].court_points
    assert session.court_clicks is None


def test_lines_near_or_behind_the_camera_are_cut_off() -> None:
    # Depth falls toward the near baseline, which is behind the camera.
    camera = np.float64([[1000, 0, 0], [0, 1000, 0], [0, -1, 0.9]])
    points = seen(BACK_GLASS, camera)

    lines = court_lines(points)
    estimate = court_estimate(points)

    near_baseline, right_side = lines[2], lines[1]
    assert len(near_baseline) == 0
    near_end = project_image_points(right_side[1:], estimate)[0]
    # Cut where depth is 5 % of the net's: 0.9 - y = 0.05 * 0.4.
    assert near_end == pytest.approx([1, 0.88], abs=1e-3)


def test_u_while_marking_steps_back_one_landmark_and_leaves_the_labels() -> None:
    session = labelled_session(frame=10)
    session.key("1")  # a hit, which u must not undo while marking
    events = list(session.segment.events)
    assert len(events) == 1
    points = seen(BACK_GLASS)
    session.key("k")
    session.click(*points["far_left_service_intersection"])
    session.click(900, 100)  # a bad click on the far service line's right end
    session.key("x")  # near service line, right end

    session.key("u")  # back to the skipped landmark
    session.key("u")  # back to the bad click, which is removed

    assert session.court_clicks == {
        "far_left_service_intersection": points["far_left_service_intersection"]
    }
    assert "far service line, right end" in session.message
    assert session.segment.events == events
