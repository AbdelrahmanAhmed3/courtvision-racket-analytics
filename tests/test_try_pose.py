import numpy as np

from scripts.try_pose import Poses, drop_still_boxes, near_team, touches_edge

JACKET = [100.0, 100.0, 140.0, 200.0, 0.6]
PLAYER = [500.0, 100.0, 560.0, 260.0, 0.9]


def moved(box, dx):
    return [box[0] + dx, box[1], box[2] + dx, box[3], box[4]]


def test_a_box_in_the_same_place_most_of_the_time_is_dropped() -> None:
    people = {frame: [JACKET, moved(PLAYER, 20 * frame)] for frame in range(11)}

    kept = drop_still_boxes(people, step=1, reach=5)

    assert kept[5] == [moved(PLAYER, 100)]


def test_a_player_who_stands_still_for_a_moment_is_kept() -> None:
    walking = {frame: [moved(PLAYER, 30 * frame)] for frame in range(8)}
    standing = {frame: [moved(PLAYER, 240)] for frame in range(8, 11)}

    kept = drop_still_boxes(walking | standing, step=1, reach=10)

    assert kept[9] == [moved(PLAYER, 240)]


def test_a_still_box_found_in_only_some_frames_is_still_dropped() -> None:
    people = {frame: [JACKET] if frame % 4 else [] for frame in range(31)}

    kept = drop_still_boxes(people, step=1, reach=10)

    assert kept[15] == []


def test_a_still_box_that_jitters_by_a_few_pixels_is_still_dropped() -> None:
    jitter = [104.0, 106.0, 139.0, 198.0, 0.5]
    people = {frame: [JACKET if frame % 2 else jitter] for frame in range(11)}

    kept = drop_still_boxes(people, step=1, reach=5)

    assert kept[5] == []


def test_a_box_touching_the_frame_is_cut() -> None:
    assert touches_edge([0, 10, 100, 100], 1920, 1080)
    assert touches_edge([10, 10, 100, 1079], 1920, 1080)
    assert not touches_edge([10, 10, 100, 100], 1920, 1080)


def test_the_near_team_is_the_two_most_confident_boxes_found_clearly() -> None:
    confidences = [0.9, 0.5, 0.7, 0.8]
    boxes = np.array([[0, 0, 10, 10, c] for c in confidences], np.float32)
    poses = Poses(
        np.array([3, 3, 3, 3]), boxes, np.zeros((4, 17, 2)), np.zeros((4, 17)), 0.0
    )

    team = near_team(poses)

    assert team.boxes[:, 4].tolist() == [np.float32(0.9), np.float32(0.8)]
