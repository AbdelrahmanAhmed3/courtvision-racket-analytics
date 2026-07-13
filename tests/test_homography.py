import numpy as np

from courtvision.geometry.homography import (
    bottom_center,
    estimate_homography,
    project_points,
    reprojection_error,
)


def test_bottom_center_uses_player_feet() -> None:
    assert bottom_center(10, 20, 30, 60) == (20, 60)


def test_ransac_homography_projects_square() -> None:
    image_points = np.array(
        [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50]],
        dtype=np.float32,
    )
    court_points = np.array(
        [[0, 0], [10, 0], [10, 20], [0, 20], [5, 10]],
        dtype=np.float32,
    )

    homography, inliers = estimate_homography(image_points, court_points)
    projected = project_points(np.array([[25, 25]], dtype=np.float32), homography)

    assert inliers.all()
    assert np.allclose(projected[0], [2.5, 5.0], atol=1e-4)
    assert reprojection_error(image_points, court_points, homography, inliers) < 1e-4
