import numpy as np
import pytest

from lookandmove import PinholeCamera, look_down_rotation, reduced_pixel_size


@pytest.fixture
def camera():
    return PinholeCamera.looking_down([0.24, -0.06, 0.905], np.deg2rad(-58.3))


def test_reduced_pixel_size_matches_the_lab_value():
    assert reduced_pixel_size() == pytest.approx(9.8241e-6, rel=1e-4)


def test_look_down_rotation_is_proper_and_points_down():
    R = look_down_rotation(0.7)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-12)


def test_optical_axis_hits_the_image_centre(camera):
    p = camera.position - np.array([0, 0, 0.5])
    np.testing.assert_allclose(camera.project(p), [320, 240], atol=1e-9)


def test_project_backproject_roundtrip(camera):
    pts = np.array([[0.3, 0.1, 0.4], [0.1, -0.2, 0.4], [0.25, 0.01, -0.25]])
    for p in pts:
        f = camera.project(p)
        depth = camera.to_camera(p)[2]
        np.testing.assert_allclose(camera.backproject(f, depth), p, atol=1e-12)
    assert camera.project(pts).shape == (3, 2)


def test_image_jacobian_is_the_derivative_of_the_pinhole_model(camera):
    c = np.array([0.03, -0.05, 0.5])
    eps = 1e-8
    lam = camera.focal_length
    J = np.empty((2, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        sp, sm = (c + d)[:2] * lam / (c + d)[2], (c - d)[:2] * lam / (c - d)[2]
        J[:, i] = (sp - sm) / (2 * eps)
    np.testing.assert_allclose(camera.image_jacobian(c), J, rtol=1e-6)


def test_scale_on_the_working_plane(camera):
    depth = camera.depth_of_plane(0.40)
    assert depth == pytest.approx(0.505)
    assert camera.metres_per_pixel(depth) == pytest.approx(depth * reduced_pixel_size() / 6e-3)
