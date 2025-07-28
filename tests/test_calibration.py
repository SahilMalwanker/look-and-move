import numpy as np
import pytest

from lookandmove import PinholeCamera, calibrate_least_squares, calibrate_three_points, lab

TRUE = PinholeCamera.looking_down([0.25, -0.05, 0.90], np.deg2rad(-58.0))


def test_three_point_calibration_recovers_a_known_camera():
    pts = [lab.P_A, lab.P_B, lab.P_C]
    f = TRUE.project(np.array(pts))
    cal = calibrate_three_points(*f, *pts)
    assert np.rad2deg(cal.yaw) == pytest.approx(-58.0, abs=1e-9)
    assert cal.depth == pytest.approx(0.50, abs=1e-12)
    np.testing.assert_allclose(cal.position, TRUE.position, atol=1e-12)
    np.testing.assert_allclose(cal.rotation, TRUE.rotation, atol=1e-12)
    assert np.abs(cal.residuals_px).max() < 1e-9


def test_roi_origin_only_shifts_the_camera_position():
    pts = [lab.P_A, lab.P_B, lab.P_C]
    f = TRUE.project(np.array(pts))
    roi = np.array([180.0, 20.0])
    cal = calibrate_three_points(*(f - roi), *pts, roi_origin=roi)
    np.testing.assert_allclose(cal.position, TRUE.position, atol=1e-12)
    wrong = calibrate_three_points(*(f - roi), *pts)
    assert cal.yaw == pytest.approx(wrong.yaw)
    assert np.linalg.norm(wrong.position - TRUE.position) > 0.1


def test_least_squares_calibration_with_noise():
    rng = np.random.default_rng(0)
    grid = np.array([[x, y, 0.40] for x in (0.15, 0.25, 0.35) for y in (-0.10, 0.0, 0.10)])
    f = TRUE.project(grid) + rng.normal(scale=0.3, size=(len(grid), 2))
    cal = calibrate_least_squares(f, grid)
    assert np.rad2deg(cal.yaw) == pytest.approx(-58.0, abs=0.2)
    assert cal.depth == pytest.approx(0.50, abs=2e-3)
    np.testing.assert_allclose(cal.position, TRUE.position, atol=3e-3)
    assert np.sqrt((cal.residuals_px**2).sum(axis=1).mean()) < 0.6


def test_lab_measurements():
    cal = calibrate_three_points(lab.F_A, lab.F_B, lab.F_C, lab.P_A, lab.P_B, lab.P_C)
    assert np.rad2deg(cal.yaw) == pytest.approx(-58.3, abs=0.1)
    assert cal.metres_per_pixel * 1e3 == pytest.approx(0.8275, abs=1e-4)
    assert cal.depth * 1e3 == pytest.approx(505.4, abs=0.1)
    np.testing.assert_allclose(cal.position * 1e3, [244.8, -60.3, 905.4], atol=0.1)
    assert abs(np.rad2deg(cal.axis_skew)) < 1.0
    assert cal.metres_per_pixel_y / cal.metres_per_pixel == pytest.approx(1.06, abs=0.01)
