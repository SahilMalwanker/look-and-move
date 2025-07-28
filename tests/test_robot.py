import numpy as np
import pytest

from lookandmove import ReBeL, lab, pose_matrix


def test_all_five_lab_test_cases_put_the_ball_at_the_same_point():
    robot = ReBeL()
    for q in np.deg2rad(lab.TEST_CASES_DEG):
        np.testing.assert_allclose(robot.tcp(q), [0.250, 0.010, 0.400], atol=1e-4)


def test_test_cases_differ_in_orientation():
    robot = ReBeL()
    angles = np.array([robot.pose_vector(q)[3:] for q in np.deg2rad(lab.TEST_CASES_DEG)])
    pitch = np.rad2deg(angles[:, 1])
    np.testing.assert_allclose(pitch[:3], -90.0, atol=0.02)
    np.testing.assert_allclose(pitch[3:], -30.0, atol=0.02)


def test_jacobian_matches_finite_differences():
    robot = ReBeL()
    q = np.deg2rad(lab.TEST_CASES_DEG[1])
    J = robot.jacobian(q)
    eps = 1e-7
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        v = (robot.tcp(q + dq) - robot.tcp(q - dq)) / (2 * eps)
        np.testing.assert_allclose(J[:3, i], v, atol=1e-6)


@pytest.mark.parametrize("xyz", [(0.40, 0.01, 0.40), (0.20, 0.08, 0.40), (0.33, -0.12, 0.40)])
def test_numerical_ik_reaches_lab_style_poses(xyz):
    robot = ReBeL()
    T = pose_matrix(*xyz, *np.deg2rad(lab.ORIENTATION_DEG))
    q = robot.ik(T, np.deg2rad(lab.TEST_CASES_DEG[0]))
    np.testing.assert_allclose(robot.fk(q), T, atol=1e-8)
