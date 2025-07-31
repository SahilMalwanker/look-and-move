import numpy as np
import pytest

from lookandmove import closed_loop_poles, convergence_time, critical_angle, loop_gain, spectral_radius


@pytest.mark.parametrize("g", [0.05, 0.4, 1.2])
@pytest.mark.parametrize("theta", [0.0, 0.5, 1.2, 2.0])
def test_no_delay_has_a_closed_form(g, theta):
    assert spectral_radius(g, theta, 0) == pytest.approx(abs(1 - g * np.exp(1j * theta)))


@pytest.mark.parametrize("g", [0.05, 0.3, 1.0, 1.5])
def test_no_delay_boundary_is_g_equals_two_cos_theta(g):
    theta_c = critical_angle(g, 0)
    assert 2 * np.cos(theta_c) == pytest.approx(g, abs=1e-5)


def test_small_gain_tolerates_almost_ninety_degrees():
    assert np.rad2deg(critical_angle(loop_gain(1.0, 17.5), delay_frames=1)) == pytest.approx(85.1, abs=0.2)
    assert np.rad2deg(critical_angle(1e-4, 0)) == pytest.approx(90.0, abs=0.01)


def test_delay_shrinks_the_stable_region():
    for g in (0.1, 0.3, 0.6):
        assert critical_angle(g, 2) < critical_angle(g, 1) < critical_angle(g, 0)


def test_one_frame_delay_goes_unstable_at_unit_gain():
    assert spectral_radius(0.99, 0.0, 1) < 1.0 < spectral_radius(1.01, 0.0, 1)


def test_poles_and_convergence_time():
    assert len(closed_loop_poles(0.2, 0.3, 2)) == 3
    t = convergence_time(0.1, 0.0, 10.0, 0, reduction=np.e)
    assert t == pytest.approx(1.0 / (10.0 * -np.log(0.9)))
    assert convergence_time(1.0, np.deg2rad(120), 10.0) == float("inf")
