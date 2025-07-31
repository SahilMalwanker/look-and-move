import numpy as np
import pytest

from lookandmove import (
    IdealMeasurement,
    PinholeCamera,
    ReBeL,
    Renderer,
    ServoConfig,
    VisionMeasurement,
    lab,
    pose_matrix,
    simulate,
)

ROBOT = ReBeL()
TRUE = PinholeCamera.looking_down([0.245, -0.060, 0.905], np.deg2rad(-58.3))
Q0 = ROBOT.ik(pose_matrix(0.33, -0.12, 0.40, *np.deg2rad(lab.ORIENTATION_DEG)), np.deg2rad(lab.TEST_CASES_DEG[0]))
TARGET = TRUE.project([0.20, 0.08, 0.40])


def run(yaw_error_deg, gain=3.0, duration=8.0, measure=None):
    model = PinholeCamera.looking_down(TRUE.position, np.deg2rad(-58.3 + yaw_error_deg))
    meas = measure or IdealMeasurement(TRUE, noise_px=0.25, seed=1)
    return simulate(ROBOT, Q0, TARGET, model, meas, ServoConfig(gain=gain, duration=duration))


def distance_from_line(start, end, points):
    d = (end - start) / np.linalg.norm(end - start)
    rel = points - start
    return np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0])


def test_calibrated_camera_converges_on_a_straight_image_path():
    res = run(0.0)
    assert res.converged and res.settling_time() < 4.0
    assert distance_from_line(res.feature_true[0], res.target, res.feature_true).max() < 10.0


def test_moderate_yaw_error_still_converges_but_slower_and_curved():
    calibrated, rotated = run(0.0), run(45.0)
    assert rotated.converged
    assert rotated.settling_time() > calibrated.settling_time()
    assert distance_from_line(rotated.feature_true[0], TARGET, rotated.feature_true).max() > 40.0


def test_yaw_error_beyond_ninety_degrees_loses_the_ball():
    res = run(110.0)
    assert res.lost and not res.converged


def test_height_and_orientation_are_held():
    res = run(30.0)
    assert np.abs(res.tcp[:, 2] - 0.40).max() < 5e-3
    assert abs(res.tcp[-1, 2] - 0.40) < 1e-4


@pytest.mark.slow
def test_full_vision_pipeline_tracks_the_ball():
    renderer = Renderer(TRUE)
    meas = VisionMeasurement(renderer, target_point=TRUE.backproject(TARGET, TRUE.depth_of_plane(-0.252)), seed=3)
    res = run(0.0, duration=2.0, measure=meas)
    assert np.abs(res.feature - res.feature_true).max() < 0.5
    assert res.error_px[-1] < 0.3 * res.error_px[0]
