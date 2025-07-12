"""Five-minute tour of the API (the output is quoted in the README)."""

from __future__ import annotations

import numpy as np
from _common import CALIBRATION, Q_START, ROBOT, TARGET_PX, TRUE_CAMERA

from lookandmove import (
    IdealMeasurement,
    PinholeCamera,
    ServoConfig,
    calibrate_three_points,
    critical_angle,
    lab,
    loop_gain,
    reduced_pixel_size,
    simulate,
)

print(f"reduced pixel size: {reduced_pixel_size() * 1e6:.2f} um")
print("ball centre of the five test cases [mm]:")
for q in np.deg2rad(lab.TEST_CASES_DEG):
    print("   ", np.round(ROBOT.tcp(q) * 1e3, 1))

cal = calibrate_three_points(lab.F_A, lab.F_B, lab.F_C, lab.P_A, lab.P_B, lab.P_C)
print(f"camera yaw {np.rad2deg(cal.yaw):.1f} deg, {cal.metres_per_pixel * 1e3:.4f} mm/px, depth {cal.depth * 1e3:.1f} mm")
print(f"camera position {np.round(cal.position * 1e3, 1)} mm")

for yaw_error in (0, 45, 85):
    belief = PinholeCamera.looking_down(TRUE_CAMERA.position, CALIBRATION.yaw + np.deg2rad(yaw_error))
    res = simulate(ROBOT, Q_START, TARGET_PX, belief, IdealMeasurement(TRUE_CAMERA), ServoConfig(gain=1.0, duration=12.0))
    state = "lost the ball" if res.lost else f"settled in {res.settling_time():.1f} s"
    print(f"K_P = 1, yaw error {yaw_error:2d} deg: {state}")

print(f"analytic limit for K_P = 1: {np.rad2deg(critical_angle(loop_gain(1.0, 17.5), delay_frames=1)):.1f} deg")
