"""Numbers from the lab: test poses, taught positions and the image features measured on the real setup."""

from __future__ import annotations

import numpy as np

# Prep task 4.1d: five joint configurations (deg).
TEST_CASES_DEG = np.array(
    [
        [8.88, -127.32, 171.91, 167.63, 46.09, 8.65],
        [-45.73, -107.44, 161.60, -89.57, 90.59, -54.16],
        [50.06, -104.34, 159.33, 87.09, 94.15, 54.89],
        [37.53, -104.47, 183.48, 50.94, 91.77, 125.39],
        [-33.41, -107.24, 185.63, -51.75, 88.23, -127.44],
    ]
)

# Feature (pixel) of the ball measured at each test case (task 5.2, ROI active).
# The fifth x reading repeats the pose's A angle (45.01) and is treated as a transcription slip.
TEST_CASE_FEATURES = np.array([[61.0, 191.0], [67.0, 183.0], [61.0, 183.0], [59.0, 181.0], [np.nan, 180.0]])

# Task 5.3.1: robot positions (m, base frame) and the measured features (px).
P_A = np.array([0.250, 0.010, 0.400])
P_B = np.array([0.400, 0.010, 0.400])
P_C = np.array([0.250, 0.150, 0.400])
F_A = np.array([251.0, 190.0])
F_B = np.array([345.0, 35.0])
F_C = np.array([116.0, 105.0])

# Tool orientation (A, B, C) used throughout the lab, deg.
ORIENTATION_DEG = np.array([180.0, -90.0, 0.0])
