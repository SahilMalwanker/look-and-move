# MATLAB originals

The MATLAB scripts written for the lab's preparation tasks, kept as they were for reference.
The maintained, tested implementation is the Python package in [`src/lookandmove`](../src/lookandmove).

| File | Purpose |
| --- | --- |
| `rebel_prep_kinematics_vision.m` | DH table of the igus ReBeL with the 60 mm ball tool, symbolic $T_0^6$, Euler angles, the five test cases, the reduced pixel size and the symbolic image Jacobian $J_v$. |
| `camera_axes_from_features.m` | Projects the A→C feature vector onto the image directions of the robot axes (task 5.3.1). |

**Requirements:** MATLAB with the Symbolic Math Toolbox. `rebel_prep_kinematics_vision.m` defines its local function
`DH_transform` in the middle of the script, which needs MATLAB R2024a or newer; on older releases move the function to the end of the file.

## Notes from the port

- The DH table in the script is correct: all five test cases place the ball at (250, 10, 400) mm
  (`tests/test_robot.py`).
- The script prints the Euler angles in the order (roll, pitch, yaw). `ReBeL.pose_vector` returns
  $(x, y, z, A, B, C)$ with $R = R_z(A)\,R_y(B)\,R_x(C)$, so the first and last angle appear swapped between the two.
