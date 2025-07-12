# Theory notes

## 1. Pinhole camera and image coordinates

A point $c = (c_x, c_y, c_z)$ in camera coordinates (origin in the pinhole, $z$ along the optical axis) lands on the sensor at

$$
s = \frac{\lambda}{c_z}\begin{bmatrix} c_x \\ c_y \end{bmatrix},\qquad \lambda = 6\ \text{mm}.
$$

Pixels follow from the pixel pitch and the image centre. The 4056-pixel-wide sensor is read out at 640 columns, so one pixel
covers $d_{Px,red} = 1.55\ \mu\text{m}\cdot 4056/640 = 9.82\ \mu\text{m}$:

$$
f^{(img)} = \frac{s}{d_{Px,red}} + \begin{bmatrix}320\\240\end{bmatrix},\qquad f^{(img)} = f + O_{ROI}.
$$

The camera software reports $f$ relative to the region of interest. Moving the ROI changes $f$ but not $f^{(img)}$.

## 2. Image Jacobian

$$
J_v = \frac{\partial s}{\partial c} = \begin{bmatrix} \lambda/c_z & 0 & -c_x\lambda/c_z^2 \\ 0 & \lambda/c_z & -c_y\lambda/c_z^2 \end{bmatrix}.
$$

For motion in a plane at known depth only the left $2\times2$ block matters, $J_v = (\lambda/c_z)\,I$.

## 3. From a pixel error to joint velocities

Chaining the image Jacobian, the camera rotation and the robot Jacobian (lab eq. 10):

$$
\Delta q = J_0^{-1}\,A_0^{cam}\,J_v^{-1}\,\Delta f\, d_{Px,red},\qquad \dot q_s = K_P\,\Delta q ,
$$

where $A_0^{cam}$ holds the camera axes in robot coordinates (it maps a camera-frame displacement into the base frame).

Vision only provides the $x$–$y$ part of the Cartesian motion. In the simulation the robot keeps the ball height and the
tool orientation with a small Cartesian P-term, which the lab formula leaves implicit.

## 4. Why a P controller is enough

The joint velocity loops turn the robot into an integrator from velocity set-point to position. The outer loop therefore
already contains an integrator, and a constant target is reached without steady-state error by a pure P controller.

## 5. Hand-eye calibration from three points

Moving the robot from A to B is a pure $+x_0$ step, from A to C a pure $+y_0$ step. The two image vectors give the robot axes
as seen by the camera:

$$
x_0^{(img)} = \frac{f_B - f_A}{\lVert f_B - f_A\rVert},\qquad y_0^{(img)} = \frac{f_C - f_A}{\lVert f_C - f_A\rVert},\qquad
A_{cam}^{0} = \big[x_0^{(img)}\ \ y_0^{(img)}\ \ x_0^{(img)}\times y_0^{(img)}\big],
$$

projected onto the nearest rotation. These columns are the robot axes in camera coordinates; the camera axes in robot
coordinates are $A_0^{cam} = (A_{cam}^{0})^{\mathsf T}$. The image scale gives the depth, $r = d_{rob}/d_{img}$ and
$c_z = \lambda\, r / d_{Px,red}$. Back-projecting $f_A$ at depth $c_z$ gives $p_{cam,A}^{(cam)}$, and the camera position is
$p_{0,cam}^{(0)} = P_A - A_0^{cam}\,p_{cam,A}^{(cam)}$. With more points, `calibrate_least_squares` fits the same model
(a mirrored 2-D similarity, because a camera looking down sees the robot's $x$–$y$ plane flipped) in closed form.

## 6. Stability under calibration errors

Assume the inner velocity loops are fast. If the controller believes the camera yaw is off by $\Delta\theta$ and the depth by a
factor $k$, one camera period $T$ moves the feature by $g\,R(\Delta\theta)\,e$ with $g = K_P\,T\,k$. Written as a complex number,
with a processing delay of $d$ frames, the image error obeys

$$
e_{k+1} = e_k - g\,e^{i\Delta\theta}\,e_{k-d},\qquad z^{d+1} - z^{d} + g\,e^{i\Delta\theta} = 0 .
$$

* $d = 0$: $|1 - g e^{i\Delta\theta}| < 1 \iff g < 2\cos\Delta\theta$. For small gains, any yaw error below 90° converges.
* Each frame of latency shrinks the region. With $d = 1$ the calibrated loop goes unstable at $g = 1$.
* The convergence rate scales with $\cos\Delta\theta$ and the path in the image is a spiral with pitch angle $\Delta\theta$.
* A wrong depth or pixel size only scales $g$: it changes speed, not convergence, unless $g$ gets large.

In the simulation the 30 ms velocity loop adds roughly half a frame of extra lag, which is why the simulated boundary at high
gain lies between the one- and two-frame curves of the stability map.
