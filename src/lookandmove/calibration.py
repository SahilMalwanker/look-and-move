"""Hand-eye calibration of a downward-looking camera from robot positions and image features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import PinholeCamera, look_down_rotation, reduced_pixel_size


@dataclass
class Calibration:
    """Camera extrinsics and scale recovered from robot/image correspondences (SI units)."""

    rotation: np.ndarray  # camera axes in the base frame (A_0^cam)
    yaw: float  # rotation of the camera about the vertical axis [rad]
    metres_per_pixel: float  # image scale on the working plane
    depth: float  # pinhole to working plane, c_z [m]
    position: np.ndarray  # camera origin in the base frame, p_0,cam [m]
    residuals_px: np.ndarray  # reprojection residuals of the input features
    axis_skew: float = 0.0  # deviation of the measured image axes from 90 deg [rad]
    metres_per_pixel_y: float | None = None  # same scale measured along the robot y axis (consistency check)

    @property
    def pose(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.position
        return T

    def camera(
        self, focal_length: float = 6e-3, pixel_size: float | None = None, width: int = 640, height: int = 480
    ) -> PinholeCamera:
        return PinholeCamera(
            pose=self.pose,
            focal_length=focal_length,
            pixel_size=reduced_pixel_size() if pixel_size is None else pixel_size,
            width=width,
            height=height,
        )


def _closest_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _reprojection(cal: Calibration, features, points, focal_length, pixel_size, image_size) -> np.ndarray:
    cam = cal.camera(focal_length, pixel_size, *image_size)
    return cam.project(np.asarray(points, dtype=float)) - np.asarray(features, dtype=float)


def calibrate_three_points(
    f_a,
    f_b,
    f_c,
    p_a,
    p_b,
    p_c,
    focal_length: float = 6e-3,
    pixel_size: float | None = None,
    image_size: tuple[int, int] = (640, 480),
    roi_origin=(0.0, 0.0),
) -> Calibration:
    """The lab procedure: B - A is a pure +x move and C - A a pure +y move of the robot.

    Features are given relative to the ROI origin, as reported by the camera software.
    """
    pixel_size = reduced_pixel_size() if pixel_size is None else pixel_size
    f_a, f_b, f_c = (np.asarray(f, dtype=float) + np.asarray(roi_origin, dtype=float) for f in (f_a, f_b, f_c))
    p_a, p_b, p_c = (np.asarray(p, dtype=float) for p in (p_a, p_b, p_c))
    x_img, y_img = f_b - f_a, f_c - f_a
    x0 = np.r_[x_img / np.linalg.norm(x_img), 0.0]
    y0 = np.r_[y_img / np.linalg.norm(y_img), 0.0]
    skew = np.arccos(np.clip(x0 @ y0, -1.0, 1.0)) - np.pi / 2
    # Robot axes seen in the camera frame form A_cam^0; orthonormalise the noisy measurement.
    A_cam_0 = _closest_rotation(np.column_stack([x0, y0, np.cross(x0, y0)]))
    rotation = A_cam_0.T
    scale_x = np.linalg.norm(p_b - p_a) / np.linalg.norm(x_img)
    scale_y = np.linalg.norm(p_c - p_a) / np.linalg.norm(y_img)
    m_per_px = scale_x  # the lab uses the A-B pair; see residuals for the y pair
    depth = focal_length * m_per_px / pixel_size
    center = np.array(image_size, dtype=float) / 2.0
    c_a = np.r_[(f_a - center) * m_per_px, depth]
    position = p_a - rotation @ c_a
    cal = Calibration(
        rotation=rotation,
        yaw=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        metres_per_pixel=float(m_per_px),
        depth=float(depth),
        position=position,
        residuals_px=np.zeros((3, 2)),
        axis_skew=float(skew),
        metres_per_pixel_y=float(scale_y),
    )
    cal.residuals_px = _reprojection(cal, [f_a, f_b, f_c], [p_a, p_b, p_c], focal_length, pixel_size, image_size)
    return cal


def calibrate_least_squares(
    features,
    points,
    focal_length: float = 6e-3,
    pixel_size: float | None = None,
    image_size: tuple[int, int] = (640, 480),
) -> Calibration:
    """Fit yaw, scale and position from N >= 2 points on one horizontal plane (2-D similarity with mirror)."""
    pixel_size = reduced_pixel_size() if pixel_size is None else pixel_size
    F = np.asarray(features, dtype=float)
    P = np.asarray(points, dtype=float)
    if np.ptp(P[:, 2]) > 1e-6:
        raise ValueError("all calibration points must lie on one horizontal plane")
    xy = P[:, :2]
    xm, fm = xy.mean(axis=0), F.mean(axis=0)
    X, Y = xy - xm, F - fm
    # Looking down mirrors the image: f = s * M(yaw) * xy + t with M = [[c, s], [s, -c]].
    a = (Y[:, 0] * X[:, 0] - Y[:, 1] * X[:, 1]).sum()
    b = (Y[:, 0] * X[:, 1] + Y[:, 1] * X[:, 0]).sum()
    yaw = np.arctan2(b, a)
    M = np.array([[np.cos(yaw), np.sin(yaw)], [np.sin(yaw), -np.cos(yaw)]])
    scale = float(np.hypot(a, b) / (X**2).sum())
    t = fm - scale * M @ xm
    center = np.array(image_size, dtype=float) / 2.0
    m_per_px = 1.0 / scale
    depth = focal_length * m_per_px / pixel_size
    o_xy = M @ (center - t) / scale
    cal = Calibration(
        rotation=look_down_rotation(yaw),
        yaw=float(yaw),
        metres_per_pixel=m_per_px,
        depth=float(depth),
        position=np.r_[o_xy, P[0, 2] + depth],
        residuals_px=np.zeros_like(F),
    )
    cal.residuals_px = _reprojection(cal, F, P, focal_length, pixel_size, image_size)
    return cal
