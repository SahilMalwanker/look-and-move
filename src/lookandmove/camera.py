"""Pinhole camera with the lab's intrinsics (6 mm lens, 640 x 480 crop of a 4056 x 3040 sensor)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot import rotx, rotz

FULL_WIDTH_PX = 4056
FULL_PIXEL_SIZE = 1.55e-6


def reduced_pixel_size(pixel_size: float = FULL_PIXEL_SIZE, full_width: int = FULL_WIDTH_PX, used_width: int = 640) -> float:
    """Effective pixel pitch when the full sensor is resampled to ``used_width`` columns."""
    return pixel_size * full_width / used_width


def look_down_rotation(yaw: float) -> np.ndarray:
    """Camera axes in the base frame for a camera looking straight down (z_cam = -z_0)."""
    return rotz(yaw) @ rotx(np.pi)


@dataclass(frozen=True, eq=False)
class PinholeCamera:
    """Ideal pinhole camera. ``pose`` is the camera frame expressed in the robot base frame."""

    pose: np.ndarray
    focal_length: float = 6e-3
    pixel_size: float = reduced_pixel_size()
    width: int = 640
    height: int = 480

    @classmethod
    def looking_down(cls, position, yaw: float, **kwargs) -> PinholeCamera:
        T = np.eye(4)
        T[:3, :3] = look_down_rotation(yaw)
        T[:3, 3] = position
        return cls(pose=T, **kwargs)

    @property
    def rotation(self) -> np.ndarray:
        return self.pose[:3, :3]

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def center(self) -> np.ndarray:
        return np.array([self.width / 2.0, self.height / 2.0])

    def to_camera(self, p_base) -> np.ndarray:
        """Base-frame point(s) -> camera coordinates c = (cx, cy, cz)."""
        p = np.atleast_2d(np.asarray(p_base, dtype=float))
        c = (p - self.position) @ self.rotation
        return c[0] if np.ndim(p_base) == 1 else c

    def project(self, p_base) -> np.ndarray:
        """Pixel coordinates (u right, v down, origin top-left) of base-frame point(s)."""
        c = np.atleast_2d(self.to_camera(p_base))
        s = c[:, :2] * self.focal_length / c[:, 2:3]  # sensor coordinates, eq. (5)
        f = s / self.pixel_size + self.center
        return f[0] if np.ndim(p_base) == 1 else f

    def backproject(self, pixel, depth: float) -> np.ndarray:
        """Base-frame point at camera depth ``depth`` that projects to ``pixel``."""
        f = np.asarray(pixel, dtype=float)
        c_xy = (f - self.center) * self.pixel_size * depth / self.focal_length
        return self.position + self.rotation @ np.r_[c_xy, depth]

    def depth_of_plane(self, z_plane: float) -> float:
        """Camera depth of the horizontal plane z = z_plane (camera looking down)."""
        return float(self.position[2] - z_plane)

    def metres_per_pixel(self, depth: float) -> float:
        return depth * self.pixel_size / self.focal_length

    def image_jacobian(self, c) -> np.ndarray:
        """ds/dc of the pinhole model (2 x 3), eq. (6) of the lab handout."""
        cx, cy, cz = c
        lam = self.focal_length
        return np.array([[lam / cz, 0.0, -cx * lam / cz**2], [0.0, lam / cz, -cy * lam / cz**2]])

    def with_pose(self, pose: np.ndarray) -> PinholeCamera:
        return PinholeCamera(
            pose=pose, focal_length=self.focal_length, pixel_size=self.pixel_size, width=self.width, height=self.height
        )
