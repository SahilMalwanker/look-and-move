"""igus ReBeL 6-DoF kinematics: DH model, tool pose (x, y, z, A, B, C), Jacobian and numerical IK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def euler_zyx(R: np.ndarray) -> np.ndarray:
    """(A, B, C) in radians with R = Rz(A) Ry(B) Rx(C)."""
    return np.array(
        [
            np.arctan2(R[1, 0], R[0, 0]),
            np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])),
            np.arctan2(R[2, 1], R[2, 2]),
        ]
    )


def pose_matrix(x: float, y: float, z: float, a: float, b: float, c: float) -> np.ndarray:
    """Homogeneous pose from position (m) and Z-Y-X Euler angles (rad)."""
    T = np.eye(4)
    T[:3, :3] = rotz(a) @ roty(b) @ rotx(c)
    T[:3, 3] = [x, y, z]
    return T


def rotation_error(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    E = R_des @ R.T
    angle = np.arccos(np.clip((np.trace(E) - 1.0) / 2.0, -1.0, 1.0))
    w = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if angle < 1e-9:
        return 0.5 * w
    if np.pi - angle < 1e-6:
        B = (E + np.eye(3)) / 2.0
        axis = B[:, int(np.argmax(np.diag(B)))]
        return angle * axis / np.linalg.norm(axis)
    return angle / (2.0 * np.sin(angle)) * w


@dataclass(frozen=True)
class ReBeL:
    """DH model of the igus ReBeL 6-DoF with the lab's 60 mm ball tool.

    K0 sits on the first joint axis at shoulder height (252 mm above the mounting flange).
    """

    a2: float = 0.237
    d4: float = 0.297
    d6: float = 0.126
    tool_length: float = 0.060

    @property
    def dh(self) -> np.ndarray:
        """Rows (d, a, alpha); theta_i = q_i."""
        return np.array(
            [
                [0.0, 0.0, -np.pi / 2],
                [0.0, self.a2, 0.0],
                [0.0, 0.0, np.pi / 2],
                [self.d4, 0.0, np.pi / 2],
                [0.0, 0.0, -np.pi / 2],
                [self.d6, 0.0, 0.0],
            ]
        )

    def frames(self, q) -> np.ndarray:
        """Poses of K0..K6 and the tool point, shape (8, 4, 4)."""
        q = np.asarray(q, dtype=float)
        out = np.empty((8, 4, 4))
        T = np.eye(4)
        out[0] = T
        for i, (d, a, alpha) in enumerate(self.dh):
            T = T @ dh_transform(q[i], d, a, alpha)
            out[i + 1] = T
        tool = np.eye(4)
        tool[2, 3] = self.tool_length
        out[7] = T @ tool
        return out

    def fk(self, q) -> np.ndarray:
        """Tool-centre-point pose (centre of the ball)."""
        return self.frames(q)[7]

    def tcp(self, q) -> np.ndarray:
        return self.fk(q)[:3, 3]

    def pose_vector(self, q) -> np.ndarray:
        """(x, y, z, A, B, C) with metres and radians."""
        T = self.fk(q)
        return np.r_[T[:3, 3], euler_zyx(T[:3, :3])]

    def jacobian(self, q) -> np.ndarray:
        """Geometric Jacobian of the tool point, base frame, rows [v; w]."""
        F = self.frames(q)
        p = F[7][:3, 3]
        J = np.empty((6, 6))
        for i in range(6):
            z, o = F[i][:3, 2], F[i][:3, 3]
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return J

    def ik(self, T_des: np.ndarray, q0, iterations: int = 300, tol: float = 1e-10, damping: float = 1e-3) -> np.ndarray:
        """Damped least-squares IK from the start guess q0 (raises if it does not converge)."""
        q = np.asarray(q0, dtype=float).copy()
        for _ in range(iterations):
            T = self.fk(q)
            err = np.r_[T_des[:3, 3] - T[:3, 3], rotation_error(T[:3, :3], T_des[:3, :3])]
            if err @ err < tol**2:
                return q
            J = self.jacobian(q)
            q += J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        raise RuntimeError("IK did not converge")
