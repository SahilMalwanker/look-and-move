"""Dynamic look-and-move: a slow camera loop commanding the robot's fast joint-velocity loops."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .camera import PinholeCamera
from .render import Renderer
from .robot import ReBeL, rotation_error
from .vision import match_template


@dataclass
class ServoConfig:
    gain: float = 1.0  # K_P [1/s]; the lab used K_P = 1
    frame_rate: float = 17.5  # camera + image processing [frames/s]
    latency_frames: int = 1  # a measurement is used one frame after it was taken
    ipo_cycle: float = 0.010  # robot interpolation cycle T_IPO [s]
    velocity_time_constant: float = 0.03  # first-order model of the joint PI velocity loops [s]
    hold_gain: float = 5.0  # Cartesian P gain that keeps height and orientation [1/s]
    duration: float = 6.0  # [s]
    joint_speed_limit: float = 1.0  # [rad/s]


@dataclass
class ServoResult:
    t: np.ndarray  # (n,) robot cycle time stamps
    q: np.ndarray  # (n, 6) joint angles
    tcp: np.ndarray  # (n, 3) ball centre
    frame_t: np.ndarray  # (m,) camera frame time stamps
    feature: np.ndarray  # (m, 2) measured feature [px], NaN when tracking was lost
    feature_true: np.ndarray  # (m, 2) exact projection [px]
    target: np.ndarray  # (2,) desired feature [px]
    frames: list = field(default_factory=list)  # rendered images (only if requested)

    @property
    def error_px(self) -> np.ndarray:
        return np.linalg.norm(self.feature_true - self.target, axis=1)

    @property
    def lost(self) -> bool:
        return bool(np.isnan(self.feature).any())

    def settling_time(self, tolerance_px: float = 1.0) -> float:
        """First time after which the image error stays below the tolerance (inf if never)."""
        outside = np.flatnonzero(self.error_px > tolerance_px)
        if outside.size == 0:
            return 0.0
        if outside[-1] == len(self.error_px) - 1:
            return float("inf")
        return float(self.frame_t[outside[-1] + 1])

    @property
    def converged(self) -> bool:
        return np.isfinite(self.settling_time())


class IdealMeasurement:
    """Exact projection of the ball centre plus Gaussian pixel noise; None once the ball leaves the image."""

    def __init__(self, camera: PinholeCamera, noise_px: float = 0.25, seed: int = 0, margin_px: float = 20.0):
        self.camera = camera
        self.noise_px = noise_px
        self.margin_px = margin_px
        self.rng = np.random.default_rng(seed)

    def __call__(self, robot: ReBeL, q: np.ndarray):
        f = self.camera.project(robot.tcp(q))
        m = self.margin_px
        if not (m <= f[0] <= self.camera.width - m and m <= f[1] <= self.camera.height - m):
            return None, None
        return f + self.rng.normal(scale=self.noise_px, size=2), None


class VisionMeasurement:
    """Full pipeline: render the frame, then find the ball by template matching in an adaptive ROI."""

    def __init__(
        self,
        renderer: Renderer,
        target_point=None,
        roi_size: int | None = 160,
        min_score: float = 0.5,
        seed: int = 0,
        keep_frames: bool = False,
    ):
        self.renderer = renderer
        self.template = renderer.teach_template()
        self.target_point = target_point
        self.roi_size = roi_size
        self.min_score = min_score
        self.rng = np.random.default_rng(seed)
        self.keep_frames = keep_frames
        self.last = None

    @property
    def camera(self) -> PinholeCamera:
        return self.renderer.camera

    def __call__(self, robot: ReBeL, q: np.ndarray):
        frames = robot.frames(q)
        img = self.renderer.render(frames[7][:3, 3], self.target_point, frames[:, :3, 3], rng=self.rng)
        roi = None
        if self.roi_size is not None and self.last is not None:
            half = self.roi_size // 2
            roi = (int(self.last[0]) - half, int(self.last[1]) - half, self.roi_size, self.roi_size)
        match = match_template(img, self.template, roi)
        kept = img if self.keep_frames else None
        if match.score < self.min_score:
            self.last = None
            return None, kept
        self.last = match.feature
        return match.feature, kept


def look_and_move_command(
    robot: ReBeL,
    q: np.ndarray,
    feature,
    target,
    camera_model: PinholeCamera,
    plane_z: float,
    gain: float,
    hold_pose: np.ndarray | None = None,
    hold_gain: float = 5.0,
) -> np.ndarray:
    """Joint-velocity set-point dq_s = K_P J0^-1 A_0^cam J_v^-1 (f_d - f) dPx (lab eq. 10 and 12).

    Vision only supplies the x-y motion; ``hold_pose`` lets the robot keep height and orientation.
    """
    v_xy = np.zeros(2)
    if feature is not None:
        ds = (np.asarray(target, float) - np.asarray(feature, float)) * camera_model.pixel_size
        depth = camera_model.depth_of_plane(plane_z)
        dc_cam = np.r_[ds * depth / camera_model.focal_length, 0.0]
        v_xy = gain * (camera_model.rotation @ dc_cam)[:2]
    v_z, w = 0.0, np.zeros(3)
    if hold_pose is not None:
        T = robot.fk(q)
        v_z = hold_gain * (hold_pose[2, 3] - T[2, 3])
        w = hold_gain * rotation_error(T[:3, :3], hold_pose[:3, :3])
    J = robot.jacobian(q)
    return J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), np.r_[v_xy, v_z, w])


def simulate(
    robot: ReBeL,
    q0,
    target_px,
    camera_model: PinholeCamera,
    measure: Callable,
    config: ServoConfig | None = None,
    true_camera: PinholeCamera | None = None,
) -> ServoResult:
    """Closed-loop dynamic look-and-move simulation.

    ``camera_model`` is the controller's (possibly wrong) belief about the camera; ``measure``
    returns what the real camera sees. ``true_camera`` (default: ``measure.camera``) is used for logging.
    """
    cfg = config or ServoConfig()
    q = np.asarray(q0, dtype=float).copy()
    hold = robot.fk(q)
    truth = true_camera or measure.camera
    qd = np.zeros(6)
    alpha = 1.0 - np.exp(-cfg.ipo_cycle / cfg.velocity_time_constant)
    frame_period = 1.0 / cfg.frame_rate
    pending: deque = deque()
    n = int(round(cfg.duration / cfg.ipo_cycle)) + 1
    ts, qs, tcps, ft, fm, ftrue, frames = [], [], [], [], [], [], []
    next_frame = 0.0
    feature_in_use = None
    for k in range(n):
        t = k * cfg.ipo_cycle
        if t >= next_frame - 1e-9:
            feature, img = measure(robot, q)
            pending.append((next_frame + cfg.latency_frames * frame_period, feature))
            ft.append(next_frame)
            fm.append(np.full(2, np.nan) if feature is None else feature)
            ftrue.append(truth.project(robot.tcp(q)))
            if img is not None:
                frames.append(img)
            next_frame += frame_period
        while pending and pending[0][0] <= t + 1e-9:
            _, feature_in_use = pending.popleft()
        qd_cmd = look_and_move_command(
            robot, q, feature_in_use, target_px, camera_model, hold[2, 3], cfg.gain, hold, cfg.hold_gain
        )
        peak = np.abs(qd_cmd).max()
        if peak > cfg.joint_speed_limit:
            qd_cmd *= cfg.joint_speed_limit / peak
        ts.append(t)
        qs.append(q.copy())
        tcps.append(robot.tcp(q))
        qd += alpha * (qd_cmd - qd)
        q = q + qd * cfg.ipo_cycle
    return ServoResult(
        t=np.array(ts),
        q=np.array(qs),
        tcp=np.array(tcps),
        frame_t=np.array(ft),
        feature=np.array(fm),
        feature_true=np.array(ftrue),
        target=np.asarray(target_px, dtype=float),
        frames=frames,
    )
