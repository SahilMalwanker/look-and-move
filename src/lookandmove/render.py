"""Synthetic top-down camera frames: textured table, target button, robot links and the tracked ball."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import PinholeCamera


@dataclass(frozen=True)
class Scene:
    """Geometry and appearance of the work cell (metres, base frame)."""

    table_z: float = -0.252
    ball_radius: float = 0.011
    target_radius: float = 0.024
    link_radius: float = 0.034
    grid_spacing: float = 0.05
    texture_seed: int = 3
    noise: float = 0.012
    blur_sigma: float = 0.7


def _resize_bilinear(a: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = np.linspace(0, a.shape[0] - 1, height)
    xs = np.linspace(0, a.shape[1] - 1, width)
    rows = np.array([np.interp(xs, np.arange(a.shape[1]), r) for r in a])
    return np.array([np.interp(ys, np.arange(a.shape[0]), c) for c in rows.T]).T


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(img, r, mode="edge")
    tmp = sum(k[i] * padded[:, i : i + img.shape[1]] for i in range(2 * r + 1))
    return sum(k[i] * tmp[i : i + img.shape[0], :] for i in range(2 * r + 1))


class Renderer:
    """Renders 8-bit grey frames as seen by ``camera`` (painter's algorithm, anti-aliased edges)."""

    def __init__(self, camera: PinholeCamera, scene: Scene | None = None):
        self.camera = camera
        self.scene = scene or Scene()
        H, W = camera.height, camera.width
        self.v, self.u = np.mgrid[0:H, 0:W].astype(float)
        self.background = self._background()

    def _table_points(self) -> np.ndarray:
        cam, s = self.camera, self.scene
        rays = np.stack(
            [
                (self.u - cam.center[0]) * cam.pixel_size / cam.focal_length,
                (self.v - cam.center[1]) * cam.pixel_size / cam.focal_length,
                np.ones_like(self.u),
            ],
            axis=-1,
        )
        dirs = rays @ cam.rotation.T
        t = (s.table_z - cam.position[2]) / dirs[..., 2]
        return cam.position + dirs * t[..., None]

    def _background(self) -> np.ndarray:
        s = self.scene
        H, W = self.camera.height, self.camera.width
        rng = np.random.default_rng(s.texture_seed)
        mottle = _resize_bilinear(rng.normal(size=(8, 10)), H, W)
        img = 0.66 + 0.035 * mottle
        pts = self._table_points()
        for axis in (0, 1):
            frac = np.abs((pts[..., axis] / s.grid_spacing + 0.5) % 1.0 - 0.5) * s.grid_spacing
            img -= 0.07 * np.clip(1.6e-3 - frac, 0.0, 1.6e-3) / 1.6e-3
        rr = ((self.u - W / 2) ** 2 + (self.v - H / 2) ** 2) / (W / 2) ** 2
        return img * (1.0 - 0.12 * rr)

    def _radius_px(self, radius: float, point) -> float:
        depth = self.camera.to_camera(point)[2]
        return radius * self.camera.focal_length / (depth * self.camera.pixel_size)

    def _paint_disk(self, img, center, radius, shade):
        x0, x1 = int(max(center[0] - radius - 2, 0)), int(min(center[0] + radius + 3, img.shape[1]))
        y0, y1 = int(max(center[1] - radius - 2, 0)), int(min(center[1] + radius + 3, img.shape[0]))
        if x0 >= x1 or y0 >= y1:
            return
        du, dv = self.u[y0:y1, x0:x1] - center[0], self.v[y0:y1, x0:x1] - center[1]
        dist = np.hypot(du, dv)
        cover = np.clip(radius + 0.5 - dist, 0.0, 1.0)
        colour = shade(dist / max(radius, 1e-6), du / max(radius, 1e-6), dv / max(radius, 1e-6))
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - cover) + colour * cover

    def _paint_segment(self, img, a, b, radius):
        lo = np.floor(np.minimum(a, b) - radius - 2).astype(int)
        hi = np.ceil(np.maximum(a, b) + radius + 3).astype(int)
        x0, y0 = max(lo[0], 0), max(lo[1], 0)
        x1, y1 = min(hi[0], img.shape[1]), min(hi[1], img.shape[0])
        if x0 >= x1 or y0 >= y1:
            return
        pu, pv = self.u[y0:y1, x0:x1], self.v[y0:y1, x0:x1]
        ab = b - a
        t = np.clip(((pu - a[0]) * ab[0] + (pv - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0.0, 1.0)
        dist = np.hypot(pu - (a[0] + t * ab[0]), pv - (a[1] + t * ab[1]))
        cover = np.clip(radius + 0.5 - dist, 0.0, 1.0)
        colour = 0.30 + 0.16 * np.clip(1.0 - (dist / radius) ** 2, 0.0, 1.0)
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - cover) + colour * cover

    def render(self, tcp, target=None, link_points=None, rng=None) -> np.ndarray:
        """Frame with the ball at base-frame point ``tcp``; ``link_points`` is an (n, 3) polyline of the arm."""
        s, cam = self.scene, self.camera
        img = self.background.copy()
        if target is not None:
            c = cam.project(target)
            r = self._radius_px(s.target_radius, target)
            self._paint_disk(img, c, r, lambda d, du, dv: np.where(d < 0.62, 0.20, np.where(d < 0.8, 0.92, 0.25)))
        if link_points is not None:
            pts = np.asarray(link_points, dtype=float)
            order = np.argsort(0.5 * (pts[:-1, 2] + pts[1:, 2]))
            for i in order:
                a, b = cam.project(pts[i]), cam.project(pts[i + 1])
                if np.linalg.norm(b - a) < 0.5:
                    continue
                self._paint_segment(img, a, b, self._radius_px(s.link_radius, 0.5 * (pts[i] + pts[i + 1])))
        c = cam.project(tcp)
        r = self._radius_px(s.ball_radius, tcp)

        def ball(d, du, dv):
            dome = np.sqrt(np.clip(1.0 - d**2, 0.0, 1.0))
            spec = np.exp(-((du + 0.35) ** 2 + (dv + 0.35) ** 2) / 0.05)
            body = 0.55 + 0.35 * dome + 0.25 * spec
            return np.where(d > 0.82, 0.08, np.clip(body, 0.0, 1.0))

        self._paint_disk(img, c, r, ball)
        img = _gaussian_blur(img, s.blur_sigma)
        if rng is not None and s.noise > 0:
            img = img + rng.normal(scale=s.noise, size=img.shape)
        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)

    def teach_template(self, size: int = 41, plane_z: float = 0.40) -> np.ndarray:
        """Template of the ball rendered exactly at the image centre (the camera's 'teach mode')."""
        depth = self.camera.depth_of_plane(plane_z)
        tcp = self.camera.backproject(self.camera.center, depth)
        frame = self.render(tcp)
        cx, cy = (int(v) for v in self.camera.center)
        r = size // 2
        return frame[cy - r : cy + r + 1, cx - r : cx + r + 1].copy()
