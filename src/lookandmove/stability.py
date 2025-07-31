"""Why a badly calibrated camera still works: discrete-time analysis of the look-and-move loop.

With a fast inner velocity loop, one camera period T moves the feature by g * R(dtheta) * e, where
g = K_P * T * (estimated depth / true depth) and dtheta is the error in the estimated camera yaw.
Writing the 2-D image error as a complex number e, a processing delay of d frames gives

    e[k+1] = e[k] - g * exp(i * dtheta) * e[k-d],

whose characteristic polynomial is z^(d+1) - z^d + g * exp(i * dtheta).
"""

from __future__ import annotations

import numpy as np


def loop_gain(kp: float, frame_rate: float, depth_ratio: float = 1.0) -> float:
    """Per-frame gain g = K_P T (c_z,estimated / c_z,true)."""
    return kp / frame_rate * depth_ratio


def closed_loop_poles(g: float, dtheta: float, delay_frames: int = 0) -> np.ndarray:
    coeffs = np.zeros(delay_frames + 2, dtype=complex)
    coeffs[0], coeffs[1] = 1.0, -1.0
    coeffs[-1] += g * np.exp(1j * dtheta)
    return np.roots(coeffs)


def spectral_radius(g: float, dtheta: float, delay_frames: int = 0) -> float:
    """Largest pole magnitude; the image error shrinks per frame by this factor (< 1 means convergence)."""
    return float(np.abs(closed_loop_poles(g, dtheta, delay_frames)).max())


def stability_map(gains, angles, delay_frames: int = 0) -> np.ndarray:
    """Spectral radius on a grid, shape (len(gains), len(angles))."""
    return np.array([[spectral_radius(g, a, delay_frames) for a in angles] for g in gains])


def critical_angle(g: float, delay_frames: int = 0, tol: float = 1e-6) -> float:
    """Largest |yaw error| (rad) for which the loop still converges; 0 if it is unstable even when calibrated."""
    if spectral_radius(g, 0.0, delay_frames) >= 1.0:
        return 0.0
    lo, hi = 0.0, np.pi
    if spectral_radius(g, hi, delay_frames) < 1.0:
        return np.pi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if spectral_radius(g, mid, delay_frames) < 1.0 else (lo, mid)
    return lo


def convergence_time(g: float, dtheta: float, frame_rate: float, delay_frames: int = 0, reduction: float = 100.0) -> float:
    """Seconds until the image error has shrunk by ``reduction`` (inf if the loop does not converge)."""
    rho = spectral_radius(g, dtheta, delay_frames)
    if rho >= 1.0:
        return float("inf")
    return float(np.log(reduction) / -np.log(rho) / frame_rate)
