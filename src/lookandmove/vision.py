"""Template matching by zero-mean normalised cross-correlation, with ROI handling and sub-pixel peaks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Match:
    """Result of one template search.

    ``feature_roi`` is what the camera software reports (relative to the ROI origin);
    ``feature`` is the same point in full-image pixel coordinates, f_img = f + O_ROI.
    """

    feature_roi: np.ndarray
    feature: np.ndarray
    score: float
    score_map: np.ndarray
    roi: tuple[int, int, int, int]


def _window_sums(a: np.ndarray, h: int, w: int) -> np.ndarray:
    s = np.pad(a, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    return s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]


def ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean NCC for every valid template position (like OpenCV's TM_CCOEFF_NORMED)."""
    img = np.asarray(image, dtype=float)
    tpl = np.asarray(template, dtype=float)
    H, W = img.shape
    h, w = tpl.shape
    if h > H or w > W:
        raise ValueError("template larger than the search region")
    t0 = tpl - tpl.mean()
    t_norm = np.sqrt((t0**2).sum())
    # Circular convolution with the flipped template; the valid block is free of wrap-around.
    spec = np.fft.rfft2(img) * np.fft.rfft2(t0[::-1, ::-1], s=(H, W))
    corr = np.fft.irfft2(spec, s=(H, W))[h - 1 :, w - 1 :]
    n = h * w
    s1 = _window_sums(img, h, w)
    s2 = _window_sums(img * img, h, w)
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    denom = t_norm * np.sqrt(var)
    out = np.zeros_like(corr)
    ok = denom > 1e-9 * max(t_norm, 1.0)
    out[ok] = corr[ok] / denom[ok]
    return out


def _parabola_offset(m1: float, m0: float, p1: float) -> float:
    den = m1 - 2.0 * m0 + p1
    return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5))


def match_template(image: np.ndarray, template: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> Match:
    """Find ``template`` in ``image`` (optionally only inside roi = (x0, y0, width, height))."""
    H, W = image.shape
    x0, y0, rw, rh = roi if roi is not None else (0, 0, W, H)
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    rw, rh = min(int(rw), W - x0), min(int(rh), H - y0)
    region = image[y0 : y0 + rh, x0 : x0 + rw]
    scores = ncc_map(region, template)
    iy, ix = np.unravel_index(int(np.argmax(scores)), scores.shape)
    dx = _parabola_offset(scores[iy, ix - 1], scores[iy, ix], scores[iy, ix + 1]) if 0 < ix < scores.shape[1] - 1 else 0.0
    dy = _parabola_offset(scores[iy - 1, ix], scores[iy, ix], scores[iy + 1, ix]) if 0 < iy < scores.shape[0] - 1 else 0.0
    h, w = template.shape
    feature_roi = np.array([ix + dx + (w - 1) / 2.0, iy + dy + (h - 1) / 2.0])
    return Match(
        feature_roi=feature_roi,
        feature=feature_roi + np.array([x0, y0]),
        score=float(scores[iy, ix]),
        score_map=scores,
        roi=(x0, y0, rw, rh),
    )


def cut_template(image: np.ndarray, center, size: int) -> np.ndarray:
    """Square template of odd ``size`` centred on ``center`` (rounded to the nearest pixel)."""
    cx, cy = (int(round(v)) for v in center)
    r = size // 2
    return image[cy - r : cy + r + 1, cx - r : cx + r + 1].copy()
