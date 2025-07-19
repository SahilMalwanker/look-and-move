"""Stability map of dynamic look-and-move versus gain and camera-yaw error, checked by simulation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import CALIBRATION, PALETTE, Q_START, ROBOT, TARGET_PX, TRUE_CAMERA, save
from matplotlib.colors import LogNorm

from lookandmove import IdealMeasurement, PinholeCamera, ServoConfig, convergence_time, critical_angle, loop_gain, simulate

FRAME_RATE = 17.5


def classify(kp: float, dtheta_deg: float, duration: float = 25.0) -> str:
    model = PinholeCamera.looking_down(TRUE_CAMERA.position, CALIBRATION.yaw + np.deg2rad(dtheta_deg))
    res = simulate(
        ROBOT, Q_START, TARGET_PX, model, IdealMeasurement(TRUE_CAMERA, 0.25, seed=1), ServoConfig(gain=kp, duration=duration)
    )
    if res.lost:
        return "lost"
    return "converged" if res.converged else "slow"


def main() -> None:
    gains = np.logspace(np.log10(0.3), np.log10(30), 160)
    angles = np.linspace(0, 180, 181)
    t_conv = np.array([[convergence_time(loop_gain(k, FRAME_RATE), np.deg2rad(a), FRAME_RATE, 1) for a in angles] for k in gains])
    masked = np.ma.masked_invalid(t_conv)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_facecolor("#fde8e8")
    im = ax.pcolormesh(angles, gains, masked, norm=LogNorm(0.2, 60), cmap="viridis_r", shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("time to shrink the image error 100\u00d7  [s]")
    for delay, style in ((0, (0, (5, 3))), (1, "-"), (2, (0, (1.5, 2)))):
        crit = [np.rad2deg(critical_angle(loop_gain(k, FRAME_RATE), delay)) for k in gains]
        ax.plot(
            crit,
            gains,
            color=PALETTE["ink"],
            lw=1.6,
            ls=style,
            label=f"stability limit, {delay} frame{'s' if delay != 1 else ''} latency",
        )
    ax.axhline(1.0, color="white", lw=1.2, ls=(0, (2, 2)))
    ax.text(3, 1.07, "lab gain K_P = 1 s\u207b\u00b9", color="white", fontsize=8.5, ha="left", va="bottom", fontweight="bold")
    ax.text(140, 12, "diverges", color=PALETTE["red"], fontsize=12, fontweight="bold", ha="center")

    markers = {
        "converged": ("o", PALETTE["green"], "simulation: converged"),
        "slow": ("o", "white", "simulation: not settled (spiral / limit cycle)"),
    }
    markers["lost"] = ("X", PALETTE["red"], "simulation: ball left the image")
    seen = set()
    for kp in (1.0, 3.0, 10.0):
        for dth in (0, 30, 50, 65, 75, 85, 95, 120):
            outcome = classify(kp, dth)
            m, c, label = markers[outcome]
            ax.scatter(
                dth,
                kp,
                marker=m,
                s=58,
                color=c,
                edgecolor=PALETTE["ink"],
                lw=0.9,
                zorder=5,
                label=None if outcome in seen else label,
            )
            seen.add(outcome)
            print(f"K_P = {kp:5.1f} 1/s  yaw error {dth:4d} deg -> {outcome}")
    ax.set_yscale("log")
    ax.set_xlim(0, 180)
    ax.set_ylim(gains[0], gains[-1])
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_xlabel("error in the estimated camera yaw  \u0394\u03b8  [deg]")
    ax.set_ylabel("vision gain K_P  [1/s]")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8)
    ax.set_title("Dynamic look-and-move at 17.5 frames/s: how wrong may the camera calibration be?", loc="left")
    save(fig, "stability_map.png")
    plt.close(fig)
    for kp in (1.0, 3.0, 10.0):
        g = loop_gain(kp, FRAME_RATE)
        print(f"K_P = {kp:4.1f}: g = {g:.3f}, critical yaw error {np.rad2deg(critical_angle(g, 1)):.1f} deg (1 frame latency)")


if __name__ == "__main__":
    main()
