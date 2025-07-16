"""Hand-eye calibration from the three positions taught in the lab (task 5.3)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import CALIBRATION, PALETTE, ROBOT, TRUE_CAMERA, save

from lookandmove import Renderer, calibrate_least_squares, lab


def main() -> None:
    cal = CALIBRATION
    ls = calibrate_least_squares([lab.F_A, lab.F_B, lab.F_C], [lab.P_A, lab.P_B, lab.P_C])
    print(f"three-point: yaw {np.rad2deg(cal.yaw):.2f} deg, scale {cal.metres_per_pixel * 1e3:.4f} mm/px (x) ", end="")
    print(
        f"{cal.metres_per_pixel_y * 1e3:.4f} mm/px (y), depth {cal.depth * 1e3:.1f} mm, skew {np.rad2deg(cal.axis_skew):.2f} deg"
    )
    print(f"             camera at {np.round(cal.position * 1e3, 1)} mm, residuals {np.round(cal.residuals_px, 2).tolist()} px")
    print(
        f"least squares: yaw {np.rad2deg(ls.yaw):.2f} deg, scale {ls.metres_per_pixel * 1e3:.4f} mm/px, "
        f"depth {ls.depth * 1e3:.1f} mm"
    )
    print(f"             camera at {np.round(ls.position * 1e3, 1)} mm, residuals {np.round(ls.residuals_px, 2).tolist()} px")

    q_a = np.deg2rad(lab.TEST_CASES_DEG[0])
    frame = Renderer(TRUE_CAMERA).render(lab.P_A, link_points=ROBOT.frames(q_a)[:, :3, 3], rng=np.random.default_rng(0))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax0.imshow(frame, cmap="gray", vmin=0, vmax=255)
    feats = {"A": lab.F_A, "B": lab.F_B, "C": lab.F_C}
    for name, f in feats.items():
        ax0.scatter(*f, s=120, facecolor="none", edgecolor=PALETTE["gold"], lw=2.2, zorder=5)
        ax0.text(f[0] + 9, f[1] - 9, name, color=PALETTE["gold"], fontsize=12, fontweight="bold")
    for end, label, colour in ((lab.F_B, "x\u2080", PALETTE["red"]), (lab.F_C, "y\u2080", PALETTE["green"])):
        ax0.annotate("", xy=end, xytext=lab.F_A, arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": colour})
        mid = 0.55 * (end - lab.F_A) + lab.F_A
        ax0.text(mid[0] + 6, mid[1] + 12, label, color=colour, fontsize=12, fontweight="bold")
    proj3 = TRUE_CAMERA.project(np.array([lab.P_A, lab.P_B, lab.P_C]))
    projls = ls.camera().project(np.array([lab.P_A, lab.P_B, lab.P_C]))
    ax0.scatter(*proj3.T, marker="+", s=110, color=PALETTE["blue"], lw=2, zorder=6, label="re-projected, three-point model")
    ax0.scatter(*projls.T, marker="x", s=70, color=PALETTE["purple"], lw=2, zorder=6, label="re-projected, least-squares model")
    ax0.scatter([], [], s=120, facecolor="none", edgecolor=PALETTE["gold"], lw=2.2, label="features measured in the lab")
    ax0.legend(loc="lower right", facecolor="white", frameon=True, framealpha=0.9)
    ax0.set_title("Robot axes as seen by the camera (640 \u00d7 480)", loc="left")
    ax0.grid(False)
    ax0.set_xlim(0, 640)
    ax0.set_ylim(480, 0)

    corners = np.array([[0, 0], [640, 0], [640, 480], [0, 480], [0, 0]], dtype=float)
    footprint = np.array([TRUE_CAMERA.backproject(c, cal.depth) for c in corners])
    ax1.fill(footprint[:, 0] * 1e3, footprint[:, 1] * 1e3, color=PALETTE["teal"], alpha=0.10, lw=0)
    ax1.plot(
        footprint[:, 0] * 1e3, footprint[:, 1] * 1e3, color=PALETTE["teal"], lw=1.2, label="camera field of view at z = 400 mm"
    )
    cam_xy = cal.position[:2] * 1e3
    for axis, colour, name in ((0, PALETTE["red"], "x_cam"), (1, PALETTE["green"], "y_cam")):
        d = cal.rotation[:2, axis] * 90
        ax1.annotate("", xy=cam_xy + d, xytext=cam_xy, arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": colour})
        ax1.text(*(cam_xy + d * 1.15), name, color=colour, fontsize=9, ha="center", va="center")
    ax1.scatter(*cam_xy, s=60, color=PALETTE["ink"], zorder=5)
    ax1.text(cam_xy[0] + 12, cam_xy[1] - 32, f"camera\n{cal.position[2] * 1e3:.0f} mm above K\u2080", fontsize=8.5)
    for name, p in (("A", lab.P_A), ("B", lab.P_B), ("C", lab.P_C)):
        ax1.scatter(p[0] * 1e3, p[1] * 1e3, s=60, color=PALETTE["gold"], edgecolor=PALETTE["ink"], zorder=6)
        ax1.text(p[0] * 1e3 + 10, p[1] * 1e3 + 8, name, fontsize=11, fontweight="bold")
    ax1.scatter(0, 0, s=220, marker="s", color=PALETTE["muted"], zorder=4)
    ax1.text(14, -30, "robot base", fontsize=8.5, color=PALETTE["muted"])
    ax1.set_aspect("equal")
    ax1.set_xlabel("x\u2080 [mm]")
    ax1.set_ylabel("y\u2080 [mm]")
    ax1.set_title(
        f"Recovered set-up: camera yawed {np.rad2deg(cal.yaw):.1f}\u00b0, {cal.depth * 1e3:.0f} mm above the ball", loc="left"
    )
    ax1.legend(loc="upper right")
    fig.suptitle("Three taught points are enough to calibrate the camera", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "calibration.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
