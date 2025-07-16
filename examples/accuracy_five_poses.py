"""One ball position, five joint configurations: what the camera says about the robot's accuracy (task 5.2)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import CALIBRATION, PALETTE, ROBOT, save

from lookandmove import lab

COLOURS = [PALETTE["teal"], PALETTE["orange"], PALETTE["blue"], PALETTE["purple"], PALETTE["red"]]


def main() -> None:
    Q = np.deg2rad(lab.TEST_CASES_DEG)
    fig = plt.figure(figsize=(12.5, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for i, q in enumerate(Q):
        pts = ROBOT.frames(q)[:, :3, 3]
        ax.plot(*pts[:7].T, color=COLOURS[i], lw=4, alpha=0.85, solid_capstyle="round", label=f"test case {i + 1}")
        ax.plot(*pts[6:].T, color=PALETTE["ink"], lw=2)
        ax.scatter(*pts[1:7].T, color=PALETTE["ink"], s=10)
    ax.scatter(*lab.P_A, s=120, color=PALETTE["gold"], edgecolor=PALETTE["ink"], zorder=10)
    ax.plot([0, 0], [0, 0], [-0.252, 0], color=PALETTE["muted"], lw=8, alpha=0.6)
    ax.set_xlim(-0.2, 0.45)
    ax.set_ylim(-0.3, 0.35)
    ax.set_zlim(-0.25, 0.55)
    ax.set_box_aspect((0.65, 0.65, 0.8), zoom=1.15)
    ax.view_init(elev=18, azim=-120)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Five configurations, one ball position", loc="left")

    ax2 = fig.add_subplot(1, 2, 2)
    f = lab.TEST_CASE_FEATURES
    mm = CALIBRATION.metres_per_pixel * 1e3
    valid = ~np.isnan(f).any(axis=1)
    centre = f[valid].mean(axis=0)
    for i, (x, y) in enumerate(f):
        if np.isnan(x):
            ax2.axhline(y, color=COLOURS[i], lw=1.5, ls=(0, (4, 3)), label=f"test case {i + 1} (only y recorded)")
        else:
            ax2.scatter(x, y, s=110, color=COLOURS[i], edgecolor=PALETTE["ink"], zorder=5, label=f"test case {i + 1}")
    d = np.linalg.norm(f[valid][:, None] - f[valid][None], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    a, b = f[valid][i], f[valid][j]
    ax2.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "<->", "lw": 1.3, "color": PALETTE["ink"]})
    ax2.text(
        *(0.5 * (a + b) + [-0.6, 0.0]),
        f"{d.max():.1f} px\n\u2248 {d.max() * mm:.1f} mm",
        fontsize=10,
        fontweight="bold",
        ha="right",
    )
    rep = 1.0 / mm
    ax2.add_patch(plt.Circle(centre, rep, color=PALETTE["green"], alpha=0.25, lw=0))
    ax2.text(
        centre[0] + 1.6, centre[1] + 0.4, "\u00b11 mm repeatability\n(datasheet)", fontsize=8, color=PALETTE["green"], ha="left"
    )
    ax2.set_aspect("equal")
    ax2.set_xlim(52, 72)
    ax2.set_ylim(194, 177)
    ax2.set_xlabel("feature x [px]  (ROI coordinates)")
    ax2.set_ylabel("feature y [px]")
    ax2.set_title(f"Measured feature  (1 px \u2248 {mm:.2f} mm)", loc="left")
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Absolute accuracy: same commanded position, up to 8 mm apart", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "five_poses.png")
    plt.close(fig)
    print(f"largest spread {d.max():.1f} px = {d.max() * mm:.1f} mm (scale {mm:.4f} mm/px)")


if __name__ == "__main__":
    main()
