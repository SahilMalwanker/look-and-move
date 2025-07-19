"""Hero animation: the full look-and-move loop with a camera whose yaw estimate is 45 deg off."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import CALIBRATION, MEDIA, PALETTE, Q_START, ROBOT, TABLE_Z, TARGET_PX, TRUE_CAMERA, save_gif
from matplotlib.patches import Rectangle

from lookandmove import IdealMeasurement, PinholeCamera, Renderer, ServoConfig, VisionMeasurement, simulate

GAIN = 3.0
DURATION = 7.0
STRIDE = 2
SHOWN = 45
OTHERS = (0, 70, 85)
COLOURS = {0: PALETTE["blue"], 45: PALETTE["teal"], 70: PALETTE["orange"], 85: PALETTE["purple"]}


def model_camera(yaw_error_deg: float) -> PinholeCamera:
    return PinholeCamera.looking_down(TRUE_CAMERA.position, CALIBRATION.yaw + np.deg2rad(yaw_error_deg))


def main() -> None:
    cfg = ServoConfig(gain=GAIN, duration=DURATION)
    target_point = TRUE_CAMERA.backproject(TARGET_PX, TRUE_CAMERA.depth_of_plane(TABLE_Z))
    renderer = Renderer(TRUE_CAMERA)
    vision = VisionMeasurement(renderer, target_point=target_point, seed=5)
    runs = {SHOWN: simulate(ROBOT, Q_START, TARGET_PX, model_camera(SHOWN), vision, cfg)}
    for err in OTHERS:
        runs[err] = simulate(ROBOT, Q_START, TARGET_PX, model_camera(err), IdealMeasurement(TRUE_CAMERA, 0.25, seed=1), cfg)
    for err, res in sorted(runs.items()):
        print(f"yaw error {err:3d} deg: settled after {res.settling_time():.2f} s, lost = {res.lost}")
    shown = runs[SHOWN]
    # The controller saw noisy frames; the animation shows the same scenes without sensor noise (smaller GIF).
    cycle_of_frame = np.searchsorted(shown.t, shown.frame_t - 1e-9)
    frames = [renderer.render(ROBOT.tcp(shown.q[i]), target_point, ROBOT.frames(shown.q[i])[:, :3, 3]) for i in cycle_of_frame]
    n = len(frames)

    fig = plt.figure(figsize=(11, 5.0))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.3, 1],
        height_ratios=[1, 1.3],
        hspace=0.42,
        wspace=0.14,
        left=0.02,
        right=0.98,
        top=0.88,
        bottom=0.08,
    )
    fig.suptitle(
        f"Dynamic look-and-move (K_P = {GAIN:g} s\u207b\u00b9, 17.5 fps) with the camera yaw mis-calibrated by {SHOWN}\u00b0",
        fontsize=12,
        fontweight="bold",
    )
    ax = fig.add_subplot(gs[:, 0])
    img = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255)
    start = shown.feature_true[0]
    ax.plot([start[0], TARGET_PX[0]], [start[1], TARGET_PX[1]], color="white", lw=1.2, ls=(0, (4, 3)))
    ax.scatter(*TARGET_PX, marker="+", s=260, color=PALETTE["red"], lw=2.4, zorder=6)
    (trail,) = ax.plot([], [], color=PALETTE["teal"], lw=2.2)
    half = vision.template.shape[0] / 2
    box = Rectangle((0, 0), 2 * half, 2 * half, fill=False, ec=PALETTE["green"], lw=2)
    ax.add_patch(box)
    clock = ax.text(10, 22, "", color="white", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Synthetic camera frame, template match (green), target (red)", loc="left", fontsize=10)

    axe = fig.add_subplot(gs[0, 1])
    axp = fig.add_subplot(gs[1, 1])
    err_lines, path_lines = {}, {}
    for err in (0, SHOWN, *OTHERS[1:]):
        c = COLOURS[err]
        (err_lines[err],) = axe.semilogy([], [], color=c, lw=1.8, label=f"\u0394\u03b8 = {err}\u00b0")
        (path_lines[err],) = axp.plot([], [], color=c, lw=1.8)
    axe.set_xlim(0, DURATION)
    axe.set_ylim(0.1, 600)
    axe.set_ylabel("image error [px]")
    axe.set_xlabel("time [s]", labelpad=1)
    axe.legend(ncol=2, loc="lower left", fontsize=8, handlelength=1.2, columnspacing=0.8)
    axe.set_title("Distance to the target in the image", loc="left", fontsize=10)
    axp.add_patch(Rectangle((0, 0), 640, 480, fill=False, ec=PALETTE["muted"], lw=1))
    axp.scatter(*TARGET_PX, marker="+", s=120, color=PALETTE["red"], lw=2, zorder=5)
    axp.set_xlim(-20, 660)
    axp.set_ylim(500, -20)
    axp.set_aspect("equal")
    axp.set_xticks([0, 320, 640])
    axp.set_yticks([0, 240, 480])
    axp.set_title("Ball path in the image", loc="left", fontsize=10)

    def update(k: int):
        img.set_data(frames[k])
        f = shown.feature[k]
        if np.all(np.isfinite(f)):
            box.set_xy((f[0] - half, f[1] - half))
        trail.set_data(shown.feature_true[: k + 1, 0], shown.feature_true[: k + 1, 1])
        clock.set_text(f"t = {shown.frame_t[k]:.2f} s")
        for err, res in runs.items():
            m = min(k + 1, len(res.frame_t))
            err_lines[err].set_data(res.frame_t[:m], np.maximum(res.error_px[:m], 0.1))
            path_lines[err].set_data(res.feature_true[:m, 0], res.feature_true[:m, 1])
        return [img, box, trail, clock, *err_lines.values(), *path_lines.values()]

    picks = list(range(0, n, STRIDE))
    if picks[-1] != n - 1:
        picks.append(n - 1)
    save_gif(fig, update, picks, "look_and_move.gif", fps=17.5 / STRIDE, dpi=82, colors=128)
    update(n - 1)
    fig.savefig(MEDIA / "look_and_move.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
