"""Template matching on a synthetic frame: score map, sub-pixel peak and the effect of the ROI (task 5.1)."""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from _common import PALETTE, ROBOT, TARGET_PX, TRUE_CAMERA, save
from matplotlib.patches import Rectangle

from lookandmove import Renderer, match_template, pose_matrix


def main() -> None:
    renderer = Renderer(TRUE_CAMERA)
    q = ROBOT.ik(pose_matrix(0.30, -0.05, 0.40, np.pi, -np.pi / 2, 0.0), np.deg2rad([8.88, -127.32, 171.91, 167.63, 46.09, 8.65]))
    tcp = ROBOT.tcp(q)
    target_point = TRUE_CAMERA.backproject(TARGET_PX, TRUE_CAMERA.depth_of_plane(-0.252))
    frame = renderer.render(tcp, target_point, ROBOT.frames(q)[:, :3, 3], rng=np.random.default_rng(4))
    template = renderer.teach_template()
    truth = TRUE_CAMERA.project(tcp)

    t0 = time.perf_counter()
    full = match_template(frame, template)
    t_full = time.perf_counter() - t0
    roi = (int(truth[0]) - 80, int(truth[1]) - 80, 160, 160)
    t0 = time.perf_counter()
    small = match_template(frame, template, roi)
    t_roi = time.perf_counter() - t0
    print(f"truth {truth}, full frame {full.feature} ({t_full * 1e3:.1f} ms), ROI {small.feature} ({t_roi * 1e3:.1f} ms)")
    print(f"ROI reports {small.feature_roi} = feature - ROI origin {roi[:2]}")

    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.42, 1.35], wspace=0.12)
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(frame, cmap="gray", vmin=0, vmax=255)
    ax0.add_patch(Rectangle(roi[:2], roi[2], roi[3], fill=False, ec=PALETTE["gold"], lw=2, ls=(0, (4, 2))))
    h = template.shape[0]
    ax0.add_patch(Rectangle(full.feature - h / 2, h, h, fill=False, ec=PALETTE["green"], lw=2))
    ax0.scatter(*TARGET_PX, marker="+", s=200, color=PALETTE["red"], lw=2)
    ax0.text(roi[0], roi[1] - 8, f"ROI, origin O_ROI = {roi[:2]}", color=PALETTE["gold"], fontsize=8.5, fontweight="bold")
    ax0.text(TARGET_PX[0] + 10, TARGET_PX[1] + 26, "target button", color=PALETTE["red"], fontsize=8.5, fontweight="bold")
    ax0.set_title("Synthetic camera frame", loc="left")
    ax0.grid(False)
    ax0.set_xticks([])
    ax0.set_yticks([])

    axt = fig.add_subplot(gs[1])
    axt.imshow(template, cmap="gray", vmin=0, vmax=255)
    axt.set_title("Template\n(teach mode)", loc="left")
    axt.grid(False)
    axt.set_xticks([])
    axt.set_yticks([])

    ax2 = fig.add_subplot(gs[2])
    im = ax2.imshow(
        full.score_map,
        cmap="magma",
        vmin=-0.3,
        vmax=1.0,
        extent=(h / 2, h / 2 + full.score_map.shape[1], h / 2 + full.score_map.shape[0], h / 2),
    )
    ax2.scatter(*full.feature, s=160, facecolor="none", edgecolor=PALETTE["green"], lw=2)
    fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02, label="normalised cross-correlation")
    err = np.linalg.norm(full.feature - truth)
    ax2.set_title(f"Score map: peak {full.score:.2f}, error {err:.2f} px after sub-pixel fit", loc="left")
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.text(
        0.5,
        0.08,
        f"Full frame: {t_full * 1e3:.0f} ms.  160 \u00d7 160 ROI: {t_roi * 1e3:.0f} ms, "
        f"reports f = {np.round(small.feature_roi, 1)} px, "
        f"and f + O_ROI = {np.round(small.feature, 1)} px is the same image point.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["ink"],
    )
    fig.suptitle("Finding the ball: zero-mean normalised cross-correlation", fontsize=13, fontweight="bold")
    save(fig, "template_matching.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
