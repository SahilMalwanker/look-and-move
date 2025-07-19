"""Regenerate every figure and animation in docs/media."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "calibration_from_lab_data.py",
    "accuracy_five_poses.py",
    "template_matching.py",
    "stability_map.py",
    "servo_animation.py",
]

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    for name in SCRIPTS:
        print(f"== {name}")
        runpy.run_path(str(HERE / name), run_name="__main__")
