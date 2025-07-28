"""lookandmove: image-based visual servoing of a 6-DoF arm with a single overhead camera."""

from . import lab
from .calibration import Calibration, calibrate_least_squares, calibrate_three_points
from .camera import PinholeCamera, look_down_rotation, reduced_pixel_size
from .render import Renderer, Scene
from .robot import ReBeL, euler_zyx, pose_matrix
from .servo import IdealMeasurement, ServoConfig, ServoResult, VisionMeasurement, look_and_move_command, simulate
from .stability import closed_loop_poles, convergence_time, critical_angle, loop_gain, spectral_radius, stability_map
from .vision import Match, cut_template, match_template, ncc_map

__version__ = "1.0.0"

__all__ = [
    "Calibration",
    "IdealMeasurement",
    "Match",
    "PinholeCamera",
    "ReBeL",
    "Renderer",
    "Scene",
    "ServoConfig",
    "ServoResult",
    "VisionMeasurement",
    "calibrate_least_squares",
    "calibrate_three_points",
    "closed_loop_poles",
    "convergence_time",
    "critical_angle",
    "cut_template",
    "euler_zyx",
    "lab",
    "look_and_move_command",
    "look_down_rotation",
    "loop_gain",
    "match_template",
    "ncc_map",
    "pose_matrix",
    "reduced_pixel_size",
    "simulate",
    "spectral_radius",
    "stability_map",
]
