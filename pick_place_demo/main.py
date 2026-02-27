#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Automatic pick-and-place grasp demo entry script.

    Usage:
        conda activate isaaclab
        python main.py --enable_cameras
        python main.py --enable_cameras --num_envs 4 --show-markers
        python main.py --enable_cameras --show-markers --enable-tilt-compensation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to Python path for local imports.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTANT: The AppLauncher must be added and launched here (do not move).
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Automatic pick-and-place grasp demo.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Logging level. Default: INFO",
)
parser.add_argument(
    "--log-file",
    type=str,
    default=None,
    help="Log file path. If None, logs only to console. Default: None",
)
parser.add_argument(
    "--reset-interval-steps",
    type=int,
    default=900,
    help="Reset interval in steps (default: 900 steps = 30 seconds at 30Hz). "
    "This allows enough time for the robot to complete a full pick-place cycle.",
)
parser.add_argument(
    "--grasp-stabilization-time",
    type=int,
    default=20,
    help="Steps to wait after closing gripper (default: 20 steps = 0.67 seconds at 30Hz). "
    "This gives the gripper time to fully close and stabilize before lifting.",
)
parser.add_argument(
    "--place-stabilization-time",
    type=int,
    default=120,
    help="Steps to wait after opening gripper (default: 120 steps = 1 second at 120Hz). "
    "This gives the gripper time to fully open and stabilize before moving.",
)
parser.add_argument(
    "--stabilization-steps",
    type=int,
    default=20,
    help="Steps to wait after reset for physics stabilization (default: 20 steps = 0.167s at 120Hz). "
    "Recommended: 10-30 steps.",
)
parser.add_argument(
    "--show-markers",
    action="store_true",
    default=False,
    help="Show visualization markers (default: False)",
)

# Object pose perturbation.
parser.add_argument(
    "--obj-pos-noise",
    type=float,
    default=0.005,
    help="Object position noise std in meters (default: 0.005 = ±5mm). "
    "Recommended: 0.003-0.008 (3-8mm). Set to 0 to disable.",
)
parser.add_argument(
    "--obj-yaw-noise",
    type=float,
    default=0.03,
    help="Object yaw rotation noise std in radians (default: 0.03). "
    "Recommended: 0.035-0.087 (2-5 degrees). Set to 0 to disable.",
)
parser.add_argument(
    "--obj-size-rand-scale",
    type=float,
    default=5.0,
    help="Object-size-aware spawn range scale factor (default: 5.0). "
    "Larger values increase XY randomization range proportional to object footprint.",
)
parser.add_argument(
    "--obj-spawn-edge-margin",
    type=float,
    default=0.02,
    help="Safety margin to keep object away from table edges in meters (default: 0.02).",
)

# Container (KLT) reset randomization.
parser.add_argument(
    "--klt-pos-range-x",
    type=float,
    default=0.0,
    help="Half range of KLT randomization along X axis in meters (default: 0.0). "
    "If <= 0, uses full constrained X range.",
)
parser.add_argument(
    "--klt-pos-range-y",
    type=float,
    default=0.0,
    help="Half range of KLT randomization along Y axis in meters (default: 0.0). "
    "If <= 0, uses full constrained Y range.",
)
parser.add_argument(
    "--klt-yaw-rand-deg",
    type=float,
    default=30.0,
    help="Max absolute KLT yaw randomization in degrees (default: 30.0). "
    "One shared yaw in [-v, v] is sampled per reset and applied to all parallel envs.",
)
parser.add_argument(
    "--obj-klt-min-dist",
    type=float,
    default=0.15,
    help="Minimum XY separation between object and KLT at reset in meters (default: 0.24). "
    "Increase this if object starts too close to the container.",
)

# Robot initial joint perturbation.
parser.add_argument(
    "--joint-noise",
    type=float,
    default=0.01,
    help="Robot initial joint position noise std in radians (default: 0.01). "
    "Recommended: 0.01-0.02. Set to 0 to disable.",
)

# Timing randomization.
parser.add_argument(
    "--random-hold-min",
    type=int,
    default=3,
    help="Minimum random hold steps at key phases (default: 3). "
    "Recommended: 3-5. Adds timing diversity to trajectories. Set to 0 to disable.",
)
parser.add_argument(
    "--random-hold-max",
    type=int,
    default=8,
    help="Maximum random hold steps at key phases (default: 8). "
    "Recommended: 8-10. Must be >= random-hold-min. Set to 0 to disable.",
)

# Control noise.
parser.add_argument(
    "--control-noise",
    type=float,
    default=0.0,
    help="Control command noise std (default: 0.0). "
    "Recommended: 0.01-0.02. Affects grasp timing and contact. Set to 0 to disable.",
)
parser.add_argument(
    "--disable-non-gripper-collision-check",
    action="store_false",
    dest="enable_non_gripper_collision_check",
    help="Disable non-gripper collision detection.",
)
parser.add_argument(
    "--non-gripper-collision-force-threshold",
    type=float,
    default=20.0,
    help="Force threshold (N) to classify non-gripper collision (default: 20.0).",
)
parser.add_argument(
    "--non-gripper-collision-log-interval",
    type=int,
    default=120,
    help="Log interval (steps) for collision debug output when collision check is enabled (default: 120).",
)

# Tilt-aware grasping configuration.
parser.add_argument(
    "--enable-tilt-compensation",
    action="store_true",
    default=False,
    help="Enable tilt-aware grasping compensation for overturned/tilted objects. "
    "Default: False",
)
parser.add_argument(
    "--tilt-threshold",
    type=float,
    default=15.0,
    help="Minimum tilt angle (degrees) to trigger compensation. Default: 15.0",
)
parser.add_argument(
    "--max-tilt-for-grasp",
    type=float,
    default=75.0,
    help="Maximum tilt angle (degrees) to attempt grasping. Default: 75.0",
)
parser.add_argument(
    "--grasp-strategy",
    type=str,
    default="vertical",
    choices=["vertical", "aligned"],
    help="Grasp strategy for tilted objects.",
)
parser.add_argument(
    "--align-tcp-xy",
    action="store_true",
    default=False,
    help="Align TCP XY-axes with object XY-axes.",
)
parser.add_argument(
    "--no-align-tcp-xy",
    action="store_false",
    dest="align_tcp_xy",
    help="Disable TCP XY alignment.",
)

# Grasp position calibration.
parser.add_argument(
    "--grasp-x-offset",
    type=float,
    default=-0.015,
    help="X-axis offset for grasp position calibration in meters.",
)
parser.add_argument(
    "--grasp-y-offset",
    type=float,
    default=0.0,
    help="Y-axis offset for grasp position calibration in meters.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app (location must remain here per user's constraint)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Remaining imports (after app launch) -------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
import torch

from env.pick_place_env import TableTopSceneCfg
from core.simulation import run_simulator, DEFAULT_DT

# Fixed configuration
ROBOT_TYPE = "Agilebot"
EE_OFFSET_X = 0.0
EE_OFFSET_Y = 0.0
EE_OFFSET_Z = 0.23
ee_offset = torch.tensor([EE_OFFSET_X, EE_OFFSET_Y, EE_OFFSET_Z])
LOG_SEPARATOR = "=" * 50
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def print_startup_parameters(args: argparse.Namespace, ee_offset: torch.Tensor) -> None:
    """Display key startup parameters."""
    logger = logging.getLogger(__name__)
    logger.info(LOG_SEPARATOR)
    logger.info("Startup Configuration")
    logger.info(LOG_SEPARATOR)
    logger.info(f"Robot: {ROBOT_TYPE} | Envs: {args.num_envs}")
    logger.info(f"EE Offset: {ee_offset.tolist()} m")
    logger.info(
        "Reset Interval: %d steps | Stabilization: %d steps",
        args.reset_interval_steps,
        args.stabilization_steps,
    )
    if args.enable_non_gripper_collision_check:
        logger.info(
            "Collision Check: enabled | Threshold: %.2f N",
            args.non_gripper_collision_force_threshold,
        )
    else:
        logger.info("Collision Check: disabled")
    logger.info(LOG_SEPARATOR)


def get_initial_joint_position(
    robot: Any,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Get initial joint positions for the robot."""
    arm_joints = torch.tensor(
        [0.0, 1.57, -1.57, 1.57, -1.57, 0.0], device=device, dtype=torch.float32
    )
    num_joints = robot.data.default_joint_pos.shape[1]
    if num_joints < 6:
        raise ValueError(f"Robot has only {num_joints} joints, expected at least 6")

    padding = torch.zeros(max(0, num_joints - 6), device=device, dtype=torch.float32)
    if padding.numel() > 0:
        arm_joints = torch.cat([arm_joints, padding], dim=0)
    return arm_joints.unsqueeze(0).expand(num_envs, -1).clone()


def setup_logging(log_level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure logging handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def main() -> None:
    """Entry point: build sim, scene and run the simulator loop."""
    log_level = getattr(logging, args_cli.log_level.upper(), logging.INFO)
    setup_logging(log_level=log_level, log_file=args_cli.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting pick-place grasp demo")
    logger.info(f"Log level: {args_cli.log_level}")
    if args_cli.log_file:
        logger.info(f"Log file: {args_cli.log_file}")

    print_startup_parameters(args_cli, ee_offset)

    sim_cfg = sim_utils.SimulationCfg(dt=DEFAULT_DT, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    logger.info("Setup complete...")

    robot = scene["robot"]
    initial_joint_pos = get_initial_joint_position(robot, args_cli.num_envs, sim.device)

    run_simulator(
        sim,
        scene,
        args_cli,
        simulation_app,
        ee_offset,
        initial_joint_pos,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
