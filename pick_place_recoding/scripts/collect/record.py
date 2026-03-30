#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an automatic pick-and-place controller with data recording capabilities.

This script combines the automatic pick-and-place functionality from main.py with the
data recording capabilities from teleop_se3_agent_with_recording.py to create a fully
autonomous pick-and-place system that records its demonstrations to an HDF5 dataset.

Usage:
    # switch the conda environment to isaaclab
    conda activate isaaclab

    # Basic execution
    python scripts/collect/record.py

    # Record demonstrations
    python scripts/collect/record.py --record --num_demos 10 --dataset_file ./my_dataset.hdf5  --enable_cameras

    # Resume recording on existing dataset
    python scripts/collect/record.py --record --num_demos 10 --dataset_file ./my_dataset.hdf5 --resume

    # Multi-environment data collection
    python scripts/collect/record.py --record --num_envs 4 --num_demos 50 --fps 30

    # With Pinocchio IK enabled
    python scripts/collect/record.py --record --num_demos 10 --enable_pinocchio

    # Recommended
    python scripts/collect/record.py --record --enable_pinocchio --enable_cameras  --num_envs 24 --num_demos 250 --no-align-tcp-xy  --resume    --enable-non-gripper-collision-check   --non-gripper-collision-force-threshold 20.0 --dataset_file ./datasets/agilebot_500_no_align.hdf5


Arguments:
    --robot (str):
        Name of the robot. Default: Agilebot

    --num_envs (int):
        Number of environments to spawn. Default: 1

    --record (flag):
        Enable data recording mode. Default: False

    --dataset_file (str):
        File path to export recorded demos. Default: ./datasets/isaac_dataset.hdf5

    --resume (flag):
        Resume recording from existing dataset file. If False and file exists, will raise an error. Default: False

    --num_demos (int):
        Number of demonstrations to record. Set to 0 for infinite. Default: 0

    --fps (int):
        FPS for video recording. Default: 30

    --dof (int):
        Degrees of freedom to retain in joint data. Default: 6

    --enable_pinocchio (flag):
        Enable Pinocchio for IK controllers. Default: False



    --device (str):
        Computation device (inherited from AppLauncher). Default: cuda

    --headless (flag):
        Run in headless mode without GUI (inherited from AppLauncher). Default: False

    --enable_cameras (flag):
        Enable camera rendering for visual observations. Required when using cameras in the scene. Default: False

    --action_frame (str):
        Reference frame for EE6D actions. Default: base (same as world frame)

    --task (str):
        Language instruction describing the task to perform. Default: 'Put the tom '

    --stabilization-steps (int):
        Steps to wait after reset for physics stabilization. Default: 20 (0.167s at 120Hz)
        Recommended: 10-30 steps. More steps = more stable but slower data collection.

    --obj-pos-noise (float):
        Object position noise std in meters. Default: 0.005 (±5mm)
        Recommended: 0.003-0.008 (3-8mm). Set to 0 to disable.

    --obj-yaw-noise (float):
        Object yaw rotation noise std in radians. Default: 0.03 (≈1.7°)
        Recommended: 0.035-0.087 (2-5 degrees). Set to 0 to disable.

    --joint-noise (float):
        Robot initial joint position noise std in radians. Default: 0.01
        Recommended: 0.01-0.02. Set to 0 to disable.
        This significantly improves trajectory diversity.

    --random-hold-min (int):
        Minimum random hold steps at key phases. Default: 3
        Recommended: 3-5. Adds timing diversity to trajectories. Set to 0 to disable.

    --random-hold-max (int):
        Maximum random hold steps at key phases. Default: 8
        Recommended: 8-10. Must be >= random-hold-min. Set to 0 to disable.

    --control-noise (float):
        Control command noise std. Default: 0.01
        Recommended: 0.01-0.02. Affects grasp timing and contact. Set to 0 to disable.

Examples:
    # Record 10 demos with 4 environments (all randomization enabled by default)
    python scripts/collect/record.py --record --num_envs 4 --num_demos 10 --enable_cameras

    # Record 20 demos, custom output path, 60 FPS
    python scripts/collect/record.py --record --num_demos 20 --dataset_file ./demo_60fps.hdf5 --fps 60

    # Infinite recording until manually stopped
    python scripts/collect/record.py --record --dataset_file ./continuous.hdf5

    # Resume recording to add more demos to existing dataset
    python scripts/collect/record.py --record --num_demos 10 --dataset_file ./existing_dataset.hdf5 --resume

    # Disable all randomization (deterministic behavior)
    python scripts/collect/record.py --record --num_envs 4 --num_demos 10 --enable_cameras \
        --obj-pos-noise 0.0 --obj-yaw-noise 0.0 \
        --joint-noise 0.0 --random-hold-min 0 --random-hold-max 0 --control-noise 0.0

    # Custom randomization parameters
    python scripts/collect/record.py --record --num_envs 4 --num_demos 10 --enable_cameras \
        --obj-pos-noise 0.008 --obj-yaw-noise 0.07 \
        --joint-noise 0.015 --random-hold-min 5 --random-hold-max 10
    
    #recommend
    python scripts/collect/record.py --record --enable_pinocchio --enable_cameras --headless  --num_env 32  --num_demo 500 --no-align-tcp-xy  --resume    --enable-non-gripper-collision-check   --non-gripper-collision-force-threshold 20.0 --dataset_file ./datasets/agilebot_500_no_align.hdf5


"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to Python path for local imports after moving entry scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



# IMPORTANT: The AppLauncher must be added and launched here (do not move).
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Automatic pick-and-place controller with data recording capabilities."
)
parser.add_argument(
    "--num_envs", type=int, default=32, help="Number of environments to spawn."
)
# Data recording parameters
parser.add_argument(
    "--record",
    action="store_true",
    default=False,
    help="Whether to enable record function",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/isaac_dataset.hdf5",
    help="File path to export recorded demos.",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume recording from existing dataset file. If False and file exists, will raise an error.",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for infinite.",
)
parser.add_argument("--fps", type=int, default=30, help="FPS for video recording.")
parser.add_argument(
    "--dof",
    type=int,
    default=6,
    help="Degrees of freedom to retain in joint data (e.g., joint_pos, joint_vel). If None, retains all DoF."
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio for IK controllers. Default: False",
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
# Reset interval
parser.add_argument(
    "--reset-interval-steps",
    type=int,
    default=900,
    help="Reset interval in steps (default: 900 steps = 30 seconds at 30Hz). "
         "This allows enough time for the robot to complete a full pick-place cycle.",
)
# Grasp stabilization time
parser.add_argument(
    "--grasp-stabilization-time",
    type=int,
    default=20,
    help="Steps to wait after closing gripper (default: 20 steps = 0.67 seconds at 30Hz). "
         "This gives the gripper time to fully close and stabilize before lifting.",
)
# Place stabilization time
parser.add_argument(
    "--place-stabilization-time",
    type=int,
    default=120,
    help="Steps to wait after opening gripper (default: 120 steps = 1 second at 120Hz). "
         "This gives the gripper time to fully open and stabilize before moving.",
)
# Simulation stabilization steps after reset
parser.add_argument(
    "--stabilization-steps",
    type=int,
    default=20,
    help="Steps to wait after reset for physics stabilization (default: 20 steps = 0.167s at 120Hz). "
         "Recommended: 10-30 steps. More steps = more stable but slower data collection.",
)
parser.add_argument(
    "--buffer-size",
    type=int,
    default=3,
    help="Number of seconds of data to buffer before sending to background process (default: 3).",
)
parser.add_argument(
    "--max-queue-size",
    type=int,
    default=10,
    help="Maximum size of the queue for overflow prevention (default: 10).",
)
# Task/language instruction
parser.add_argument(
    "--task",
    type=str,
    default="put tomato soup can into container",
    help="Language instruction describing the task to perform.",
)
# Action frame for EE6D
parser.add_argument(
    "--action_frame",
    type=str,
    default="base",
    help="Reference frame for EE6D actions. Default: base (same as world frame)",
)
# Visualization markers
parser.add_argument(
    "--show-markers",
    action="store_true",
    default=False,
    help="Show visualization markers (default: False)",
)



# Object pose perturbation (already implemented, but adding control)
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
    help="Object yaw rotation noise std in radians (default: 0.03 ≈ 1.7°). "
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

# Container (KLT) reset randomization
parser.add_argument(
    "--klt-pos-range-x",
    type=float,
    default=0.005,
    help="Half range of KLT randomization along X axis in meters (default: 0.005). "
         "If <= 0, disables X-axis randomization.",
)
parser.add_argument(
    "--klt-pos-range-y",
    type=float,
    default=0.005,
    help="Half range of KLT randomization along Y axis in meters (default: 0.005). "
         "If <= 0, disables Y-axis randomization.",
)
parser.add_argument(
    "--klt-yaw-rand-deg",
    type=float,
    default=2.0,
    help="Max absolute KLT yaw randomization in degrees (default: 2.0). "
         "One shared yaw in [-v, v] is sampled per reset and applied to all parallel envs.",
)
parser.add_argument(
    "--obj-klt-min-dist",
    type=float,
    default=0.15,
    help="Minimum XY separation between object and KLT at reset in meters (default: 0.15). "
         "Increase this if object starts too close to the container.",
)
parser.add_argument(
    "--obj-klt-min-dist-max-attempts",
    type=int,
    default=50,
    help="Maximum strict resampling attempts to satisfy obj-klt min distance (default: 50). "
         "If exhausted, reset is marked invalid and skipped.",
)
parser.add_argument(
    "--max-invalid-reset-streak",
    type=int,
    default=20,
    help="Abort if consecutive invalid resets reach this limit (default: 20).",
)
parser.add_argument(
    "--global-z-rot-max-deg",
    type=float,
    default=0.0,
    help="Shared global Z-rotation range per reset in degrees (default: 0.0, disabled). "
         "If >0, samples one angle in [-v, v] for all envs.",
)

# Robot initial joint perturbation (IMPORTANT for trajectory diversity)
parser.add_argument(
    "--joint-noise",
    type=float,
    default=0.01,
    help="Robot initial joint position noise std in radians (default: 0.01). "
         "Recommended: 0.01-0.02. Set to 0 to disable. "
         "This significantly improves trajectory diversity.",
)

# Timing randomization (often overlooked but important for VLA)
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

# Control noise (optional, use with caution)
parser.add_argument(
    "--control-noise",
    type=float,
    default=0.,
    help="Control command noise std (default: 0.01). "
         "Recommended: 0.01-0.02. Affects grasp timing and contact. Set to 0 to disable.",
)
parser.add_argument(
    "--enable-non-gripper-collision-check",
    action="store_true",
    default=True,
    help="Enable non-gripper collision detection during grasp phases. "
         "When enabled, contact forces on non-gripper robot bodies are monitored. "
         "Default: enabled.",
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

# ==============================================================================
# Tilt-Aware Grasping Configuration
# ==============================================================================
parser.add_argument(
    "--enable-tilt-compensation",
    action="store_true",
    default=False,
    help="Enable tilt-aware grasping compensation for overturned/tilted objects. "
         "When enabled, the gripper orientation is adjusted to maintain horizontal "
         "grasping even when the object's Z-axis is not vertical. Default: False",
)
parser.add_argument(
    "--tilt-threshold",
    type=float,
    default=15.0,
    help="Minimum tilt angle (degrees) to trigger compensation. "
         "Objects tilted less than this angle use standard grasp. Default: 15.0",
)
parser.add_argument(
    "--max-tilt-for-grasp",
    type=float,
    default=75.0,
    help="Maximum tilt angle (degrees) to attempt grasping. "
         "Objects tilted more than this are considered ungraspable. Default: 75.0",
)
parser.add_argument(
    "--grasp-strategy",
    type=str,
    default="vertical",
    choices=["vertical", "aligned"],
    help="Grasp strategy for tilted objects. "
         "'vertical': Gripper stays vertical (recommended for slight tilts). "
         "'aligned': Gripper follows object tilt (for large angle tilts). "
         "Default: vertical",
)
parser.add_argument(
    "--align-tcp-xy",
    action="store_true",
    default=False,
    help="Align TCP XY-axes with object XY-axes (default: False). "
         "When True (use --align-tcp-xy): TCP_X = Object_X, TCP_Y = Object_Y. "
         "When False: TCP_X = -Object_Y, TCP_Y = Object_X. "
         "This enables 'grasping both ends' mode for cylindrical objects. "
         "TCP_Z always = -Object_Z (grasp from above).",
)
parser.add_argument(
    "--no-align-tcp-xy",
    action="store_false",
    dest="align_tcp_xy",
    help="Disable TCP XY alignment. TCP_X will align with -Object_Y instead. "
         "Use this for 'grasping both ends' mode.",
)

# ==============================================================================
# Grasp Position Calibration (for gripper centering)
# ==============================================================================
parser.add_argument(
    "--grasp-x-offset",
    type=float,
    default=-0.015,

    help="X-axis offset for grasp position calibration (meters). "
         "Positive = forward from calculated grasp point. "
         "Use to compensate for TCP definition mismatch or gripper asymmetry. "
         "Typical range: -0.02 to 0.02. Default: 0.0",
)
parser.add_argument(
    "--grasp-y-offset",
    type=float,
    default=0.0,
    help="Y-axis offset for grasp position calibration (meters). "
         "Positive = left from calculated grasp point. "
         "CRITICAL for gripper centering: if gripper grasps off-center, adjust this. "
         "If left finger contacts first, use positive value. "
         "If right finger contacts first, use negative value. "
         "Typical range: -0.02 to 0.02. Default: 0.0",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app (location must remain here per user's constraint)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Remaining imports (after app launch) -------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
import torch
# project-specific imports
from env.pick_place_env import TableTopSceneCfg



from core.simulation import run_simulator, DEFAULT_DT

# Fixed configuration
ROBOT_TYPE = "Agilebot"  # Robot type (fixed)

# Fixed end-effector offset: distance from flange to grasp point (gripper length)
EE_OFFSET_X = 0.0
EE_OFFSET_Y = 0.0
EE_OFFSET_Z = 0.23  # Length of the gripper in meters
ee_offset = torch.tensor([EE_OFFSET_X, EE_OFFSET_Y, EE_OFFSET_Z])
LOG_SEPARATOR = "=" * 50
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def print_startup_parameters(args: argparse.Namespace, ee_offset: torch.Tensor) -> None:
    """Display key startup parameters."""
    logger = logging.getLogger(__name__)
    logger.info(LOG_SEPARATOR)
    logger.info("Startup Configuration")
    logger.info(LOG_SEPARATOR)
    logger.info(f"Robot: {ROBOT_TYPE} | Envs: {args.num_envs} | FPS: {args.fps}")
    logger.info(f"Recording: {args.record} | File: {args.dataset_file}")
    if args.record:
        demo_target = args.num_demos if args.num_demos > 0 else "Infinite"
        logger.info(f"Demos: {demo_target} | Resume: {args.resume}")
    logger.info(f"EE Offset: {ee_offset.tolist()} m")
    logger.info(f"Task: {args.task}")
    if args.enable_non_gripper_collision_check:
        logger.info(
            "Collision Check: enabled | Threshold: "
            f"{args.non_gripper_collision_force_threshold:.2f} N"
        )
    else:
        logger.info("Collision Check: disabled")
    logger.info(LOG_SEPARATOR)


def get_initial_joint_position(
    robot: Any,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Get the initial joint position for the robot.

    Args:
        robot: Robot entity (used to get the number of joints).
        num_envs: Number of environments.
        device: Device to use for tensors.

    Returns:
        initial_joint_pos: Tensor of shape (num_envs, num_joints)
    """
    # Initial arm joint positions: [0.0, pi/2, -pi/2, pi/2, -pi/2, 0.0]
    arm_joints = torch.tensor(
        [0.0, torch.pi / 2, -torch.pi / 2, torch.pi / 2, -torch.pi / 2, 0.0],
        device=device,
        dtype=torch.float32,
    )

    # Get actual joint count from robot
    num_joints = robot.data.default_joint_pos.shape[1]

    if num_joints < 6:
        raise ValueError(f"Robot has only {num_joints} joints, expected at least 6")

    # Pad with zeros for gripper joints if present
    padding = torch.zeros(max(0, num_joints - 6), device=device, dtype=torch.float32)
    if padding.numel() > 0:
        arm_joints = torch.cat([arm_joints, padding], dim=0)

    return arm_joints.unsqueeze(0).expand(num_envs, -1).clone()


def setup_logging(log_level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure logging system.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Log file path, if None only output to console
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # If needed, add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def main() -> None:
    """Entry point: build sim, scene and run the simulator loop."""
    # Setup logging
    log_level = getattr(logging, args_cli.log_level.upper(), logging.INFO)
    setup_logging(log_level=log_level, log_file=args_cli.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting pick-place recording system")
    logger.info(f"Log level: {args_cli.log_level}")
    if args_cli.log_file:
        logger.info(f"Log file: {args_cli.log_file}")

    # Display startup parameters
    print_startup_parameters(args_cli, ee_offset)

    sim_cfg = sim_utils.SimulationCfg(dt=DEFAULT_DT, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera to a stable view
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Build scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    # set robot in scene
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    logger.info("Setup complete...")

    # Get the initial joint position from the robot configuration
    # This ensures we use the configured initial pose instead of USD defaults
    robot = scene["robot"]
    initial_joint_pos = get_initial_joint_position(robot, args_cli.num_envs, sim.device)

    # run the main loop
    run_simulator(
        sim,
        scene,
        args_cli,
        simulation_app,
        ee_offset,
        initial_joint_pos,
        show_markers=args_cli.show_markers,
    )  # type: ignore


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
