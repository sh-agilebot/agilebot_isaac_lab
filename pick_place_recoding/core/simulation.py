#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Main simulation module for automatic pick-and-place with data recording.
"""

import sys
from pathlib import Path

# Add project root to Python path for local imports when used as a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import torch
from typing import Any

logger = logging.getLogger(__name__)

# Import non-omniverse modules at module level
from controller.pick_place_controller import ParallelPickPlaceController
from controller.phases.state_machine import GraspPhase

from core.common_utils import reset_everything, compute_grasp_and_place, make_markers
from core.math_utils import quat_to_rot6d


# Constants
DEFAULT_DT = 1 / 120
CENTER_OFFSET = torch.tensor([0.0, 0.0, 0.015])  # Object center to grasp point offset
GRASP_XY_OFFSET = torch.tensor([0.0, 0.0])  # Gripper centering calibration

# Drop detection thresholds
DROP_DISTANCE_THRESHOLD = 0.10  # 10cm - if TCP-cube distance exceeds this, consider dropped
DROP_HEIGHT_THRESHOLD = 0.05   # 5cm - if cube drops significantly from held position

# Gripper bodies are excluded from collision checks by name.
GRIPPER_BODY_NAME_KEYWORDS = ("gripper", "finger", "knuckle", "pad", "arg2f")
COLLISION_CHECK_PHASES = [phase for phase in GraspPhase if phase != GraspPhase.DONE]


def build_phase_mask(
    current_phases: torch.Tensor,
    phases: list[GraspPhase],
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean mask for envs whose phase is in `phases`."""
    mask = torch.zeros(current_phases.shape[0], dtype=torch.bool, device=device)
    for phase in phases:
        mask |= current_phases == phase.value
    return mask


def get_camera_intrinsics(scene) -> dict:
    """Extract camera intrinsic matrices from scene sensors."""
    camera_intrinsics = {}
    for sensor_name, sensor in scene.sensors.items():
        if hasattr(sensor, 'data') and hasattr(sensor.data, 'intrinsic_matrices'):
            intrinsic_matrix = sensor.data.intrinsic_matrices[0].cpu().numpy()
            if "main_camera" in sensor_name.lower():
                camera_intrinsics["main_camera"] = intrinsic_matrix.tolist()
            elif "wrist" in sensor_name.lower():
                camera_intrinsics["wrist_camera"] = intrinsic_matrix.tolist()
    return camera_intrinsics


def check_object_dropped(
    cube_pos_w: torch.Tensor,
    tcp_pos_w: torch.Tensor,
    current_phases: torch.Tensor,
    grasp_offset_recorded: torch.Tensor,
    cube_height_at_grasp: torch.Tensor,
    drop_flags: torch.Tensor,
    device: torch.device,
) -> tuple:
    """Check if object has dropped during gripper-closed phases.
    
    Drop detection logic:
    - During LIFT, MOVE_TO_PLACE, PLACE_DESCENT phases (after gripper firmly closed),
      the TCP and cube should maintain relative position.
    - If the distance between TCP and cube exceeds threshold, mark as dropped.
    - GRASP_CLOSED phase is excluded to allow gripper stabilization.
    
    Args:
        cube_pos_w: Cube world position [N, 3]
        tcp_pos_w: TCP world position [N, 3]
        current_phases: Current phase for each env [N,]
        grasp_offset_recorded: Recorded TCP-cube offset when grasp established [N, 3]
        cube_height_at_grasp: Cube height when grasp established [N,]
        drop_flags: Current drop status [N,]
        device: Torch device
        
    Returns:
        Tuple of (updated_drop_flags, updated_grasp_offset, updated_cube_height)
    """
    # Phases where gripper is firmly closed and cube should follow TCP
    # NOTE: GRASP_CLOSED is excluded - gripper is still stabilizing
    GRIPPER_FIRM_PHASES = [
        GraspPhase.LIFT,
        GraspPhase.MOVE_TO_PLACE,
        GraspPhase.PLACE_DESCENT,
    ]
    
    # Record baseline when first entering LIFT phase (gripper is now firm)
    entering_lift = (current_phases == GraspPhase.LIFT.value) & (cube_height_at_grasp == 0)
    if entering_lift.any():
        # Record TCP-cube offset as baseline when starting to lift
        grasp_offset_recorded[entering_lift] = (
            tcp_pos_w[entering_lift] - cube_pos_w[entering_lift]
        )
        cube_height_at_grasp[entering_lift] = cube_pos_w[entering_lift, 2]
    
    # Check if in firm-grip phase
    in_firm_phase = build_phase_mask(current_phases, GRIPPER_FIRM_PHASES, device)
    
    # Check for drop during firm phases
    if in_firm_phase.any() and (~drop_flags).any():
        check_mask = in_firm_phase & (~drop_flags)
        
        # Check TCP-cube distance change from baseline
        current_offset = tcp_pos_w - cube_pos_w
        offset_diff = (current_offset - grasp_offset_recorded).norm(dim=1)
        distance_exceeded = offset_diff > DROP_DISTANCE_THRESHOLD
        
        # Mark as dropped
        newly_dropped = check_mask & distance_exceeded
        drop_flags = drop_flags | newly_dropped
        
        # Log drops for debugging
        if newly_dropped.any():
            dropped_envs = torch.where(newly_dropped)[0]
            for env_idx in dropped_envs[:3]:  # Log first 3
                phase_name = GraspPhase(current_phases[env_idx].item()).name
                logger.info(
                    f"Env {env_idx}: Drop detected in phase {phase_name}, "
                    f"offset_diff={offset_diff[env_idx]:.4f}m"
                )
    
    return drop_flags, grasp_offset_recorded, cube_height_at_grasp


def resolve_non_gripper_body_ids(collision_sensor, device: torch.device) -> tuple[torch.Tensor, list[str], list[str]]:
    """Resolve non-gripper body ids from a whole-robot contact sensor."""
    all_body_names = collision_sensor.body_names
    non_gripper_body_ids: list[int] = []
    excluded_body_names: list[str] = []

    for body_id, body_name in enumerate(all_body_names):
        lower_name = body_name.lower()
        is_gripper_body = any(keyword in lower_name for keyword in GRIPPER_BODY_NAME_KEYWORDS)
        if is_gripper_body:
            excluded_body_names.append(body_name)
        else:
            non_gripper_body_ids.append(body_id)

    if not non_gripper_body_ids:
        raise RuntimeError(
            "No non-gripper robot bodies resolved for collision check. "
            f"Resolved bodies: {all_body_names}"
        )

    return torch.tensor(non_gripper_body_ids, device=device, dtype=torch.long), all_body_names, excluded_body_names


def check_non_gripper_collision(
    contact_forces_w: torch.Tensor,
    current_phases: torch.Tensor,
    non_gripper_body_ids: torch.Tensor,
    collision_flags: torch.Tensor,
    force_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect collisions on non-gripper bodies during active grasp phases.

    Returns:
        Tuple of:
            - updated collision flags [N]
            - newly collided env mask [N]
            - max non-gripper contact force per env [N]
            - max-force body id in original sensor body index [N]
    """
    in_check_phase = build_phase_mask(current_phases, COLLISION_CHECK_PHASES, device)

    non_gripper_forces = contact_forces_w.index_select(1, non_gripper_body_ids)
    non_gripper_force_norm = torch.norm(non_gripper_forces, dim=-1)
    max_force_per_env, max_local_ids = non_gripper_force_norm.max(dim=1)
    max_body_ids = non_gripper_body_ids[max_local_ids]

    is_collision = in_check_phase & (max_force_per_env > force_threshold)
    newly_collided = is_collision & (~collision_flags)
    updated_collision_flags = collision_flags | is_collision
    return updated_collision_flags, newly_collided, max_force_per_env, max_body_ids


def stabilize_simulation(sim, scene, sim_dt: float, steps: int) -> None:
    """Run simulation steps for physics stabilization."""
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


def perform_episode_reset(
    robot, grasp_controller, cubes, scene, device, sim, sim_dt,
    initial_joint_pos, small_KLT, args, center_offset, grasp_xy_offset,
    robot_entity_cfg, num_envs
):
    """Perform full episode reset with stabilization and grasp pose computation.
    
    Returns:
        Tuple of (cube_initial_pos_list, cube_initial_quat_list, nominal_pos_local,
                  grasp_pos, grasp_quat, place_pos, place_quat)
    """
    from isaaclab.utils.math import quat_apply
    
    # Reset robot and objects
    pos_list, quat_list, nominal_pos_local, is_valid_reset = reset_everything(
        robot, grasp_controller, cubes, scene, device, initial_joint_pos, small_KLT,
        joint_noise_std=args.joint_noise,
        obj_pos_noise_std=args.obj_pos_noise,
        obj_yaw_noise_std=args.obj_yaw_noise,
        obj_size_rand_scale=args.obj_size_rand_scale,
        obj_spawn_edge_margin=args.obj_spawn_edge_margin,
        klt_pos_range_x=args.klt_pos_range_x,
        klt_pos_range_y=args.klt_pos_range_y,
        klt_yaw_rand_deg=args.klt_yaw_rand_deg,
        klt_object_min_dist=args.obj_klt_min_dist,
        klt_object_min_dist_max_attempts=args.obj_klt_min_dist_max_attempts,
        global_z_rot_max_deg=args.global_z_rot_max_deg,
        random_hold_min=args.random_hold_min,
        random_hold_max=args.random_hold_max,
    )
    if not is_valid_reset:
        return None
    
    # Stabilize simulation
    stabilize_simulation(sim, scene, sim_dt, args.stabilization_steps)
    
    # Compute grasp & place poses
    cube = cubes[0]
    cube_pos_w = cube.data.root_pos_w.clone()
    cube_quat_w = cube.data.root_quat_w.clone()
    klt_pos_w = small_KLT.data.root_pos_w.clone()
    
    # Get end-effector pose for symmetry-aware grasp selection
    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    ee_pos_w = ee_pose_w[:, 0:3]
    ee_quat_w = ee_pose_w[:, 3:7]
    
    grasp_pos, grasp_quat, place_pos, place_quat = compute_grasp_and_place(
        cube_pos_w, cube_quat_w, scene.env_origins, center_offset,
        nominal_pos_local, quat_list[0], device, klt_pos_w,
        ee_pos_w=ee_pos_w, ee_quat_w=ee_quat_w,
        consider_symmetry=True,
        enable_tilt_compensation=args.enable_tilt_compensation,
        tilt_threshold=args.tilt_threshold,
        max_tilt_for_grasp=args.max_tilt_for_grasp,
        grasp_xy_offset=grasp_xy_offset,
        grasp_strategy=args.grasp_strategy,
        align_tcp_xy=args.align_tcp_xy,
    )
    
    return pos_list, quat_list, nominal_pos_local, grasp_pos, grasp_quat, place_pos, place_quat

def create_ee6d_data_collector(args):
    """Create EE6D data collector with HDF5 I/O."""
    from core.ee6d_background_data_collector import EE6DDataCollector
    return EE6DDataCollector(
        dataset_file=args.dataset_file,
        fps=args.fps,
        dof=args.dof,
        sim_frequency=int(1 / DEFAULT_DT),
        resume=args.resume,
        action_frame=args.action_frame,
        buffer_size=args.buffer_size,
        max_queue_size=args.max_queue_size,
        language_instruction=args.task
    )

def run_simulator(
    sim, scene, args: argparse.Namespace,
    simulation_app, ee_offset: torch.Tensor = None,
    initial_joint_pos: torch.Tensor = None,
    show_markers: bool = False,
) -> None:
    """Main simulation loop."""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply

    # Extract scene entities
    robot = scene["robot"]
    tomato_soup_can = scene["tomato_soup_can"]
    small_KLT = scene["small_KLT"]
    cubes = [tomato_soup_can]
    collision_sensor = scene.sensors.get("collision_sensor")

    # Setup robot entity cfg
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=[".*"], body_names=["robotiq_arg2f_base_link"]
    )
    robot_entity_cfg.resolve(scene)

    # Create grasping controller
    grasp_controller = ParallelPickPlaceController(
        robot, scene, sim, robot_entity_cfg, "Agilebot", ee_offset,
        grasp_stabilization_time=args.grasp_stabilization_time,
        place_stabilization_time=args.place_stabilization_time
    )

    # Initialize data collector
    data_collector = None
    sim_dt = sim.get_physics_dt()
    if args.record:
        data_collector = create_ee6d_data_collector(args)
        print(f"Using HDF5 I/O (buffer_size={args.buffer_size}s)")
        
    markers = make_markers(scene.num_envs, show_markers=args.show_markers)
    device = sim.device
    num_envs = scene.num_envs

    collision_check_enabled = bool(args.enable_non_gripper_collision_check and collision_sensor is not None)
    collision_log_interval = max(1, int(args.non_gripper_collision_log_interval))
    non_gripper_body_ids = torch.empty(0, dtype=torch.long, device=device)
    collision_sensor_body_names: list[str] = []
    if args.enable_non_gripper_collision_check and collision_sensor is None:
        logger.warning(
            "Collision check requested but 'collision_sensor' is missing in scene config. "
            "Non-gripper collision check will be disabled."
        )
    elif collision_check_enabled:
        non_gripper_body_ids, collision_sensor_body_names, excluded_gripper_bodies = resolve_non_gripper_body_ids(
            collision_sensor, device
        )
        logger.info(
            "Non-gripper collision check enabled: monitored_bodies=%d, excluded_gripper_bodies=%d, threshold=%.2fN",
            non_gripper_body_ids.numel(),
            len(excluded_gripper_bodies),
            args.non_gripper_collision_force_threshold,
        )
    
    # Preallocate tensors
    env_success_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
    current_episode_successful = False
    grasp_pos = torch.zeros((num_envs, 3), device=device)
    grasp_quat = torch.zeros((num_envs, 4), device=device)
    place_pos = torch.zeros((num_envs, 3), device=device)
    place_quat = torch.zeros((num_envs, 4), device=device)
    cube_initial_pos = [torch.zeros((num_envs, 3), device=device)]
    cube_initial_quat = [torch.zeros((num_envs, 4), device=device)]
    nominal_pos_local = torch.zeros(3, device=device)

    center_offset = CENTER_OFFSET.to(device)
    grasp_xy_offset = GRASP_XY_OFFSET.to(device)
    grasp_xy_offset[0] = args.grasp_x_offset
    grasp_xy_offset[1] = args.grasp_y_offset
    if torch.any(grasp_xy_offset != 0):
        print(f"[INFO] Grasp XY offset: X={grasp_xy_offset[0]:.4f}m, Y={grasp_xy_offset[1]:.4f}m")

    step_counter = 0
    current_recorded_demo_count = 0
    recording_active = False
    env_task_completed = torch.zeros(num_envs, device=device, dtype=torch.bool)
    arm_cmds = None
    gripper_cmds = None
    prev_state = None
    invalid_reset_streak = 0
    
    # Drop detection tensors
    env_dropped_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
    grasp_offset_recorded = torch.zeros((num_envs, 3), device=device)
    cube_height_at_grasp = torch.zeros(num_envs, device=device)
    # Collision detection tensors
    env_collision_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)

    # Start recording if enabled
    if args.record:
        data_collector.start_recording(num_envs=num_envs)
        recording_active = True
        print(f"Started recording with {num_envs} environments...")

    def _perform_reset_with_guard(reset_tag: str):
        nonlocal invalid_reset_streak
        while True:
            reset_result = perform_episode_reset(
                robot,
                grasp_controller,
                cubes,
                scene,
                device,
                sim,
                sim_dt,
                initial_joint_pos,
                small_KLT,
                args,
                center_offset,
                grasp_xy_offset,
                robot_entity_cfg,
                num_envs,
            )
            if reset_result is not None:
                if invalid_reset_streak > 0:
                    logger.info(
                        "[RESET] Recovered after %d invalid reset attempts (%s).",
                        invalid_reset_streak,
                        reset_tag,
                    )
                invalid_reset_streak = 0
                return reset_result

            invalid_reset_streak += 1
            logger.warning(
                "[RESET] Invalid reset skipped (%s). Consecutive invalid resets: %d/%d",
                reset_tag,
                invalid_reset_streak,
                args.max_invalid_reset_streak,
            )
            if invalid_reset_streak >= args.max_invalid_reset_streak:
                raise RuntimeError(
                    "Exceeded maximum consecutive invalid resets "
                    f"({args.max_invalid_reset_streak}) during {reset_tag}."
                )

    # Initial reset
    print("[INFO] Performing initial reset...")
    pos_list, quat_list, nominal_pos_local, g_pos, g_quat, p_pos, p_quat = _perform_reset_with_guard("initial")
    cube_initial_pos[0][:] = pos_list[0]
    cube_initial_quat[0][:] = quat_list[0]
    grasp_pos[:] = g_pos
    grasp_quat[:] = g_quat
    place_pos[:] = p_pos
    place_quat[:] = p_quat
    print("[INFO] Initial reset complete")

    # Main loop
    try:
        while simulation_app.is_running():
            # Check demo count
            if args.num_demos > 0 and current_recorded_demo_count >= args.num_demos:
                print(f"Completed recording {args.num_demos} demos. Exiting.")
                simulation_app.close()
                break
            
            cube = cubes[0]
            arm_cmds = None
            gripper_cmds = None
    
            # Reset logic
            if step_counter > 0 and step_counter % args.reset_interval_steps == 0:
                step_counter = 0
                
                if recording_active:
                    camera_intrinsics = get_camera_intrinsics(scene)
                    data_collector.stop_recording(
                        success=current_episode_successful, 
                        camera_intrinsics=camera_intrinsics,
                        flange_to_tcp_offset=grasp_controller.ee_offset.cpu().numpy(),
                        env_success_flags=env_success_flags.cpu().numpy()
                    )
                    
                    successful_envs = torch.sum(env_success_flags).item()
                    dropped_envs = torch.sum(env_dropped_flags).item()
                    collision_envs = torch.sum(env_collision_flags).item()
                    done_envs = torch.sum(env_task_completed).item()
                    
                    if successful_envs > 0:
                        current_recorded_demo_count += successful_envs
                        print(f"Saved {successful_envs} demos (total: {current_recorded_demo_count}/{args.num_demos})")
                        if dropped_envs > 0 or collision_envs > 0:
                            print(
                                f"  [STATS] Done: {done_envs}/{num_envs}, Dropped: {dropped_envs}, "
                                f"Collision: {collision_envs}, Success: {successful_envs}"
                            )
                    else:
                        print(
                            "Discarded demo - no successful environments "
                            f"(Done: {done_envs}, Dropped: {dropped_envs}, Collision: {collision_envs})"
                        )
                    
                    if args.num_demos > 0 and current_recorded_demo_count >= args.num_demos:
                        print(f"Completed recording {args.num_demos} demos. Exiting.")
                        break
    
                    if args.num_demos <= 0 or current_recorded_demo_count < args.num_demos:
                        data_collector.start_recording(num_envs=num_envs)
                        print(f"Started recording demo #{current_recorded_demo_count + 1}...")
                
                current_episode_successful = False
                env_success_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
                env_task_completed = torch.zeros(num_envs, device=device, dtype=torch.bool)
                # Reset drop detection state
                env_dropped_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
                grasp_offset_recorded = torch.zeros((num_envs, 3), device=device)
                cube_height_at_grasp = torch.zeros(num_envs, device=device)
                env_collision_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
    
                # Perform reset
                pos_list, quat_list, nominal_pos_local, g_pos, g_quat, p_pos, p_quat = _perform_reset_with_guard("episode")
                cube_initial_pos[0][:] = pos_list[0]
                cube_initial_quat[0][:] = quat_list[0]
                grasp_pos[:] = g_pos
                grasp_quat[:] = g_quat
                place_pos[:] = p_pos
                place_quat[:] = p_quat
                prev_state = None
                done_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
            else:
                # Normal control
                arm_cmds, gripper_cmds = grasp_controller.compute(
                    grasp_pos, grasp_quat, place_pos, place_quat, count=step_counter
                )
                
                if args.control_noise > 0:
                    arm_cmds = arm_cmds + torch.randn_like(arm_cmds) * args.control_noise
                
                robot.set_joint_position_target(arm_cmds, joint_ids=robot_entity_cfg.joint_ids)
                robot.set_joint_position_target(gripper_cmds, joint_ids=grasp_controller.gripper_entity_cfg.joint_ids)
                
                done_mask = grasp_controller.state_machine.is_done()
                
                # === Drop Detection ===
                # Get current TCP and cube positions
                ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                tcp_pos_w = ee_pose_w[:, 0:3] + quat_apply(
                    ee_pose_w[:, 3:7], grasp_controller.ee_offset.unsqueeze(0).expand(num_envs, -1)
                )
                cube_pos_w = cube.data.root_pos_w
                current_phases = grasp_controller.state_machine.get_current_phases()
                
                # Check for drops
                env_dropped_flags, grasp_offset_recorded, cube_height_at_grasp = check_object_dropped(
                    cube_pos_w, tcp_pos_w, current_phases,
                    grasp_offset_recorded, cube_height_at_grasp, env_dropped_flags, device
                )
    
                # === Non-gripper collision detection ===
                if collision_check_enabled and collision_sensor is not None:
                    env_collision_flags, newly_collided, max_force_per_env, max_body_ids = check_non_gripper_collision(
                        contact_forces_w=collision_sensor.data.net_forces_w,
                        current_phases=current_phases,
                        non_gripper_body_ids=non_gripper_body_ids,
                        collision_flags=env_collision_flags,
                        force_threshold=args.non_gripper_collision_force_threshold,
                        device=device,
                    )
    
                    if newly_collided.any():
                        collided_envs = torch.where(newly_collided)[0]
                        for env_idx in collided_envs[:3]:
                            body_id = int(max_body_ids[env_idx].item())
                            body_name = collision_sensor_body_names[body_id]
                            phase_name = GraspPhase(current_phases[env_idx].item()).name
                            logger.warning(
                                "Env %d: Non-gripper collision detected in phase %s on body %s, force=%.3fN",
                                env_idx.item(),
                                phase_name,
                                body_name,
                                max_force_per_env[env_idx].item(),
                            )
    
                    if step_counter % collision_log_interval == 0:
                        collision_count = torch.sum(env_collision_flags).item()
                        max_force_value = max_force_per_env.max().item()
                        print(
                            "[DEBUG] Collision check: "
                            f"COLLIDED={collision_count}/{num_envs}, MAX_FORCE={max_force_value:.3f}N"
                        )
                
                # Success = done AND not dropped AND no non-gripper collision
                true_success_mask = done_mask & (~env_dropped_flags) & (~env_collision_flags)
                env_success_flags = env_success_flags | true_success_mask
                
                # Debug output
                if step_counter % 100 == 0:
                    dropped_count = torch.sum(env_dropped_flags).item()
                    collision_count = torch.sum(env_collision_flags).item()
                    done_count = torch.sum(done_mask).item()
                    true_success_count = torch.sum(true_success_mask).item()
                    print(
                        f"[DEBUG] Step {step_counter}: DONE={done_count}/{num_envs}, "
                        f"DROPPED={dropped_count}, COLLISION={collision_count}, TRUE_SUCCESS={true_success_count}"
                    )
                
                env_task_completed = env_task_completed | done_mask | env_collision_flags
                if torch.any(true_success_mask):
                    current_episode_successful = True
                
                if env_task_completed.all():
                    print(f"[INFO] All envs completed at step {step_counter}. Early reset.")
                    step_counter = args.reset_interval_steps - 1
    
            scene.write_data_to_sim()
            
            # Recording logic
            if recording_active:
                sampling_interval = data_collector.actual_sampling_interval if data_collector else max(1, round(120 / args.fps))
                
                if step_counter % sampling_interval == 0:
                    # Build observation
                    target_dof = (args.dof or 6) + 1
                    joint_pos_full = robot.data.joint_pos
                    joint_pos_data = joint_pos_full[:, :target_dof]
                    
                    obs = {"observation.state": joint_pos_data}
                    
                    if data_collector and data_collector.previous_actions is not None:
                        obs["observation.state.actions"] = data_collector.previous_actions
                    
                    # Add camera images
                    for sensor_name, sensor in scene.sensors.items():
                        if hasattr(sensor, 'data') and hasattr(sensor.data, 'output') and "rgb" in sensor.data.output:
                            rgb_data = sensor.data.output["rgb"]
                            if "main_camera" in sensor_name.lower():
                                obs["observation.images.main_cam"] = rgb_data
                            elif "wrist" in sensor_name.lower():
                                obs["observation.images.wrist_cam"] = rgb_data
                    
                    # Compute TCP pose
                    flange_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                    tcp_pos_w = flange_pose_w[:, 0:3] + quat_apply(
                        flange_pose_w[:, 3:7], grasp_controller.ee_offset.unsqueeze(0).expand(num_envs, -1)
                    )
                    tcp_quat_w = flange_pose_w[:, 3:7]
                    tcp_pos_local = tcp_pos_w - scene.env_origins
                    
                    # Gripper state
                    if gripper_cmds is not None:
                        gripper_min = grasp_controller.open_gripper_pos[0].item()
                        gripper_max = grasp_controller.closed_gripper_pos[0].item()
                        gripper_state = torch.clamp((gripper_cmds[:, :1] - gripper_min) / (gripper_max - gripper_min), 0.0, 1.0)
                    else:
                        gripper_state = torch.zeros((num_envs, 1), device=device)
                    
                    # Build proprio and joint_state
                    tcp_rot6d = quat_to_rot6d(tcp_quat_w)
                    proprio = torch.cat([tcp_pos_local, tcp_rot6d, gripper_state], dim=1)
                    arm_joints = joint_pos_full[:, :6]
                    finger_joint = joint_pos_full[:, 6:7] if joint_pos_full.shape[1] >= 7 else torch.zeros((num_envs, 1), device=device)
                    joint_state = torch.cat([arm_joints, finger_joint], dim=1)
                    
                    current_state = torch.cat([proprio, joint_state], dim=1)
                    
                    # Delayed recording: action_t = state_{t+1}
                    if prev_state is not None:
                        obs["observation.proprio"] = prev_state[:, :10]
                        obs["observation.joint_state"] = prev_state[:, 10:]
                        actions = proprio
                        obs["action_joint"] = joint_state
                        
                        rewards = torch.zeros(num_envs, device=device)
                        dones = done_mask | env_collision_flags
                        if data_collector:
                            data_collector.record_step(obs, actions, rewards, dones, {})
                    
                    prev_state = current_state
    
            sim.step()
            step_counter += 1
            scene.update(sim_dt)
    
            # Visualization
            if markers is not None:
                ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
                ee_pos_with_offset = ee_pose_w[:, 0:3] + quat_apply(
                    ee_pose_w[:, 3:7], grasp_controller.ee_offset.unsqueeze(0).expand(num_envs, -1)
                )
                markers["tcp"].visualize(ee_pos_with_offset, ee_pose_w[:, 3:7])
                markers["grasp"].visualize(grasp_pos + scene.env_origins, grasp_quat)
                markers["place"].visualize(place_pos + scene.env_origins, place_quat)
                for m, c in zip(markers["cubes"], cubes):
                    m.visualize(c.data.root_pos_w, c.data.root_quat_w)
                markers["klt"].visualize(small_KLT.data.root_pos_w, small_KLT.data.root_quat_w)
    finally:
        if data_collector is not None:
            data_collector.close()
