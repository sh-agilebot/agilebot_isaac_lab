#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Main simulation module for automatic pick-and-place demo execution.
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

logger = logging.getLogger(__name__)

# Import non-omniverse modules at module level
from controller.pick_place_controller import ParallelPickPlaceController
from controller.phases.state_machine import GraspPhase

from core.common_utils import reset_everything, compute_grasp_and_place, make_markers


# Constants
DEFAULT_DT = 1 / 120
CENTER_OFFSET = torch.tensor([0.0, 0.0, 0.015])  # Object center to grasp point offset
GRASP_XY_OFFSET = torch.tensor([0.0, 0.0])  # Gripper centering calibration

# Drop detection threshold
DROP_DISTANCE_THRESHOLD = 0.10  # 10cm - if TCP-cube distance exceeds this, consider dropped

# Strict collision allowance:
# only these two Robotiq contact-pad links are allowed as non-violating links.
# all other contact-sensor bodies (robot/container/object, etc.) are monitored for violations.
ROBOTIQ_ALLOWED_CONTACT_BODY_NAME_PAIRS = (
    # Preferred naming in some Robotiq USDs
    ("left_inner_finger_pad", "right_inner_finger_pad"),
    # Fallback naming in this project USD
    ("left_inner_finger", "right_inner_finger"),
)
# Fallback keyword rules (still strict: exactly one left link and one right link).
ROBOTIQ_ALLOWED_CONTACT_SIDE_KEYWORD_PAIRS = (
    (("left", "inner", "finger", "pad"), ("right", "inner", "finger", "pad")),
    (("left", "inner", "finger"), ("right", "inner", "finger")),
)
COLLISION_CHECK_PHASES = [phase for phase in GraspPhase if phase != GraspPhase.DONE]
COLLISION_HIGHLIGHT_MATERIAL_PATH = "/World/Looks/non_gripper_collision_highlight_red"


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


def _find_unique_body_id_by_keywords(
    body_names_lower: list[str], required_keywords: tuple[str, ...]
) -> int | None:
    """Find exactly one body id containing all required keywords."""
    matches = [
        body_id
        for body_id, body_name in enumerate(body_names_lower)
        if all(keyword in body_name for keyword in required_keywords)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_allowed_robotiq_contact_body_ids(body_names_lower: list[str]) -> list[int]:
    """Resolve exactly two allowed Robotiq contact-link ids (left + right)."""
    # Priority 1: explicit left/right link-name pairs.
    for left_name, right_name in ROBOTIQ_ALLOWED_CONTACT_BODY_NAME_PAIRS:
        left_id = _find_unique_body_id_by_keywords(body_names_lower, (left_name,))
        right_id = _find_unique_body_id_by_keywords(body_names_lower, (right_name,))
        if left_id is not None and right_id is not None and left_id != right_id:
            return sorted([left_id, right_id])

    # Priority 2: strict keyword-pair fallback.
    for left_keywords, right_keywords in ROBOTIQ_ALLOWED_CONTACT_SIDE_KEYWORD_PAIRS:
        left_id = _find_unique_body_id_by_keywords(body_names_lower, left_keywords)
        right_id = _find_unique_body_id_by_keywords(body_names_lower, right_keywords)
        if left_id is not None and right_id is not None and left_id != right_id:
            return sorted([left_id, right_id])

    return []


def resolve_restricted_collision_body_ids(
    collision_sensor, device: torch.device
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Resolve body ids that are NOT allowed to collide.

    Only two Robotiq contact-pad links are treated as allowed links:
    left/right inner finger contact links (pad naming preferred when available).
    All other bodies exposed by `collision_sensor` are monitored as collision violations.
    """
    all_body_names = collision_sensor.body_names
    body_names_lower = [name.lower() for name in all_body_names]

    allowed_body_ids = _resolve_allowed_robotiq_contact_body_ids(body_names_lower)
    if len(allowed_body_ids) != 2:
        raise RuntimeError(
            "Failed to resolve exactly two allowed Robotiq contact links for collision allowance. "
            f"Resolved allowed ids: {allowed_body_ids}, body names: {all_body_names}"
        )

    monitored_body_ids = [
        body_id for body_id in range(len(all_body_names)) if body_id not in allowed_body_ids
    ]
    if not monitored_body_ids:
        raise RuntimeError(
            "No monitored bodies remain after applying strict Robotiq contact-pad allowance. "
            f"Body names: {all_body_names}"
        )

    allowed_body_names = [all_body_names[body_id] for body_id in allowed_body_ids]
    return (
        torch.tensor(monitored_body_ids, device=device, dtype=torch.long),
        all_body_names,
        allowed_body_names,
    )


def check_monitored_body_collision(
    contact_forces_w: torch.Tensor,
    current_phases: torch.Tensor,
    monitored_body_ids: torch.Tensor,
    collision_flags: torch.Tensor,
    force_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect collisions on monitored (disallowed) bodies during active grasp phases.

    Returns:
        Tuple of:
            - updated collision flags [N]
            - newly collided env mask [N]
            - max monitored-body contact force per env [N]
            - max-force body id in original sensor body index [N]
    """
    in_check_phase = build_phase_mask(current_phases, COLLISION_CHECK_PHASES, device)

    monitored_forces = contact_forces_w.index_select(1, monitored_body_ids)
    monitored_force_norm = torch.norm(monitored_forces, dim=-1)
    max_force_per_env, max_local_ids = monitored_force_norm.max(dim=1)
    max_body_ids = monitored_body_ids[max_local_ids]

    is_collision = in_check_phase & (max_force_per_env > force_threshold)
    newly_collided = is_collision & (~collision_flags)
    updated_collision_flags = collision_flags | is_collision
    return updated_collision_flags, newly_collided, max_force_per_env, max_body_ids


def get_collision_contact_forces_w(collision_sensor) -> torch.Tensor:
    """Get contact forces for collision checks, preferring filtered force matrix when configured."""
    force_matrix_w = collision_sensor.data.force_matrix_w
    if force_matrix_w is not None and force_matrix_w.numel() > 0:
        # Aggregate forces over filtered targets: [N, B, F, 3] -> [N, B, 3]
        return force_matrix_w.sum(dim=2)
    return collision_sensor.data.net_forces_w


def resolve_collision_body_prim_paths(collision_sensor, num_envs: int) -> list[str]:
    """Resolve contact-sensor body prim paths in flattened env-major order."""
    body_prim_paths = [str(path) for path in collision_sensor.body_physx_view.prim_paths]
    expected = num_envs * collision_sensor.num_bodies
    if len(body_prim_paths) < expected:
        raise RuntimeError(
            "Collision sensor body prim path count is smaller than expected: "
            f"{len(body_prim_paths)} < {expected}."
        )
    return body_prim_paths[:expected]


def ensure_collision_highlight_material() -> str:
    """Create (if needed) and return a red material path for collision highlighting."""
    import isaaclab.sim as sim_utils
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()
    if not stage.GetPrimAtPath(COLLISION_HIGHLIGHT_MATERIAL_PATH).IsValid():
        red_mat_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.0, 0.0),
            emissive_color=(0.15, 0.0, 0.0),
            roughness=0.2,
            metallic=0.0,
        )
        red_mat_cfg.func(COLLISION_HIGHLIGHT_MATERIAL_PATH, red_mat_cfg)
    return COLLISION_HIGHLIGHT_MATERIAL_PATH


def reset_collision_highlight_materials(highlighted_prim_original_materials: dict[str, str | None]) -> None:
    """Restore original visual materials for highlighted collision bodies."""
    if not highlighted_prim_original_materials:
        return

    import isaaclab.sim as sim_utils
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdShade

    stage = get_current_stage()
    for prim_path, original_material_path in highlighted_prim_original_materials.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        try:
            if original_material_path is not None:
                # Restore previous direct material binding.
                sim_utils.bind_visual_material(
                    prim_path,
                    original_material_path,
                    stronger_than_descendants=True,
                )
            else:
                # No previous direct binding: clear the one introduced by collision highlight.
                UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().ClearTargets(True)
        except Exception as exc:
            logger.warning(
                "Failed to restore collision highlight material on %s: %s",
                prim_path,
                exc,
            )

    highlighted_prim_original_materials.clear()


def _get_direct_visual_material_path(prim) -> str | None:
    """Get the direct visual material binding target path of a prim."""
    from pxr import UsdShade

    binding_rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
    targets = binding_rel.GetTargets()
    if len(targets) == 0:
        return None
    return str(targets[0])


def highlight_collided_bodies(
    newly_collided: torch.Tensor,
    max_body_ids: torch.Tensor,
    collision_body_prim_paths: list[str],
    num_bodies: int,
    highlight_material_path: str,
    highlighted_prim_original_materials: dict[str, str | None],
) -> None:
    """Highlight newly collided robot bodies in red."""
    if not newly_collided.any():
        return

    import isaaclab.sim as sim_utils
    from isaacsim.core.utils.stage import get_current_stage

    collided_env_ids = torch.where(newly_collided)[0].tolist()
    stage = get_current_stage()
    for env_id in collided_env_ids:
        body_id = int(max_body_ids[env_id].item())
        flat_index = env_id * num_bodies + body_id
        if flat_index >= len(collision_body_prim_paths):
            logger.warning(
                "Skip collision highlight for env=%d, body_id=%d due to invalid flat index=%d.",
                env_id,
                body_id,
                flat_index,
            )
            continue

        body_prim_path = collision_body_prim_paths[flat_index]
        if body_prim_path in highlighted_prim_original_materials:
            continue

        try:
            # IMPORTANT:
            # Do not change instanceability at runtime (e.g. make_uninstanceable), otherwise PhysX views
            # can be invalidated and articulation writes will fail in the next step.
            prim = stage.GetPrimAtPath(body_prim_path)
            if not prim.IsValid():
                continue

            # Instance proxies are not safely editable at runtime in this loop.
            if prim.IsInstanceProxy():
                logger.debug(
                    "Skip collision highlight for instance-proxy prim: %s. Motion halt still applies.",
                    body_prim_path,
                )
                continue

            original_material_path = _get_direct_visual_material_path(prim)

            sim_utils.bind_visual_material(
                body_prim_path,
                highlight_material_path,
                stronger_than_descendants=True,
            )
            highlighted_prim_original_materials[body_prim_path] = original_material_path
        except Exception as exc:
            logger.warning(
                "Failed to highlight collided body at %s: %s",
                body_prim_path,
                exc,
            )


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
    # Reset robot and objects
    pos_list, quat_list, nominal_pos_local = reset_everything(
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
        random_hold_min=args.random_hold_min,
        random_hold_max=args.random_hold_max,
    )
    
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

def run_simulator(
    sim, scene, args: argparse.Namespace,
    simulation_app, ee_offset: torch.Tensor = None,
    initial_joint_pos: torch.Tensor = None,
) -> None:
    """Main simulation loop."""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.math import quat_apply

    # Extract scene entities
    robot = scene["robot"]
    tomato_soup_can = scene["tomato_soup_can"]
    small_KLT = scene["small_KLT"]
    cubes = [tomato_soup_can]
    robot_collision_sensor = scene.sensors.get("collision_sensor_robot")

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

    sim_dt = sim.get_physics_dt()
        
    markers = make_markers(scene.num_envs, show_markers=args.show_markers)
    device = sim.device
    num_envs = scene.num_envs

    collision_check_enabled = False
    collision_log_interval = max(1, int(args.non_gripper_collision_log_interval))
    collision_sources: list[dict[str, object]] = []
    collision_highlight_material_path: str | None = None
    highlighted_collision_body_materials: dict[str, str | None] = {}
    if args.enable_non_gripper_collision_check:
        # Enforce violations only on robot bodies.
        # The robot collision sensor is configured with filtered targets (container/object),
        # so table contacts are excluded from strict collision checks.
        configured_sensors = [
            ("robot", robot_collision_sensor),
        ]
        for source_name, source_sensor in configured_sensors:
            if source_sensor is None:
                logger.warning(
                    "Collision check requested but '%s' is missing in scene config. This source is skipped.",
                    f"collision_sensor_{source_name}",
                )
                continue

            if source_name == "robot":
                monitored_body_ids, body_names, allowed_collision_bodies = resolve_restricted_collision_body_ids(
                    source_sensor,
                    device,
                )
                logger.info(
                    "Collision source '%s': allowed_bodies=%s, monitored_bodies=%d, threshold=%.2fN",
                    source_name,
                    allowed_collision_bodies,
                    monitored_body_ids.numel(),
                    args.non_gripper_collision_force_threshold,
                )
            else:
                body_names = source_sensor.body_names
                if len(body_names) == 0:
                    logger.warning(
                        "Collision source '%s' has zero bodies. This source is skipped.",
                        source_name,
                    )
                    continue
                monitored_body_ids = torch.arange(len(body_names), dtype=torch.long, device=device)
                logger.info(
                    "Collision source '%s': monitored_bodies=%d, threshold=%.2fN",
                    source_name,
                    monitored_body_ids.numel(),
                    args.non_gripper_collision_force_threshold,
                )

            collision_sources.append(
                {
                    "name": source_name,
                    "sensor": source_sensor,
                    "monitored_body_ids": monitored_body_ids,
                    "body_names": body_names,
                    "body_prim_paths": resolve_collision_body_prim_paths(source_sensor, num_envs),
                }
            )

        collision_check_enabled = len(collision_sources) > 0
        if not collision_check_enabled:
            logger.warning("Collision check requested but no valid collision sensor source is available.")
        else:
            collision_highlight_material_path = ensure_collision_highlight_material()
    
    # Preallocate tensors
    env_success_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
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
    env_task_completed = torch.zeros(num_envs, device=device, dtype=torch.bool)
    arm_cmds = None
    gripper_cmds = None
    
    # Drop detection tensors
    env_dropped_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
    grasp_offset_recorded = torch.zeros((num_envs, 3), device=device)
    cube_height_at_grasp = torch.zeros(num_envs, device=device)
    # Collision detection tensors
    env_collision_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)

    # Initial reset
    print("[INFO] Performing initial reset...")
    pos_list, quat_list, nominal_pos_local, g_pos, g_quat, p_pos, p_quat = perform_episode_reset(
        robot, grasp_controller, cubes, scene, device, sim, sim_dt,
        initial_joint_pos, small_KLT, args, center_offset, grasp_xy_offset,
        robot_entity_cfg, num_envs
    )
    reset_collision_highlight_materials(highlighted_collision_body_materials)
    cube_initial_pos[0][:] = pos_list[0]
    cube_initial_quat[0][:] = quat_list[0]
    grasp_pos[:] = g_pos
    grasp_quat[:] = g_quat
    place_pos[:] = p_pos
    place_quat[:] = p_quat
    print("[INFO] Initial reset complete")

    # Main loop
    while simulation_app.is_running():
        cube = cubes[0]
        arm_cmds = None
        gripper_cmds = None

        # Reset logic
        if step_counter > 0 and step_counter % args.reset_interval_steps == 0:
            step_counter = 0

            successful_envs = torch.sum(env_success_flags).item()
            dropped_envs = torch.sum(env_dropped_flags).item()
            collision_envs = torch.sum(env_collision_flags).item()
            done_envs = torch.sum(env_task_completed).item()
            print(
                f"[EPISODE] Done: {done_envs}/{num_envs}, "
                f"Dropped: {dropped_envs}, Collision: {collision_envs}, "
                f"Success: {successful_envs}"
            )

            env_success_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
            env_task_completed = torch.zeros(num_envs, device=device, dtype=torch.bool)
            # Reset drop detection state
            env_dropped_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)
            grasp_offset_recorded = torch.zeros((num_envs, 3), device=device)
            cube_height_at_grasp = torch.zeros(num_envs, device=device)
            env_collision_flags = torch.zeros(num_envs, device=device, dtype=torch.bool)

            # Perform reset
            pos_list, quat_list, nominal_pos_local, g_pos, g_quat, p_pos, p_quat = perform_episode_reset(
                robot, grasp_controller, cubes, scene, device, sim, sim_dt,
                initial_joint_pos, small_KLT, args, center_offset, grasp_xy_offset,
                robot_entity_cfg, num_envs
            )
            reset_collision_highlight_materials(highlighted_collision_body_materials)
            cube_initial_pos[0][:] = pos_list[0]
            cube_initial_quat[0][:] = quat_list[0]
            grasp_pos[:] = g_pos
            grasp_quat[:] = g_quat
            place_pos[:] = p_pos
            place_quat[:] = p_quat
            done_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
        else:
            # Normal control
            arm_cmds, gripper_cmds = grasp_controller.compute(
                grasp_pos, grasp_quat, place_pos, place_quat, count=step_counter
            )
            
            if args.control_noise > 0:
                arm_cmds = arm_cmds + torch.randn_like(arm_cmds) * args.control_noise

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

            # === Collision detection (robot strict; filtered to container/object targets) ===
            if collision_check_enabled:
                max_force_value = 0.0
                for source in collision_sources:
                    source_name = source["name"]
                    source_sensor = source["sensor"]
                    source_monitored_body_ids = source["monitored_body_ids"]
                    source_body_names = source["body_names"]
                    source_body_prim_paths = source["body_prim_paths"]

                    env_collision_flags, newly_collided, max_force_per_env, max_body_ids = check_monitored_body_collision(
                        contact_forces_w=get_collision_contact_forces_w(source_sensor),
                        current_phases=current_phases,
                        monitored_body_ids=source_monitored_body_ids,
                        collision_flags=env_collision_flags,
                        force_threshold=args.non_gripper_collision_force_threshold,
                        device=device,
                    )

                    max_force_value = max(max_force_value, max_force_per_env.max().item())

                    if newly_collided.any():
                        if collision_highlight_material_path is not None:
                            highlight_collided_bodies(
                                newly_collided=newly_collided,
                                max_body_ids=max_body_ids,
                                collision_body_prim_paths=source_body_prim_paths,
                                num_bodies=source_sensor.num_bodies,
                                highlight_material_path=collision_highlight_material_path,
                                highlighted_prim_original_materials=highlighted_collision_body_materials,
                            )

                        collided_envs = torch.where(newly_collided)[0]
                        for env_idx in collided_envs[:3]:
                            body_id = int(max_body_ids[env_idx].item())
                            body_name = source_body_names[body_id]
                            phase_name = GraspPhase(current_phases[env_idx].item()).name
                            logger.warning(
                                "Env %d: Collision violation (%s) in phase %s on body %s, force=%.3fN",
                                env_idx.item(),
                                source_name,
                                phase_name,
                                body_name,
                                max_force_per_env[env_idx].item(),
                            )

                if step_counter % collision_log_interval == 0:
                    collision_count = torch.sum(env_collision_flags).item()
                    print(
                        "[DEBUG] Collision check: "
                        f"COLLIDED={collision_count}/{num_envs}, MAX_FORCE={max_force_value:.3f}N"
                    )

            # Freeze motion in collided envs: keep current joint positions and stop further motion commands.
            if env_collision_flags.any():
                halted_env_ids = torch.where(env_collision_flags)[0]
                arm_cmds[halted_env_ids] = robot.data.joint_pos[halted_env_ids][:, robot_entity_cfg.joint_ids]
                gripper_cmds[halted_env_ids] = robot.data.joint_pos[halted_env_ids][
                    :, grasp_controller.gripper_entity_cfg.joint_ids
                ]

            robot.set_joint_position_target(arm_cmds, joint_ids=robot_entity_cfg.joint_ids)
            robot.set_joint_position_target(gripper_cmds, joint_ids=grasp_controller.gripper_entity_cfg.joint_ids)
            
            # Success = done AND not dropped AND no collision violation
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
            
            # Collision envs are halted in-place; only "done" envs contribute to early reset.
            env_task_completed = env_task_completed | done_mask

            if env_task_completed.all() and (~env_collision_flags).all():
                print(f"[INFO] All envs completed at step {step_counter}. Early reset.")
                step_counter = args.reset_interval_steps - 1

        scene.write_data_to_sim()

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
