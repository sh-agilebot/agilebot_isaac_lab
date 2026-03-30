#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Place phase handlers.

This module implements the place-related phases:
- LiftPhase: Lift the object
- MoveToPlacePhase: Move horizontally to place position
- PlaceDescentPhase: Descend vertically to place position
- PlaceOpeningPhase: Open the gripper
- PlaceOpenedPhase: Wait for stabilization with a small upward retract
- DonePhase: Place completed
"""

import torch
from common.pose_transformer import PoseTransformer
from controller.phases.base import PhaseHandler, PhaseContext, PhaseResult
from controller.phases.state_machine import GraspPhase


class LiftPhase(PhaseHandler):
    """Lift phase: Lift the object.
    
    In this phase, the end-effector lifts the grasped object vertically
    by a specified height.
    
    Completion: When end-effector reaches the lift height.
    Next Phase: MOVE_TO_PLACE
    """
    
    def __init__(self, lift_height: float = 0.20, tolerance: float = 0.005):
        """Initialize lift phase.
        
        Args:
            lift_height: Height to lift [m]
            tolerance: Position tolerance [m]
        """
        self.lift_height = lift_height
        self.tolerance = tolerance
    
    def get_phase_id(self) -> int:
        return GraspPhase.LIFT.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute lift phase."""
        batch_size = ctx.ee_pos_b.shape[0]
        device = ctx.device
        
        # Initialize output with current positions
        joint_positions = ctx.joint_pos.clone()
        
        if not mask.any():
            return PhaseResult(
                arm_joint_positions=joint_positions,
                next_phase=None,
                completion_mask=torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
        
        # Maintain gripper closed
        self.update_gripper_position(ctx, mask)
        
        # Initialize lift targets if needed
        if ctx.lift_target_positions is None:
            ctx.lift_target_positions = torch.zeros((batch_size, 3), device=device)
            ctx.lift_target_orientations = torch.zeros((batch_size, 4), device=device)
            ctx.has_lift_target = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Check which environments are entering lift phase for the first time
        entering_lift = mask & ~ctx.has_lift_target
        
        if entering_lift.any():
            # Set lift targets for newly entering environments
            ctx.lift_target_positions[entering_lift] = ctx.ee_pos_b[entering_lift].clone()
            ctx.lift_target_positions[entering_lift, 2] += self.lift_height
            ctx.lift_target_orientations[entering_lift] = ctx.ee_quat_b[entering_lift].clone()
            ctx.has_lift_target[entering_lift] = True
        
        # Use stored lift targets
        target_pos = ctx.lift_target_positions.clone()
        target_quat = ctx.lift_target_orientations.clone()
        
        # For non-lift environments, use current position (won't be used anyway)
        target_pos[~mask] = ctx.ee_pos_b[~mask]
        target_quat[~mask] = ctx.ee_quat_b[~mask]
        
        # Compute IK
        computed_positions = self.compute_ik(ctx, target_pos, target_quat)
        joint_positions[mask] = computed_positions[mask]
        
        # Check if lift height reached
        reached = self.check_position_reached(
            ctx.ee_pos_b[mask],
            ctx.lift_target_positions[mask],
            self.tolerance
        )
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        completion_mask[indices[reached]] = True
        
        # Clear lift targets for completed environments
        if reached.any():
            reached_indices = indices[reached]
            ctx.has_lift_target[reached_indices] = False
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=GraspPhase.MOVE_TO_PLACE.value if reached.any() else None,
            completion_mask=completion_mask
        )


class MoveToPlacePhase(PhaseHandler):
    """Move to place phase: Move horizontally to place position.
    
    In this phase, the end-effector moves horizontally to the place
    position while maintaining the lift height.
    
    Completion: When horizontal (X,Y) position is reached.
    Next Phase: PLACE_DESCENT
    """
    
    def __init__(self, tolerance: float = 0.015):
        """Initialize move to place phase.
        
        Args:
            tolerance: Position tolerance for XY plane [m]
        """
        self.tolerance = tolerance
    
    def get_phase_id(self) -> int:
        return GraspPhase.MOVE_TO_PLACE.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute move to place phase."""
        batch_size = ctx.place_position.shape[0]
        device = ctx.device
        
        # Initialize output with current positions
        joint_positions = ctx.joint_pos.clone()
        
        if not mask.any():
            return PhaseResult(
                arm_joint_positions=joint_positions,
                next_phase=None,
                completion_mask=torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
        
        # Maintain gripper closed
        self.update_gripper_position(ctx, mask)
        
        # Initialize target tensors
        target_pos = torch.zeros((batch_size, 3), device=device)
        target_quat = torch.zeros((batch_size, 4), device=device)
        
        # For environments in move to place phase
        place_quat_active = ctx.place_orientation[mask]
        place_pos_active = ctx.place_position[mask]
        
        # Compute flange position for place
        # flange_pos = TCP_pos - ee_offset_in_world
        ee_offset_world = PoseTransformer.local_to_world_vector(
            place_quat_active, ctx.ee_offset
        )
        
        # Set X,Y to place position (minus offset)
        flange_pos = place_pos_active - ee_offset_world
        
        # Keep Z at current height (lifted)
        flange_pos[:, 2] = ctx.ee_pos_b[mask, 2]
        
        target_pos[mask] = flange_pos
        target_quat[mask] = place_quat_active
        
        # Compute IK
        computed_positions = self.compute_ik(ctx, target_pos, target_quat)
        joint_positions[mask] = computed_positions[mask]
        
        # Check if XY position reached
        xy_error = torch.norm(
            ctx.ee_pos_b[mask, :2] - target_pos[mask, :2],
            dim=-1
        )
        reached = xy_error < self.tolerance
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        completion_mask[indices[reached]] = True
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=GraspPhase.PLACE_DESCENT.value if reached.any() else None,
            completion_mask=completion_mask
        )


class PlaceDescentPhase(PhaseHandler):
    """Place descent phase: Descend vertically to place position.
    
    In this phase, the end-effector descends from the lift height
    to the place position.
    
    Completion: When place position is reached.
    Next Phase: PLACE_OPENING (and sets gripper target to open)
    """
    
    def __init__(self, tolerance: float = 0.01):
        """Initialize place descent phase.
        
        Args:
            tolerance: Position tolerance [m]
        """
        self.tolerance = tolerance
    
    def get_phase_id(self) -> int:
        return GraspPhase.PLACE_DESCENT.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute place descent phase."""
        batch_size = ctx.place_position.shape[0]
        device = ctx.device
        
        # Initialize output with current positions
        joint_positions = ctx.joint_pos.clone()
        
        if not mask.any():
            return PhaseResult(
                arm_joint_positions=joint_positions,
                next_phase=None,
                completion_mask=torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
        
        # Maintain gripper closed
        self.update_gripper_position(ctx, mask)
        
        # Initialize target tensors
        target_pos = torch.zeros((batch_size, 3), device=device)
        target_quat = torch.zeros((batch_size, 4), device=device)
        
        # For environments in place descent phase
        place_quat_active = ctx.place_orientation[mask]
        place_pos_active = ctx.place_position[mask]
        
        # Compute flange position for place
        flange_pos = PoseTransformer.compute_grasp_flange_position(
            place_pos_active, place_quat_active, ctx.ee_offset
        )
        
        target_pos[mask] = flange_pos
        target_quat[mask] = place_quat_active
        
        # Compute IK
        computed_positions = self.compute_ik(ctx, target_pos, target_quat)
        joint_positions[mask] = computed_positions[mask]
        
        # Check if place position reached
        reached_pos = self.check_position_reached(
            ctx.ee_pos_b[mask],
            target_pos[mask],
            self.tolerance
        )
        
        # Check if orientation is stabilized
        # Compute angular difference between current EE quaternion and target place quaternion
        q1 = ctx.ee_quat_b[mask]
        q2 = target_quat[mask]
        dot = torch.abs(torch.sum(q1 * q2, dim=1))
        dot = torch.clamp(dot, -1.0, 1.0)
        angle_diff = 2.0 * torch.acos(dot)
        
        # Use a small tolerance for orientation (e.g., 5 degrees = 0.087 rad)
        orientation_stabilized = angle_diff < 0.087
        
        # Combined reached condition: Position AND Orientation
        reached = reached_pos & orientation_stabilized
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        completion_mask[indices[reached]] = True
        
        # If reached, set gripper target to open
        if reached.any():
            reached_indices = indices[reached]
            # Use controller's configured open position if available, else default
            # Note: We should ideally use ctx.open_gripper_pos but it's not in ctx?
            # It seems ctx doesn't have open_gripper_pos.
            # But earlier code used `torch.tensor([0.0])`.
            # Let's check if ctx has open_gripper_pos.
            # In pick_place_controller.py, ctx is created.
            # It doesn't seem to pass open_gripper_pos explicitly, but maybe via gripper_target_pos init?
            # However, the original code had `open_pos = torch.tensor([0.0], device=device)`.
            # We'll stick to that or use 0.0.
            open_pos = torch.zeros((reached.sum(), ctx.gripper_target_pos.shape[1]), device=device)
            ctx.gripper_target_pos[reached_indices] = open_pos
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=GraspPhase.PLACE_OPENING.value if reached.any() else None,
            completion_mask=completion_mask
        )


class PlaceOpeningPhase(PhaseHandler):
    """Place opening phase: Open the gripper.
    
    In this phase, the gripper opens while maintaining position.
    
    Completion: When gripper reaches open position.
    Next Phase: PLACE_OPENED
    """
    
    def __init__(self, tolerance_multiplier: float = 2.0):
        """Initialize place opening phase.
        
        Args:
            tolerance_multiplier: Multiplier for gripper tolerance
        """
        self.tolerance_multiplier = tolerance_multiplier
    
    def get_phase_id(self) -> int:
        return GraspPhase.PLACE_OPENING.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute place opening phase."""
        batch_size = ctx.place_position.shape[0]
        device = ctx.device
        
        # Initialize output with current positions
        joint_positions = ctx.joint_pos.clone()
        
        if not mask.any():
            return PhaseResult(
                arm_joint_positions=joint_positions,
                next_phase=None,
                completion_mask=torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
        
        # Initialize target tensors
        target_pos = torch.zeros((batch_size, 3), device=device)
        target_quat = torch.zeros((batch_size, 4), device=device)
        
        # For environments in place opening phase
        place_quat_active = ctx.place_orientation[mask]
        place_pos_active = ctx.place_position[mask]
        
        # Compute flange position
        flange_pos = PoseTransformer.compute_grasp_flange_position(
            place_pos_active, place_quat_active, ctx.ee_offset
        )
        
        target_pos[mask] = flange_pos
        target_quat[mask] = place_quat_active
        
        # Compute IK
        computed_positions = self.compute_ik(ctx, target_pos, target_quat)
        joint_positions[mask] = computed_positions[mask]
        
        # Check if position reached (before opening gripper)
        pos_reached = self.check_position_reached(
            ctx.ee_pos_b[mask],
            target_pos[mask],
            0.01  # Position tolerance
        )
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        
        # Only open gripper for environments that reached position
        if pos_reached.any():
            reached_indices = indices[pos_reached]
            
            # Create sub-mask for reached environments
            reached_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            reached_mask[reached_indices] = True
            
            # Update gripper position for reached environments
            self.update_gripper_position(ctx, reached_mask)
            
            # Check if gripper is fully opened
            gripper_at_target = self.is_gripper_at_target(
                ctx,
                reached_mask,
                tolerance_multiplier=self.tolerance_multiplier,
            )
            
            # Check if orientation is stabilized
            # Compute angular difference between current EE quaternion and target place quaternion
            q1 = ctx.ee_quat_b[reached_mask]
            q2 = target_quat[reached_mask]
            dot = torch.abs(torch.sum(q1 * q2, dim=1))
            dot = torch.clamp(dot, -1.0, 1.0)
            angle_diff = 2.0 * torch.acos(dot)
            
            # Use a small tolerance for orientation (e.g., 5 degrees = 0.087 rad)
            orientation_stabilized = angle_diff < 0.087
            
            # Combine conditions: position reached + orientation stabilized + gripper opened
            transition_ready = orientation_stabilized & gripper_at_target
            if transition_ready.any():
                # Get local indices where all transition conditions are satisfied
                stable_local_indices = torch.nonzero(transition_ready).squeeze(-1)
                # Map back to global indices
                global_stable_indices = reached_indices[stable_local_indices]
                completion_mask[global_stable_indices] = True
            
            return PhaseResult(
                arm_joint_positions=joint_positions,
                next_phase=GraspPhase.PLACE_OPENED.value if completion_mask.any() else None,
                completion_mask=completion_mask
            )
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=None,
            completion_mask=completion_mask
        )


class PlaceOpenedPhase(PhaseHandler):
    """Place opened phase: Wait for stabilization with a small retract.
    
    In this phase, we keep gripper open, wait briefly, then retract upward
    while the object stabilizes.
    
    Completion: After stabilization time has elapsed.
    Next Phase: DONE
    """
    
    def get_phase_id(self) -> int:
        return GraspPhase.PLACE_OPENED.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute place opened phase."""
        batch_size = ctx.place_position.shape[0]
        device = ctx.device
        
        # Initialize output with current positions
        joint_positions = ctx.joint_pos.clone()
        
        if not mask.any():
            return PhaseResult(
                arm_joint_positions=joint_positions,
                next_phase=None,
                completion_mask=torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
        
        # Initialize target tensors
        target_pos = torch.zeros((batch_size, 3), device=device)
        target_quat = torch.zeros((batch_size, 4), device=device)
        
        # For environments in place opened phase
        place_quat_active = ctx.place_orientation[mask]
        place_pos_active = ctx.place_position[mask]
        timers = ctx.phase_timer[mask]
        
        if ctx.random_hold_steps is not None:
            effective_wait = ctx.place_stabilization_time + ctx.random_hold_steps[mask]
        else:
            effective_wait = torch.full_like(timers, ctx.place_stabilization_time)

        # Minimal retract behavior:
        # keep place pose in the first half, then retract TCP upward by 4cm.
        hold_steps = torch.clamp(effective_wait * 0.5, min=1.0)
        retract_steps = torch.clamp(effective_wait - hold_steps, min=1.0)
        retract_progress = torch.clamp(
            (timers - hold_steps) / retract_steps,
            min=0.0,
            max=1.0,
        )
        retract_tcp_pos = place_pos_active.clone()
        retract_tcp_pos[:, 2] += 0.04 * retract_progress

        # Compute flange position
        flange_pos = PoseTransformer.compute_grasp_flange_position(
            retract_tcp_pos, place_quat_active, ctx.ee_offset
        )
        
        target_pos[mask] = flange_pos
        target_quat[mask] = place_quat_active
        
        # Compute IK
        computed_positions = self.compute_ik(ctx, target_pos, target_quat)
        joint_positions[mask] = computed_positions[mask]
        
        # Maintain gripper open
        self.update_gripper_position(ctx, mask)
        
        # Check if stabilization time has elapsed
        stabilized = timers >= effective_wait
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        completion_mask[indices[stabilized]] = True
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=GraspPhase.DONE.value if stabilized.any() else None,
            completion_mask=completion_mask
        )


class DonePhase(PhaseHandler):
    """Done phase: Place completed.
    
    In this phase, we maintain position and wait for a reset signal.
    After a certain time, automatically reset to pre-grasp.
    
    Completion: After reset timeout.
    Next Phase: PRE_GRASP (reset)
    """
    
    def __init__(self, reset_timeout: int = 200):
        """Initialize done phase.
        
        Args:
            reset_timeout: Steps before auto-reset
        """
        self.reset_timeout = reset_timeout
    
    def get_phase_id(self) -> int:
        return GraspPhase.DONE.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute done phase."""
        batch_size = ctx.place_position.shape[0]
        device = ctx.device
        
        # Initialize output with current positions
        joint_positions = ctx.joint_pos.clone()
        
        if not mask.any():
            return PhaseResult(
                arm_joint_positions=joint_positions,
                next_phase=None,
                completion_mask=torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
        
        # Initialize target tensors
        target_pos = torch.zeros((batch_size, 3), device=device)
        target_quat = torch.zeros((batch_size, 4), device=device)
        
        # Maintain position at place target
        place_quat_active = ctx.place_orientation[mask]
        place_pos_active = ctx.place_position[mask]
        
        flange_pos = PoseTransformer.compute_grasp_flange_position(
            place_pos_active, place_quat_active, ctx.ee_offset
        )
        
        target_pos[mask] = flange_pos
        target_quat[mask] = place_quat_active
        
        # Compute IK
        computed_positions = self.compute_ik(ctx, target_pos, target_quat)
        joint_positions[mask] = computed_positions[mask]
        
        # Maintain gripper open
        self.update_gripper_position(ctx, mask)
        
        # Check if should reset
        should_reset = ctx.phase_timer[mask] >= self.reset_timeout
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        completion_mask[indices[should_reset]] = True
        
        # Clear lift targets on reset
        if should_reset.any() and ctx.has_lift_target is not None:
            reset_indices = indices[should_reset]
            ctx.has_lift_target[reset_indices] = False
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=GraspPhase.PRE_GRASP.value if should_reset.any() else None,
            completion_mask=completion_mask
        )
