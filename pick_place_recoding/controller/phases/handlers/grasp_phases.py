#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp phase handlers.

This module implements the grasp-related phases:
- PreGraspPhase: Move above the object
- MoveDownPhase: Move down to grasp position  
- GraspClosingPhase: Close the gripper
- GraspClosedPhase: Wait for gripper stabilization
"""

import torch
from common.pose_transformer import PoseTransformer
from controller.phases.base import PhaseHandler, PhaseContext, PhaseResult
from controller.phases.state_machine import GraspPhase


class PreGraspPhase(PhaseHandler):
    """Pre-grasp phase: Move above the object.
    
    In this phase, the end-effector moves to a position above the object
    (along the negative local Z axis) before descending for the grasp.
    
    Completion: When end-effector reaches the pre-grasp position.
    Next Phase: MOVE_DOWN
    """
    
    def __init__(self, approach_height: float = 0.3, tolerance: float = 0.01):
        """Initialize pre-grasp phase.
        
        Args:
            approach_height: Height offset for pre-grasp approach [m]
            tolerance: Position tolerance for completion check [m]
        """
        self.approach_height = approach_height
        self.tolerance = tolerance
    
    def get_phase_id(self) -> int:
        return GraspPhase.PRE_GRASP.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute pre-grasp phase."""
        batch_size = ctx.grasp_position.shape[0]
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
        
        # Compute pre-grasp position (approach point above object)
        # Using PoseTransformer for clear semantics
        approach_offset = torch.tensor(
            [0.0, 0.0, -self.approach_height],
            device=device,
            dtype=ctx.grasp_position.dtype
        )
        
        # For environments in pre-grasp phase
        grasp_quat_active = ctx.grasp_orientation[mask]
        grasp_pos_active = ctx.grasp_position[mask]
        
        # Compute approach offset in world frame
        # Note: We want the TCP to be at grasp_pos + approach_offset
        # But we command the flange, so we need to convert
        
        # Step 1: Compute TCP position at approach point
        # approach_offset_local is in grasp frame (negative Z = above)
        approach_offset_world = PoseTransformer.local_to_world_vector(
            grasp_quat_active, approach_offset
        )
        tcp_approach_pos = grasp_pos_active + approach_offset_world
        
        # Step 2: Convert TCP position to flange position
        flange_pos = PoseTransformer.compute_grasp_flange_position(
            tcp_approach_pos, grasp_quat_active, ctx.ee_offset
        )
        
        target_pos[mask] = flange_pos
        target_quat[mask] = grasp_quat_active
        
        # Compute IK
        computed_positions = self.compute_ik(ctx, target_pos, target_quat)
        joint_positions[mask] = computed_positions[mask]
        
        # Check if position reached for active environments
        reached = self.check_position_reached(
            ctx.ee_pos_b[mask],
            target_pos[mask],
            self.tolerance
        )
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        completion_mask[indices[reached]] = True
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=GraspPhase.MOVE_DOWN.value if reached.any() else None,
            completion_mask=completion_mask
        )


class MoveDownPhase(PhaseHandler):
    """Move down phase: Move to grasp position.
    
    In this phase, the end-effector descends from the pre-grasp position
    to the actual grasp position.
    
    Completion: When end-effector reaches the grasp position.
    Next Phase: GRASP_CLOSING (and sets gripper target to closed)
    """
    
    def __init__(self, tolerance: float = 0.005):
        """Initialize move down phase.
        
        Args:
            tolerance: Position tolerance for completion check [m]
        """
        self.tolerance = tolerance
    
    def get_phase_id(self) -> int:
        return GraspPhase.MOVE_DOWN.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute move down phase."""
        batch_size = ctx.grasp_position.shape[0]
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
        
        # For environments in move down phase
        grasp_quat_active = ctx.grasp_orientation[mask]
        grasp_pos_active = ctx.grasp_position[mask]
        
        # Compute flange position for grasp
        flange_pos = PoseTransformer.compute_grasp_flange_position(
            grasp_pos_active, grasp_quat_active, ctx.ee_offset
        )
        
        target_pos[mask] = flange_pos
        target_quat[mask] = grasp_quat_active
        
        # Compute IK
        computed_positions = self.compute_ik(ctx, target_pos, target_quat)
        joint_positions[mask] = computed_positions[mask]
        
        # Check if position reached
        reached = self.check_position_reached(
            ctx.ee_pos_b[mask],
            target_pos[mask],
            self.tolerance
        )
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        completion_mask[indices[reached]] = True
        
        # If reached, set gripper target to closed position
        if reached.any():
            reached_indices = indices[reached]
            closed_pos = torch.tensor([0.7], device=device)  # Default closed position
            ctx.gripper_target_pos[reached_indices] = closed_pos
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=GraspPhase.GRASP_CLOSING.value if reached.any() else None,
            completion_mask=completion_mask
        )


class GraspClosingPhase(PhaseHandler):
    """Grasp closing phase: Close the gripper.
    
    In this phase, the gripper moves towards the closed position.
    Arm position is maintained.
    
    Completion: When gripper reaches target position.
    Next Phase: GRASP_CLOSED
    """
    
    def get_phase_id(self) -> int:
        return GraspPhase.GRASP_CLOSING.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute grasp closing phase."""
        batch_size = ctx.grasp_position.shape[0]
        device = ctx.device
        
        # Maintain current arm position (hold position)
        joint_positions = ctx.joint_pos.clone()
        
        if not mask.any():
            return PhaseResult(
                arm_joint_positions=joint_positions,
                next_phase=None,
                completion_mask=torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
        
        # Update gripper position towards closed target
        self.update_gripper_position(ctx, mask)
        
        # Check if gripper reached target
        at_target = self.is_gripper_at_target(ctx, mask)
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        completion_mask[indices[at_target]] = True
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=GraspPhase.GRASP_CLOSED.value if at_target.any() else None,
            completion_mask=completion_mask
        )


class GraspClosedPhase(PhaseHandler):
    """Grasp closed phase: Wait for stabilization.
    
    In this phase, we maintain the closed gripper position and wait
    for stabilization before lifting.
    
    Completion: After stabilization time has elapsed.
    Next Phase: LIFT
    """
    
    def get_phase_id(self) -> int:
        return GraspPhase.GRASP_CLOSED.value
    
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute grasp closed phase."""
        batch_size = ctx.grasp_position.shape[0]
        device = ctx.device
        
        # Maintain current arm position
        joint_positions = ctx.joint_pos.clone()
        
        if not mask.any():
            return PhaseResult(
                arm_joint_positions=joint_positions,
                next_phase=None,
                completion_mask=torch.zeros(batch_size, dtype=torch.bool, device=device)
            )
        
        # Maintain gripper at closed position
        self.update_gripper_position(ctx, mask)
        
        # Check if stabilization time has elapsed
        timers = ctx.phase_timer[mask]
        
        # Compute effective wait time
        if ctx.random_hold_steps is not None:
            effective_wait = ctx.grasp_stabilization_time + ctx.random_hold_steps[mask]
        else:
            effective_wait = torch.full_like(timers, ctx.grasp_stabilization_time)
        
        stabilized = timers >= effective_wait
        
        # Build full completion mask
        completion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        indices = torch.nonzero(mask).squeeze(-1)
        completion_mask[indices[stabilized]] = True
        
        return PhaseResult(
            arm_joint_positions=joint_positions,
            next_phase=GraspPhase.LIFT.value if stabilized.any() else None,
            completion_mask=completion_mask
        )
