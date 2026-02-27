#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Base classes for phase-based controller architecture.

This module defines the abstract base class for all grasp phases and the context
object that holds shared state across phases.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import torch
from isaaclab.controllers import DifferentialIKController


@dataclass
class PhaseResult:
    """Result of executing a phase.
    
    Attributes:
        arm_joint_positions: Computed joint positions for arm [batch, num_joints]
        next_phase: The phase to transition to (None = stay in current phase)
        completion_mask: Boolean mask indicating which environments completed [batch,]
        extra_data: Optional additional data from phase execution
    """
    arm_joint_positions: torch.Tensor
    next_phase: Optional[int] = None
    completion_mask: torch.Tensor = field(default_factory=lambda: torch.tensor(False))
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseContext:
    """Shared context for all phases.
    
    This context object holds all the state and dependencies needed by phases
    to perform their computations. It's passed to each phase handler.
    
    Attributes:
        # Robot state
        ee_pos_b: End-effector position in base frame [batch, 3]
        ee_quat_b: End-effector orientation in base frame [w,x,y,z] [batch, 4]
        jacobian: Jacobian matrix [batch, 6, num_joints]
        joint_pos: Current joint positions [batch, num_joints]
        
        # Controller
        diff_ik_controller: Differential IK controller instance
        
        # Targets
        grasp_position: Grasp target position (TCP) [batch, 3]
        grasp_orientation: Grasp target orientation [w,x,y,z] [batch, 4]
        place_position: Place target position (TCP) [batch, 3]
        place_orientation: Place target orientation [w,x,y,z] [batch, 4]
        
        # Gripper state
        gripper_current_pos: Current gripper position [batch, num_gripper_joints]
        gripper_target_pos: Target gripper position [batch, num_gripper_joints]
        gripper_speed: Speed of gripper movement per step
        gripper_tolerance: Tolerance for gripper position check
        
        # Configuration
        ee_offset: Offset from flange to TCP in local frame [3,]
        robot_type: Type of robot (e.g., "Agilebot")
        
        # Phase state
        phase_timer: Timer for current phase [batch,]
        
        # Optional persistent state
        lift_target_positions: Storage for lift targets [batch, 3]
        lift_target_orientations: Storage for lift orientations [batch, 4]
        has_lift_target: Flag for lift target initialization [batch,]
        
        # Device
        device: torch.device
    """
    # Robot state
    ee_pos_b: torch.Tensor
    ee_quat_b: torch.Tensor
    jacobian: torch.Tensor
    joint_pos: torch.Tensor
    
    # Controller
    diff_ik_controller: DifferentialIKController
    
    # Targets
    grasp_position: torch.Tensor
    grasp_orientation: torch.Tensor
    place_position: torch.Tensor
    place_orientation: torch.Tensor
    
    # Gripper state
    gripper_current_pos: torch.Tensor
    gripper_target_pos: torch.Tensor
    gripper_speed: float
    gripper_tolerance: float
    
    # Configuration
    ee_offset: torch.Tensor
    robot_type: str
    
    # Phase state
    phase_timer: torch.Tensor
    
    # Persistent state (mutable)
    lift_target_positions: Optional[torch.Tensor] = None
    lift_target_orientations: Optional[torch.Tensor] = None
    has_lift_target: Optional[torch.Tensor] = None
    
    # Timing configuration
    grasp_stabilization_time: int = 20
    place_stabilization_time: int = 30
    random_hold_steps: Optional[torch.Tensor] = None
    
    # Device (derived from tensors)
    @property
    def device(self) -> torch.device:
        """Get torch device from joint_pos tensor."""
        return self.joint_pos.device


class PhaseHandler(ABC):
    """Abstract base class for grasp phase handlers.
    
    Each phase handler implements the behavior for a specific grasp phase,
    including target computation, completion checking, and state transitions.
    
    The handler receives a PhaseContext with all necessary state and returns
    a PhaseResult with computed joint positions and transition information.
    
    Implementations should follow these patterns:
    1. Use mask-based computation for batch processing
    2. Compute target pose for the phase
    3. Use IK controller to compute joint positions
    4. Check completion conditions
    5. Return appropriate PhaseResult
    """
    
    @abstractmethod
    def execute(self, ctx: PhaseContext, mask: torch.Tensor) -> PhaseResult:
        """Execute this phase for the environments specified by mask.
        
        Args:
            ctx: Shared phase context with all state and dependencies
            mask: Boolean tensor indicating which environments are in this phase [batch,]
            
        Returns:
            PhaseResult with computed positions and transition info
        """
        pass
    
    @abstractmethod
    def get_phase_id(self) -> int:
        """Get the unique identifier for this phase.
        
        Returns:
            Integer phase ID (should match GraspPhase enum)
        """
        pass
    
    def compute_ik(
        self,
        ctx: PhaseContext,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IK for a target pose.
        
        This is a helper method that standardizes IK computation across phases.
        
        Args:
            ctx: Phase context with current state and IK controller
            target_pos: Target flange position [batch, 3]
            target_quat: Target flange orientation [w,x,y,z] [batch, 4]
            
        Returns:
            Computed joint positions [batch, num_joints]
        """
        # Set command for IK controller
        command = torch.cat([target_pos, target_quat], dim=-1)
        ctx.diff_ik_controller.set_command(command)
        
        # Compute joint positions
        return ctx.diff_ik_controller.compute(
            ctx.ee_pos_b, ctx.ee_quat_b, ctx.jacobian, ctx.joint_pos
        )
    
    def check_position_reached(
        self,
        current_pos: torch.Tensor,
        target_pos: torch.Tensor,
        tolerance: float,
    ) -> torch.Tensor:
        """Check if position has been reached within tolerance.
        
        Args:
            current_pos: Current position [batch, 3]
            target_pos: Target position [batch, 3]
            tolerance: Position tolerance [m]
            
        Returns:
            Boolean tensor indicating reached positions [batch,]
        """
        pos_error = torch.norm(current_pos - target_pos, dim=-1)
        return pos_error < tolerance
    
    def update_gripper_position(
        self,
        ctx: PhaseContext,
        mask: torch.Tensor,
    ) -> None:
        """Update gripper position towards target (in-place).
        
        Args:
            ctx: Phase context with gripper state
            mask: Boolean mask for environments to update [batch,]
        """
        gripper_diff = ctx.gripper_target_pos[mask] - ctx.gripper_current_pos[mask]
        move_magnitude = torch.min(
            torch.full_like(gripper_diff, ctx.gripper_speed),
            torch.abs(gripper_diff)
        )
        gripper_move = torch.sign(gripper_diff) * move_magnitude
        ctx.gripper_current_pos[mask] += gripper_move
    
    def is_gripper_at_target(
        self,
        ctx: PhaseContext,
        mask: torch.Tensor,
        tolerance_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Check if gripper has reached target position.
        
        Args:
            ctx: Phase context with gripper state
            mask: Boolean mask for environments to check [batch,]
            tolerance_multiplier: Multiplier for base tolerance
            
        Returns:
            Boolean tensor indicating gripper at target [batch,]
        """
        gripper_diff = ctx.gripper_target_pos[mask] - ctx.gripper_current_pos[mask]
        tolerance = ctx.gripper_tolerance * tolerance_multiplier
        return torch.any(torch.abs(gripper_diff) < tolerance, dim=-1)
