#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Refactored parallel pick-and-place controller with improved architecture.

This module implements a parallel controller using a phase-based architecture:
- PoseTransformer: Unified coordinate frame transformations with explicit wxyz quaternion handling
- PhaseHandler: Strategy pattern for grasp phases
- StateMachine: Centralized phase transition management

Quaternion Format:
    IsaacLab uses [w, x, y, z] (wxyz) format for quaternions.
    See common.pose_transformer for transformation utilities.

Architecture:
    The controller is organized into three layers:
    1. ParallelPickPlaceController: Main coordinator
    2. Phase Handlers: Individual phase implementations
    3. Utilities: PoseTransformer for coordinate transformations
"""

from typing import Tuple, Optional, Dict
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from common.pose_transformer import PoseTransformer
from controller.phases.base import PhaseContext
from controller.phases.state_machine import StateMachine, GraspPhase
from controller.phases.handlers import (
    PreGraspPhase,
    MoveDownPhase,
    GraspClosingPhase,
    GraspClosedPhase,
    LiftPhase,
    MoveToPlacePhase,
    PlaceDescentPhase,
    PlaceOpeningPhase,
    PlaceOpenedPhase,
    DonePhase,
)


class ParallelPickPlaceController:
    """Parallel pick-and-place controller with phase-based architecture.
    
    This controller implements a state machine for grasp phases, with each phase
    encapsulated as a separate handler. Coordinate transformations are handled
    by PoseTransformer with explicit quaternion format documentation.
    
    Args:
        robot: The robot asset
        scene: The interactive scene
        sim: The simulation instance
        robot_entity_cfg: Configuration for the robot entity
        robot_type: Type of robot (e.g., "Agilebot")
        ee_offset: End-effector offset tensor [3,]
        grasp_stabilization_time: Steps to wait after closing gripper
        place_stabilization_time: Steps to wait after opening gripper
        
    Example:
        >>> controller = ParallelPickPlaceController(
        ...     robot=robot,
        ...     scene=scene,
        ...     sim=sim,
        ...     robot_entity_cfg=robot_entity_cfg,
        ...     robot_type="Agilebot",
        ...     ee_offset=torch.tensor([0.0, 0.0, 0.15]),
        ... )
        >>> arm_pos, gripper_pos = controller.compute(
        ...     grasp_position=grasp_pos,
        ...     grasp_orientation=grasp_quat,
        ...     place_position=place_pos,
        ...     place_orientation=place_quat,
        ... )
    """
    
    def __init__(
        self,
        robot,
        scene,
        sim,
        robot_entity_cfg: SceneEntityCfg,
        robot_type: str,
        ee_offset: Optional[torch.Tensor] = None,
        grasp_stabilization_time: int = 20,
        place_stabilization_time: int = 30,
    ):
        """Initialize the controller."""
        self.robot = robot
        self.scene = scene
        self.sim = sim
        self.robot_type = robot_type
        
        # Create IK controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
        )
        
        # Robot configuration
        self.robot_entity_cfg = robot_entity_cfg
        if robot.is_fixed_base:
            self.ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = robot_entity_cfg.body_ids[0]
        
        # Gripper configuration
        self._setup_gripper(robot_type, sim.device)
        
        # End-effector offset
        if ee_offset is not None:
            self.ee_offset = ee_offset.to(sim.device)
        else:
            # Default offset for Agilebot
            self.ee_offset = torch.tensor([0.0, 0.0, 0.15], device=sim.device)
        
        # State machine
        self.state_machine = StateMachine(
            num_envs=scene.num_envs,
            device=sim.device,
            initial_phase=GraspPhase.PRE_GRASP
        )
        
        # Phase handlers
        self.phase_handlers: Dict[int, object] = self._create_phase_handlers(
            grasp_stabilization_time,
            place_stabilization_time,
        )
        
        # Configuration
        self.grasp_stabilization_time = grasp_stabilization_time
        self.place_stabilization_time = place_stabilization_time
        self.random_hold_steps: Optional[torch.Tensor] = None
        
        # Lift state (managed by LiftPhase, stored here for persistence)
        self.lift_target_positions: Optional[torch.Tensor] = None
        self.lift_target_orientations: Optional[torch.Tensor] = None
        self.has_lift_target: Optional[torch.Tensor] = None
    
    def _setup_gripper(self, robot_type: str, device: torch.device) -> None:
        """Configure gripper parameters based on robot type."""
        if robot_type == "Agilebot":
            self.gripper_joint_names = ["finger_joint"]
            self.open_gripper_pos = torch.tensor([0.0], device=device)
            self.closed_gripper_pos = torch.tensor([0.7], device=device)
        else:
            raise ValueError(f"Robot type {robot_type} not supported. Valid: Agilebot")
        
        # Resolve gripper joints
        self.gripper_entity_cfg = SceneEntityCfg(
            "robot", joint_names=self.gripper_joint_names
        )
        self.gripper_entity_cfg.resolve(self.scene)
        
        # Gripper state tensors
        self.gripper_current_pos = torch.zeros(
            (self.scene.num_envs, len(self.gripper_joint_names)),
            device=device
        )
        self.gripper_target_pos = torch.zeros(
            (self.scene.num_envs, len(self.gripper_joint_names)),
            device=device
        )
        
        # Compute speed and tolerance based on range
        gripper_range = self.closed_gripper_pos[0].item() - self.open_gripper_pos[0].item()
        self.gripper_speed = gripper_range * 0.02  # 2% per step
        self.gripper_tolerance = gripper_range * 0.0125  # 1.25%
    
    def _create_phase_handlers(
        self,
        grasp_stabilization_time: int,
        place_stabilization_time: int,
    ) -> Dict[int, object]:
        """Create and return phase handler instances."""
        # Tolerance configuration
        # Relaxed tolerances to prevent wobbling and stuck states
        # Previous values (0.001) were too tight for stable simulation
        pos_tolerance = 0.005  # 5mm for general positioning
        xy_tolerance = 0.01    # 1cm for XY positioning
        
        handlers = {
            GraspPhase.PRE_GRASP.value: PreGraspPhase(
                approach_height=0.10,
                tolerance=0.01  # 1cm for pre-grasp (approach)
            ),
            GraspPhase.MOVE_DOWN.value: MoveDownPhase(
                tolerance=0.002  # 2mm for final grasp position (tight but reachable)
            ),
            GraspPhase.GRASP_CLOSING.value: GraspClosingPhase(),
            GraspPhase.GRASP_CLOSED.value: GraspClosedPhase(),
            GraspPhase.LIFT.value: LiftPhase(
                lift_height=0.20,
                tolerance=pos_tolerance
            ),
            GraspPhase.MOVE_TO_PLACE.value: MoveToPlacePhase(
                tolerance=xy_tolerance
            ),
            GraspPhase.PLACE_DESCENT.value: PlaceDescentPhase(
                tolerance=0.01
            ),
            GraspPhase.PLACE_OPENING.value: PlaceOpeningPhase(
                tolerance_multiplier=2.0
            ),
            GraspPhase.PLACE_OPENED.value: PlaceOpenedPhase(),
            GraspPhase.DONE.value: DonePhase(
                reset_timeout=200
            ),
        }
        return handlers
    
    def reset(
        self,
        random_hold_min: int = 0,
        random_hold_max: int = 0
    ) -> None:
        """Reset the controller.
        
        Args:
            random_hold_min: Minimum random hold steps
            random_hold_max: Maximum random hold steps
        """
        self.diff_ik_controller.reset()
        self.state_machine.reset(GraspPhase.PRE_GRASP)
        
        # Reset gripper positions
        self.gripper_current_pos[:] = self.open_gripper_pos
        self.gripper_target_pos[:] = self.open_gripper_pos
        
        # Reset lift targets
        if self.lift_target_positions is not None:
            self.lift_target_positions.fill_(0.0)
        if self.lift_target_orientations is not None:
            self.lift_target_orientations.fill_(0.0)
        if self.has_lift_target is not None:
            self.has_lift_target.fill_(False)
        
        # Setup random hold steps
        if random_hold_min > 0 and random_hold_max >= random_hold_min:
            self.random_hold_steps = torch.randint(
                random_hold_min, random_hold_max + 1,
                (self.scene.num_envs,),
                device=self.sim.device
            )
        else:
            self.random_hold_steps = None
    
    def compute(
        self,
        grasp_position: torch.Tensor,
        grasp_orientation: torch.Tensor,
        place_position: torch.Tensor,
        place_orientation: Optional[torch.Tensor] = None,
        count: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute joint commands for grasping and placing.
        
        Args:
            grasp_position: Target position to grasp [num_envs, 3]
            grasp_orientation: Target orientation [w,x,y,z] [num_envs, 4]
            place_position: Position to place [num_envs, 3]
            place_orientation: Orientation to place [w,x,y,z] [num_envs, 4]
            count: Simulation step count
            
        Returns:
            Tuple of (arm_joint_positions, gripper_joint_positions)
        """
        # Default place orientation to grasp orientation if not provided
        if place_orientation is None:
            place_orientation = grasp_orientation
        
        # Get current robot state
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
        ]
        ee_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        
        # Convert to base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        
        # Step state machine timer
        self.state_machine.step()
        
        # Create phase context
        ctx = PhaseContext(
            ee_pos_b=ee_pos_b,
            ee_quat_b=ee_quat_b,
            jacobian=jacobian,
            joint_pos=joint_pos,
            diff_ik_controller=self.diff_ik_controller,
            grasp_position=grasp_position,
            grasp_orientation=grasp_orientation,
            place_position=place_position,
            place_orientation=place_orientation,
            gripper_current_pos=self.gripper_current_pos,
            gripper_target_pos=self.gripper_target_pos,
            gripper_speed=self.gripper_speed,
            gripper_tolerance=self.gripper_tolerance,
            ee_offset=self.ee_offset,
            robot_type=self.robot_type,
            phase_timer=self.state_machine.phase_timers,
            lift_target_positions=self.lift_target_positions,
            lift_target_orientations=self.lift_target_orientations,
            has_lift_target=self.has_lift_target,
            grasp_stabilization_time=self.grasp_stabilization_time,
            place_stabilization_time=self.place_stabilization_time,
            random_hold_steps=self.random_hold_steps,
        )
        
        # Initialize output
        arm_joint_positions = joint_pos.clone()
        
        # Process each phase
        for phase_id, handler in self.phase_handlers.items():
            phase = GraspPhase(phase_id)
            mask = self.state_machine.get_mask(phase)
            
            if not mask.any():
                continue
            
            # Execute phase handler
            result = handler.execute(ctx, mask)
            
            # Update joint positions for this phase's environments
            arm_joint_positions[mask] = result.arm_joint_positions[mask]
            
            # Handle phase transitions
            if result.next_phase is not None and result.completion_mask.any():
                self.state_machine.transition(
                    phase,
                    GraspPhase(result.next_phase),
                    result.completion_mask
                )
        
        # Update lift state from context
        self.lift_target_positions = ctx.lift_target_positions
        self.lift_target_orientations = ctx.lift_target_orientations
        self.has_lift_target = ctx.has_lift_target
        
        return arm_joint_positions, self.gripper_current_pos.clone()
