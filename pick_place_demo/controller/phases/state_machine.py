#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""State machine for managing grasp phases.

This module defines the phase enumeration and state machine for coordinating
transitions between grasp phases.
"""

from enum import IntEnum
from typing import Dict, Optional, List, Tuple
import torch


class GraspPhase(IntEnum):
    """Enumeration of grasp phases.
    
    The phases progress in order:
    1. PRE_GRASP: Move above the object
    2. MOVE_DOWN: Move down to grasp position
    3. GRASP_CLOSING: Closing gripper
    4. GRASP_CLOSED: Gripper closed, wait for stabilization
    5. LIFT: Lift the object
    6. MOVE_TO_PLACE: Move horizontally to place position (at lift height)
    7. PLACE_DESCENT: Descend vertically to place position
    8. PLACE_OPENING: Opening gripper to place
    9. PLACE_OPENED: Gripper opened, wait for stabilization
    10. DONE: Place completed
    """
    PRE_GRASP = 0
    MOVE_DOWN = 1
    GRASP_CLOSING = 2
    GRASP_CLOSED = 3
    LIFT = 4
    MOVE_TO_PLACE = 5
    PLACE_DESCENT = 6
    PLACE_OPENING = 7
    PLACE_OPENED = 8
    DONE = 9
    
    @classmethod
    def get_default_sequence(cls) -> List["GraspPhase"]:
        """Get the default phase sequence."""
        return [
            cls.PRE_GRASP,
            cls.MOVE_DOWN,
            cls.GRASP_CLOSING,
            cls.GRASP_CLOSED,
            cls.LIFT,
            cls.MOVE_TO_PLACE,
            cls.PLACE_DESCENT,
            cls.PLACE_OPENING,
            cls.PLACE_OPENED,
            cls.DONE,
        ]


class StateMachine:
    """State machine for managing grasp phase transitions.
    
    The state machine coordinates transitions between grasp phases based on
    completion conditions. It maintains the current phase for each environment
    and handles phase timer management.
    
    Attributes:
        current_phases: Current phase for each environment [batch,]
        phase_timers: Timer for each environment's current phase [batch,]
        device: torch device
    """
    
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        initial_phase: GraspPhase = GraspPhase.PRE_GRASP,
    ):
        """Initialize the state machine.
        
        Args:
            num_envs: Number of parallel environments
            device: torch device
            initial_phase: Starting phase for all environments
        """
        self.num_envs = num_envs
        self.device = device
        self.current_phases = torch.full(
            (num_envs,),
            initial_phase.value,
            dtype=torch.int,
            device=device
        )
        self.phase_timers = torch.zeros(num_envs, device=device)
    
    def reset(
        self,
        initial_phase: GraspPhase = GraspPhase.PRE_GRASP,
        env_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset the state machine.
        
        Args:
            initial_phase: Phase to reset to
            env_mask: Optional mask for selective reset [batch,]
        """
        if env_mask is None:
            self.current_phases.fill_(initial_phase.value)
            self.phase_timers.fill_(0)
        else:
            self.current_phases[env_mask] = initial_phase.value
            self.phase_timers[env_mask] = 0
    
    def step(self) -> None:
        """Increment phase timers for all environments."""
        self.phase_timers += 1
    
    def get_mask(self, phase: GraspPhase) -> torch.Tensor:
        """Get boolean mask for environments in a specific phase.
        
        Args:
            phase: Phase to check
            
        Returns:
            Boolean tensor [batch,] indicating environments in phase
        """
        return self.current_phases == phase.value
    
    def get_active_mask(self, phases: List[GraspPhase]) -> torch.Tensor:
        """Get mask for environments in any of the specified phases.
        
        Args:
            phases: List of phases to check
            
        Returns:
            Boolean tensor [batch,] indicating environments in any of the phases
        """
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for phase in phases:
            mask |= (self.current_phases == phase.value)
        return mask
    
    def transition(
        self,
        current_phase: GraspPhase,
        next_phase: GraspPhase,
        completion_mask: torch.Tensor,
    ) -> None:
        """Execute phase transition for environments that have completed.
        
        Args:
            current_phase: Phase to transition from
            next_phase: Phase to transition to
            completion_mask: Boolean mask indicating which envs completed [batch,]
        """
        # Only consider environments that are in the current phase
        current_mask = self.get_mask(current_phase)
        transition_mask = current_mask & completion_mask
        
        # Update phases
        self.current_phases[transition_mask] = next_phase.value
        
        # Reset timers for transitioned environments
        self.phase_timers[transition_mask] = 0
    
    def transition_multi(
        self,
        transitions: List[Tuple[GraspPhase, GraspPhase, torch.Tensor]],
    ) -> None:
        """Execute multiple phase transitions in batch.
        
        Args:
            transitions: List of (from_phase, to_phase, completion_mask) tuples
        """
        for from_phase, to_phase, completion_mask in transitions:
            self.transition(from_phase, to_phase, completion_mask)
    
    def get_timer(self, phase: GraspPhase) -> torch.Tensor:
        """Get phase timers for environments in a specific phase.
        
        Args:
            phase: Phase to get timers for
            
        Returns:
            Phase timers masked to only include environments in phase
        """
        mask = self.get_mask(phase)
        timers = self.phase_timers.clone()
        timers[~mask] = 0
        return timers
    
    def get_current_phases(self) -> torch.Tensor:
        """Get current phases for all environments.
        
        Returns:
            Current phase IDs [batch,]
        """
        return self.current_phases.clone()
    
    def is_done(self, done_phase: GraspPhase = GraspPhase.DONE) -> torch.Tensor:
        """Check which environments have completed the full grasp sequence.
        
        Args:
            done_phase: Phase considered as "done"
            
        Returns:
            Boolean tensor [batch,] indicating done environments
        """
        return self.get_mask(done_phase)
