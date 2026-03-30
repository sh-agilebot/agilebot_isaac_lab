#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Pose transformation utilities with explicit quaternion format handling.

This module provides unified tools for coordinate frame transformations,
with clear documentation on quaternion formats and transformation semantics.

Quaternion Format:
    IsaacLab uses [w, x, y, z] format (wxyz) for quaternions, where:
    - w: scalar (cos(θ/2))
    - x, y, z: vector components (axis * sin(θ/2))

    This is the same format used by: ROS2, PyTorch3D, and scipy.spatial.transform.Rotation
    But different from: numpy-quaternion (xyzw), some robotics libraries

Transformation Semantics:
    - quat_apply(quat, vec): Rotates vector FROM local frame TO world frame
      vec_world = quat * vec_local * quat_conj
    
    - quat_apply_inverse(quat, vec): Rotates vector FROM world frame TO local frame
      vec_local = quat_conj * vec_world * quat

Coordinate Frames:
    - World frame: Global simulation coordinate system
    - Base frame: Robot base coordinate system (attached to robot root)
    - TCP frame: Tool Center Point (end-effector tip)
    - Flange frame: Robot wrist flange (mounting point)

Typical workflow:
    1. Target pose is specified in TCP frame (where gripper should be)
    2. Convert TCP pose to Flange pose by applying ee_offset
    3. Use Flange pose for IK computation
"""

from typing import Tuple
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse


class PoseTransformer:
    """Unified pose transformation utilities for robotic manipulation.
    
    This class provides static methods for coordinate frame transformations
    with explicit documentation on quaternion formats and transformation semantics.
    
    All methods support batch operations with shape [batch_size, ...].
    
    Example:
        >>> # Convert TCP target to Flange target for IK
        >>> flange_pos, flange_quat = PoseTransformer.tcp_to_flange(
        ...     tcp_pos, tcp_quat, ee_offset
        ... )
        >>> 
        >>> # Apply local offset in world frame
        >>> offset_world = PoseTransformer.local_to_world_vector(
        ...     quat, offset_local
        ... )
    """
    
    @staticmethod
    def tcp_to_flange(
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor,
        ee_offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert TCP (Tool Center Point) pose to Flange pose.
        
        The TCP is at the tip of the gripper where it interacts with objects.
        The Flange is at the robot's wrist mounting point.
        
        Relationship: Flange = TCP - ee_offset (rotated by orientation)
        
        Args:
            tcp_pos: TCP position in base/world frame [batch, 3]
            tcp_quat: TCP orientation quaternion [w,x,y,z] [batch, 4]
            ee_offset: Offset from flange to TCP in TCP local frame [3,]
            
        Returns:
            Tuple of (flange_pos, flange_quat) - quat unchanged, only pos modified
            
        Example:
            >>> tcp_pos = torch.tensor([[0.5, 0.0, 0.1]])  # TCP target
            >>> tcp_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity
            >>> ee_offset = torch.tensor([0.0, 0.0, 0.15])  # 15cm gripper length
            >>> flange_pos, flange_quat = PoseTransformer.tcp_to_flange(
            ...     tcp_pos, tcp_quat, ee_offset
            ... )
            >>> # flange_pos will be [0.5, 0.0, -0.05] (TCP - offset)
        """
        # ee_offset is defined in TCP local frame
        # To subtract it from TCP position, we need to rotate it to world frame
        # quat_apply rotates FROM local TO world
        ee_offset_world = PoseTransformer.local_to_world_vector(tcp_quat, ee_offset)
        flange_pos = tcp_pos - ee_offset_world
        return flange_pos, tcp_quat.clone()
    
    @staticmethod
    def flange_to_tcp(
        flange_pos: torch.Tensor,
        flange_quat: torch.Tensor,
        ee_offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Flange pose to TCP (Tool Center Point) pose.
        
        Relationship: TCP = Flange + ee_offset (rotated by orientation)
        
        Args:
            flange_pos: Flange position in base/world frame [batch, 3]
            flange_quat: Flange orientation quaternion [w,x,y,z] [batch, 4]
            ee_offset: Offset from flange to TCP in flange local frame [3,]
            
        Returns:
            Tuple of (tcp_pos, tcp_quat) - quat unchanged, only pos modified
        """
        ee_offset_world = PoseTransformer.local_to_world_vector(flange_quat, ee_offset)
        tcp_pos = flange_pos + ee_offset_world
        return tcp_pos, flange_quat.clone()
    
    @staticmethod
    def local_to_world_vector(
        quat: torch.Tensor,
        vec_local: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate vector FROM local frame TO world frame.
        
        This uses quat_apply which implements: vec_world = quat * vec_local * quat_conj
        
        Args:
            quat: Orientation quaternion [w,x,y,z] [batch, 4]
            vec_local: Vector in local frame [batch, 3] or [3,]
            
        Returns:
            Vector in world frame [batch, 3]
            
        Example:
            >>> quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity
            >>> vec_local = torch.tensor([[0.0, 0.0, 0.1]])  # 10cm along local Z
            >>> vec_world = PoseTransformer.local_to_world_vector(quat, vec_local)
            >>> # vec_world is also [0.0, 0.0, 0.1]
        """
        # Handle scalar expansion
        if vec_local.dim() == 1:
            vec_local = vec_local.unsqueeze(0).expand(quat.shape[0], -1)
        return quat_apply(quat, vec_local)
    
    @staticmethod
    def world_to_local_vector(
        quat: torch.Tensor,
        vec_world: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate vector FROM world frame TO local frame.
        
        This uses quat_apply_inverse which implements: vec_local = quat_conj * vec_world * quat
        
        Args:
            quat: Orientation quaternion [w,x,y,z] [batch, 4]
            vec_world: Vector in world frame [batch, 3] or [3,]
            
        Returns:
            Vector in local frame [batch, 3]
            
        Example:
            >>> # Object moving with velocity [1, 0, 0] in world frame
            >>> quat = object_orientation  # Object's orientation
            >>> vel_world = torch.tensor([[1.0, 0.0, 0.0]])
            >>> vel_local = PoseTransformer.world_to_local_vector(quat, vel_world)
            >>> # vel_local is velocity expressed in object's local frame
        """
        # Handle scalar expansion
        if vec_world.dim() == 1:
            vec_world = vec_world.unsqueeze(0).expand(quat.shape[0], -1)
        return quat_apply_inverse(quat, vec_world)
    
    @staticmethod
    def apply_offset_in_local_frame(
        position: torch.Tensor,
        quat: torch.Tensor,
        offset_local: torch.Tensor,
    ) -> torch.Tensor:
        """Apply an offset defined in local frame to a world position.
        
        This is equivalent to: position + rotate(offset_local, quat)
        
        Args:
            position: Base position in world frame [batch, 3]
            quat: Orientation quaternion [w,x,y,z] [batch, 4]
            offset_local: Offset in local frame [3,]
            
        Returns:
            New position with offset applied [batch, 3]
            
        Example:
            >>> # Move 10cm "up" in local Z direction
            >>> pos = torch.tensor([[0.5, 0.0, 0.1]])
            >>> quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            >>> offset = torch.tensor([0.0, 0.0, 0.1])
            >>> new_pos = PoseTransformer.apply_offset_in_local_frame(pos, quat, offset)
        """
        offset_world = PoseTransformer.local_to_world_vector(quat, offset_local)
        return position + offset_world
    
    @staticmethod
    def compute_pre_grasp_position(
        grasp_pos: torch.Tensor,
        grasp_quat: torch.Tensor,
        ee_offset: torch.Tensor,
        approach_height: float = 0.3,
    ) -> torch.Tensor:
        """Compute pre-grasp position (approach point above object).
        
        The pre-grasp position is offset along the negative Z direction
        of the grasp frame (typically downward for top-down grasping).
        
        Args:
            grasp_pos: Grasp position (TCP) [batch, 3]
            grasp_quat: Grasp orientation [w,x,y,z] [batch, 4]
            ee_offset: Offset from flange to TCP [3,]
            approach_height: Height offset for pre-grasp approach [m]
            
        Returns:
            Pre-grasp flange position [batch, 3]
            
        Formula:
            pre_grasp_tcp = grasp_pos - (approach_height along local Z)
            pre_grasp_flange = tcp_to_flange(pre_grasp_tcp, ...)
        """
        # Offset in grasp local frame (along negative Z, i.e., "above")
        approach_offset_local = torch.tensor(
            [0.0, 0.0, -approach_height],
            device=grasp_pos.device,
            dtype=grasp_pos.dtype
        )
        
        # Compute approach point in TCP frame
        pre_grasp_tcp = PoseTransformer.apply_offset_in_local_frame(
            grasp_pos, grasp_quat, approach_offset_local
        )
        
        # Convert to flange frame
        pre_grasp_flange, _ = PoseTransformer.tcp_to_flange(
            pre_grasp_tcp, grasp_quat, ee_offset
        )
        
        return pre_grasp_flange
    
    @staticmethod
    def compute_grasp_flange_position(
        grasp_pos: torch.Tensor,
        grasp_quat: torch.Tensor,
        ee_offset: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flange position for grasp (TCP position with offset applied).
        
        Args:
            grasp_pos: Grasp position (TCP) [batch, 3]
            grasp_quat: Grasp orientation [w,x,y,z] [batch, 4]
            ee_offset: Offset from flange to TCP [3,]
            
        Returns:
            Grasp flange position [batch, 3]
        """
        flange_pos, _ = PoseTransformer.tcp_to_flange(
            grasp_pos, grasp_quat, ee_offset
        )
        return flange_pos


def normalize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length.
    
    Args:
        quat: Quaternion [w,x,y,z] [..., 4]
        
    Returns:
        Normalized quaternion [..., 4]
    """
    return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)


def quat_distance(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """Compute angular distance between two quaternions.
    
    Args:
        quat1: First quaternion [w,x,y,z] [batch, 4]
        quat2: Second quaternion [w,x,y,z] [batch, 4]
        
    Returns:
        Angular distance in radians [batch,]
    """
    dot = torch.sum(quat1 * quat2, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)
