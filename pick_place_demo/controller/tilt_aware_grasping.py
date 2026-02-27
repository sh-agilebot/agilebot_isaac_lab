#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tilt-aware grasping controller for handling overturned/tilted objects.

This module provides algorithms for:
- Detecting object tilt from quaternion orientation
- Computing gripper pose compensation for horizontal grasping
- Stable grasping strategies for various tilt angles

Usage:
    from controller.tilt_aware_grasping import TiltAwareGrasping
    
    grasping = TiltAwareGrasping(device=sim.device)
    compensated_pose = grasping.compute_compensated_grasp_pose(
        object_pos, object_quat, max_tilt_angle=45.0
    )
"""

import torch
import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TiltInfo:
    """Information about object tilt state.
    
    Attributes:
        tilt_angle: Tilt angle in degrees (0 = upright, 90 = horizontal, 180 = upside down)
        tilt_axis: Unit vector representing the tilt rotation axis in world frame
        is_tilted: True if object is significantly tilted (angle > threshold)
        is_upside_down: True if object's Z-axis points downward (angle > 90°)
        z_axis_world: Object's local Z-axis in world coordinates
    """
    tilt_angle: torch.Tensor  # degrees, shape (N,)
    tilt_axis: torch.Tensor   # shape (N, 3)
    is_tilted: torch.Tensor   # shape (N,) bool
    is_upside_down: torch.Tensor  # shape (N,) bool
    z_axis_world: torch.Tensor    # shape (N, 3)


class TiltAwareGrasping:
    """
    Tilt-aware grasping controller for handling objects that are not upright.
    
    This class provides methods to:
    1. Detect object tilt from quaternion
    2. Compute optimal gripper orientation for stable grasping
    3. Adapt grasp strategy based on tilt severity
    
    Key Concepts:
    - Object's Z-axis orientation determines if it's tilted/upright/upside-down
    - Gripper should approach from the side of the object's "top" surface
    - For severely tilted objects (>45°), use side-grasp instead of top-grasp
    """
    
    # Tilt thresholds for different grasping strategies
    TILT_THRESHOLD_SLIGHT = 15.0    # degrees - minor tilt, standard grasp
    TILT_THRESHOLD_MODERATE = 30.0  # degrees - moderate tilt, start compensation
    TILT_THRESHOLD_SEVERE = 45.0    # degrees - severe tilt, side-grasp recommended
    TILT_THRESHOLD_HORIZONTAL = 75.0 # degrees - nearly horizontal, full side-grasp
    
    def __init__(
        self,
        device: torch.device,
        tilt_threshold: float = 15.0,
        max_grasp_offset: float = 0.05,
        gripper_finger_length: float = 0.08,
    ):
        """Initialize tilt-aware grasping controller.
        
        Args:
            device: Torch device for tensor operations
            tilt_threshold: Minimum tilt angle (degrees) to trigger compensation
            max_grasp_offset: Maximum lateral offset for compensated grasping (m)
            gripper_finger_length: Length of gripper fingers (m)
        """
        self.device = device
        self.tilt_threshold = tilt_threshold
        self.max_grasp_offset = max_grasp_offset
        self.gripper_finger_length = gripper_finger_length
        
        # Pre-compute common tensors
        self._world_z = torch.tensor([0.0, 0.0, 1.0], device=device)
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    
    def detect_tilt(
        self,
        object_quat: torch.Tensor,
        tilt_threshold: Optional[float] = None,
    ) -> TiltInfo:
        """
        Detect object tilt from its quaternion orientation.
        
        The tilt angle is computed as the angle between the object's local Z-axis
        and the world Z-axis. This indicates how much the object has tipped over.
        
        Args:
            object_quat: Object orientation quaternion [w, x, y, z], shape (N, 4)
            tilt_threshold: Custom threshold for "tilted" classification
            
        Returns:
            TiltInfo containing tilt angle, axis, and classification
        """
        if tilt_threshold is None:
            tilt_threshold = self.tilt_threshold
        
        # Normalize quaternion
        quat_norm = object_quat / torch.norm(object_quat, dim=-1, keepdim=True)
        
        # Extract object's local Z-axis in world frame
        # For quaternion [w, x, y, z], rotating [0, 0, 1] gives:
        # z_world = [2(xz + wy), 2(yz - wx), 1 - 2(x² + y²)]
        w, x, y, z = quat_norm[..., 0], quat_norm[..., 1], quat_norm[..., 2], quat_norm[..., 3]
        
        z_axis_world = torch.stack([
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ], dim=-1)
        
        # Normalize to unit vector
        z_axis_world = z_axis_world / torch.norm(z_axis_world, dim=-1, keepdim=True)
        
        # Compute tilt angle using dot product with world Z
        # cos(angle) = z_axis_world · world_z = z_axis_world[..., 2]
        cos_angle = torch.clamp(z_axis_world[..., 2], -1.0, 1.0)
        tilt_angle_rad = torch.acos(cos_angle)
        tilt_angle_deg = tilt_angle_rad * 180.0 / math.pi
        
        # Compute tilt axis (rotation axis from upright to current)
        # tilt_axis = world_z × z_axis_world, normalized
        tilt_axis = torch.cross(
            self._world_z.unsqueeze(0).expand_as(z_axis_world),
            z_axis_world,
            dim=-1
        )
        tilt_axis_norm = torch.norm(tilt_axis, dim=-1, keepdim=True)
        # Avoid division by zero for nearly upright objects
        tilt_axis = torch.where(
            tilt_axis_norm > 1e-6,
            tilt_axis / tilt_axis_norm,
            torch.zeros_like(tilt_axis)
        )
        
        # Classify tilt state
        is_tilted = tilt_angle_deg > tilt_threshold
        is_upside_down = tilt_angle_deg > 90.0
        
        return TiltInfo(
            tilt_angle=tilt_angle_deg,
            tilt_axis=tilt_axis,
            is_tilted=is_tilted,
            is_upside_down=is_upside_down,
            z_axis_world=z_axis_world,
        )
    
    def compute_compensated_grasp_pose(
        self,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        max_tilt_angle: float = 90.0,
        grasp_height_ratio: float = 0.5,
        approach_from_side: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, TiltInfo]:
        """
        Compute gripper pose with tilt compensation for horizontal grasping.
        
        This method adjusts the gripper's approach direction and orientation
        to maintain a stable grasp on tilted objects.
        
        Strategy:
        1. Detect object tilt angle and axis
        2. For slight tilt (<15°): Use standard top-down grasp with minor compensation
        3. For moderate tilt (15°-45°): Apply partial rotation compensation
        4. For severe tilt (>45°): Use side-grasp approach
        
        Args:
            object_pos: Object position in world frame, shape (N, 3)
            object_quat: Object orientation quaternion [w, x, y, z], shape (N, 4)
            max_tilt_angle: Maximum tilt angle to attempt grasping (degrees)
            grasp_height_ratio: Ratio of grasp height (0=bottom, 1=top), default 0.5
            approach_from_side: If True, approach from the side of tilt axis
            
        Returns:
            Tuple of:
            - grasp_position: Compensated grasp position, shape (N, 3)
            - grasp_quat: Compensated gripper orientation, shape (N, 4)
            - tilt_info: Tilt detection results
        """
        # Detect tilt
        tilt_info = self.detect_tilt(object_quat)
        
        # Get batch size
        batch_size = object_pos.shape[0]
        
        # Initialize output tensors
        grasp_position = object_pos.clone()
        grasp_quat = torch.zeros((batch_size, 4), device=self.device)
        
        for i in range(batch_size):
            tilt_angle = tilt_info.tilt_angle[i].item()
            
            # Skip if tilt exceeds maximum
            if tilt_angle > max_tilt_angle:
                # Return default pose (may fail in practice)
                grasp_quat[i] = self._identity_quat
                continue
            
            # Determine grasp strategy based on tilt severity
            if tilt_angle < self.TILT_THRESHOLD_SLIGHT:
                # Slight tilt: Standard top-down grasp
                pos, quat = self._compute_slight_tilt_grasp(
                    object_pos[i], object_quat[i], tilt_info, i
                )
            elif tilt_angle < self.TILT_THRESHOLD_SEVERE:
                # Moderate tilt: Partial compensation
                pos, quat = self._compute_moderate_tilt_grasp(
                    object_pos[i], object_quat[i], tilt_info, i, grasp_height_ratio
                )
            else:
                # Severe tilt: Side-grasp
                pos, quat = self._compute_severe_tilt_grasp(
                    object_pos[i], object_quat[i], tilt_info, i, 
                    grasp_height_ratio, approach_from_side
                )
            
            grasp_position[i] = pos
            grasp_quat[i] = quat
        
        return grasp_position, grasp_quat, tilt_info
    
    def _compute_slight_tilt_grasp(
        self,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        tilt_info: TiltInfo,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute grasp for slightly tilted objects (<15°).
        
        Use standard top-down approach with minor orientation adjustment.
        """
        # Standard approach: gripper Z-axis aligned with world Z
        # Small rotation to match object orientation slightly
        
        # Compute small rotation to compensate tilt
        compensation_angle = tilt_info.tilt_angle[idx] * math.pi / 180.0
        compensation_axis = tilt_info.tilt_axis[idx]
        
        # Create compensation quaternion (rotate opposite to tilt)
        half_angle = -compensation_angle * 0.5  # Partial compensation
        compensation_quat = self._axis_angle_to_quat(
            compensation_axis, half_angle
        )
        
        # Base gripper orientation (top-down)
        base_quat = torch.tensor([0.0, 0.7071, 0.7071, 0.0], device=self.device)
        
        # Apply compensation
        grasp_quat = self._quat_multiply(base_quat.unsqueeze(0), 
                                          compensation_quat.unsqueeze(0)).squeeze(0)
        
        return object_pos, grasp_quat
    
    def _compute_moderate_tilt_grasp(
        self,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        tilt_info: TiltInfo,
        idx: int,
        grasp_height_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute grasp for moderately tilted objects (15°-45°).
        
        Apply partial orientation compensation while maintaining gripper stability.
        """
        tilt_angle_rad = tilt_info.tilt_angle[idx] * math.pi / 180.0
        tilt_axis = tilt_info.tilt_axis[idx]
        
        # Compensation factor: linear interpolation from 0.3 to 0.7
        # More compensation for larger tilts
        compensation_factor = 0.3 + 0.4 * (
            (tilt_info.tilt_angle[idx] - self.TILT_THRESHOLD_SLIGHT) /
            (self.TILT_THRESHOLD_SEVERE - self.TILT_THRESHOLD_SLIGHT)
        )
        
        # Compute compensation rotation
        compensation_angle = -tilt_angle_rad * compensation_factor
        compensation_quat = self._axis_angle_to_quat(tilt_axis, compensation_angle)
        
        # Compute lateral offset based on tilt
        # Gripper should be positioned to the side of the object
        lateral_offset = self.max_grasp_offset * torch.sin(tilt_angle_rad * compensation_factor)
        
        # Offset direction: perpendicular to tilt axis and world Z
        offset_dir = torch.cross(tilt_axis, self._world_z, dim=-1)
        offset_dir = offset_dir / (torch.norm(offset_dir) + 1e-6)
        
        # Apply offset to grasp position
        grasp_pos = object_pos + offset_dir * lateral_offset
        
        # Base gripper orientation
        base_quat = torch.tensor([0.0, 0.7071, 0.7071, 0.0], device=self.device)
        
        # Apply compensation
        grasp_quat = self._quat_multiply(base_quat.unsqueeze(0),
                                          compensation_quat.unsqueeze(0)).squeeze(0)
        
        return grasp_pos, grasp_quat
    
    def _compute_severe_tilt_grasp(
        self,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        tilt_info: TiltInfo,
        idx: int,
        grasp_height_ratio: float,
        approach_from_side: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute grasp for severely tilted objects (>45°).
        
        Use side-grasp approach: gripper approaches from the side,
        maintaining horizontal grip on the tilted object.
        """
        tilt_angle_rad = tilt_info.tilt_angle[idx] * math.pi / 180.0
        tilt_axis = tilt_info.tilt_axis[idx]
        z_axis_world = tilt_info.z_axis_world[idx]
        
        # For severely tilted objects, approach from the direction
        # that the object's "top" surface is facing
        
        # Compute approach direction: along the object's tilted Z-axis
        approach_dir = z_axis_world
        
        # Compute gripper orientation to maintain horizontal grip
        # The gripper's X-axis should align with approach direction
        # The gripper's Z-axis should be perpendicular (horizontal)
        
        # Create rotation from approach direction
        # gripper_x = approach_dir
        # gripper_z = tilt_axis (rotation axis, perpendicular to approach)
        # gripper_y = gripper_z × gripper_x
        
        gripper_x = approach_dir
        gripper_z = tilt_axis
        gripper_y = torch.cross(gripper_z, gripper_x, dim=-1)
        
        # Ensure orthogonalization
        gripper_x = gripper_x / (torch.norm(gripper_x) + 1e-6)
        gripper_y = gripper_y / (torch.norm(gripper_y) + 1e-6)
        gripper_z = torch.cross(gripper_x, gripper_y, dim=-1)
        gripper_z = gripper_z / (torch.norm(gripper_z) + 1e-6)
        
        # Construct rotation matrix
        rot_matrix = torch.stack([gripper_x, gripper_y, gripper_z], dim=-1)
        
        # Convert to quaternion
        grasp_quat = self._rotmat_to_quat(rot_matrix.unsqueeze(0)).squeeze(0)
        
        # Compute approach offset
        # Position gripper at a distance from the object center
        approach_distance = 0.15  # meters
        
        if approach_from_side:
            # Approach from the side perpendicular to tilt axis
            approach_offset = approach_dir * approach_distance * 0.5
        else:
            # Approach from above with compensation
            approach_offset = self._world_z * approach_distance * 0.3
        
        grasp_pos = object_pos + approach_offset
        
        return grasp_pos, grasp_quat
    
    def compute_stable_grasp_points(
        self,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        object_dims: torch.Tensor,
        tilt_info: Optional[TiltInfo] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optimal grasp contact points on tilted objects.
        
        For tilted cylindrical objects (cans), the optimal grasp points
        should be on opposite sides of the cylinder, aligned with the
        object's tilted orientation.
        
        Args:
            object_pos: Object center position, shape (N, 3)
            object_quat: Object orientation quaternion, shape (N, 4)
            object_dims: Object dimensions [radius, height], shape (N, 2)
            tilt_info: Pre-computed tilt info (optional)
            
        Returns:
            Tuple of:
            - left_contact: Left finger contact point, shape (N, 3)
            - right_contact: Right finger contact point, shape (N, 3)
        """
        if tilt_info is None:
            tilt_info = self.detect_tilt(object_quat)
        
        batch_size = object_pos.shape[0]
        
        # Get object's local X-axis in world frame (for grasp direction)
        quat_norm = object_quat / torch.norm(object_quat, dim=-1, keepdim=True)
        w, x, y, z = quat_norm[..., 0], quat_norm[..., 1], quat_norm[..., 2], quat_norm[..., 3]
        
        # Object's local X-axis in world frame
        obj_x_axis = torch.stack([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ], dim=-1)
        
        # For tilted objects, use the projection of world X or Y
        # onto the object's equatorial plane (perpendicular to Z)
        obj_z_axis = tilt_info.z_axis_world
        
        # Choose grasp axis perpendicular to object Z
        # Use the tilt axis direction for consistent grasping
        grasp_axis = tilt_info.tilt_axis
        
        # Ensure grasp axis is perpendicular to object Z
        grasp_axis = grasp_axis - torch.sum(grasp_axis * obj_z_axis, dim=-1, keepdim=True) * obj_z_axis
        grasp_axis = grasp_axis / (torch.norm(grasp_axis, dim=-1, keepdim=True) + 1e-6)
        
        # Compute contact points at the object's radius
        radius = object_dims[..., 0]  # cylinder radius
        
        # For tilted objects, adjust height based on tilt
        # Contact points should be at the "equator" of the tilted cylinder
        height_offset = torch.zeros((batch_size, 3), device=self.device)
        
        left_contact = object_pos + grasp_axis * radius.unsqueeze(-1) + height_offset
        right_contact = object_pos - grasp_axis * radius.unsqueeze(-1) + height_offset
        
        return left_contact, right_contact
    
    def _axis_angle_to_quat(
        self, axis: torch.Tensor, angle: float
    ) -> torch.Tensor:
        """Convert axis-angle representation to quaternion.
        
        Args:
            axis: Rotation axis (unit vector), shape (3,)
            angle: Rotation angle in radians
            
        Returns:
            Quaternion [w, x, y, z]
        """
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)
        
        quat = torch.zeros(4, device=self.device)
        quat[0] = cos_half
        quat[1:4] = axis * sin_half
        
        return quat / (torch.norm(quat) + 1e-8)
    
    def _quat_multiply(
        self, q1: torch.Tensor, q2: torch.Tensor
    ) -> torch.Tensor:
        """Multiply two quaternions.
        
        Args:
            q1: First quaternion [w, x, y, z], shape (N, 4) or (4,)
            q2: Second quaternion [w, x, y, z], shape (N, 4) or (4,)
            
        Returns:
            Product quaternion, same shape as inputs
        """
        if q1.dim() == 1:
            q1 = q1.unsqueeze(0)
            single_input = True
        else:
            single_input = False
            
        if q2.dim() == 1:
            q2 = q2.unsqueeze(0)
        
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        result = torch.zeros_like(q1)
        result[..., 0] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        result[..., 1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        result[..., 2] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        result[..., 3] = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        if single_input:
            result = result.squeeze(0)
        
        return result
    
    def _rotmat_to_quat(self, rotmat: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion [w, x, y, z].
        
        Args:
            rotmat: Rotation matrix, shape (N, 3, 3)
            
        Returns:
            Quaternion [w, x, y, z], shape (N, 4)
        """
        batch_size = rotmat.shape[0]
        
        r00, r01, r02 = rotmat[:, 0, 0], rotmat[:, 0, 1], rotmat[:, 0, 2]
        r10, r11, r12 = rotmat[:, 1, 0], rotmat[:, 1, 1], rotmat[:, 1, 2]
        r20, r21, r22 = rotmat[:, 2, 0], rotmat[:, 2, 1], rotmat[:, 2, 2]
        
        trace = r00 + r11 + r22
        
        quat = torch.zeros((batch_size, 4), device=self.device)
        
        # Case 1: trace > 0
        mask1 = trace > 0
        s = torch.sqrt(trace + 1.0) * 2.0
        quat[mask1, 0] = 0.25 * s[mask1]
        quat[mask1, 1] = (r21 - r12)[mask1] / s[mask1]
        quat[mask1, 2] = (r02 - r20)[mask1] / s[mask1]
        quat[mask1, 3] = (r10 - r01)[mask1] / s[mask1]
        
        # Case 2: r00 is largest diagonal
        mask2 = (~mask1) & (r00 > r11) & (r00 > r22)
        s = torch.sqrt(1.0 + r00 - r11 - r22) * 2.0
        quat[mask2, 0] = (r21 - r12)[mask2] / s[mask2]
        quat[mask2, 1] = 0.25 * s[mask2]
        quat[mask2, 2] = (r01 + r10)[mask2] / s[mask2]
        quat[mask2, 3] = (r02 + r20)[mask2] / s[mask2]
        
        # Case 3: r11 is largest diagonal
        mask3 = (~mask1) & (~mask2) & (r11 > r22)
        s = torch.sqrt(1.0 + r11 - r00 - r22) * 2.0
        quat[mask3, 0] = (r02 - r20)[mask3] / s[mask3]
        quat[mask3, 1] = (r01 + r10)[mask3] / s[mask3]
        quat[mask3, 2] = 0.25 * s[mask3]
        quat[mask3, 3] = (r12 + r21)[mask3] / s[mask3]
        
        # Case 4: r22 is largest diagonal
        mask4 = (~mask1) & (~mask2) & (~mask3)
        s = torch.sqrt(1.0 + r22 - r00 - r11) * 2.0
        quat[mask4, 0] = (r10 - r01)[mask4] / s[mask4]
        quat[mask4, 1] = (r02 + r20)[mask4] / s[mask4]
        quat[mask4, 2] = (r12 + r21)[mask4] / s[mask4]
        quat[mask4, 3] = 0.25 * s[mask4]
        
        # Normalize
        quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
        
        return quat


class SensorConfiguration:
    """
    Sensor configuration recommendations for tilt-aware grasping.
    
    This class provides guidance on sensor setup for reliable
    object pose detection in real-world applications.
    """
    
    @staticmethod
    def get_recommended_sensors() -> Dict[str, Any]:
        """Get recommended sensor configuration for tilt detection.
        
        Returns:
            Dictionary with sensor specifications and setup guidelines
        """
        return {
            "vision_sensors": {
                "rgb_camera": {
                    "purpose": "Object detection and pose estimation",
                    "resolution": "1280x720 or higher",
                    "frame_rate": "30+ fps",
                    "placement": "Above workspace, angled 30-45° from vertical",
                    "quantity": "1-2 cameras for stereo or multi-view",
                },
                "depth_camera": {
                    "purpose": "3D pose estimation and tilt detection",
                    "type": "Structured light or Time-of-Flight",
                    "resolution": "640x480 or higher",
                    "range": "0.3m - 2.0m",
                    "accuracy": "< 5mm at 1m distance",
                    "placement": "Above workspace or wrist-mounted",
                },
                "wrist_camera": {
                    "purpose": "Close-range pose refinement during approach",
                    "resolution": "640x480",
                    "placement": "Mounted on robot end-effector",
                    "fov": "60-90°",
                },
            },
            "force_torque_sensor": {
                "purpose": "Contact detection and grasp force control",
                "type": "6-axis F/T sensor",
                "placement": "Between robot wrist and gripper",
                "range": "±50N force, ±5Nm torque",
                "sample_rate": "1000+ Hz",
            },
            "proximity_sensors": {
                "purpose": "Pre-contact object detection",
                "type": "Infrared or ultrasonic",
                "range": "0.05m - 0.5m",
                "placement": "On gripper fingers",
            },
            "imu": {
                "purpose": "Object dynamics and collision detection",
                "type": "6-axis or 9-axis IMU",
                "placement": "On gripper or object (if feasible)",
                "sample_rate": "200+ Hz",
            },
        }
    
    @staticmethod
    def get_pose_estimation_pipeline() -> Dict[str, Any]:
        """Get recommended pose estimation pipeline.
        
        Returns:
            Dictionary describing the pose estimation workflow
        """
        return {
            "step_1_object_detection": {
                "method": "Deep learning (YOLO, Mask R-CNN)",
                "input": "RGB image",
                "output": "Bounding box, segmentation mask",
            },
            "step_2_3d_reconstruction": {
                "method": "Depth fusion or stereo matching",
                "input": "Depth image + detection mask",
                "output": "3D point cloud of object",
            },
            "step_3_pose_estimation": {
                "method": "ICP, PPF, or learning-based (DenseFusion)",
                "input": "Point cloud + CAD model",
                "output": "6DoF pose (position + quaternion)",
            },
            "step_4_tilt_analysis": {
                "method": "Quaternion decomposition",
                "input": "Object quaternion",
                "output": "Tilt angle, tilt axis, grasp strategy",
            },
            "step_5_real_time_tracking": {
                "method": "Kalman filter or particle filter",
                "purpose": "Smooth pose estimates, predict motion",
                "update_rate": "30+ Hz",
            },
        }
    
    @staticmethod
    def get_calibration_requirements() -> Dict[str, Any]:
        """Get calibration requirements for accurate pose estimation.
        
        Returns:
            Dictionary with calibration specifications
        """
        return {
            "camera_intrinsics": {
                "method": "Chessboard or circle grid pattern",
                "frequency": "Once per setup change",
                "accuracy": "< 0.5 pixel reprojection error",
            },
            "camera_extrinsics": {
                "method": "Hand-eye calibration (Tsai method)",
                "frequency": "Once per setup change",
                "accuracy": "< 2mm position, < 1° orientation",
            },
            "robot_kinematics": {
                "method": "Robot calibration routine",
                "frequency": "Monthly or after collision",
                "accuracy": "< 1mm TCP accuracy",
            },
            "gripper_geometry": {
                "method": "Manual measurement + CAD verification",
                "parameters": "Finger length, width, open/close range",
            },
        }


def integrate_with_controller(
    controller: 'ParallelPickPlaceController',
    tilt_aware_grasping: TiltAwareGrasping,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate tilt-aware grasping with existing pick-place controller.
    
    This function modifies the grasp orientation computed by the standard
    controller to account for object tilt, enabling horizontal grasping
    of tilted objects.
    
    Args:
        controller: Existing ParallelPickPlaceController instance
        tilt_aware_grasping: TiltAwareGrasping instance
        object_pos: Object position tensor
        object_quat: Object orientation quaternion
        
    Returns:
        Tuple of modified (grasp_position, grasp_quaternion)
    """
    # Detect tilt
    tilt_info = tilt_aware_grasping.detect_tilt(object_quat)
    
    # Only modify for significantly tilted objects
    if torch.any(tilt_info.is_tilted):
        # Compute compensated pose
        comp_pos, comp_quat, _ = tilt_aware_grasping.compute_compensated_grasp_pose(
            object_pos, object_quat
        )
        
        # Blend with original based on tilt severity
        blend_factor = torch.clamp(
            (tilt_info.tilt_angle - tilt_aware_grasping.TILT_THRESHOLD_SLIGHT) / 
            (tilt_aware_grasping.TILT_THRESHOLD_SEVERE - tilt_aware_grasping.TILT_THRESHOLD_SLIGHT),
            0.0, 1.0
        )
        
        # For now, return compensated pose for tilted objects
        # In practice, you might want to blend or validate with IK feasibility
        return comp_pos, comp_quat
    else:
        # Return original (would be computed by standard controller)
        return object_pos, object_quat


# Example usage and testing
if __name__ == "__main__":
    """Test tilt-aware grasping algorithms."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create tilt-aware grasping controller
    grasping = TiltAwareGrasping(device=device)
    
    # Test case 1: Upright object
    upright_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    tilt_info = grasping.detect_tilt(upright_quat)
    print(f"\nTest 1 - Upright object:")
    print(f"  Tilt angle: {tilt_info.tilt_angle[0]:.2f}°")
    print(f"  Is tilted: {tilt_info.is_tilted[0]}")
    
    # Test case 2: 30° tilted object
    # Rotation of 30° around X-axis
    angle = 30 * math.pi / 180
    tilted_quat = torch.tensor([[
        math.cos(angle/2), math.sin(angle/2), 0.0, 0.0
    ]], device=device)
    tilt_info = grasping.detect_tilt(tilted_quat)
    print(f"\nTest 2 - 30° tilted object:")
    print(f"  Tilt angle: {tilt_info.tilt_angle[0]:.2f}°")
    print(f"  Is tilted: {tilt_info.is_tilted[0]}")
    
    # Test case 3: 60° tilted object (severe)
    angle = 60 * math.pi / 180
    severe_quat = torch.tensor([[
        math.cos(angle/2), math.sin(angle/2), 0.0, 0.0
    ]], device=device)
    pos = torch.tensor([[0.5, 0.3, 0.2]], device=device)
    grasp_pos, grasp_quat, tilt_info = grasping.compute_compensated_grasp_pose(
        pos, severe_quat
    )
    print(f"\nTest 3 - 60° tilted object (severe):")
    print(f"  Tilt angle: {tilt_info.tilt_angle[0]:.2f}°")
    print(f"  Grasp position: {grasp_pos[0]}")
    print(f"  Grasp quaternion: {grasp_quat[0]}")
    
    # Test case 4: Upside-down object (180°)
    angle = 180 * math.pi / 180
    inverted_quat = torch.tensor([[
        math.cos(angle/2), 0.0, math.sin(angle/2), 0.0
    ]], device=device)
    tilt_info = grasping.detect_tilt(inverted_quat)
    print(f"\nTest 4 - Upside-down object:")
    print(f"  Tilt angle: {tilt_info.tilt_angle[0]:.2f}°")
    print(f"  Is upside-down: {tilt_info.is_upside_down[0]}")
    
    print("\n[OK] Tilt-aware grasping tests completed")
    
    # Print sensor configuration
    print("\n" + "="*60)
    print("Recommended Sensor Configuration:")
    print("="*60)
    sensors = SensorConfiguration.get_recommended_sensors()
    for sensor_type, config in sensors.items():
        print(f"\n{sensor_type.upper()}:")
        if isinstance(config, dict):
            # Check if this is a nested config (like vision_sensors) or flat config (like imu)
            if any(isinstance(v, dict) for v in config.values()):
                for name, specs in config.items():
                    if isinstance(specs, dict):
                        print(f"  {name}: {specs.get('purpose', 'N/A')}")
            else:
                for key, value in config.items():
                    print(f"  {key}: {value}")
