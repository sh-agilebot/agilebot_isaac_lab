#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Math utilities for EE6D data processing.

This module provides quaternion to 6D rotation conversion functions
compatible with IsaacLab and X-VLA requirements.
"""

import torch
import numpy as np
from typing import Union


def quat_to_rot6d(quaternions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert quaternions to 6D rotation representation.
    
    This function converts a quaternion [w, x, y, z] or [x, y, z, w] to a 6D rotation
    representation consisting of the first two rows of a rotation matrix.
    
    Args:
        quaternions: Quaternion tensor/array of shape (..., 4)
                    Can be either [w, x, y, z] or [x, y, z, w] format
        
    Returns:
        rot6d: 6D rotation representation of shape (..., 6)
               Contains the first two rows of the rotation matrix flattened
    """
    if isinstance(quaternions, np.ndarray):
        quaternions = torch.from_numpy(quaternions)
    
    # Preserve device
    device = quaternions.device
    
    # Ensure input has correct shape
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with 4 components, got shape {quaternions.shape}")
    
    # Normalize quaternion
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    # Extract components (IsaacLab uses [w, x, y, z] format)
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    
    # Compute rotation matrix elements
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # First row of rotation matrix
    r00 = 1 - 2*(yy + zz)
    r01 = 2*(xy - wz)
    r02 = 2*(xz + wy)
    
    # Second row of rotation matrix  
    r10 = 2*(xy + wz)
    r11 = 1 - 2*(xx + zz)
    r12 = 2*(yz - wx)
    
    # Create 6D representation (first two rows flattened)
    rot6d = torch.stack([r00, r01, r02, r10, r11, r12], dim=-1)
    
    # Ensure output is on the same device as input
    rot6d = rot6d.to(device)
    
    if isinstance(quaternions, torch.Tensor):
        return rot6d
    else:
        return rot6d.numpy()


def rot6d_to_rotmat(rot6d: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert 6D rotation representation back to rotation matrix.
    
    Args:
        rot6d: 6D rotation representation of shape (..., 6)
        
    Returns:
        rotmat: Full 3x3 rotation matrix of shape (..., 3, 3)
    """
    if isinstance(rot6d, np.ndarray):
        rot6d = torch.from_numpy(rot6d)
    
    # Preserve device
    device = rot6d.device
    
    # Reshape to extract rows
    x1 = rot6d[..., :3]  # First row
    x2 = rot6d[..., 3:6]  # Second row
    
    # Normalize rows
    x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
    x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
    
    # Compute third row using cross product
    x3 = torch.cross(x1, x2, dim=-1)
    
    # Stack rows to form rotation matrix
    rotmat = torch.stack([x1, x2, x3], dim=-2)
    
    # Ensure output is on the same device as input
    rotmat = rotmat.to(device)
    
    if isinstance(rot6d, torch.Tensor):
        return rotmat
    else:
        return rotmat.numpy()


def compute_pose_difference(pose1: torch.Tensor, pose2: torch.Tensor) -> torch.Tensor:
    """
    Compute the pose difference between two poses.
    
    Args:
        pose1: First pose [pos(3), quat(4)]
        pose2: Second pose [pos(3), quat(4)]
        
    Returns:
        pose_diff: Pose difference [pos_diff(3), quat_diff(4)]
    """
    device = pose1.device
    
    pos1 = pose1[:3]
    quat1 = pose1[3:7]
    pos2 = pose2[:3]
    quat2 = pose2[3:7]
    
    pos_diff = pos2 - pos1
    
    quat1_inv = torch.cat([quat1[0:1], -quat1[1:4]])
    quat_diff = quaternion_multiply(quat2, quat1_inv)
    
    return torch.cat([pos_diff, quat_diff]).to(device)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions (using IsaacLab [w, x, y, z] format).

    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]

    Returns:
        q_product: Quaternion product [w, x, y, z]
    """
    device = q1.device
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.tensor([w, x, y, z], device=device)


def rotmat_to_quat(rotmat: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert rotation matrix to quaternion (IsaacLab [w, x, y, z] format).

    This function converts a 3x3 rotation matrix to a quaternion [w, x, y, z].
    Uses the Shepperd's method for numerical stability.

    Args:
        rotmat: Rotation matrix of shape (..., 3, 3)

    Returns:
        quaternion: Quaternion [w, x, y, z] of shape (..., 4)
    """
    if isinstance(rotmat, np.ndarray):
        rotmat = torch.from_numpy(rotmat)
    
    # Ensure input has correct shape
    if rotmat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix with shape (..., 3, 3), got {rotmat.shape}")
    
    # Extract matrix elements
    r00, r01, r02 = rotmat[..., 0, 0], rotmat[..., 0, 1], rotmat[..., 0, 2]
    r10, r11, r12 = rotmat[..., 1, 0], rotmat[..., 1, 1], rotmat[..., 1, 2]
    r20, r21, r22 = rotmat[..., 2, 0], rotmat[..., 2, 1], rotmat[..., 2, 2]
    
    # Compute quaternion components using Shepperd's method
    # Choose the case with the largest diagonal element for numerical stability
    trace = r00 + r11 + r22
    
    # Initialize quaternion tensor
    batch_shape = rotmat.shape[:-2]
    quaternion = torch.zeros(batch_shape + (4,), dtype=rotmat.dtype, device=rotmat.device)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
        quaternion[mask1] = torch.stack([w[mask1], x[mask1], y[mask1], z[mask1]], dim=-1)
    
    # Case 2: r00 is the largest diagonal element
    mask2 = (~mask1) & (r00 > r11) & (r00 > r22)
    if mask2.any():
        s = torch.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
        quaternion[mask2] = torch.stack([w[mask2], x[mask2], y[mask2], z[mask2]], dim=-1)
    
    # Case 3: r11 is the largest diagonal element
    mask3 = (~mask1) & (~mask2) & (r11 > r22)
    if mask3.any():
        s = torch.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
        quaternion[mask3] = torch.stack([w[mask3], x[mask3], y[mask3], z[mask3]], dim=-1)
    
    # Case 4: r22 is the largest diagonal element
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        s = torch.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s
        quaternion[mask4] = torch.stack([w[mask4], x[mask4], y[mask4], z[mask4]], dim=-1)
    
    # Normalize quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    
    if isinstance(rotmat, torch.Tensor):
        return quaternion
    else:
        return quaternion.numpy()
