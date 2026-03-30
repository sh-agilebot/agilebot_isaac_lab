#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
EE6D utilities for X-VLA compatible data recording.

This module provides functions to convert robot poses and actions to the
EE6D 10D format compatible with X-VLA training.
"""

import torch
import numpy as np
from typing import Union
from core.math_utils import quat_to_rot6d, rot6d_to_rotmat


def compute_ee6d_delta(
    current_pose: torch.Tensor, target_pose: torch.Tensor, action_frame: str = "base"
) -> torch.Tensor:
    """
    Compute EE6D delta action from current to target pose (without gripper).

    The gripper delta should be computed separately by the caller since it
    depends on the gripper state transition logic.

    Args:
        current_pose: Current end-effector pose [pos(3), quat(4)] or (N, 7) for batch
        target_pose: Target end-effector pose [pos(3), quat(4)] or (N, 7) for batch
        action_frame: Reference frame for delta computation

    Returns:
        ee6d_delta: 9D action vector [dx, dy, dz, dr6d_0...dr6d_5] (pos + rot, no gripper)
    """
    # Extract position and orientation (works for both single and batch)
    current_pos = current_pose[..., :3]
    current_quat = current_pose[..., 3:7]
    target_pos = target_pose[..., :3]
    target_quat = target_pose[..., 3:7]

    # Compute position delta
    pos_delta = target_pos - current_pos

    # Convert quaternions to rotation matrices
    current_rot6d = quat_to_rot6d(current_quat)
    target_rot6d = quat_to_rot6d(target_quat)

    # Convert 6D rotation to rotation matrices
    current_rotmat = rot6d_to_rotmat(current_rot6d)
    target_rotmat = rot6d_to_rotmat(target_rot6d)

    # Compute exact rotation delta using rotation matrix composition
    # delta_rotmat = R_current^T @ R_target
    # Use transpose(-2, -1) to handle both single and batch dimensions
    delta_rotmat = current_rotmat.transpose(-2, -1) @ target_rotmat

    # Convert delta rotation matrix to 6D representation
    # Extract first two rows and flatten to 6D, preserving batch dimensions
    rot_delta = delta_rotmat[..., :2, :].reshape(delta_rotmat.shape[:-2] + (6,))

    # Combine into 9D action vector (pos + rot, no gripper)
    # Caller should add gripper delta separately
    ee6d_delta = torch.cat(
        [
            pos_delta,  # [..., 3] position delta
            rot_delta,  # [..., 6] rotation 6D delta
        ],
        dim=-1,
    )

    return ee6d_delta


def pose_to_ee6d_proprio(
    pose: torch.Tensor, gripper_state: Union[float, torch.Tensor] = 0.0
) -> torch.Tensor:
    """
    Convert robot pose to EE6D proprioceptive state.

    Args:
        pose: End-effector pose [pos(3), quat(4)]
        gripper_state: Gripper opening state [0.0, 1.0]

    Returns:
        ee6d_proprio: 10D proprioceptive vector [x, y, z, r6d_0...r6d_5, gripper]
    """
    # Extract position and orientation
    pos = pose[:3]
    quat = pose[3:7]

    # Convert quaternion to 6D rotation
    rot6d = quat_to_rot6d(quat)

    # Ensure gripper state is tensor on same device as pose
    if isinstance(gripper_state, (float, int)):
        gripper_state = torch.tensor([float(gripper_state)], device=pose.device)
    else:
        if gripper_state.dim() == 0:
            gripper_state = gripper_state.unsqueeze(0)
        gripper_state = gripper_state.to(pose.device)

    # Combine into 10D proprioceptive vector
    ee6d_proprio = torch.cat(
        [
            pos,  # [3] position
            rot6d,  # [6] rotation 6D
            gripper_state,  # [1] gripper state
        ]
    )

    return ee6d_proprio


def create_ee6d_metadata(
    domain_id: int = 0,
    action_frame: str = "base",
    translation_unit: str = "meter",
    gripper_semantics: str = "1=close,0=open",
) -> dict:
    """
    Create metadata dictionary for EE6D dataset.

    Args:
        domain_id: Domain identifier
        action_frame: Reference frame for actions
        translation_unit: Unit for position measurements
        gripper_semantics: Gripper state semantics

    Returns:
        metadata: Dictionary with EE6D dataset metadata
    """
    metadata = {
        "spec_version": "xvla_dataset_spec_v1",
        "description": "EE6D 10D dataset compatible with X-VLA fine-tuning",
        "domains": {
            str(domain_id): {
                "name": "isaac_franka",
                "hz": 30,
                "action_frame": action_frame,
                "translation_unit": translation_unit,
                "gripper_semantics": gripper_semantics,
                "camera_map": {"image0": "front_rgb", "image1": "wrist_rgb"},
            }
        },
        "action_space": {
            "name": "EE6D_AbsoluteLocal_10D",
            "vector_dim": 10,
            "fields": [
                "x",
                "y",
                "z",
                "r6d_0",
                "r6d_1",
                "r6d_2",
                "r6d_3",
                "r6d_4",
                "r6d_5",
                "gripper",
            ],
            "semantics": {
                "translation": {
                    "indices": [0, 1, 2],
                    "type": "absolute_local",
                    "unit": translation_unit,
                    "frame": action_frame,
                },
                "rotation": {
                    "indices": [3, 4, 5, 6, 7, 8],
                    "type": "rot6d",
                    "note": "Two 3D vectors are orthonormalized to form a valid rotation matrix.",
                },
                "gripper": {
                    "index": 9,
                    "type": "scalar",
                    "recommended_range": [0.0, 1.0],
                    "threshold_note": f"Threshold and open/close meaning: {gripper_semantics}",
                },
            },
            "legacy_aliases": ["EE6D_Delta_10D"],
            "note": "Action stores next-step absolute local EE6D state (action_t = proprio_{t+1}).",
        },
        "proprio": {
            "note": "EE6D proprioceptive state for single-arm robot",
            "schema": {
                "type": "object",
                "properties": {
                    "ee_pos": {"type": "array", "items": "float", "length": 3},
                    "ee_rot6d": {"type": "array", "items": "float", "length": 6},
                    "gripper_state": {"type": "number", "range": [0.0, 1.0]},
                },
            },
        },
        "xvla_compatibility": {
            "action_mode": "auto",
            "real_action_dim": 10,
            "max_action_dim": 20,
        },
    }

    return metadata


def validate_ee6d_action(action: torch.Tensor) -> bool:
    """
    Validate EE6D action format.

    Args:
        action: Action vector to validate

    Returns:
        is_valid: Whether action conforms to EE6D 10D format
    """
    if action.dim() == 0:
        action = action.unsqueeze(0)

    # Check dimension
    if action.shape[-1] != 10:
        print(f"Invalid action dimension: {action.shape[-1]}, expected 10")
        return False

    # Check for NaN or Inf
    if torch.isnan(action).any() or torch.isinf(action).any():
        print("Action contains NaN or Inf values")
        return False

    # Check gripper range
    gripper = action[..., 9]
    if (gripper < -1.0).any() or (gripper > 2.0).any():
        print(
            f"Gripper values out of reasonable range: [{gripper.min():.3f}, {gripper.max():.3f}]"
        )
        return False

    return True


def normalize_gripper_state(gripper_state: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Normalize gripper state to [0, 1] range.

    Args:
        gripper_state: Raw gripper state

    Returns:
        normalized_gripper: Normalized gripper state in [0, 1]
    """
    if isinstance(gripper_state, (float, int)):
        gripper_state = torch.tensor([float(gripper_state)])
    elif gripper_state.dim() == 0:
        gripper_state = gripper_state.unsqueeze(0)

    # Assuming input is in [-1, 1] range, normalize to [0, 1]
    normalized = (gripper_state + 1.0) / 2.0
    normalized = torch.clamp(normalized, 0.0, 1.0)

    return normalized
