#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility functions for pick and place demo functionality.

Includes tilt-aware grasping support for handling overturned/tilted objects.
"""

import logging
import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_apply_inverse, quat_mul

logger = logging.getLogger(__name__)

# Cache prim paths to avoid stage traversal every reset.
_TABLE_PRIM_PATH_CACHE: Optional[str] = None
_KLT_PRIM_PATH_CACHE: Optional[str] = None
_OBJECT_PRIM_PATH_CACHE: Dict[str, Optional[str]] = {}
_OBJECT_BASE_Z_LOCAL_CACHE: Dict[str, float] = {}
_OBJECT_BASE_XY_LOCAL_CACHE: Dict[str, Tuple[float, float]] = {}
_KLT_BASE_Z_LOCAL_CACHE: Optional[float] = None
_KLT_BASE_XY_LOCAL_CACHE: Optional[Tuple[float, float]] = None
_KLT_BASE_QUAT_LOCAL_CACHE: Optional[Tuple[float, float, float, float]] = None
SPAWN_X_MIN_LOCAL = 0.25
SPAWN_X_MAX_LOCAL = 0.75
SPAWN_Y_MIN_LOCAL = -0.30
SPAWN_Y_MAX_LOCAL = 0.30

# =============================================================================
# Tilt Detection Functions
# =============================================================================

def detect_object_tilt(
    object_quat: torch.Tensor,
    tilt_threshold: float = 15.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Detect object tilt from quaternion orientation.
    
    The tilt angle is the angle between the object's local Z-axis and world Z-axis.
    This indicates how much the object has tipped over from upright position.
    
    Args:
        object_quat: Object orientation quaternion [w, x, y, z], shape (N, 4)
        tilt_threshold: Threshold in degrees to consider object as "tilted"
        
    Returns:
        Tuple of:
        - tilt_angle_deg: Tilt angle in degrees, shape (N,)
        - is_tilted: Boolean tensor indicating if object is tilted, shape (N,)
        - z_axis_world: Object's Z-axis in world frame, shape (N, 3)
    """
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
    
    # Normalize
    z_axis_world = z_axis_world / torch.norm(z_axis_world, dim=-1, keepdim=True)
    
    # Compute tilt angle using dot product with world Z (which is z_axis_world[..., 2])
    cos_angle = torch.clamp(z_axis_world[..., 2], -1.0, 1.0)
    tilt_angle_rad = torch.acos(cos_angle)
    tilt_angle_deg = tilt_angle_rad * 180.0 / math.pi
    
    # Determine if tilted
    is_tilted = tilt_angle_deg > tilt_threshold
    
    return tilt_angle_deg, is_tilted, z_axis_world


def compute_tilt_compensation(
    object_quat: torch.Tensor,
    tilt_angle_deg: torch.Tensor,
    z_axis_world: torch.Tensor,
    device: torch.device,
    tilt_threshold_slight: float = 15.0,
    tilt_threshold_severe: float = 45.0,
) -> torch.Tensor:
    """
    Compute gripper orientation compensation for tilted objects.
    
    Strategy:
    - Slight tilt (<15°): Minor compensation, standard top-down grasp
    - Moderate tilt (15°-45°): Partial rotation compensation
    - Severe tilt (>45°): Full side-grasp approach
    
    Args:
        object_quat: Object orientation quaternion [w, x, y, z], shape (N, 4)
        tilt_angle_deg: Tilt angle in degrees, shape (N,)
        z_axis_world: Object's Z-axis in world frame, shape (N, 3)
        device: Torch device
        tilt_threshold_slight: Threshold for slight tilt
        tilt_threshold_severe: Threshold for severe tilt (side-grasp)
        
    Returns:
        Compensation quaternion to apply to base gripper orientation
    """
    batch_size = object_quat.shape[0]
    
    # Compute tilt axis (rotation axis from upright to current)
    world_z = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(batch_size, -1)
    tilt_axis = torch.cross(world_z, z_axis_world, dim=-1)
    tilt_axis_norm = torch.norm(tilt_axis, dim=-1, keepdim=True)
    # Avoid division by zero for nearly upright objects
    tilt_axis = torch.where(
        tilt_axis_norm > 1e-6,
        tilt_axis / tilt_axis_norm,
        torch.zeros_like(tilt_axis)
    )
    
    # Convert tilt angle to radians
    tilt_angle_rad = tilt_angle_deg * math.pi / 180.0
    
    # Compute compensation factor based on tilt severity
    # 0-15°: 0.0-0.3, 15-45°: 0.3-0.7, >45°: 0.7-1.0
    compensation_factor = torch.where(
        tilt_angle_deg < tilt_threshold_slight,
        tilt_angle_deg / tilt_threshold_slight * 0.3,
        torch.where(
            tilt_angle_deg < tilt_threshold_severe,
            0.3 + 0.4 * (tilt_angle_deg - tilt_threshold_slight) / (tilt_threshold_severe - tilt_threshold_slight),
            torch.clamp(0.7 + (tilt_angle_deg - tilt_threshold_severe) / 45.0 * 0.3, 0.7, 1.0)
        )
    )
    
    # Compute compensation quaternion (rotate opposite to tilt)
    compensation_angle = -tilt_angle_rad * compensation_factor
    half_angle = compensation_angle / 2.0
    
    # Create quaternion from axis-angle: [cos(half), axis * sin(half)]
    sin_half = torch.sin(half_angle).unsqueeze(-1)
    cos_half = torch.cos(half_angle).unsqueeze(-1)
    
    compensation_quat = torch.zeros((batch_size, 4), device=device)
    compensation_quat[:, 0] = cos_half.squeeze(-1)
    compensation_quat[:, 1:4] = tilt_axis * sin_half
    
    # Normalize
    compensation_quat = compensation_quat / (torch.norm(compensation_quat, dim=-1, keepdim=True) + 1e-8)
    
    return compensation_quat


def make_markers(num_envs: int, show_markers: bool = False) -> Optional[Dict[str, VisualizationMarkers]]:
    """Create and return markers used for visualization."""
    if not show_markers:
        return None

    frame_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    tcp_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/tcp_marker"))

    target_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    target_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    target_marker = VisualizationMarkers(target_cfg.replace(prim_path="/Visuals/target_marker"))

    grasp_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    grasp_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    grasp_marker = VisualizationMarkers(grasp_cfg.replace(prim_path="/Visuals/grasp_target"))

    place_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    place_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    place_marker = VisualizationMarkers(place_cfg.replace(prim_path="/Visuals/place_target"))

    cube_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    cube_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    cube_markers = [
        VisualizationMarkers(cube_cfg.replace(prim_path=f"/Visuals/cube_{i+1}_marker"))
        for i in range(3)
    ]

    klt_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    klt_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
    klt_marker = VisualizationMarkers(klt_cfg.replace(prim_path="/Visuals/klt_marker"))

    return {
        "tcp": tcp_marker, "target": target_marker, "grasp": grasp_marker,
        "place": place_marker, "cubes": cube_markers, "klt": klt_marker
    }


def randomize_object_states(
    num_envs: int, 
    env_origins: torch.Tensor, 
    device: torch.device,
    pos_noise_std: float = 0.005,
    yaw_noise_std: float = 0.0,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    base_z: Optional[float] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Generate randomized root states for the object (tomato can)."""
    states: List[torch.Tensor] = []

    if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
        base_x = torch.rand(1, device=device) * (x_max - x_min) + x_min
        base_y = torch.rand(1, device=device) * (y_max - y_min) + y_min
    else:
        base_x = torch.rand(1, device=device) * 0.2 + 0.4
        base_y = torch.rand(1, device=device) * 0.3 - 0.15

    if base_z is None:
        base_z_tensor = torch.tensor([0.03], device=device)
    else:
        base_z_tensor = torch.tensor([base_z], device=device)
    base_pos_local = torch.cat([base_x, base_y, base_z_tensor])
    
    state = torch.zeros((num_envs, 13), device=device)
    pos_noise = torch.randn(num_envs, 3, device=device) * pos_noise_std
    pos_noise[:, 2] = 0.0

    x_local = base_pos_local[0] + pos_noise[:, 0]
    y_local = base_pos_local[1] + pos_noise[:, 1]
    if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
        x_local = torch.clamp(x_local, x_min, x_max)
        y_local = torch.clamp(y_local, y_min, y_max)

    state[:, 0] = x_local + env_origins[:, 0]
    state[:, 1] = y_local + env_origins[:, 1]
    state[:, 2] = base_pos_local[2] + env_origins[:, 2]
    
    if yaw_noise_std > 0:
        yaw_angles = torch.randn(num_envs, device=device) * yaw_noise_std
        half_yaw = yaw_angles / 2
        yaw_quat = torch.zeros((num_envs, 4), device=device)
        yaw_quat[:, 0] = torch.cos(half_yaw)
        yaw_quat[:, 3] = torch.sin(half_yaw)
        base_rot = torch.tensor([0.70711, 0.0, 0.0, 0.70711], device=device)
        base_rot_expanded = base_rot.unsqueeze(0).expand(num_envs, -1)
        state[:, 3:7] = quat_mul(yaw_quat, base_rot_expanded)
    else:
        base_rot = torch.tensor([0.70711, 0.0, 0.0, 0.70711], device=device)
        state[:, 3:7] = base_rot.unsqueeze(0).expand(num_envs, -1)
    
    state[:, 7:13] = 0.0
    states.append(state)
    return states, base_pos_local


def randomize_klt_state(
    num_envs: int,
    env_origins: torch.Tensor,
    device: torch.device,
    pos_range_x: float = 0.03,
    pos_range_y: float = 0.03,
    yaw_delta_deg: float = 0.0,
    x_min: float = 0.46,
    x_max: float = 0.64,
    y_min: float = 0.20,
    y_max: float = 0.34,
    base_z: float = 0.076,
    base_x: Optional[float] = None,
    base_y: Optional[float] = None,
    base_quat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate randomized root state for KLT bin with XY bounds.

    Note:
        KLT randomization (x/y/yaw) is sampled once per reset and shared by all
        parallel environments. Per-env differences should come from perturbation.
    """
    # Use provided base (typically default_root_state) as nominal pose.
    # Fallback to bounds center if not provided.
    if base_x is None:
        base_x = 0.5 * (x_min + x_max)
    if base_y is None:
        base_y = 0.5 * (y_min + y_max)
    # Clamp by requested range and feasible half-width from bounds.
    max_half_x = max(0.0, 0.5 * (x_max - x_min))
    max_half_y = max(0.0, 0.5 * (y_max - y_min))
    safe_range_x = min(max(pos_range_x, 0.0), max_half_x)
    safe_range_y = min(max(pos_range_y, 0.0), max_half_y)
    # Randomization is shared across all parallel envs.
    dx = (torch.rand(1, device=device) * 2.0 - 1.0) * safe_range_x
    dy = (torch.rand(1, device=device) * 2.0 - 1.0) * safe_range_y
    x_local_shared = torch.clamp(torch.tensor(base_x, device=device) + dx, x_min, x_max)
    y_local_shared = torch.clamp(torch.tensor(base_y, device=device) + dy, y_min, y_max)
    x_local = x_local_shared.expand(num_envs)
    y_local = y_local_shared.expand(num_envs)

    klt_state = torch.zeros((num_envs, 13), device=device)
    klt_state[:, 0] = x_local + env_origins[:, 0]
    klt_state[:, 1] = y_local + env_origins[:, 1]
    klt_state[:, 2] = base_z + env_origins[:, 2]

    if base_quat is None:
        base_quat_tensor = torch.tensor([0.70711, 0.0, 0.0, 0.70711], device=device, dtype=torch.float32)
    else:
        base_quat_tensor = base_quat.to(device=device, dtype=torch.float32)
    base_quat_tensor = base_quat_tensor / (torch.norm(base_quat_tensor) + 1.0e-8)
    base_quat_batch = base_quat_tensor.unsqueeze(0).expand(num_envs, -1)

    if abs(yaw_delta_deg) > 1.0e-6:
        half_yaw = 0.5 * math.radians(yaw_delta_deg)
        yaw_quat = torch.tensor(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(num_envs, -1)
        klt_state[:, 3:7] = quat_mul(yaw_quat, base_quat_batch)
    else:
        klt_state[:, 3:7] = base_quat_batch

    klt_state[:, 7:13] = 0.0
    return klt_state


def _find_first_prim_path_by_suffix(stage: Any, suffix: str) -> Optional[str]:
    """Find prim path by suffix, preferring env_0 for stable local-frame math."""
    first_match: Optional[str] = None
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.endswith(suffix):
            if "/env_0/" in prim_path:
                return prim_path
            if first_match is None:
                first_match = prim_path
    return first_match


def _get_object_half_size_from_bbox(
    scene: Any,
    object_suffix: str = "/tomato_soup_can",
) -> Tuple[float, float, float]:
    """Get object half size (x,y,z) from world AABB. Uses size only, not position."""
    global _OBJECT_PRIM_PATH_CACHE

    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception as e:
        raise RuntimeError(f"[RESET] Cannot import USD modules for object size: {e}") from e

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("[RESET] Object size query failed: USD stage is None.")

    if _OBJECT_PRIM_PATH_CACHE.get(object_suffix) is None:
        _OBJECT_PRIM_PATH_CACHE[object_suffix] = _find_first_prim_path_by_suffix(stage, object_suffix)
    object_prim_path = _OBJECT_PRIM_PATH_CACHE.get(object_suffix)
    if object_prim_path is None:
        raise RuntimeError(f"[RESET] Object size query failed: cannot find prim for suffix {object_suffix}.")

    object_prim = stage.GetPrimAtPath(object_prim_path)
    if not object_prim.IsValid():
        raise RuntimeError(f"[RESET] Object size query failed: invalid prim {object_prim_path}.")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )
    object_range = bbox_cache.ComputeWorldBound(object_prim).GetRange()
    object_min = object_range.GetMin()
    object_max = object_range.GetMax()
    half_x = 0.5 * (float(object_max[0]) - float(object_min[0]))
    half_y = 0.5 * (float(object_max[1]) - float(object_min[1]))
    half_z = 0.5 * (float(object_max[2]) - float(object_min[2]))
    return half_x, half_y, half_z


def _compute_dynamic_klt_xy_bounds(scene: Any) -> Optional[Tuple[float, float, float, float]]:
    """Compute table-safe KLT XY bounds from world-space AABBs (env_0 converted to local frame)."""
    global _TABLE_PRIM_PATH_CACHE, _KLT_PRIM_PATH_CACHE

    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception as e:
        logger.warning(f"[RESET] Cannot import USD modules for dynamic KLT bounds: {e}")
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        logger.warning("[RESET] Dynamic object bounds: USD stage is None.")
        return None

    # Resolve and cache prim paths (env_0)
    if _TABLE_PRIM_PATH_CACHE is None:
        _TABLE_PRIM_PATH_CACHE = _find_first_prim_path_by_suffix(stage, "/Table")
    if _KLT_PRIM_PATH_CACHE is None:
        _KLT_PRIM_PATH_CACHE = _find_first_prim_path_by_suffix(stage, "/small_KLT")
    if _TABLE_PRIM_PATH_CACHE is None or _KLT_PRIM_PATH_CACHE is None:
        return None

    table_prim = stage.GetPrimAtPath(_TABLE_PRIM_PATH_CACHE)
    klt_prim = stage.GetPrimAtPath(_KLT_PRIM_PATH_CACHE)
    if not table_prim.IsValid() or not klt_prim.IsValid():
        return None

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )
    table_range = bbox_cache.ComputeWorldBound(table_prim).GetRange()
    klt_range = bbox_cache.ComputeWorldBound(klt_prim).GetRange()

    table_min = table_range.GetMin()
    table_max = table_range.GetMax()
    klt_min = klt_range.GetMin()
    klt_max = klt_range.GetMax()

    # Convert env_0 world bounds to local bounds.
    env0_origin = scene.env_origins[0]
    env0_x = float(env0_origin[0].item())
    env0_y = float(env0_origin[1].item())

    table_x_min_local = float(table_min[0]) - env0_x
    table_x_max_local = float(table_max[0]) - env0_x
    table_y_min_local = float(table_min[1]) - env0_y
    table_y_max_local = float(table_max[1]) - env0_y

    klt_half_x = 0.5 * (float(klt_max[0]) - float(klt_min[0]))
    klt_half_y = 0.5 * (float(klt_max[1]) - float(klt_min[1]))

    # Keep a generous margin from table edge to avoid edge contacts and falls.
    edge_margin = 0.04
    x_min = table_x_min_local + klt_half_x + edge_margin
    x_max = table_x_max_local - klt_half_x - edge_margin
    y_min = table_y_min_local + klt_half_y + edge_margin
    y_max = table_y_max_local - klt_half_y - edge_margin

    # User-requested hard constraints to prevent out-of-table samples.
    x_min = max(x_min, SPAWN_X_MIN_LOCAL)
    x_max = min(x_max, SPAWN_X_MAX_LOCAL)
    y_min = max(y_min, SPAWN_Y_MIN_LOCAL)
    y_max = min(y_max, SPAWN_Y_MAX_LOCAL)

    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, x_max, y_min, y_max


def _compute_dynamic_object_spawn_config(
    scene: Any,
    object_suffix: str = "/tomato_soup_can",
    size_range_scale: float = 3.0,
    edge_margin: float = 0.02,
    center_xy_local: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float, float, float]:
    """Compute object XY spawn bounds and Z height from scene AABBs.

    Returns local-frame values for env_0:
    (x_min, x_max, y_min, y_max, base_z)
    """
    global _TABLE_PRIM_PATH_CACHE, _OBJECT_PRIM_PATH_CACHE

    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception as e:
        raise RuntimeError(f"[RESET] Cannot import USD modules for dynamic object bounds: {e}") from e

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("[RESET] Dynamic object bounds failed: USD stage is None.")

    if _TABLE_PRIM_PATH_CACHE is None:
        _TABLE_PRIM_PATH_CACHE = _find_first_prim_path_by_suffix(stage, "/Table")
    # Retry lookup when cache misses (None), instead of pinning a permanent miss.
    if _OBJECT_PRIM_PATH_CACHE.get(object_suffix) is None:
        _OBJECT_PRIM_PATH_CACHE[object_suffix] = _find_first_prim_path_by_suffix(stage, object_suffix)

    object_prim_path = _OBJECT_PRIM_PATH_CACHE.get(object_suffix)
    if _TABLE_PRIM_PATH_CACHE is None or object_prim_path is None:
        raise RuntimeError(
            "[RESET] Dynamic object bounds failed: prim path lookup failed "
            f"(table={_TABLE_PRIM_PATH_CACHE}, object={object_prim_path})."
        )

    table_prim = stage.GetPrimAtPath(_TABLE_PRIM_PATH_CACHE)
    object_prim = stage.GetPrimAtPath(object_prim_path)
    if not object_prim.IsValid():
        # Cached path may become stale after stage updates; refresh once.
        _OBJECT_PRIM_PATH_CACHE[object_suffix] = _find_first_prim_path_by_suffix(stage, object_suffix)
        object_prim_path = _OBJECT_PRIM_PATH_CACHE.get(object_suffix)
        if object_prim_path is None:
            raise RuntimeError(
                "[RESET] Dynamic object bounds failed: cached object prim path invalid "
                "and refresh lookup failed."
            )
        object_prim = stage.GetPrimAtPath(object_prim_path)
    if not table_prim.IsValid() or not object_prim.IsValid():
        raise RuntimeError(
            "[RESET] Dynamic object bounds failed: prim invalid "
            f"(table_valid={table_prim.IsValid()}, object_valid={object_prim.IsValid()})."
        )

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )
    table_range = bbox_cache.ComputeWorldBound(table_prim).GetRange()
    object_range = bbox_cache.ComputeWorldBound(object_prim).GetRange()

    table_min = table_range.GetMin()
    table_max = table_range.GetMax()
    object_min = object_range.GetMin()
    object_max = object_range.GetMax()

    env0_origin = scene.env_origins[0]
    env0_x = float(env0_origin[0].item())
    env0_y = float(env0_origin[1].item())
    env0_z = float(env0_origin[2].item())

    table_x_min_local = float(table_min[0]) - env0_x
    table_x_max_local = float(table_max[0]) - env0_x
    table_y_min_local = float(table_min[1]) - env0_y
    table_y_max_local = float(table_max[1]) - env0_y
    table_top_z_local = float(table_max[2]) - env0_z

    object_half_x = 0.5 * (float(object_max[0]) - float(object_min[0]))
    object_half_y = 0.5 * (float(object_max[1]) - float(object_min[1]))
    object_half_z = 0.5 * (float(object_max[2]) - float(object_min[2]))
    object_center_z_local = 0.5 * (float(object_min[2]) + float(object_max[2])) - env0_z

    x_min = table_x_min_local + object_half_x + edge_margin
    x_max = table_x_max_local - object_half_x - edge_margin
    y_min = table_y_min_local + object_half_y + edge_margin
    y_max = table_y_max_local - object_half_y - edge_margin

    if x_min >= x_max or y_min >= y_max:
        raise RuntimeError(
            "[RESET] Dynamic object bounds failed: invalid table-safe workspace "
            f"x=[{x_min:.3f},{x_max:.3f}], y=[{y_min:.3f},{y_max:.3f}]."
        )

    # Keep the final object bounds inside requested hard limits.
    # `size_range_scale` and `center_xy_local` are kept for backward compatibility of function signature.
    x_min = max(x_min, SPAWN_X_MIN_LOCAL)
    x_max = min(x_max, SPAWN_X_MAX_LOCAL)
    y_min = max(y_min, SPAWN_Y_MIN_LOCAL)
    y_max = min(y_max, SPAWN_Y_MAX_LOCAL)

    if x_min >= x_max or y_min >= y_max:
        raise RuntimeError(
            "[RESET] Dynamic object bounds failed after hard-constraint intersection: "
            f"table_safe_x=[{table_x_min_local:.3f},{table_x_max_local:.3f}], "
            f"table_safe_y=[{table_y_min_local:.3f},{table_y_max_local:.3f}], "
            f"hard_x=[{SPAWN_X_MIN_LOCAL:.3f},{SPAWN_X_MAX_LOCAL:.3f}], "
            f"hard_y=[{SPAWN_Y_MIN_LOCAL:.3f},{SPAWN_Y_MAX_LOCAL:.3f}]."
        )

    # Keep table-based estimate as fallback only; actual reset uses cached stable root-state Z.
    base_z = table_top_z_local + object_half_z + 0.002
    return x_min, x_max, y_min, y_max, base_z


def _print_per_env_object_bbox_info(
    scene: Any,
    cubes: Optional[List[Any]] = None,
    object_suffix: str = "/tomato_soup_can",
) -> None:
    """Print per-environment object world/local position and AABB size."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception as e:
        raise RuntimeError(f"[RESET] Cannot import USD modules for per-env object bbox print: {e}") from e

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("[RESET] Per-env object bbox print failed: USD stage is None.")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
    )

    num_envs = scene.num_envs
    for env_idx in range(num_envs):
        prim_path = f"/World/envs/env_{env_idx}{object_suffix}"
        object_prim = stage.GetPrimAtPath(prim_path)
        if not object_prim.IsValid():
            raise RuntimeError(
                f"[RESET] Per-env object bbox print failed: invalid prim path for env_{env_idx}: {prim_path}"
            )

        object_range = bbox_cache.ComputeWorldBound(object_prim).GetRange()
        obj_min = object_range.GetMin()
        obj_max = object_range.GetMax()

        center_w = torch.tensor(
            [
                0.5 * (float(obj_min[0]) + float(obj_max[0])),
                0.5 * (float(obj_min[1]) + float(obj_max[1])),
                0.5 * (float(obj_min[2]) + float(obj_max[2])),
            ],
            dtype=torch.float32,
        )
        size = torch.tensor(
            [
                float(obj_max[0]) - float(obj_min[0]),
                float(obj_max[1]) - float(obj_min[1]),
                float(obj_max[2]) - float(obj_min[2]),
            ],
            dtype=torch.float32,
        )
        env_origin = scene.env_origins[env_idx].detach().cpu().to(torch.float32)
        center_local = center_w - env_origin
        root_w = None
        root_local = None
        if cubes is not None and len(cubes) > 0:
            root_w = cubes[0].data.root_pos_w[env_idx].detach().cpu().to(torch.float32)
            root_local = root_w - env_origin

        line = (
            f"[OBJ-BBOX] env={env_idx} "
            f"origin=({env_origin[0]:.4f},{env_origin[1]:.4f},{env_origin[2]:.4f}) "
            f"aabb_center_w=({center_w[0]:.4f},{center_w[1]:.4f},{center_w[2]:.4f}) "
            f"aabb_center_local=({center_local[0]:.4f},{center_local[1]:.4f},{center_local[2]:.4f}) "
            f"size=({size[0]:.4f},{size[1]:.4f},{size[2]:.4f})"
        )
        if root_w is not None and root_local is not None:
            line += (
                f" root_w=({root_w[0]:.4f},{root_w[1]:.4f},{root_w[2]:.4f}) "
                f"root_local=({root_local[0]:.4f},{root_local[1]:.4f},{root_local[2]:.4f})"
            )
        print(line)


def compute_rotation_distance(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    """Compute angular distance between two quaternions."""
    dot = torch.sum(quat1 * quat2, dim=1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


def compute_motion_cost(
    ee_pos: torch.Tensor, ee_quat: torch.Tensor,
    target_pos: torch.Tensor, target_quat: torch.Tensor,
    pos_weight: float = 1.0, rot_weight: float = 0.5,
) -> torch.Tensor:
    """Compute motion cost from current EE pose to target pose."""
    pos_dist = torch.norm(ee_pos - target_pos, dim=1)
    rot_dist = compute_rotation_distance(ee_quat, target_quat)
    return pos_weight * pos_dist + rot_weight * rot_dist


def compute_grasp_and_place(
    cube_root_pos_w: torch.Tensor,
    cube_root_quat_w: torch.Tensor,
    env_origins: torch.Tensor,
    center_offset: torch.Tensor,
    nominal_pos_local: torch.Tensor,
    cube_initial_quat: torch.Tensor,
    device: torch.device,
    klt_pos_w: torch.Tensor = None,
    ee_pos_w: torch.Tensor = None,
    ee_quat_w: torch.Tensor = None,
    consider_symmetry: bool = False,
    enable_tilt_compensation: bool = False,
    tilt_threshold: float = 15.0,
    max_tilt_for_grasp: float = 75.0,
    grasp_xy_offset: torch.Tensor = None,
    grasp_strategy: str = "vertical",
    align_tcp_xy: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute grasp and place poses for a single object.
    
    Supports two grasping strategies for tilted objects:
    
    Strategy "vertical" (default): Gripper always maintains vertical orientation,
    ignoring object tilt. Suitable for slight tilts or when gripper cannot align.
    
    Strategy "aligned": Gripper orientation follows object tilt, allowing
    side-grasping for severely tilted objects.
    
    TCP XY alignment control (for "aligned" strategy):
    - align_tcp_xy=True: TCP X-axis aligns with object X-axis (default)
    - align_tcp_xy=False: TCP X-axis aligns with negative object Y-axis
                          (for "grasping both ends" mode)
    
    Args:
        cube_root_pos_w: Object world position, shape (N, 3)
        cube_root_quat_w: Object world quaternion [w,x,y,z], shape (N, 4)
        env_origins: Environment origin offsets, shape (N, 3)
        center_offset: Offset from object center to grasp point, shape (3,)
        nominal_pos_local: Nominal place position in local frame, shape (3,)
        cube_initial_quat: Initial object quaternion for reference
        device: Torch device
        klt_pos_w: KLT bin world position (optional)
        ee_pos_w: End-effector world position for symmetry consideration
        ee_quat_w: End-effector world quaternion for symmetry consideration
        consider_symmetry: Whether to consider symmetric grasp orientations
        enable_tilt_compensation: Enable tilt-aware grasping compensation
        tilt_threshold: Minimum tilt angle (degrees) to trigger compensation
        max_tilt_for_grasp: Maximum tilt angle to attempt grasping (degrees)
        grasp_xy_offset: XY offset for gripper centering calibration, shape (2,)
                        X: forward/backward adjustment (meters)
                        Y: left/right adjustment (meters)
        grasp_strategy: Grasp strategy for tilted objects - "vertical" or "aligned"
                       "vertical": Gripper stays vertical, ignores object tilt
                       "aligned": Gripper follows object tilt orientation
        align_tcp_xy: Whether TCP XY-axes align with object XY-axes (default: True)
                     True: TCP_X = Object_X, TCP_Y = Object_Y (standard grasp)
                     False: TCP_X = -Object_Y, TCP_Y = Object_X (rotated 90°)
                     TCP_Z always = -Object_Z (grasp from above)
        
    Returns:
        Tuple of (grasp_pos, grasp_quat, place_pos, place_quat)
    """
    cube_pos_local = cube_root_pos_w.clone() - env_origins
    cube_quat = cube_root_quat_w.clone()

    offset_comp = quat_apply_inverse(cube_quat, center_offset.unsqueeze(0).expand(cube_quat.shape[0], -1))
    grasp_pos = cube_pos_local - offset_comp
    
    # Apply grasp XY offset for gripper centering calibration
    # This adjusts the grasp position in world frame to compensate for:
    # 1. TCP definition mismatch
    # 2. Gripper finger asymmetry
    # 3. Object geometry/center of mass offset
    if grasp_xy_offset is not None and torch.any(grasp_xy_offset != 0):
        # Get world X and Y axes from gripper orientation for offset application
        # The gripper approaches from above, so XY offset is in world XY plane
        grasp_pos[:, 0] += grasp_xy_offset[0]  # X offset
        grasp_pos[:, 1] += grasp_xy_offset[1]  # Y offset

    # =====================================================================
    # Compute base gripper orientation based on align_tcp_xy parameter
    # =====================================================================
    # Base flip quaternion: rotates object frame to gripper frame
    # Default (align_tcp_xy=True):
    #   We want TCP_X aligned with Object_X, TCP_Y with Object_Y, but TCP_Z pointing down (-Object_Z)
    #   Since TCP frame is usually Z-up, we need to rotate 180 degrees around X-axis (or Y-axis) to point Z down.
    #   Let's check the standard grasp orientation:
    #   If we rotate 180 around X: Y -> -Y, Z -> -Z. X stays X.
    #   So TCP_X = Object_X, TCP_Y = -Object_Y, TCP_Z = -Object_Z.
    #   If we rotate 180 around Y: X -> -X, Z -> -Z. Y stays Y.
    #   So TCP_X = -Object_X, TCP_Y = Object_Y, TCP_Z = -Object_Z.
    
    #   Previous code used [0, 0.707, 0.707, 0], which is 90 deg around X then 90 deg around Y? Or 180 around axis (0,1,1)?
    #   [0, 0.707, 0.707, 0] corresponds to axis (0, 1, 1) normalized? No.
    #   Let's stick to standard rotations.
    #   Rx(180) = [0, 1, 0, 0]
    #   Ry(180) = [0, 0, 1, 0]
    
    #   If align_tcp_xy=True (Standard):
    #   We want gripper fingers to align with object sides?
    #   Usually for parallel gripper, if we grasp a box:
    #   Gripper X axis is often the finger closing direction (or Y axis).
    #   Let's assume Robotiq: Z is approach, X is finger movement? Or Y?
    #   Usually Y is finger movement (left/right fingers).
    
    #   Let's assume we want to align TCP frame such that Z is down.
    #   And we want consistent alignment with object frame.
    
    #   Previous logic:
    #   base_flip_quat = [0, 0.707, 0.707, 0] -> This is 180 rotation around (0,1,1)/sqrt(2).
    #   This swaps X and Y?
    #   Let's re-implement with clear rotations.
    
    # Rotation to point Z down (180 around Y): [0, 0, 1, 0] -> X becomes -X, Z becomes -Z.
    # This aligns TCP_Z with -Object_Z.
    
    # If align_tcp_xy=True:
    # We want TCP to align with Object as much as possible.
    # If we use Ry(180), we get X_tcp = -X_obj, Y_tcp = Y_obj.
    # If we want X_tcp = X_obj, we need Rx(180): X_tcp = X_obj, Y_tcp = -Y_obj.
    
    # Let's use Rx(180) as base "downward" looking pose.
    # Rx(180) quaternion: [0, 1, 0, 0]
    rx_180_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).unsqueeze(0).expand(cube_quat.shape[0], -1)
    
    if align_tcp_xy:
        # Standard alignment: TCP X aligns with Object X (roughly)
        # Using Rx(180) gives X_tcp = X_obj
        base_flip_quat = rx_180_quat
        logger.debug("[GRASP] Align TCP XY: TCP_X = Object_X, TCP_Z = -Object_Z")
    else:
        # Rotated alignment: TCP X aligns with Object Y (rotated 90 deg)
        # We want to rotate 90 deg around Z-axis relative to the standard grasp.
        # Rz(90) = [0.707, 0, 0, 0.707]
        rz_90_quat = torch.tensor([0.70710678, 0.0, 0.0, 0.70710678], device=device).unsqueeze(0).expand(cube_quat.shape[0], -1)
        # q_new = q_old * q_rot_local? Or q_rot_global * q_old?
        # We want to rotate around TCP Z axis.
        base_flip_quat = quat_mul(rx_180_quat, rz_90_quat)
        logger.debug("[GRASP] No Align TCP XY: TCP_X = Object_Y, TCP_Z = -Object_Z")
    
    # =====================================================================
    # Grasp Strategy Selection
    # =====================================================================
    # Strategy "vertical": Gripper maintains vertical orientation regardless of object tilt
    # Strategy "aligned": Gripper orientation follows object tilt
    if grasp_strategy == "vertical":
        # Strategy 1: Vertical grasp mode
        # Keep the gripper upright and do not follow object tilt.
        # Only follow the object's Z-axis rotation (Yaw) to keep grasp direction aligned.
        
        # 1. Extract object Yaw rotation (Z-axis component).
        # Rotate the object's local X-axis into world frame and project onto the XY plane.
        # q = [w, x, y, z]
        # x_axis = [1 - 2(y^2 + z^2), 2(xy + wz), 2(xz - wy)]
        w, x, y, z = cube_quat[:, 0], cube_quat[:, 1], cube_quat[:, 2], cube_quat[:, 3]
        
        # Local X vector [1, 0, 0] rotated by cube_quat
        x_axis_x = 1.0 - 2.0 * (y * y + z * z)
        x_axis_y = 2.0 * (x * y + w * z)
        
        # Compute Yaw angle
        yaw = torch.atan2(x_axis_y, x_axis_x)
        half_yaw = yaw * 0.5
        
        # Build Yaw quaternion [cos(yaw/2), 0, 0, sin(yaw/2)]
        yaw_quat = torch.zeros_like(cube_quat)
        yaw_quat[:, 0] = torch.cos(half_yaw)
        yaw_quat[:, 3] = torch.sin(half_yaw)
        
        # Apply Yaw rotation to the base grasp orientation
        grasp_quat = quat_mul(yaw_quat, base_flip_quat)
        
        logger.debug("[GRASP] Using vertical strategy - gripper upright but aligned with object Yaw")
    else:
        # Strategy 2: Pose-aligned grasp
        # Let gripper orientation follow object tilt, suitable for large tilt angles.
        grasp_quat = quat_mul(cube_quat, base_flip_quat)
        logger.debug("[GRASP] Using aligned strategy - gripper follows object tilt")

    # =====================================================================
    # Tilt-Aware Grasping Compensation
    # =====================================================================
    tilt_angle_deg = None
    is_tilted = None
    
    if enable_tilt_compensation:
        if grasp_strategy == "aligned":
            from controller.tilt_aware_grasping import TiltAwareGrasping
            tilt_grasping = TiltAwareGrasping(
                device=device,
                tilt_threshold=tilt_threshold,
                max_grasp_offset=0.05
            )
            
            # Since TiltAwareGrasping base pose differs slightly from common_utils, 
            # we modify its default base_quat to match rx_180 if needed, or simply let it do its magic.
            comp_grasp_pos, comp_grasp_quat, tilt_info = tilt_grasping.compute_compensated_grasp_pose(
                grasp_pos, cube_quat, max_tilt_angle=max_tilt_for_grasp, approach_from_side=True
            )
            
            tilt_angle_deg = tilt_info.tilt_angle
            is_tilted = tilt_info.is_tilted
            graspable_mask = tilt_angle_deg <= max_tilt_for_grasp
            apply_mask = is_tilted & graspable_mask
            
            grasp_pos = torch.where(apply_mask.unsqueeze(-1).expand_as(grasp_pos), comp_grasp_pos, grasp_pos)
            grasp_quat = torch.where(apply_mask.unsqueeze(-1).expand_as(grasp_quat), comp_grasp_quat, grasp_quat)

            if torch.any(apply_mask):
                max_tilt = torch.max(tilt_angle_deg[apply_mask]).item()
                num_tilted = torch.sum(apply_mask).item()
                logger.debug(f"[TILT] {num_tilted} objects tilted (max: {max_tilt:.1f}°), applied TiltAwareGrasping")
                
        else:
            # Original vertical strategy
            # Detect object tilt
            tilt_angle_deg, is_tilted, z_axis_world = detect_object_tilt(
                cube_quat, tilt_threshold=tilt_threshold
            )
            
            # Apply compensation for tilted objects
            if torch.any(is_tilted):
                # Check if tilt is within graspable range
                graspable_mask = tilt_angle_deg <= max_tilt_for_grasp
                
                # Compute compensation quaternion
                compensation_quat = compute_tilt_compensation(
                    cube_quat, tilt_angle_deg, z_axis_world, device,
                    tilt_threshold_slight=tilt_threshold,
                    tilt_threshold_severe=45.0,
                )
                
                apply_mask = is_tilted & graspable_mask
                
                # Modify grasp orientation: apply compensation to base grasp quaternion
                # The compensation rotates the gripper partially towards the tilted object
                grasp_quat_comp = quat_mul(compensation_quat, grasp_quat)
                # Apply only where the object is tilted and graspable
                grasp_quat = torch.where(
                    apply_mask.unsqueeze(-1).expand_as(grasp_quat), 
                    grasp_quat_comp, 
                    grasp_quat
                )
                
                # Log tilt information
                if torch.any(apply_mask):
                    max_tilt = torch.max(tilt_angle_deg[apply_mask]).item()
                    num_tilted = torch.sum(apply_mask).item()
                    logger.debug(f"[TILT] {num_tilted} objects tilted (max: {max_tilt:.1f}°) - vertical mode")

    use_alternate = torch.zeros(cube_quat.shape[0], dtype=torch.bool, device=device)
    z_180_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(cube_quat.shape[0], -1)

    if consider_symmetry and ee_pos_w is not None and ee_quat_w is not None:
        ee_pos_local = ee_pos_w - env_origins
        grasp_quat_alt = quat_mul(grasp_quat, z_180_quat)
        cost_primary = compute_motion_cost(ee_pos_local, ee_quat_w, grasp_pos, grasp_quat)
        cost_alternate = compute_motion_cost(ee_pos_local, ee_quat_w, grasp_pos, grasp_quat_alt)
        use_alternate = cost_alternate < cost_primary
        grasp_quat = torch.where(use_alternate.unsqueeze(1).expand_as(grasp_quat), grasp_quat_alt, grasp_quat)

    if klt_pos_w is not None:
        klt_pos_local = klt_pos_w.clone() - env_origins
        target_offset = torch.tensor([0.0, 0.0, 0.15], device=device).unsqueeze(0).expand(klt_pos_w.shape[0], -1)
        place_pos = klt_pos_local + target_offset
    else:
        place_pos = nominal_pos_local.unsqueeze(0).expand(cube_root_pos_w.shape[0], -1)

    base_place_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    place_quat = base_place_quat.unsqueeze(0).expand(cube_root_pos_w.shape[0], -1).clone()
    place_quat = torch.where(use_alternate.unsqueeze(1).expand_as(place_quat), quat_mul(place_quat, z_180_quat), place_quat)

    return grasp_pos, grasp_quat, place_pos, place_quat


def reset_everything(
    robot: Any,
    grasp_controller: Any,
    cubes: List[Any],
    scene: Any,
    device: torch.device,
    initial_joint_pos: torch.Tensor = None,
    klt: Any = None,
    joint_noise_std: float = 0.0,
    obj_pos_noise_std: float = 0.005,
    obj_yaw_noise_std: float = 0.0,
    obj_size_rand_scale: float = 5.0,
    obj_spawn_edge_margin: float = 0.02,
    klt_pos_range_x: float = 0.0,
    klt_pos_range_y: float = 0.0,
    klt_yaw_rand_deg: float = 30.0,
    klt_object_min_dist: float = 0.24,
    random_hold_min: int = 0,
    random_hold_max: int = 0,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """Reset robot, controller and randomize cubes.
    
    Args:
        robot: Robot entity.
        grasp_controller: Grasping controller.
        cubes: List of cube entities.
        scene: Interactive scene.
        device: Device to use for tensors.
        initial_joint_pos: Custom initial joint positions.
        klt: Optional KLT bin entity to reset.
        joint_noise_std: Joint position noise std (rad). Recommended: 0.01-0.02.
        obj_pos_noise_std: Object position noise std (m). Default: 0.005.
        obj_yaw_noise_std: Object yaw noise std (rad). Default: 0.0.
        obj_size_rand_scale: Reserved (unused in manual fixed-bounds mode).
        obj_spawn_edge_margin: Reserved (unused in manual fixed-bounds mode).
        klt_pos_range_x: Half range of KLT randomization on X axis (m). Default: 0.0.
        klt_pos_range_y: Half range of KLT randomization on Y axis (m). Default: 0.0.
        klt_yaw_rand_deg: Max absolute KLT yaw randomization (deg). One shared
            angle in [-klt_yaw_rand_deg, klt_yaw_rand_deg] is sampled per reset.
        klt_object_min_dist: Minimum XY separation between object and KLT center
            after reset (m). Default: 0.24.
        random_hold_min: Minimum random hold steps for timing diversity.
        random_hold_max: Maximum random hold steps for timing diversity.
        seed: Random seed for deterministic Z-axis rotation (default: 42).
              Ensures all parallel environments get the same rotation.
    Returns:
        initial_positions: list of tensors (num_envs, 3)
        initial_quats: list of tensors (num_envs, 4)
        nominal_pos_local: tensor (3,) - consistent base position
    """
    # Z-axis random rotation (deterministic across parallel envs)
    rng = np.random.RandomState(seed)
    # Fixed manual workspace bounds from user request.
    manual_x_min, manual_x_max = SPAWN_X_MIN_LOCAL, SPAWN_X_MAX_LOCAL
    manual_y_min, manual_y_max = SPAWN_Y_MIN_LOCAL, SPAWN_Y_MAX_LOCAL
    # Keep CLI/API compatibility; these are intentionally unused in manual-bound mode.
    _ = obj_size_rand_scale, obj_spawn_edge_margin
    
    num_envs = scene.num_envs

    if initial_joint_pos is not None:
        joint_pos = initial_joint_pos.clone()
    else:
        joint_pos = robot.data.default_joint_pos.clone()
    
    if joint_noise_std > 0:
        num_arm_joints = min(6, joint_pos.shape[1])
        joint_noise = torch.randn((num_envs, num_arm_joints), device=device) * joint_noise_std
        joint_pos[:, :num_arm_joints] = joint_pos[:, :num_arm_joints] + joint_noise
        logger.info(f"[RESET] Applied joint noise (std={joint_noise_std})")
    
    joint_vel = robot.data.default_joint_vel.clone()
    
    # Reset robot first, then write initial state
    robot.reset()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # Also reset position targets so stabilization doesn't chase stale commands
    robot.set_joint_position_target(joint_pos)
    
    logger.info(f"[RESET] Robot reset for {num_envs} envs")

    if random_hold_min > 0 and random_hold_max >= random_hold_min:
        grasp_controller.reset(random_hold_min=random_hold_min, random_hold_max=random_hold_max)
    else:
        grasp_controller.reset()

    if klt is not None:
        global _KLT_BASE_Z_LOCAL_CACHE, _KLT_BASE_XY_LOCAL_CACHE, _KLT_BASE_QUAT_LOCAL_CACHE
        env_origins = scene.env_origins

        if _KLT_BASE_Z_LOCAL_CACHE is None:
            _KLT_BASE_Z_LOCAL_CACHE = float(klt.data.default_root_state[0, 2].item())
            logger.info(f"[RESET] Cached KLT base_z_local from default root state: {_KLT_BASE_Z_LOCAL_CACHE:.4f}")
        if _KLT_BASE_XY_LOCAL_CACHE is None:
            _KLT_BASE_XY_LOCAL_CACHE = (
                float(klt.data.default_root_state[0, 0].item()),
                float(klt.data.default_root_state[0, 1].item()),
            )
            logger.info(
                "[RESET] Cached KLT base_xy_local from default root state: "
                f"x={_KLT_BASE_XY_LOCAL_CACHE[0]:.4f}, y={_KLT_BASE_XY_LOCAL_CACHE[1]:.4f}"
            )
        if _KLT_BASE_QUAT_LOCAL_CACHE is None:
            _KLT_BASE_QUAT_LOCAL_CACHE = (
                float(klt.data.default_root_state[0, 3].item()),
                float(klt.data.default_root_state[0, 4].item()),
                float(klt.data.default_root_state[0, 5].item()),
                float(klt.data.default_root_state[0, 6].item()),
            )
            logger.info(
                "[RESET] Cached KLT base_quat_local from default root state: "
                f"{_KLT_BASE_QUAT_LOCAL_CACHE}"
            )

        x_min, x_max = manual_x_min, manual_x_max
        y_min, y_max = manual_y_min, manual_y_max

        # Center sampling in the constrained interval to avoid one-sided clamping.
        klt_center_x = 0.5 * (x_min + x_max)
        klt_center_y = 0.5 * (y_min + y_max)

        # Use configured range as cap, but allow full-table randomization when range <= 0.
        klt_half_span_x = max(0.0, 0.5 * (x_max - x_min))
        klt_half_span_y = max(0.0, 0.5 * (y_max - y_min))
        effective_klt_range_x = klt_half_span_x if klt_pos_range_x <= 0 else min(klt_pos_range_x, klt_half_span_x)
        effective_klt_range_y = klt_half_span_y if klt_pos_range_y <= 0 else min(klt_pos_range_y, klt_half_span_y)
        if klt_yaw_rand_deg > 0.0:
            shared_klt_yaw_delta_deg = float(np.random.uniform(-klt_yaw_rand_deg, klt_yaw_rand_deg))
        else:
            shared_klt_yaw_delta_deg = 0.0

        klt_base_quat = torch.tensor(_KLT_BASE_QUAT_LOCAL_CACHE, device=device, dtype=torch.float32)

        klt_state = randomize_klt_state(
            num_envs,
            env_origins,
            device,
            pos_range_x=effective_klt_range_x,
            pos_range_y=effective_klt_range_y,
            yaw_delta_deg=shared_klt_yaw_delta_deg,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            base_z=float(_KLT_BASE_Z_LOCAL_CACHE),
            base_x=float(klt_center_x),
            base_y=float(klt_center_y),
            base_quat=klt_base_quat,
        )
        klt_local_xy = klt_state[:, 0:2] - env_origins[:, 0:2]
        print(
            "[INFO] KLT manual randomized local XY range: "
            f"x=[{klt_local_xy[:, 0].min().item():.3f}, {klt_local_xy[:, 0].max().item():.3f}], "
            f"y=[{klt_local_xy[:, 1].min().item():.3f}, {klt_local_xy[:, 1].max().item():.3f}], "
            f"shared_yaw_delta_deg={shared_klt_yaw_delta_deg:.2f}"
        )
        klt.write_root_state_to_sim(klt_state)
        klt.reset()

    if "/tomato_soup_can" not in _OBJECT_BASE_XY_LOCAL_CACHE:
        object_ref = cubes[0]
        object_xy_local = (
            float(object_ref.data.default_root_state[0, 0].item()),
            float(object_ref.data.default_root_state[0, 1].item()),
        )
        _OBJECT_BASE_XY_LOCAL_CACHE["/tomato_soup_can"] = object_xy_local
        logger.info(
            "[RESET] Cached object base_xy_local from default root state: "
            f"x={object_xy_local[0]:.4f}, y={object_xy_local[1]:.4f}"
        )
    # Use fixed manual bounds for object spawn; disable dynamic table/bbox bounds.
    obj_x_min, obj_x_max = manual_x_min, manual_x_max
    obj_y_min, obj_y_max = manual_y_min, manual_y_max

    # Use stable object root-state Z from scene default to prevent penetration/flying.
    if "/tomato_soup_can" not in _OBJECT_BASE_Z_LOCAL_CACHE:
        object_ref = cubes[0]
        object_z_local = float(object_ref.data.default_root_state[0, 2].item())
        _OBJECT_BASE_Z_LOCAL_CACHE["/tomato_soup_can"] = object_z_local
        logger.info(f"[RESET] Cached object base_z_local from default root state: {object_z_local:.4f}")
    obj_base_z = _OBJECT_BASE_Z_LOCAL_CACHE["/tomato_soup_can"]

    print(
        "[INFO] Manual object spawn bounds: "
        f"x=[{obj_x_min:.3f}, {obj_x_max:.3f}], "
        f"y=[{obj_y_min:.3f}, {obj_y_max:.3f}], z={obj_base_z:.3f}"
    )
    logger.info(
        "[RESET] Manual object spawn bounds: "
        f"x=[{obj_x_min:.3f}, {obj_x_max:.3f}], "
        f"y=[{obj_y_min:.3f}, {obj_y_max:.3f}], z={obj_base_z:.3f}"
    )

    root_states, nominal_pos_local = randomize_object_states(
        num_envs, scene.env_origins, device,
        pos_noise_std=obj_pos_noise_std,
        yaw_noise_std=obj_yaw_noise_std,
        x_min=obj_x_min,
        x_max=obj_x_max,
        y_min=obj_y_min,
        y_max=obj_y_max,
        base_z=obj_base_z,
    )

    # Keep a minimum XY separation between object and container to avoid initial collision impulses.
    if klt is not None:
        obj_xy = root_states[0][:, 0:2]
        bin_xy = klt_state[:, 0:2]
        min_dist = max(0.0, float(klt_object_min_dist))
        if min_dist > 0.0:
            delta = obj_xy - bin_xy
            dist = torch.norm(delta, dim=1)
            need_fix = dist < min_dist
            if torch.any(need_fix):
                eps = 1.0e-6
                default_dir = torch.tensor([0.0, -1.0], device=device, dtype=obj_xy.dtype)
                safe_dir = delta / torch.clamp(dist.unsqueeze(1), min=eps)
                safe_dir = torch.where(
                    (dist < eps).unsqueeze(1),
                    default_dir.unsqueeze(0).expand_as(safe_dir),
                    safe_dir,
                )
                shift = (min_dist - dist[need_fix] + 0.01).unsqueeze(1)
                obj_xy[need_fix] = obj_xy[need_fix] + safe_dir[need_fix] * shift

                # Clamp in local frame (per-env) to keep samples inside workspace.
                obj_xy_local = obj_xy - scene.env_origins[:, 0:2]
                obj_xy_local[:, 0] = torch.clamp(obj_xy_local[:, 0], obj_x_min, obj_x_max)
                obj_xy_local[:, 1] = torch.clamp(obj_xy_local[:, 1], obj_y_min, obj_y_max)

                # If clamping still leaves close pairs, place object at the farthest workspace corner.
                obj_xy_world = obj_xy_local + scene.env_origins[:, 0:2]
                dist_after = torch.norm(obj_xy_world - bin_xy, dim=1)
                still_close = dist_after < min_dist
                if torch.any(still_close):
                    corners_local = torch.tensor(
                        [
                            [obj_x_min, obj_y_min],
                            [obj_x_min, obj_y_max],
                            [obj_x_max, obj_y_min],
                            [obj_x_max, obj_y_max],
                        ],
                        device=device,
                        dtype=obj_xy.dtype,
                    )
                    corners_world = corners_local.unsqueeze(0) + scene.env_origins[:, 0:2].unsqueeze(1)
                    corner_dist = torch.norm(corners_world - bin_xy.unsqueeze(1), dim=2)
                    farthest_corner_idx = torch.argmax(corner_dist, dim=1)
                    farthest_corner_local = corners_local[farthest_corner_idx]
                    obj_xy_local[still_close] = farthest_corner_local[still_close]

                root_states[0][:, 0:2] = obj_xy_local + scene.env_origins[:, 0:2]

    # Z-axis random rotation (deterministic across parallel envs)
    # Generate random angle in degrees [0, 360]
    z_angle_deg = rng.uniform(0.0, 360.0)
    z_angle_rad = np.deg2rad(z_angle_deg)
    
    # Create quaternion for rotation around Z-axis: [cos(theta/2), 0, 0, sin(theta/2)]
    # Note: Isaac Lab uses [w, x, y, z] format
    half_angle = z_angle_rad / 2.0
    z_rot_quat = torch.tensor([math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)], device=device)
    z_rot_quat = z_rot_quat.unsqueeze(0).expand(num_envs, -1)
    
    # Apply rotation to all objects
    for i in range(len(root_states)):
        # Apply global Z rotation: q_new = q_rot * q_old
        root_states[i][:, 3:7] = quat_mul(z_rot_quat, root_states[i][:, 3:7])

    initial_positions: List[torch.Tensor] = []
    initial_quats: List[torch.Tensor] = []
    for cube, state in zip(cubes, root_states):
        cube.write_root_state_to_sim(state)
        initial_positions.append(state[:, 0:3].clone())
        initial_quats.append(state[:, 3:7].clone())

    # Print per-env object position/size diagnostics in world and local coordinates.
    _print_per_env_object_bbox_info(scene, cubes=cubes, object_suffix="/tomato_soup_can")

    return initial_positions, initial_quats, nominal_pos_local
