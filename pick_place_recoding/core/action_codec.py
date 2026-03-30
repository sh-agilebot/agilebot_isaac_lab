#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for decoding EE6D xyz semantics."""

from __future__ import annotations

import numpy as np


def decode_action_xyz_mode(
    action: np.ndarray,
    anchor_xyz_local: np.ndarray,
    xyz_mode: str,
    action_dim: int = 10,
) -> np.ndarray:
    """Decode model action into absolute-local EE6D according to xyz mode.

    Args:
        action: Raw action vector from model.
        anchor_xyz_local: Request-time xyz anchor in local frame.
        xyz_mode: Either "relative" or "absolute".
        action_dim: Output action dimension to keep.

    Returns:
        Decoded action vector in absolute-local semantics.
    """
    decoded = np.asarray(action, dtype=np.float32)[:action_dim].copy()
    if str(xyz_mode).strip().lower() == "relative":
        decoded[:3] = np.asarray(anchor_xyz_local, dtype=np.float32).reshape(3) + decoded[:3]
    return decoded
