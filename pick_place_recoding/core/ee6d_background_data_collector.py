#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
EE6D-compatible data collector for X-VLA training with HDF5 I/O.

This module extends the DataCollector to record EE6D 10D format
compatible with X-VLA fine-tuning specifications.
"""

import argparse
import json
import logging
import os
import sys
import time
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.background_data_collector import DataCollector
from core.ee6d_utils import create_ee6d_metadata


class EE6DDataCollector(DataCollector):
    """EE6D-compatible data collector for X-VLA training with HDF5 I/O.

    Extends the DataCollector to record data in EE6D 10D format
    with proper metadata for X-VLA fine-tuning.
    """

    def __init__(self, dataset_file: str, fps: int = 30, dof=None, sampling_frequency: int = 1,
                 sim_frequency: int = 120, resume: bool = False, action_frame: str = "base",
                 buffer_size: int = 3, max_queue_size: int = 10, language_instruction: str = "pick and place object"):
        super().__init__(dataset_file, fps, dof, sampling_frequency, sim_frequency, resume,
                        buffer_size, max_queue_size)
        self.action_frame = action_frame
        self.language_instruction = language_instruction
        self.ee6d_metadata = create_ee6d_metadata(
            domain_id=19,
            action_frame=action_frame,
            translation_unit="meter",
            gripper_semantics="1=close,0=open"
        )

    def start_recording(self, num_envs=1):
        """Start recording with EE6D metadata."""
        super().start_recording(num_envs)

        logger.info(f"[EE6D] Recording with action frame: {self.action_frame}")
        logger.info("[EE6D] Dataset format: EE6D_AbsoluteLocal_10D (action_t = proprio_{t+1})")

    def record_step(self, obs, actions, rewards, dones, infos):
        """Record a single step with EE6D-specific handling.

        Records the current observation (state) and the action that will be executed 
        from this state to transition to the next state.
        """
        if not self.is_recording:
            return

        if len(self.current_episode_data) == 0:
            logger.info("[EE6D] Recording first step, observation structure:")
            self._print_obs_structure(obs, prefix="  ")

        step_data = {
            "observations": self._convert_tensor_dict(obs),
            "actions": self._convert_tensor(actions),
            "rewards": self._convert_tensor(rewards),
            "dones": self._convert_tensor(dones),
            "timestamp": time.time(),
        }

        self.current_episode_data.append(step_data)
        self.batch_buffer.append(step_data)
        self.previous_actions = self._convert_tensor(actions)

        current_time = time.time()
        time_since_flush = current_time - self.last_flush_time

        if len(self.batch_buffer) >= self.steps_per_batch or time_since_flush >= self.buffer_size:
            self._flush_buffer()
            self.last_flush_time = current_time

    def _prepare_batch_data(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare batch data for sending to background process with EE6D format.

        NOTE: Environment filtering is done at STOP_RECORDING time, not here.
        This method just batches the data as-is.

        Args:
            batch_data: List of step data dictionaries.

        Returns:
            Dictionary with batched data in steps/ format.
        """
        batched = {
            "main_cam": [],
            "wrist_cam": [],
            "joint_state": [],
            "proprio": [],
            "action": [],
            "action_joint": [],
            "dones": [],
            "timestamps": []
        }

        for step_data in batch_data:
            obs_data = step_data["observations"]

            if "observation.images.main_cam" in obs_data:
                batched["main_cam"].append(obs_data["observation.images.main_cam"])

            if "observation.images.wrist_cam" in obs_data:
                batched["wrist_cam"].append(obs_data["observation.images.wrist_cam"])

            if "observation.proprio" in obs_data:
                batched["proprio"].append(obs_data["observation.proprio"])

            if "observation.joint_state" in obs_data:
                batched["joint_state"].append(obs_data["observation.joint_state"])

            batched["action"].append(step_data["actions"])
            batched["dones"].append(step_data["dones"])
            batched["timestamps"].append(step_data["timestamp"])

            if "action_joint" in obs_data:
                batched["action_joint"].append(obs_data["action_joint"])

        for key in batched:
            try:
                batched[key] = np.array(batched[key])
            except (ValueError, TypeError):
                pass

        return batched

    def _send_stop_recording(self, success: bool, camera_intrinsics: dict,
                            flange_to_tcp_offset: np.ndarray, successful_envs: int,
                            total_envs: int, env_success_flags: List[bool]) -> bool:
        """Send stop recording message to background process with EE6D metadata."""
        # Convert env_success_flags to list if it's a numpy array
        if isinstance(env_success_flags, np.ndarray):
            env_success_flags = env_success_flags.tolist()

        metadata = {
            "episode_length": len(self.current_episode_data),
            "fps": self.fps,
            "sampling_frequency": self.sampling_frequency,
            "actual_sampling_interval": self.actual_sampling_interval,
            "effective_fps": self.fps / self.sampling_frequency if self.sampling_frequency > 0 else self.fps,
            "actual_effective_fps": self.sim_frequency / self.actual_sampling_interval,
            "timestamp": datetime.now().isoformat(),
            "success": successful_envs > 0 and success,
            "successful_envs": successful_envs,
            "total_envs": total_envs,
            "env_success_flags": env_success_flags,  # Add success flags for filtering
            "episode_id": f"episode_{self.episode_count:06d}",
            "domain_id": 19,  # AgileBot domain_id (matches DATA_DOMAIN_ID in X-VLA/datasets/domain_config.py)
            "language_instruction": self.language_instruction,
            "action_space": "EE6D_AbsoluteLocal_10D",
            "action_semantics": "absolute_local_next_state",
            "action_dim": 10,
            "action_frame": self.action_frame,
            "action_joint_dim": 7,
            "proprio_schema": json.dumps({
                "type": "object",
                "properties": {
                    "ee_pos": {"type": "array", "items": "float", "length": 3},
                    "ee_rot6d": {"type": "array", "items": "float", "length": 6},
                    "gripper_state": {"type": "number", "range": [0.0, 1.0]}
                }
            }),
            "action_schema": json.dumps({
                "type": "array",
                "items": "float",
                "length": 10,
                "description": "EE6D 10D absolute-local next state (action_t = proprio_{t+1})"
            }),
            "action_joint_schema": json.dumps({
                "type": "array",
                "items": "float",
                "length": 7,
                "description": "Joint 7D (joint1~joint6 + finger_joint)"
            }),
            "joint_schema": json.dumps({
                "type": "array",
                "items": "float",
                "length": 7,
                "description": "joint1, joint2, joint3, joint4, joint5, joint6, finger_joint"
            })
        }

        if camera_intrinsics is not None:
            if "main_camera" in camera_intrinsics:
                metadata["main_camera_intrinsic"] = camera_intrinsics["main_camera"]
            if "wrist_camera" in camera_intrinsics:
                metadata["wrist_camera_intrinsic"] = camera_intrinsics["wrist_camera"]

        if flange_to_tcp_offset is not None:
            metadata["flange_to_tcp_offset"] = flange_to_tcp_offset

        return self._put_control_message(
            {"type": "STOP_RECORDING", "metadata": metadata},
            "STOP_RECORDING",
            retries=5,
        )
