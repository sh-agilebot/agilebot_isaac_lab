#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Data collection module for recording demonstrations with HDF5 I/O.

This module implements DataCollector using multiprocessing for HDF5 I/O operations,
improving performance during high-frequency data recording.

Records state-action pairs where:
- State: Observation of the environment at time t
- Action: Control command executed at time t to transition from state t to state t+1
"""

import argparse
import json
import logging
import os
import sys
import time
import h5py
import multiprocessing as mp
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import queue

import torch

logger = logging.getLogger(__name__)

from core.background_hdf5_writer import run_background_process


class DataCollector:
    """Data collector for recording demonstrations with HDF5 I/O.

    Records state-action pairs where:
    - State: Observation of the environment at time t
    - Action: Control command executed at time t to transition from state t to state t+1

    This version uses multiprocessing to offload HDF5 I/O operations to a separate process,
    improving performance during high-frequency data recording.

    Args:
        dataset_file: Path to the HDF5 file where data will be stored.
        fps: Frames per second for recording.
        dof: Degrees of freedom to retain in joint data. If None, retains all DoF.
        sampling_frequency: Legacy parameter. The actual sampling interval is calculated as sim_frequency / fps.
        sim_frequency: Simulation frequency (e.g., 120 Hz).
        resume: Whether to resume from existing file.
        buffer_size: Number of seconds of data to buffer before sending to writer process (default: 3).
        max_queue_size: Maximum size of the queue for overflow prevention (default: 10).
    """

    @staticmethod
    def validate_camera_intrinsics(camera_intrinsics: dict) -> bool:
        """Validate camera intrinsic matrix format.

        Args:
            camera_intrinsics: Dictionary containing camera intrinsic matrices.

        Returns:
            bool: True if valid, False otherwise.

        Raises:
            ValueError: If format is invalid.
        """
        if camera_intrinsics is None:
            return True

        required_cameras = ["main_camera"]
        for camera_name in required_cameras:
            if camera_name not in camera_intrinsics:
                raise ValueError(f"Missing required camera intrinsic: {camera_name}")

            intrinsic = camera_intrinsics[camera_name]
            if isinstance(intrinsic, list):
                intrinsic = np.array(intrinsic)

            if not isinstance(intrinsic, np.ndarray):
                raise ValueError(f"Camera intrinsic for {camera_name} must be numpy array or list")

            if intrinsic.shape != (3, 3):
                raise ValueError(f"Camera intrinsic for {camera_name} must have shape (3, 3), got {intrinsic.shape}")

        return True

    def __init__(self, dataset_file: str, fps: int = 30, dof=None, sampling_frequency: int = 1, 
                 sim_frequency: int = 120, resume: bool = False, buffer_size: int = 3, 
                 max_queue_size: int = 10):
        """
        Initialize data collector with HDF5 I/O.

        Args:
            dataset_file: Path to save the HDF5 file
            fps: Desired frames per second for recording
            dof: Degrees of freedom for the robot. Must be between 1 and 20 if specified.
            sampling_frequency: Sampling frequency for data recording
            sim_frequency: Simulation frequency (e.g., 120 Hz)
            resume: Whether to resume from existing file
            buffer_size: Number of seconds of data to buffer before sending to writer process
            max_queue_size: Maximum size of the queue for overflow prevention
        """
        if dof is not None and (dof <= 0 or dof > 20):
            raise ValueError(f"dof must be between 1 and 20, got {dof}")

        self.dataset_file = dataset_file
        self.fps = fps
        self.dof = dof
        self.sampling_frequency = sampling_frequency
        self.sim_frequency = sim_frequency
        self.resume = resume
        self.buffer_size = buffer_size
        self.max_queue_size = max_queue_size

        target_interval = self.sim_frequency / fps
        actual_interval = round(target_interval)
        self.actual_sampling_interval = max(1, actual_interval)
        actual_fps = self.sim_frequency / self.actual_sampling_interval
        if abs(target_interval - actual_interval) > 0.01:
            logger.warning(f"请求的FPS {fps}, 实际FPS将为 {actual_fps:.2f} (间隔={self.actual_sampling_interval} 步)")
        else:
            logger.info(f"Data collection: Simulation frequency {self.sim_frequency}Hz, desired FPS {fps}, recording every {self.actual_sampling_interval} steps")

        output_dir = os.path.dirname(dataset_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        if os.path.exists(self.dataset_file):
            if self.resume:
                try:
                    with h5py.File(self.dataset_file, "r") as f:
                        self.episode_count = f.attrs.get("total_episodes", 0)
                        episodes = [key for key in f.keys() if key.startswith('episode_')]
                        if episodes:
                            max_idx = max(int(ep.split('_')[1]) for ep in episodes)
                            self.episode_count = max(self.episode_count, max_idx + 1)
                    logger.info(f"Resuming from existing dataset: {len(episodes)} episodes found, next episode will be {self.episode_count}")
                except Exception as e:
                    logger.warning(f"Failed to read existing dataset: {e}")
                    logger.info("Starting fresh with episode count 0")
                    self.episode_count = 0
            else:
                raise FileExistsError(
                    f"Dataset file already exists: {self.dataset_file}\n"
                    f"Please either:\n"
                    f"  1. Delete the existing file manually, or\n"
                    f"  2. Use a different output path via --dataset_file, or\n"
                    f"  3. Use the --resume flag to continue adding data to the existing file"
                )
        else:
            self.episode_count = 0

        self.is_recording = False
        self.current_episode_data = []
        self.env_success_flags = []
        self.previous_actions = None

        self.data_queue = mp.Queue(maxsize=max_queue_size)
        self.status_queue = mp.Queue()
        self.background_process = None
        self.process_started = False

        self.batch_buffer = []
        self.last_flush_time = 0
        self.steps_per_batch = int(fps * buffer_size)
        self.batch_start_index = 0

    def _start_background_process(self):
        """Start the HDF5 writer process."""
        if self.process_started:
            return

        try:
            self.background_process = mp.Process(
                target=run_background_process,
                args=(self.data_queue, self.status_queue, self.dataset_file, self.resume)
            )
            self.background_process.start()
            self.process_started = True
            logger.info("HDF5 writer process started")

            time.sleep(0.5)

            while True:
                try:
                    status = self.status_queue.get(timeout=1.0)
                    if status.get("type") == "STARTED":
                        logger.info("Writer process initialized successfully")
                        break
                    elif status.get("type") == "ERROR":
                        logger.error(f"Writer process error: {status.get('data')}")
                        raise RuntimeError(f"Writer process failed to start: {status.get('data')}")
                except queue.Empty:
                    logger.warning("Timeout waiting for writer process initialization")
                    break

        except Exception as e:
            logger.error(f"Failed to start background process: {e}")
            raise

    def _put_control_message(
        self,
        message: Dict[str, Any],
        message_name: str,
        retries: int = 3,
        timeout: float = 1.0,
    ) -> bool:
        """Send control message to writer queue with retry.

        Control messages (START/STOP/CANCEL/CLOSE) define episode boundaries.
        Retrying them reduces the chance of leaving an unfinished episode when
        the queue is temporarily full.
        """
        for attempt in range(1, retries + 1):
            try:
                self.data_queue.put(message, timeout=timeout)
                return True
            except queue.Full:
                logger.warning(
                    "%s message failed (attempt %d/%d): queue full",
                    message_name,
                    attempt,
                    retries,
                )
                if attempt < retries:
                    time.sleep(0.2 * attempt)

        logger.error("Failed to send %s message after %d attempts", message_name, retries)
        return False

    def start_recording(self, num_envs=1):
        """Start recording a new episode."""
        self._start_background_process()

        self.current_episode_data = []
        self.env_success_flags = [False] * num_envs
        self.is_recording = True
        self.previous_actions = None
        self.batch_buffer = []
        self.last_flush_time = time.time()
        self.batch_start_index = 0

        if not self._put_control_message(
            {"type": "START_RECORDING", "num_envs": num_envs},
            "START_RECORDING",
            retries=5,
        ):
            raise RuntimeError("Failed to send START_RECORDING message")

        logger.info(f"Starting to record demo #{self.episode_count + 1} with {num_envs} environments...")

    def stop_recording(self, success: bool = True, camera_intrinsics: dict = None, 
                      flange_to_tcp_offset: np.ndarray = None, env_success_flags=None):
        """Stop recording and save episode if successful.

        Args:
            success: Whether the episode was successful.
            camera_intrinsics: Dictionary containing camera intrinsic matrices.
            flange_to_tcp_offset: Offset from flange to TCP (controller.ee_offset).
            env_success_flags: List of success flags for each environment. If None, uses self.env_success_flags.
        """
        if not self.is_recording:
            return

        if camera_intrinsics is not None:
            try:
                self.validate_camera_intrinsics(camera_intrinsics)
            except ValueError as e:
                logger.warning(f"Invalid camera intrinsics: {e}")
                logger.info("Continuing with invalid intrinsics...")

        self.is_recording = False

        if env_success_flags is not None:
            success_flags_to_use = env_success_flags
        else:
            success_flags_to_use = self.env_success_flags

        successful_envs = sum(success_flags_to_use)
        total_envs = len(success_flags_to_use)

        if successful_envs > 0 and len(self.current_episode_data) > 0:
            self._flush_buffer()
            stop_sent = self._send_stop_recording(
                success,
                camera_intrinsics,
                flange_to_tcp_offset,
                successful_envs,
                total_envs,
                success_flags_to_use,
            )
            if stop_sent:
                self.episode_count += 1
                logger.info(
                    f"Successfully saved demo #{self.episode_count} with {len(self.current_episode_data)} steps "
                    f"({successful_envs}/{total_envs} environments successful)"
                )
            else:
                logger.error(
                    "Failed to send STOP_RECORDING; attempting CANCEL_RECORDING to avoid dirty data"
                )
                self._put_control_message(
                    {"type": "CANCEL_RECORDING"},
                    "CANCEL_RECORDING",
                    retries=5,
                )
        else:
            reason = "no successful environments" if successful_envs == 0 else "empty"
            logger.warning(f"Recording discarded because {reason}, data not saved")
            self._put_control_message(
                {"type": "CANCEL_RECORDING"},
                "CANCEL_RECORDING",
                retries=5,
            )

        self.current_episode_data = []
        self.env_success_flags = []

    def _send_stop_recording(self, success: bool, camera_intrinsics: dict, 
                            flange_to_tcp_offset: np.ndarray, successful_envs: int, 
                            total_envs: int, env_success_flags: List[bool]) -> bool:
        """Send stop recording message to background process."""
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
            "total_envs": total_envs
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

    def record_step(self, obs, actions, rewards, dones, infos):
        """Record a single step.

        Records the current observation (state) and the action that will be executed 
        from this state to transition to the next state.
        """
        if not self.is_recording:
            return

        if len(self.current_episode_data) == 0:
            logger.info("Recording first step, observation structure:")
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

    def _flush_buffer(self):
        """Flush buffered data to background process."""
        if not self.batch_buffer:
            return

        if self.data_queue.full():
            logger.warning("Data queue is full, applying backpressure")
            self._apply_backpressure()

        try:
            batch_data = self._prepare_batch_data(self.batch_buffer)
            self.data_queue.put({
                "type": "RECORD_BATCH",
                "data": batch_data,
                "batch_index": self.batch_start_index
            }, timeout=1.0)
            self.batch_start_index += len(self.batch_buffer)
            self.batch_buffer = []
        except queue.Full:
            logger.error("Failed to send batch data: queue full")
            self._apply_backpressure()

    def _prepare_batch_data(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare batch data for sending to background process.

        Args:
            batch_data: List of step data dictionaries.

        Returns:
            Dictionary with batched data.
        """
        batched = {
            "observations": {},
            "actions": [],
            "rewards": [],
            "dones": [],
            "timestamps": []
        }

        for step_data in batch_data:
            obs_data = step_data["observations"]
            for key, value in obs_data.items():
                if key not in batched["observations"]:
                    batched["observations"][key] = []
                batched["observations"][key].append(value)

            batched["actions"].append(step_data["actions"])
            batched["rewards"].append(step_data["rewards"])
            batched["dones"].append(step_data["dones"])
            batched["timestamps"].append(step_data["timestamp"])

        for key in batched["observations"]:
            try:
                batched["observations"][key] = np.array(batched["observations"][key])
            except (ValueError, TypeError):
                pass

        try:
            batched["actions"] = np.array(batched["actions"])
            batched["rewards"] = np.array(batched["rewards"])
            batched["dones"] = np.array(batched["dones"])
            batched["timestamps"] = np.array(batched["timestamps"])
        except (ValueError, TypeError):
            pass

        return batched

    def _apply_backpressure(self):
        """Apply backpressure by dropping least critical data."""
        if len(self.batch_buffer) > 1:
            dropped = self.batch_buffer.pop(0)
            logger.warning(f"Applied backpressure: dropped oldest step from buffer")
        else:
            logger.warning("Cannot apply backpressure: buffer too small")

    def _print_obs_structure(self, obs, prefix="", max_depth=3, current_depth=0):
        """Print observation data structure for debugging."""
        if current_depth >= max_depth:
            return

        if isinstance(obs, dict):
            for key, value in obs.items():
                if isinstance(value, dict):
                    logger.info(f"{prefix}{key}: dict ({len(value)} items)")
                    self._print_obs_structure(
                        value, prefix + "  ", max_depth, current_depth + 1
                    )
                elif torch.is_tensor(value):
                    logger.info(f"{prefix}{key}: tensor {tuple(value.shape)} {value.dtype}")
                elif isinstance(value, np.ndarray):
                    logger.info(f"{prefix}{key}: array {value.shape} {value.dtype}")
                else:
                    logger.info(f"{prefix}{key}: {type(value)}")
        else:
            logger.info(f"{prefix}Non-dict type: {type(obs)}")

    def _convert_tensor_dict(self, tensor_dict):
        """Convert tensor dictionary to numpy, handling nested dictionaries."""
        if isinstance(tensor_dict, dict):
            result = {}
            for k, v in tensor_dict.items():
                if isinstance(v, dict):
                    result[k] = self._convert_tensor_dict(v)
                else:
                    result[k] = self._convert_tensor(v, k)
            return result
        else:
            return self._convert_tensor(tensor_dict, key)

    def _convert_tensor(self, tensor, key=None):
        """Convert tensor to numpy array with proper handling of different data types."""
        if torch.is_tensor(tensor):
            data = tensor.detach().cpu().numpy()
            
            if self.dof is not None and key in ['joint_pos', 'joint_vel']:
                if len(data.shape) == 2 and data.shape[1] >= self.dof:
                    data = data[:, :self.dof]
            
            if data.dtype == np.float64:
                return data.astype(np.float32)
            return data
        elif isinstance(tensor, np.ndarray):
            if tensor.dtype == np.float64:
                return tensor.astype(np.float32)
            return tensor
        elif isinstance(tensor, (list, tuple)):
            try:
                data = np.array(tensor)
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
                return data
            except (ValueError, TypeError):
                return tensor
        elif isinstance(tensor, (int, float, bool)):
            return np.array(tensor, dtype=np.float32)
        else:
            try:
                data = np.array(tensor)
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
                return data
            except (ValueError, TypeError):
                return tensor

    def close(self):
        """Close the data collector."""
        has_inflight_episode = bool(self.is_recording or self.current_episode_data or self.batch_buffer)
        self.is_recording = False

        if has_inflight_episode:
            # Discard unfinished local buffers and cancel writer-side active episode.
            self.batch_buffer = []
            self.current_episode_data = []
            self.env_success_flags = []

        if self.process_started:
            if has_inflight_episode:
                self._put_control_message(
                    {"type": "CANCEL_RECORDING"},
                    "CANCEL_RECORDING",
                    retries=5,
                )

            self._put_control_message({"type": "CLOSE"}, "CLOSE", retries=5)

            if self.background_process is not None:
                self.background_process.join(timeout=30.0)
                if self.background_process.is_alive():
                    logger.warning("Background process did not terminate gracefully, forcing termination")
                    self.background_process.terminate()
                    self.background_process.join(timeout=2.0)
            self.process_started = False
            self.background_process = None

        logger.info(
            f"Data collection completed. Recorded {self.episode_count} demos to {self.dataset_file}"
        )
