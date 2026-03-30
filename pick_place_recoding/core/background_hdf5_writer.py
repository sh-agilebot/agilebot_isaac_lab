#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Separate process for HDF5 I/O operations.

This module runs in a separate process and handles all HDF5 file operations.
It must NOT import any Isaac libraries - only h5py, numpy, and standard library.

Message Protocol:
- START_RECORDING: Initialize new episode
- RECORD_BATCH: Add a batch of data (one second of data including images)
- STOP_RECORDING: Finalize current episode
- CANCEL_RECORDING: Drop current episode without saving metadata
- CLOSE: Shut down the writer process
"""

import h5py
import logging
import numpy as np
import os
import queue
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HDF5BackgroundProcess:
    """Separate process for handling HDF5 I/O operations.

    This class runs in a separate process and handles all HDF5 file operations.
    It must NOT import any Isaac libraries - only h5py, numpy, and standard library.

    Args:
        data_queue: Queue for receiving data from main thread.
        status_queue: Queue for sending status updates to main thread.
        dataset_file: Path to the HDF5 file where data will be stored.
        resume: Whether to resume from existing file.
    """

    def __init__(self, data_queue: queue.Queue, status_queue: queue.Queue, 
                 dataset_file: str, resume: bool = False):
        self.data_queue = data_queue
        self.status_queue = status_queue
        self.dataset_file = dataset_file
        self.resume = resume
        self.h5_file = None
        self.episode_count = 0
        self.current_episode_group = None
        self.running = False
        self.batch_counter = 0

    def run(self):
        """Main loop for the writer process."""
        logger.info("HDF5 writer process started")
        self.running = True

        while self.running:
            try:
                message = self.data_queue.get(timeout=0.1)
                self._handle_message(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in writer process: {e}")
                self._send_status("ERROR", str(e))

        self._cleanup()
        logger.info("HDF5 writer process stopped")

    def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages from main thread.

        Args:
            message: Dictionary containing message type and data.
        """
        msg_type = message.get("type")

        if msg_type == "START_RECORDING":
            self._start_recording(message)
        elif msg_type == "RECORD_BATCH":
            self._record_batch(message)
        elif msg_type == "STOP_RECORDING":
            self._stop_recording(message)
        elif msg_type == "CANCEL_RECORDING":
            self._cancel_recording()
        elif msg_type == "CLOSE":
            self._close()
        else:
            logger.warning(f"Unknown message type: {msg_type}")

    def _start_recording(self, message: Dict[str, Any]):
        """Start recording a new episode.

        Args:
            message: Dictionary containing episode metadata.
        """
        try:
            if self.h5_file is None:
                mode = "a" if self.resume else "w"
                self.h5_file = h5py.File(self.dataset_file, mode)
                if self.resume:
                    try:
                        episodes = [key for key in self.h5_file.keys() if key.startswith('episode_')]
                        if episodes:
                            max_idx = max(int(ep.split('_')[1]) for ep in episodes)
                            self.episode_count = max_idx + 1
                        else:
                            self.episode_count = 0
                    except Exception as e:
                        logger.warning(f"Failed to read existing dataset: {e}")
                        self.episode_count = 0

            self.current_episode_group = self.h5_file.create_group(f"episode_{self.episode_count:06d}")
            self.steps_group = self.current_episode_group.create_group("steps")
            self.batch_counter = 0
            self._send_status("STARTED", {"episode_id": self.episode_count})
            logger.info(f"Started recording episode {self.episode_count}")

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self._send_status("ERROR", str(e))

    def _record_batch(self, message: Dict[str, Any]):
        """Record a batch of data (one second of data including images).

        Args:
            message: Dictionary containing batch data.
        """
        try:
            if self.current_episode_group is None or self.steps_group is None:
                logger.warning("No active episode to record batch")
                return

            batch_data = message.get("data", {})
            batch_index = message.get("batch_index", 0)

            for key, value in batch_data.items():
                self._save_batch_data(self.steps_group, key, value, batch_index)

            self._send_status("BATCH_RECORDED", {"batch_index": batch_index})
            self.batch_counter += 1

        except Exception as e:
            logger.error(f"Failed to record batch: {e}")
            self._send_status("ERROR", str(e))

    def _save_batch_data(self, group: h5py.Group, key: str, value: Any, batch_index: int = 0):
        """Save batch data to HDF5 group.

        Args:
            group: HDF5 group to save data to.
            key: Data key.
            value: Data value (must be numpy array or serializable).
            batch_index: Starting index for this batch (for debugging).
        """
        try:
            if isinstance(value, np.ndarray):
                if value.dtype == np.object_:
                    logger.warning(f"Skipping object array for key {key}")
                    return
                if key in group:
                    existing_dataset = group[key]
                    current_size = existing_dataset.shape[0]
                    new_size = current_size + value.shape[0]
                    existing_dataset.resize((new_size,) + existing_dataset.shape[1:])
                    existing_dataset[current_size:new_size] = value
                else:
                    maxshape = (None,) + value.shape[1:]
                    group.create_dataset(key, data=value, maxshape=maxshape)
            elif isinstance(value, dict):
                sub_group = group.create_group(key) if key not in group else group[key]
                for sub_key, sub_value in value.items():
                    self._save_batch_data(sub_group, sub_key, sub_value, batch_index)
            elif isinstance(value, (int, float, bool)):
                if key in group:
                    existing_dataset = group[key]
                    current_size = existing_dataset.shape[0]
                    existing_dataset.resize((current_size + 1,))
                    existing_dataset[current_size] = value
                else:
                    group.create_dataset(key, data=np.array([value]), maxshape=(None,))
            elif isinstance(value, str):
                if key in group:
                    existing_dataset = group[key]
                    current_size = existing_dataset.shape[0]
                    existing_dataset.resize((current_size + 1,))
                    existing_dataset[current_size] = value.encode('utf-8')
                else:
                    string_dtype = h5py.string_dtype(encoding='utf-8')
                    group.create_dataset(key, data=np.array([value.encode('utf-8')]), 
                                        maxshape=(None,), dtype=string_dtype)
            elif isinstance(value, list):
                try:
                    value_array = np.array(value)
                    if value_array.dtype != np.object_:
                        if key in group:
                            existing_dataset = group[key]
                            current_size = existing_dataset.shape[0]
                            new_size = current_size + value_array.shape[0]
                            existing_dataset.resize((new_size,) + existing_dataset.shape[1:])
                            existing_dataset[current_size:new_size] = value_array
                        else:
                            maxshape = (None,) + value_array.shape[1:]
                            group.create_dataset(key, data=value_array, maxshape=maxshape)
                    else:
                        logger.warning(f"Skipping list with object dtype for key {key}")
                except (ValueError, TypeError):
                    logger.warning(f"Failed to convert list to array for key {key}")
            else:
                logger.warning(f"Unsupported data type for key {key}: {type(value)}")

        except Exception as e:
            logger.error(f"Failed to save batch data for key {key}: {e}")

    def _stop_recording(self, message: Dict[str, Any]):
        """Stop recording and finalize episode.

        Filters out failed environments before saving, keeping only successful ones.

        Args:
            message: Dictionary containing episode metadata including env_success_flags.
        """
        try:
            if self.current_episode_group is None:
                logger.warning("No active episode to stop")
                return

            metadata = message.get("metadata", {})
            if not metadata:
                logger.warning(
                    "Received empty STOP_RECORDING metadata for episode %06d, dropping episode as dirty data",
                    self.episode_count,
                )
                self._drop_current_episode(increment_episode_count=False)
                return

            env_success_flags = metadata.get("env_success_flags", None)

            # Filter datasets to keep only successful environments
            if env_success_flags is not None and len(env_success_flags) > 0:
                success_mask = np.array(env_success_flags, dtype=bool)
                successful_count = np.sum(success_mask)

                if successful_count > 0 and successful_count < len(success_mask):
                    # Filter each dataset in steps/ to keep only successful environments
                    self._filter_successful_environments(self.steps_group, success_mask)
                    logger.info(f"Filtered episode data: kept {successful_count}/{len(success_mask)} successful environments")
                elif successful_count == 0:
                    logger.warning(f"No successful environments in episode {self.episode_count}, deleting episode")
                    self._drop_current_episode(increment_episode_count=False)
                    return

            # Save metadata attributes
            for key, value in metadata.items():
                if isinstance(value, np.ndarray):
                    self.current_episode_group.attrs[key] = value
                else:
                    self.current_episode_group.attrs[key] = value

            self.h5_file.flush()
            # Update file-level metadata as backup in case process is terminated
            self.h5_file.attrs["total_episodes"] = self.episode_count + 1
            self.h5_file.attrs["created_at"] = datetime.now().isoformat()
            self.h5_file.flush()
            self._send_status("STOPPED", {"episode_id": self.episode_count})
            logger.info(f"Stopped recording episode {self.episode_count}")
            self.episode_count += 1
            self.current_episode_group = None

        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            self._send_status("ERROR", str(e))

    def _cancel_recording(self):
        """Cancel current recording and drop the in-progress episode."""
        if self.current_episode_group is None:
            logger.warning("No active episode to cancel")
            return
        logger.info("Cancelling episode %06d: dropping dirty data", self.episode_count)
        self._drop_current_episode(increment_episode_count=False)

    def _drop_current_episode(self, increment_episode_count: bool = False):
        """Delete current episode group and clear writer state."""
        if self.h5_file is not None:
            episode_key = f"episode_{self.episode_count:06d}"
            if episode_key in self.h5_file:
                del self.h5_file[episode_key]
                self.h5_file.flush()
        self.current_episode_group = None
        self.steps_group = None
        self.batch_counter = 0
        if increment_episode_count:
            self.episode_count += 1

    def _filter_successful_environments(self, steps_group: h5py.Group, success_mask: np.ndarray):
        """Filter datasets to keep only successful environments.

        Args:
            steps_group: HDF5 group containing step data.
            success_mask: Boolean array indicating which environments were successful.
        """
        successful_indices = np.where(success_mask)[0]

        for key in steps_group.keys():
            dataset = steps_group[key]

            if not isinstance(dataset, h5py.Dataset):
                continue

            try:
                data = dataset[()]

                # Filter based on data shape
                if data.ndim >= 2:
                    # Shape: (T, envs, ...) or (envs, ...)
                    # Filter the environment dimension (axis 0 if 2D, axis 1 if 3D+)
                    if data.ndim == 2:
                        # (envs, features) or (T, features) - need to check
                        # If second dim matches success_mask length, filter along axis 0
                        if data.shape[0] == len(success_mask):
                            data = data[success_mask, ...]
                    else:
                        # (T, envs, H, W, C) or similar - filter along axis 1
                        if data.shape[1] == len(success_mask):
                            data = data[:, success_mask, ...]

                elif data.ndim == 1 and len(data) == len(success_mask):
                    # (envs,) - filter along axis 0
                    data = data[success_mask]

                # Resize and write back filtered data
                dataset.resize(data.shape)
                dataset[...] = data

            except Exception as e:
                logger.warning(f"Failed to filter dataset {key}: {e}")
                continue

    def _close(self):
        """Close the writer process."""
        if self.current_episode_group is not None:
            logger.warning(
                "Closing writer with active episode %06d; dropping unfinished data",
                self.episode_count,
            )
            self._drop_current_episode(increment_episode_count=False)
        self.running = False

    def _cleanup(self):
        """Clean up resources."""
        if self.h5_file is not None:
            try:
                self.h5_file.attrs["total_episodes"] = self.episode_count
                self.h5_file.attrs["created_at"] = datetime.now().isoformat()
                self.h5_file.close()
            except Exception as e:
                logger.error(f"Error closing HDF5 file: {e}")
            finally:
                self.h5_file = None

    def _send_status(self, status_type: str, data: Any = None):
        """Send status update to main thread.

        Args:
            status_type: Type of status message.
            data: Optional data to include with status.
        """
        try:
            self.status_queue.put({"type": status_type, "data": data})
        except Exception as e:
            logger.error(f"Failed to send status: {e}")


def run_background_process(data_queue: queue.Queue, status_queue: queue.Queue, 
                           dataset_file: str, resume: bool = False):
    """Entry point for the writer process.

    Args:
        data_queue: Queue for receiving data from main thread.
        status_queue: Queue for sending status updates to main thread.
        dataset_file: Path to the HDF5 file where data will be stored.
        resume: Whether to resume from existing file.
    """
    process = HDF5BackgroundProcess(data_queue, status_queue, dataset_file, resume)
    process.run()
