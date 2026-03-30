#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Action-stream scheduler for continuous inference with LAAS semantics."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    """Inference output tied to the observation step that generated it."""

    obs_step: int
    ready_step: int
    latency_s: float
    decoded_actions: List[np.ndarray]


@dataclass(frozen=True)
class ScheduledAction:
    """Action scheduled for an absolute execution step."""

    action: np.ndarray
    source_obs_step: int
    chunk_index: int


class ActionStreamScheduler:
    """Timeline scheduler implementing LAAS outdated-drop and overlap-overwrite."""

    def __init__(self) -> None:
        self.timeline: Dict[int, ScheduledAction] = {}
        self.last_action: Optional[np.ndarray] = None
        self.num_dropped_outdated = 0
        self.num_overwritten_overlap = 0
        self.num_underflow_hold = 0

    def has_action_for_step(self, step: int) -> bool:
        return step in self.timeline

    def count_scheduled_from(self, step: int) -> int:
        return sum(1 for abs_step in self.timeline if abs_step >= step)

    def ingest_result(self, result: InferenceResult, current_step: int) -> None:
        for chunk_index, action in enumerate(result.decoded_actions):
            abs_step = result.obs_step + chunk_index
            if abs_step < current_step:
                self.num_dropped_outdated += 1
                continue

            if abs_step in self.timeline:
                self.num_overwritten_overlap += 1

            self.timeline[abs_step] = ScheduledAction(
                action=np.asarray(action, dtype=np.float32),
                source_obs_step=result.obs_step,
                chunk_index=chunk_index,
            )

    def pop_action_for_step(self, step: int, underflow_hold_max_steps: int = -1) -> np.ndarray:
        scheduled = self.timeline.pop(step, None)
        if scheduled is not None:
            self.last_action = scheduled.action
            return scheduled.action

        if self.last_action is None:
            raise RuntimeError("Action stream underflow before first action became available.")

        self.num_underflow_hold += 1
        if underflow_hold_max_steps >= 0 and self.num_underflow_hold > underflow_hold_max_steps:
            raise RuntimeError(
                f"Action stream underflow exceeded hold limit: {self.num_underflow_hold} > {underflow_hold_max_steps}."
            )
        return self.last_action
