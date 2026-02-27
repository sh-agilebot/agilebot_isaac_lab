#!/usr/bin/env python3
# Copyright (c) 2026, Agilebot Robotics Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-based controller architecture for pick-and-place.

This module implements a strategy pattern for grasp phases, where each phase
is encapsulated as a separate class with a unified interface.
"""

from controller.phases.base import PhaseHandler, PhaseContext, PhaseResult
from controller.phases.state_machine import StateMachine, GraspPhase

__all__ = [
    "PhaseHandler",
    "PhaseContext",
    "PhaseResult",
    "StateMachine",
    "GraspPhase",
]