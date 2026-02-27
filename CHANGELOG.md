# Changelog

[中文版本](./CHANGELOG_zh.md)

All notable changes to this project will be documented in this file.

This project uses commit dates as release dates.

## [Unreleased]

### Future Plans
- Large-scale parallel data collection scripts (VLA-oriented).
- VLA (Vision-Language-Action) training scripts.
- High-performance inference deployment scripts.
- Advanced reinforcement learning demos (multi-task, obstacle avoidance, etc.).

## [0.1.0] - 2026-02-27

### Added
- Agilebot pick-and-place demo with environment and controller workflow.
- Tilt-aware grasping logic and phased state-machine handlers.
- Documentation in English and Chinese, including usage guides and asset setup notes.
- Demo image and USD robot/gripper assets for simulation.

### Changed
- Reorganized robot asset paths under `pick_place_demo/assets/robots/`.
- Refactored demo structure and supporting modules (`core`, `common`, `controller/phases`).
