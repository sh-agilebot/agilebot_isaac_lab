# IsaacLab Pick-and-Place Demo

[中文版](./README_zh.md)

This project is a high-performance simulation demo built on **IsaacLab**, featuring an **Agilebot** robot performing autonomous pick-and-place tasks. It demonstrates a robust state-machine-based control flow, multi-environment synchronization, and advanced features like tilt-aware grasping.

![Pick-and-Place Demo (Placeholder)](../docs/assets/pick_place_demo.png)

## 🚀 Features

- **Autonomous Pick-and-Place**: Complete cycle from object detection (simulated) to grasping, lifting, and placing in a container.
- **State-Machine Control**: Modular control logic divided into clear phases (Reach, Grasp, Lift, Place, etc.).
- **Multi-Environment Support**: Efficiently run multiple parallel simulation environments.
- **Tilt Compensation**: Advanced logic to handle and grasp tilted or overturned objects.
- **Collision Detection**: Safety monitoring for non-gripper parts of the robot.
- **Randomization**: Support for position and orientation noise to test robustness.

## 📂 Directory Structure

- `main.py`: Entry point for launching the simulation and demo.
- `env/`: Contains environment and scene configurations (`pick_place_env.py`).
- `controller/`: Core logic for the pick-and-place behavior.
    - `phases/`: Implementation of individual state machine phases.
    - `pick_place_controller.py`: High-level controller orchestration.
    - `tilt_aware_grasping.py`: Specialized logic for non-vertical grasps.
- `core/`: Simulation utilities and common helper functions.
- `assets/`: Robot models (URDF/USD) and associated textures.

## 🛠️ Getting Started

### Prerequisites

- NVIDIA Isaac Sim & IsaacLab installed.
- Isaac Lab installation reference: `https://isaac-sim.github.io/IsaacLab/main/index.html`
- Conda environment configured (e.g., `isaaclab`).

### Execution

Launch the demo with default settings:
```bash
python main.py --enable_cameras
```

Run with multiple environments and visualization markers:
```bash
python main.py --enable_cameras --num_envs 4 --show-markers
```

Enable tilt-aware grasping (with 'aligned' side-grasp strategy for severe tilts):
```bash
python main.py --enable_cameras --enable-tilt-compensation --grasp-strategy aligned --show-markers
```

## ⚙️ Common CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--num_envs` | `1` | Number of parallel simulation environments. |
| `--show-markers` | `False` | Show TCP / grasp / place visualization markers. |
| `--enable_cameras` | `False` | Enable camera sensors (required for this demo). |
| `--enable-tilt-compensation` | `False` | Enable tilted/overturned object grasp compensation. |
| `--grasp-strategy` | `vertical` | Grasp strategy for tilted objects (`vertical` or `aligned`). Use `aligned` to enable side-grasping for overturned objects. |
| `--reset-interval-steps` | `900` | Automatically reset the environment after this number of simulation steps. |
| `--control-noise` | `0.0` | Joint noise (radians) applied at robot initialization. Keeps initial poses similar but not identical for VLA data collection. |
| `--disable-non-gripper-collision-check` | `False` | Disable non-gripper collision checks (enabled by default). |
| `--non-gripper-collision-force-threshold` | `20.0` | Collision classification threshold (N). |
| `--device` | IsaacLab default | Simulation device from AppLauncher (for example `cpu` / `cuda:0`). |
| `--headless` | `False` | Run without GUI (from AppLauncher). |

For the full CLI list, run:
```bash
python main.py --help
```

## 🧩 Troubleshooting

- **USD asset path errors**: confirm Agilebot USD assets are available and configured (see repo root `assets/agilebot.py`).
- **Camera issues**: this demo requires `--enable_cameras` (an Isaac Lab AppLauncher flag).
- **Headless runs**: try `python main.py --headless` (forwarded to Isaac Lab AppLauncher).
- **Performance**: increase `--num_envs` gradually and monitor GPU/VRAM usage.

---
Copyright (c) 2026, Agilebot Robotics Co., Ltd.
