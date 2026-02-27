# 🦾 Agilebot Isaac Lab

[中文版本](./README_zh.md)

**High-performance training and simulation framework for Agilebot robots based on NVIDIA Isaac Lab**

[![Website](https://img.shields.io/badge/Official-Website-blue)](https://www.sh-agilebot.com/)
[![Isaac Lab](https://img.shields.io/badge/Simulation-Isaac%20Lab-orange)](https://developer.nvidia.com/isaac-lab)
[![Agilebot](https://img.shields.io/badge/Robot-Agilebot-blue)](https://www.agilebot.com)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

## 📖 Overview

`agilebot_isaac_lab` is the official robot training and simulation repository from [Agilebot](https://www.sh-agilebot.com/). The project is deeply integrated with **NVIDIA Isaac Lab** and is designed to provide an efficient, scalable, and practical platform for embodied AI development and validation.

The repository currently includes an autonomous **Pick-and-Place** demo that showcases precise control and robust grasping behavior in complex physics simulation.

---

## 🗂️ Project Structure

```text
.
├── assets/                 # Robot asset config and setup notes (e.g., agilebot.py)
├── docs/                   # Documentation images and supporting materials
├── pick_place_demo/        # Autonomous pick-and-place demo source code
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # BSD 3-Clause license
├── README.md               # Main English README
└── README_zh.md            # Chinese README
```

Key modules:

- `assets/agilebot.py`: Agilebot robot asset definitions used by Isaac Lab.
- `assets/README.md`: English asset setup guide.
- `assets/README_zh.md`: Chinese asset setup guide.
- `pick_place_demo/main.py`: Demo entrypoint.
- `pick_place_demo/controller/`: Control logic (state machine, tilt-aware grasping, etc.).

---

## 🎬 Showcase: Autonomous Pick-and-Place

The project includes a ready-to-run pick-and-place demo. It combines **state-machine control**, **tilt-aware grasping compensation**, and **collision-triggered stop** behavior, making it a strong baseline for high-throughput data collection workflows.

![Pick-and-Place Demo](./docs/assets/pick_place_demo.png)

> 🚀 **Quick start**: Enter `pick_place_demo` and run `python main.py --enable_cameras`.
>
> 📖 **Demo details**: See the [Pick-and-Place Demo Guide](./pick_place_demo/README.md).

---

## ⚡ Quickstart

> The commands below use a common Isaac Lab Conda environment name: `isaaclab`. Replace it with your local environment name if needed.

1. Install **NVIDIA Omniverse Isaac Sim** and **Isaac Lab** (recommended versions below).
2. Activate environment:
   ```bash
   conda activate isaaclab
   ```
3. Configure Agilebot USD asset path (see "Asset Setup").
4. Run the demo:
   ```bash
   cd pick_place_demo
   python main.py --enable_cameras
   ```

---

## ✅ Recommended Versions

| Component | Recommended | Notes |
| :--- | :--- | :--- |
| Isaac Sim | `5.1` | Baseline version used by this repo |
| Isaac Lab | Matching Isaac Sim version | Must align with Isaac Sim |
| Python | `3.11` | Depends on Isaac Lab environment |

---

## 🗺️ Roadmap

Planned next milestones:

- [ ] Large-scale parallel data collection scripts (VLA-oriented)
- [ ] VLA (Vision-Language-Action) training scripts
- [ ] High-performance inference deployment scripts
- [ ] Advanced reinforcement learning demos (multi-task, obstacle avoidance, etc.)

---

## 🔗 Ecosystem Repositories

- **[`agilebot_isaac_sim`](https://github.com/sh-agilebot/agilebot_isaac_sim)**: Isaac Sim integration with environment setup and baseline examples.
- **[`agilebot_isaac_usd_assets`](https://github.com/sh-agilebot/agilebot_isaac_usd_assets)**: Unified robot digital assets (USD models, meshes, textures).

---

## 🛠️ Requirements

1. **Environment setup**: install NVIDIA Omniverse Isaac Sim and Isaac Lab.
   - Official Isaac Lab docs: `https://isaac-sim.github.io/IsaacLab/main/index.html`
2. **Conda environment**:
   ```bash
   conda activate isaaclab
   ```
3. **Asset setup**:
   - Get Agilebot digital assets from: `https://github.com/sh-agilebot/agilebot_isaac_lab`.
   - Configure `USD_PATH` in `assets/agilebot.py` and use an **absolute path** (do not use relative paths).
   - Detailed setup guide: [`assets/README.md`](./assets/README.md) (EN), [`assets/README_zh.md`](./assets/README_zh.md) (中文).
   - Asset licenses may differ from this code repository license. Verify compliance before distribution.

---

## 🧩 Troubleshooting

- **USD asset path errors**: check whether `USD_PATH` in `assets/agilebot.py` points to a valid location.
- **Camera errors / no camera data**: this demo requires `--enable_cameras` (Isaac Lab AppLauncher flag).
- **Headless mode issues**: try Isaac Lab `--headless` (also supported by this demo).

---

## 📄 License

This repository is open-sourced under the **BSD 3-Clause License**.

- Full text: [`LICENSE`](./LICENSE)
- Note: third-party resources may have separate license terms.

---

## 🤝 Contributing

Contributions are welcome.

- Open an issue for bugs or feature proposals.
- Submit pull requests for improvements.

Please read `CONTRIBUTING.md` before contributing.

---

<div align="center">

Copyright © 2026, Agilebot Robotics Co., Ltd.

</div>
