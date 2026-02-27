# 🦾 Agilebot Isaac Lab

[English Version](./README.md)

**基于 NVIDIA Isaac Lab 的捷勃特 (Agilebot) 机器人高性能训练与仿真框架**

[![Website](https://img.shields.io/badge/Official-Website-blue)](https://www.sh-agilebot.com/)
[![Isaac Lab](https://img.shields.io/badge/Simulation-Isaac%20Lab-orange)](https://developer.nvidia.com/isaac-lab)
[![Agilebot](https://img.shields.io/badge/Robot-Agilebot-blue)](https://www.agilebot.com)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

## 📖 简介

`agilebot_isaac_lab` 是[捷勃特 (Agilebot)](https://www.sh-agilebot.com/) 官方提供的机器人训练与仿真仓库。本项目深度集成 **NVIDIA Isaac Lab**，旨在为具身智能 (Embodied AI) 开发者提供一个高性能、可扩展且易于使用的学习与验证平台。

目前已实现首个**自主抓取与放置 (Pick-and-Place)** 演示案例，展示了 Agilebot 机器人在复杂物理环境中的精准控制与鲁棒抓取能力。

---

## 🗂️ 项目结构

```text
.
├── assets/                 # 机器人资产配置与说明（如 agilebot.py）
├── docs/                   # 文档图片与补充资料
├── pick_place_demo/        # 自主抓取与放置 Demo 源码
├── CONTRIBUTING.md         # 贡献指南
├── LICENSE                 # BSD 3-Clause 开源协议
├── README.md               # 英文主 README
└── README_zh.md            # 中文 README
```

关键模块：

- `assets/agilebot.py`：Isaac Lab 使用的 Agilebot 机器人资产定义。
- `assets/README.md`：英文资产配置说明文档。
- `assets/README_zh.md`：中文资产配置说明文档。
- `pick_place_demo/main.py`：Demo 启动入口。
- `pick_place_demo/controller/`：控制逻辑（状态机、倾斜补偿等）。

---

## 🎬 Showcase：自主抓取与放置 (Pick-and-Place)

本项目包含一个开箱即用的抓取演示，展示了 Agilebot 机器人在仿真环境中执行自动化的抓取循环。该演示集成了**状态机控制**、**倾斜补偿**以及**碰撞检测停机**功能，是开发高性能数据采集脚本的理想基础。

![Pick-and-Place Demo](./docs/assets/pick_place_demo.png)
> 🚀 **快速开始**：进入 `pick_place_demo` 目录并运行 `python main.py --enable_cameras`。
> 
> 📖 **详细说明**：请参阅 [Pick-and-Place Demo 指南](./pick_place_demo/README_zh.md)。

---

## ⚡ 快速开始

> 下面命令以 Isaac Lab 的常见 Conda 环境名 `isaaclab` 为例；如果你本地环境名不同，请自行替换。

1. 安装 **NVIDIA Omniverse Isaac Sim** 与 **Isaac Lab**（版本建议见下表）。
2. 激活环境：
   ```bash
   conda activate isaaclab
   ```
3. 配置 Agilebot USD 资产路径（见“资产配置”）。
4. 运行抓取 Demo：
   ```bash
   cd pick_place_demo
   python main.py --enable_cameras
   ```

---

## ✅ 版本建议

| 组件 | 建议版本 | 说明 |
| :--- | :--- | :--- |
| Isaac Sim | `5.1` | README 以此为最低建议版本 |
| Isaac Lab | 与 Isaac Sim 对应版本 | 需与 Isaac Sim 版本匹配 |
| Python | `3.11` | 取决于 Isaac Lab 环境 |

---

## 🗺️ 未来计划 (Roadmap)

我们正致力于构建更完整的具身智能生态，后续将陆续发布：

- [ ] 📥 **大规模并行数据采集脚本**: 以抓取为例子，支持并行采集大量VLA训练数据
- [ ] 🧠 **VLA (Vision-Language-Action) 训练脚本**: 支持视觉-语言-动作交互的端到端大模型训练。
- [ ] ⚡ **高性能推理部署脚本**: 将训练好的模型快速部署至仿真或真实 Agilebot 机器人。
- [ ] 🎮 **强化学习 (RL) 进阶 Demo**: 包含多任务学习、复杂环境避障等强化学习案例。

---

## 🔗 生态系统与相关仓库

本项目是 Agilebot Isaac 生态体系的一部分，建议同步参考以下仓库：

* **[`agilebot_isaac_sim`](https://github.com/sh-agilebot/agilebot_isaac_sim)**: Isaac Sim 基础集成仓库，包含环境设置与基础示例。
* **[`agilebot_isaac_usd_assets`](https://github.com/sh-agilebot/agilebot_isaac_usd_assets)**: 统一维护的机器人数字资产（USD 模型、网格及贴图）。

---

## 🛠️ 安装要求

1. **环境准备**: 确保已安装 NVIDIA Omniverse Isaac Sim 和 Isaac Lab。
   - Isaac Lab 官方安装参考：`https://isaac-sim.github.io/IsaacLab/main/index.html`
2. **Conda 环境**:
   ```bash
   conda activate isaaclab
   ```
3. **资产配置**:
   - 从以下仓库获取 Agilebot 数字资产：`https://github.com/sh-agilebot/agilebot_isaac_lab`。
   - 在 `assets/agilebot.py` 中配置 `USD_PATH`，并使用**绝对路径**（不要使用相对路径）。
   - 详细配置说明：[`assets/README.md`](./assets/README.md)（EN），[`assets/README_zh.md`](./assets/README_zh.md)（中文）。
   - 资产许可可能与本仓库代码许可不同；对外发布/分发前请先确认合规。

---

## 🧩 常见问题（Troubleshooting）

- **找不到 USD 资产 / 启动时报路径错误**：优先检查 `assets/agilebot.py` 的 `USD_PATH` 是否指向正确目录。
- **相机相关报错 / 看不到相机数据**：本 Demo 依赖相机传感器，请确保启动时包含 `--enable_cameras`（Isaac Lab AppLauncher 参数）。
- **无 GUI 跑不起来**：尝试使用 Isaac Lab 的 `--headless` 参数（Demo 侧也支持 `--headless` 透传）。

---

## 📄 开源协议（License）

本仓库代码采用 **BSD 3-Clause License** 开源协议。

- 协议全文见：[`LICENSE`](./LICENSE)
- 注意：第三方资源可能适用独立许可条款，请在使用和分发前分别确认。

---

## 🤝 贡献与支持

欢迎开发者参与到 Agilebot 机器人算法的开发与改进中！

- **提交 Issue**: 如果发现 Bug 或有功能建议，请通过 GitHub Issue 反馈。
- **Pull Requests**: 我们非常欢迎您的代码贡献！

建议先阅读贡献指南：`CONTRIBUTING.md`。

---

<div align="center">

Copyright © 2026, Agilebot Robotics Co., Ltd.

</div>
