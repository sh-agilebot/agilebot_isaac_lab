# 当前 Reset 与扰动方案（主路径，2026-03-06）

本文只描述主采集路径（`scripts/collect/record.py`）中的 reset 与扰动逻辑，代码来源：
- `scripts/collect/record.py`
- `core/simulation.py`
- `core/common_utils.py`

## 1. 触发链路与失败处理

1. `run_simulator()` 在启动时执行一次 initial reset。
2. 之后每 `reset_interval_steps`（默认 900）执行一次 episode reset。
3. reset 入口为 `perform_episode_reset()`，内部调用 `reset_everything()`。
4. 若 `reset_everything()` 判定为 invalid reset，则本次跳过并重试。
5. 连续 invalid reset 达到 `max_invalid_reset_streak`（默认 20）时抛错退出。

## 2. 单次 Reset 具体流程

### 2.1 机器人与控制器
- 机器人关节恢复到初始位姿，可叠加 `joint_noise`（默认 `0.01 rad`）。
- 控制器 reset 时可采样每个 env 的随机等待步数 `random_hold_steps`（由 `random_hold_min/max` 控制，默认 `3~8`）。

### 2.2 KLT 重置（共享采样）
- 基准位姿来自 `small_KLT.data.default_root_state`（首次缓存后复用）。
- 位置扰动在手动工作区内进行，当前固定工作区为：
  - `x ∈ [0.25, 0.75]`
  - `y ∈ [-0.30, 0.30]`
- 每次 reset 只采样一组共享 `dx/dy/yaw`，应用到所有并行 env（再叠加各自 `env_origin`）。
- `klt_pos_range_x/y <= 0` 表示该轴关闭随机化（该轴偏移为 0）。

### 2.3 目标物体重置位姿范围（不含噪声参数解释）
- 基准位姿：
  - `x_base, y_base, z_base` 来自物体 `default_root_state`（首次缓存后复用）。
  - 基准朝向使用固定 base quat（代码中为 `[0.70711, 0, 0, 0.70711]`）。
- 当前场景默认基准值（`env/pick_place_env.py`）：
  - `x_base=0.55, y_base=-0.05, z_base=0.03`
  - `yaw_base≈+90 deg`
- 位置范围（最终硬边界）：
  - `x_final ∈ [0.25, 0.75]`
  - `y_final ∈ [-0.30, 0.30]`
  - `z_final = z_base`（本流程不对 z 加噪声）
- 朝向范围：
  - yaw 没有硬 clamp 边界；由基准 yaw + 随机噪声（以及可选全局旋转）叠加得到。

### 2.4 目标物体噪声范围（每 env 独立）
- 位置噪声（XY）：
  - `eps_x, eps_y ~ N(0, obj_pos_noise^2)`，默认 `obj_pos_noise=0.005 m`。
  - 理论范围无界；工程上常用 `3σ` 估计：默认约 `±0.015 m`（`±15 mm`）。
- 姿态噪声（yaw）：
  - `eps_yaw ~ N(0, obj_yaw_noise^2)`，默认 `obj_yaw_noise=0.03 rad`（约 `1.72 deg`）。
  - 理论范围无界；默认 `3σ` 约 `±0.09 rad`（约 `±5.16 deg`）。
- 组合后公式（单个 env）：
  - `x_final = clamp(x_base + eps_x, 0.25, 0.75)`
  - `y_final = clamp(y_base + eps_y, -0.30, 0.30)`
  - `z_final = z_base`
  - `yaw_final = yaw_base + eps_yaw + yaw_global`
  - 其中 `yaw_global` 为共享项：`global_z_rot_max_deg=0` 时为 0；`>0` 时每次 reset 在 `[-v, v]` 采样一个角度并对所有 env 共享。
- 默认参数下的“常用范围（3σ）”：
  - `x_final` 约 `0.55 ± 0.015`，即 `[0.535, 0.565]`
  - `y_final` 约 `-0.05 ± 0.015`，即 `[-0.065, -0.035]`
  - `z_final = 0.03`
  - `yaw_final`（`global_z_rot_max_deg=0`）约 `90 deg ± 5.16 deg`

### 2.5 目标-KLT 最小距离约束（严格重采样）
- 约束：`||obj_xy - klt_xy|| >= obj_klt_min_dist`（默认 `0.15 m`）。
- 每次 reset 最多尝试 `obj_klt_min_dist_max_attempts`（默认 50）次。
- 必须所有 env 都满足约束才算有效 reset，否则记为 invalid。
- 当前没有角点回退等“大位移补救”分支。

### 2.6 全局 Z 轴旋转（可选）
- 当 `global_z_rot_max_deg > 0` 时，每次 reset 采样一个共享角度并作用于物体姿态（四元数）。
- 默认 `0.0`，即关闭该项。

### 2.7 物理稳定阶段
- reset 写入状态后，执行 `stabilization_steps`（默认 20）步物理推进。
- 稳定完成后再计算 grasp/place 目标并进入控制循环。

## 3. 并行环境中的“共享 vs 独立”

同一次 reset 内：
- 共享量：KLT 的 `dx/dy/yaw`，以及可选的全局 Z 旋转角。
- 独立量：物体 XY/yaw 噪声、关节噪声、控制过程中的碰撞/掉落/成功结果。

## 4. 当前默认参数（主路径）

来自 `scripts/collect/record.py`：
- `reset_interval_steps = 900`
- `stabilization_steps = 20`
- `joint_noise = 0.01`
- `random_hold_min = 3`
- `random_hold_max = 8`
- `obj_pos_noise = 0.005`
- `obj_yaw_noise = 0.03`
- `klt_pos_range_x = 0.005`
- `klt_pos_range_y = 0.005`
- `klt_yaw_rand_deg = 2.0`
- `obj_klt_min_dist = 0.15`
- `obj_klt_min_dist_max_attempts = 50`
- `global_z_rot_max_deg = 0.0`
- `max_invalid_reset_streak = 20`

## 5. 备注（和参数名相关）

- `obj_size_rand_scale`、`obj_spawn_edge_margin` 在当前主路径 reset 中保留参数但未实际参与采样（manual fixed-bounds 模式下被显式忽略）。
- `control_noise` 属于控制阶段扰动（每步加到 arm 命令），不属于 reset 采样本身。
