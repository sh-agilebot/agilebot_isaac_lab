# Current Reset and Perturbation Scheme (Main Path, 2026-03-06)

This document describes the reset and perturbation logic in the main collection path (`scripts/collect/record.py`). Code sources:
- `scripts/collect/record.py`
- `core/simulation.py`
- `core/common_utils.py`

## 1. Trigger Chain and Failure Handling

1. `run_simulator()` executes one initial reset at startup.
2. Afterwards, an episode reset is performed every `reset_interval_steps` (default 900) steps.
3. The reset entry point is `perform_episode_reset()`, which internally calls `reset_everything()`.
4. If `reset_everything()` determines an invalid reset, the current attempt is skipped and retried.
5. When consecutive invalid resets reach `max_invalid_reset_streak` (default 20), an error is raised and the program exits.

## 2. Single Reset Workflow

### 2.1 Robot and Controller
- Robot joints are restored to initial pose, with optional `joint_noise` overlay (default `0.01 rad`).
- During controller reset, a random hold step count `random_hold_steps` can be sampled for each env (controlled by `random_hold_min/max`, default `3~8`).

### 2.2 KLT Reset (Shared Sampling)
- Base pose comes from `small_KLT.data.default_root_state` (cached on first use and reused).
- Position perturbation occurs within a manual workspace, currently fixed to:
  - `x ∈ [0.25, 0.75]`
  - `y ∈ [-0.30, 0.30]`
- Each reset samples only one set of shared `dx/dy/yaw`, applied to all parallel envs (then overlaid with respective `env_origin`).
- `klt_pos_range_x/y <= 0` indicates that axis is disabled from randomization (offset is 0 for that axis).

### 2.3 Target Object Reset Pose Range (Excluding Noise Parameters)
- Base pose:
  - `x_base, y_base, z_base` come from object's `default_root_state` (cached on first use and reused).
  - Base orientation uses a fixed base quat (`[0.70711, 0, 0, 0.70711]` in code).
- Current scene default base values (`env/pick_place_env.py`):
  - `x_base=0.55, y_base=-0.05, z_base=0.03`
  - `yaw_base≈+90 deg`
- Position range (final hard boundaries):
  - `x_final ∈ [0.25, 0.75]`
  - `y_final ∈ [-0.30, 0.30]`
  - `z_final = z_base` (no z-axis noise in this workflow)
- Orientation range:
  - yaw has no hard clamp boundary; it's derived from base yaw + random noise (and optional global rotation) overlay.

### 2.4 Target Object Noise Range (Per-Env Independent)
- Position noise (XY):
  - `eps_x, eps_y ~ N(0, obj_pos_noise^2)`, default `obj_pos_noise=0.005 m`.
  - Theoretical range is unbounded; engineering commonly uses `3σ` estimate: default approx `±0.015 m` (`±15 mm`).
- Orientation noise (yaw):
  - `eps_yaw ~ N(0, obj_yaw_noise^2)`, default `obj_yaw_noise=0.03 rad` (approx `1.72 deg`).
  - Theoretical range is unbounded; default `3σ` approx `±0.09 rad` (approx `±5.16 deg`).
- Combined formula (single env):
  - `x_final = clamp(x_base + eps_x, 0.25, 0.75)`
  - `y_final = clamp(y_base + eps_y, -0.30, 0.30)`
  - `z_final = z_base`
  - `yaw_final = yaw_base + eps_yaw + yaw_global`
  - Where `yaw_global` is a shared term: `0` when `global_z_rot_max_deg=0`; when `>0`, samples an angle in `[-v, v]` per reset and shares across all envs.
- "Common range (3σ)" under default parameters:
  - `x_final` approx `0.55 ± 0.015`, i.e., `[0.535, 0.565]`
  - `y_final` approx `-0.05 ± 0.015`, i.e., `[-0.065, -0.035]`
  - `z_final = 0.03`
  - `yaw_final` (with `global_z_rot_max_deg=0`) approx `90 deg ± 5.16 deg`

### 2.5 Target-KLT Minimum Distance Constraint (Strict Resampling)
- Constraint: `||obj_xy - klt_xy|| >= obj_klt_min_dist` (default `0.15 m`).
- Each reset attempts at most `obj_klt_min_dist_max_attempts` (default 50) times.
- All envs must satisfy the constraint for a valid reset; otherwise marked as invalid.
- Currently no "large displacement remediation" branches like corner fallback.

### 2.6 Global Z-Axis Rotation (Optional)
- When `global_z_rot_max_deg > 0`, each reset samples a shared angle applied to object pose (quaternion).
- Default `0.0`, meaning this feature is disabled.

### 2.7 Physics Stabilization Phase
- After writing reset state, executes `stabilization_steps` (default 20) physics steps.
- After stabilization, calculates grasp/place targets and enters control loop.

## 3. "Shared vs Independent" in Parallel Environments

Within the same reset:
- Shared quantities: KLT's `dx/dy/yaw`, and optional global Z rotation angle.
- Independent quantities: object XY/yaw noise, joint noise, collision/drop/success outcomes during control.

## 4. Current Default Parameters (Main Path)

From `scripts/collect/record.py`:
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

## 5. Notes (Related to Parameter Names)

- `obj_size_rand_scale`, `obj_spawn_edge_margin` are preserved as parameters in the main reset path but don't actually participate in sampling (explicitly ignored in manual fixed-bounds mode).
- `control_noise` belongs to control-phase perturbation (added to arm commands each step), not part of reset sampling itself.
