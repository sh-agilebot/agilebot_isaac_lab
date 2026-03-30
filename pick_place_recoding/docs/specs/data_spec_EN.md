# X-VLA Data Collection Specification (AgileBot)

[Back to README](../../README.md) | [中文版本](data_spec_CN.md)

## 1. Overview

This document defines the standard for Isaac Lab data collection for X-VLA model fine-tuning.

> **Core Principle**: Use the same rot6d format for both collection and training
> - Store rot6d format (10D) directly during collection
> - Interpolate directly in rot6d space during training
> - Avoid euler → rot6d conversion for a simpler workflow

## 2. Data Format

### 2.1 Proprio Format

| Dimension | Index | Meaning | Description |
|-----------|-------|---------|-------------|
| Position | 0-2 | x, y, z | TCP position in world coordinates (meters) |
| Rotation | 3-8 | rot6d | **6D rotation representation** (first two rows of rotation matrix) **absolute** |
| Gripper | 9 | gripper | 0.0=open, 1.0=closed |

**Format**: `proprio [T, 10]` = `xyz(3) + rot6d(6) + gripper(1)`

> **Important**: proprio stores **absolute pose**, representing the true TCP pose in world coordinates. During training, the delta representation of xyz will be used.

### 2.2 Action Format

| Dimension | Index | Meaning | Description |
|-----------|-------|---------|-------------|
| Position | 0-2 | x, y, z | Next time step TCP position in world coordinates (meters) |
| Rotation | 3-8 | rot6d | Next time step 6D rotation representation |
| Gripper | 9 | gripper | Next time step gripper state |

**Format**: `action [T, 10]` = `xyz(3) + rot6d(6) + gripper(1)`

> **Core Definition**: `action_t = state_{t+1}` (action is the complete state of the next time step)

### 2.3 Why Use rot6d

From a mathematical perspective, rot6d interpolation is superior to Euler angles:

| Comparison | Euler Angles | rot6d |
|------------|--------------|-------|
| Gimbal Lock | Has problems | No problems |
| Continuity | May jump | Smooth continuous |
| Interpolation Quality | Poor at singularities | Stable |
| Paper Support | - | [Rotation Continuity](https://zhouyisjtu.github.io/RotationContinuity/) |

## 3. HDF5 File Format Specification

### 3.1 File Structure

```
datasets.hdf5
│
├── episode_000001/
│   ├── @episode_length: int          # Trajectory length (number of frames)
│   ├── @fps: float                   # Sampling frequency (default 30)
│   ├── @domain_id: int               # Domain ID (19)
│   ├── @language_instruction: str    # Task instruction
│   ├── @success: bool                 # Task success flag
│   │
│   └── steps/
│       ├── main_cam: [T, H, W, 3] uint8     # Main camera RGB image
│       ├── wrist_cam: [T, H, W, 3] uint8    # Wrist camera RGB image
│       ├── proprio: [T, 10] float32         # Current absolute pose (xyz+rot6d+grip)
│       ├── action: [T, 10] float32          # Next time step absolute pose (xyz+rot6d+grip)
│       ├── joint_state: [T, 7] float32       # Joint angles
│       ├── action_joint: [T, 7] float32      # Next time step joint angles
│       ├── dones: [T] bool                  # Completion flag
│       └── timestamps: [T] float64          # Timestamps
│
├── episode_000002/
│   └── ...
```

### 3.2 Episode Attributes

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `@episode_length` | int | ✅ | Trajectory length (number of frames) |
| `@fps` | float | ✅ | Sampling frequency, 30 Hz |
| `@domain_id` | int | ✅ | Domain ID, 19 |
| `@language_instruction` | str | ✅ | Task language instruction |
| `@success` | bool | ✅ | Task success flag |

### 3.3 Data Format

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `proprio` | `[T, 10]` | `float32` | xyz(3) + rot6d(6) + gripper(1) |
| `action` | `[T, 10]` | `float32` | Next time step absolute pose |
| `joint_state` | `[T, 7]` | `float32` | joint1~joint6 + finger_joint |
| `action_joint` | `[T, 7]` | `float32` | Next time step joint angles |

### 3.4 Parallel Environment Format

| Field | Single Env | Parallel Envs (4 envs) |
|-------|------------|------------------------|
| `proprio` | `[T, 10]` | `[T, 4, 10]` |
| `action` | `[T, 10]` | `[T, 4, 10]` |
| `images` | `[T, H, W, 3]` | `[T, 4, H, W, 3]` |

## 4. 6D Rotation Representation

### 4.1 Mathematical Principle

rot6d is the first two rows of the rotation matrix:

```
Rotation matrix R:
[ r00, r01, r02 ]   ← First row
[ r10, r11, r12 ]   ← Second row
[ r20, r21, r22 ]   ← Third row (cross product)

rot6d = [r00, r01, r02, r10, r11, r12]
```

### 4.2 Conversion Functions

**Quaternion → rot6d** (used during collection):
```python
import torch
from core.math_utils import quat_to_rot6d

# IsaacLab uses [w, x, y, z] quaternion format
quat = torch.randn(1, 4)
rot6d = quat_to_rot6d(quat)  # [1, 6]
```

## 5. Collection Implementation

### 5.1 Data Recording Code

```python
from core.ee6d_utils import pose_to_ee6d_proprio

def record_step():
    # Get TCP pose
    tcp_pos_w = ...       # [N, 3] position
    tcp_quat_w = ...      # [N, 4] quaternion [w, x, y, z]

    # Gripper state [0, 1]
    gripper_state = ...   # [N, 1] 0.0=open, 1.0=closed

    # Combine proprio [N, 10] - store absolute pose
    proprio = pose_to_ee6d_proprio(
        torch.cat([tcp_pos_w, tcp_quat_w], dim=-1),
        gripper_state
    )

    # Store
    data_collector.record_step(obs, proprio, ...)
```

### 5.2 Collection Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Sampling Rate** | 30 Hz | Collect 1 frame every 4 rendered frames |
| **Future Window** | 1.0 second | 30 frames @ 30Hz |
| **Index Margin** | 15 frames | Safety margin |
| **domain_id** | 19 | agilebot |

## 6. Training Label Generation Workflow

### 6.1 Action Definition

**Core Definition**: `action_t = state_{t+1}` (action is the complete state of the next time step)

```
Timeline (N total transitions):
Collection: state_0, state_1, state_2, ..., state_N

Recording:
record_0: obs=state_0,      action=state_1
record_1: obs=state_1,      action=state_2
record_2: obs=state_2,      action=state_3
...
record_{N-1}: obs=state_{N-1}, action=state_N

Note: state_N only appears as action, not as obs (no next state exists)
```

### 6.2 Data Flow

```
Collection: proprio [T, 10] rot6d format (absolute pose)
            action [T, 10] rot6d format (absolute pose)
         ↓ Handler reads directly
left [T, 10] rot6d
         ↓ Linear interpolation
seq [total_points, 10] interpolation result (absolute pose)
         ↓
Training sample: abs_trajectory [total_points, 10] (absolute pose)
```

### 6.3 Interpolation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **fps** | 30 Hz | Collection frame rate |
| **qdur** | 1.0 second | Prediction window duration |
| **num_actions** | 10 | Number of predicted action steps |
| **margin** | 15 frames | Index margin |

## 7. Index Margin Explanation

### 7.1 What is Index Margin

Index margin (margin) ensures there is sufficient future data from the sampling start point to generate training labels.

```
Trajectory length T = 300 collection frames (10 seconds at 30Hz)
Index margin margin = 15 frames (0.5 seconds at 30Hz)
Future window qdur = 1.0 second = 30 frames

Valid sampling start: range(0, T - margin) = range(0, 285)
```

### 7.2 Frequency Relationship

```
Render/Inference: 120 Hz
    ↓ Collect 1 frame every 4 frames
Collection:      30 Hz
    ↓ Index margin calculated at collection frequency
Margin:          15 frames = 0.5 second @ 30Hz
```

## 8. Meta.json Configuration

```json
{
  "dataset_name": "agilebot",
  "observation_key": ["main_cam", "wrist_cam"],
  "language_instruction_key": "language_instruction",
  "datalist": [
    {
      "hdf5_path": "/path/to/datasets/agilebot_dataset.hdf5",
      "episode_path": "episode_000001",
      "env_id": 0
    }
  ]
}
```

## 9. Data Quality Check

### 9.1 Post-Collection Verification

```python
import h5py
import numpy as np

f = h5py.File("agilebot_dataset.hdf5", "r")
ep = f["episode_000001"]
steps = ep["steps"]

# Check shapes
assert steps["proprio"].shape[1] == 10, "proprio must be 10D"
assert steps["action"].shape[1] == 10, "action must be 10D"
assert len(steps["main_cam"].shape) == 4, "images must be 4D"

# Check attributes
assert ep.attrs["domain_id"] == 19
assert ep.attrs["fps"] == 30

print("Data format validation passed!")
```

### 9.2 Checklist

- [ ] `proprio` shape is `[T, 10]`
- [ ] `action` shape is `[T, 10]`
- [ ] rot6d part is the first two rows of rotation matrix
- [ ] gripper is in [0, 1] range
- [ ] `fps` is 30 Hz
- [ ] `domain_id` is 19

## 10. FAQ

### Q1: Why not use Euler angles?

**A**: rot6d is mathematically superior, no gimbal lock, smooth interpolation.

### Q2: Will conversion affect performance during collection?

**A**: Minimal impact, quaternion to rot6d is a simple matrix operation.

### Q3: How to use during inference?

**A**: AgileBot model outputs **absolute pose action**, used directly as the next state.

### Q4: Why use linear interpolation for position?

**A**:
- **Sufficient sampling density**: At 30Hz sampling, position changes between adjacent frames are small, linear interpolation is smooth enough
- **Simple implementation**: Linear interpolation is computationally efficient and easy to understand and maintain

## References

- [X-VLA GitHub](https://github.com/2toINF/X-VLA)
- [Rotation Continuity Paper](https://zhouyisjtu.github.io/RotationContinuity/)
