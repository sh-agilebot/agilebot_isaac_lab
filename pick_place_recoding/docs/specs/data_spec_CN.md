# X-VLA 数据采集规范（AgileBot）

[返回中文 README](../../README_CN.md) | [English Version](data_spec_EN.md)

## 一、概述

本文档定义了 Isaac Lab 采集数据以用于 X-VLA 模型微调的规范标准。

> **核心原则**：采集和训练使用相同的 rot6d 格式
> - 采集时直接存储 rot6d 格式（10 维）
> - 训练时直接在 rot6d 空间插值
> - 避免 euler → rot6d 转换，流程更简洁

## 二、数据格式

### 2.1 Proprio 格式

| 维度 | 索引 | 含义 | 说明 |
|-----|------|------|------|
| 位置 | 0-2 | x, y, z | TCP 在世界坐标系中的位置（米） |
| 旋转 | 3-8 | rot6d | **6D 旋转表示**（旋转矩阵前两行）**绝对量** |
| 夹爪 | 9 | gripper | 0.0=打开, 1.0=闭合 |

**格式**：`proprio [T, 10]` = `xyz(3) + rot6d(6) + gripper(1)`

> **重要**：proprio 存储的是 **绝对位姿**，表示 TCP 在世界坐标系下的真实姿态。训练时候将使用xyz的增量表示

### 2.2 Action 格式

| 维度 | 索引 | 含义 | 说明 |
|-----|------|------|------|
| 位置 | 0-2 | x, y, z | 下一时刻 TCP 在世界坐标系中的位置（米） |
| 旋转 | 3-8 | rot6d | 下一时刻 6D 旋转表示 |
| 夹爪 | 9 | gripper | 下一时刻夹爪状态 |

**格式**：`action [T, 10]` = `xyz(3) + rot6d(6) + gripper(1)`

> **核心定义**：`action_t = state_{t+1}`（action 是下一个时刻的完整状态）

### 2.3 为什么用 rot6d

从数学角度，rot6d 插值优于欧拉角：

| 对比项 | 欧拉角 | rot6d |
|--------|--------|-------|
| 万向锁 | 有问题 | 无问题 |
| 连续性 | 可能跳变 | 平滑连续 |
| 插值效果 | 奇异点差 | 稳定 |
| 论文支持 | - | [Rotation Continuity](https://zhouyisjtu.github.io/RotationContinuity/) |

## 三、HDF5 文件格式规范

### 3.1 文件结构

```
datasets.hdf5
│
├── episode_000001/
│   ├── @episode_length: int          # 轨迹长度（帧数）
│   ├── @fps: float                   # 采样频率（默认 30）
│   ├── @domain_id: int               # 域 ID (19)
│   ├── @language_instruction: str    # 任务指令
│   ├── @success: bool                 # 任务是否成功
│   │
│   └── steps/
│       ├── main_cam: [T, H, W, 3] uint8     # 主相机 RGB 图像
│       ├── wrist_cam: [T, H, W, 3] uint8    # 手腕相机 RGB 图像
│       ├── proprio: [T, 10] float32         # 当前绝对位姿（xyz+rot6d+grip）
│       ├── action: [T, 10] float32          # 下一时刻绝对位姿（xyz+rot6d+grip）
│       ├── joint_state: [T, 7] float32       # 关节角度
│       ├── action_joint: [T, 7] float32      # 下一时刻关节角度
│       ├── dones: [T] bool                  # 完成标志
│       └── timestamps: [T] float64          # 时间戳
│
├── episode_000002/
│   └── ...
```

### 3.2 Episode 属性

| 属性名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `@episode_length` | int | ✅ | 轨迹长度（帧数） |
| `@fps` | float | ✅ | 采样频率，30 Hz |
| `@domain_id` | int | ✅ | 域 ID，19 |
| `@language_instruction` | str | ✅ | 任务语言指令 |
| `@success` | bool | ✅ | 任务是否成功 |

### 3.3 数据格式

| 字段名 | 形状 | 类型 | 说明 |
|--------|------|------|------|
| `proprio` | `[T, 10]` | `float32` | xyz(3) + rot6d(6) + gripper(1) |
| `action` | `[T, 10]` | `float32` | 下一时刻绝对位姿 |
| `joint_state` | `[T, 7]` | `float32` | joint1~joint6 + finger_joint |
| `action_joint` | `[T, 7]` | `float32` | 下一时刻关节角度 |

### 3.4 并行环境格式

| 字段名 | 单环境 | 并行环境（4 envs） |
|--------|--------|-------------------|
| `proprio` | `[T, 10]` | `[T, 4, 10]` |
| `action` | `[T, 10]` | `[T, 4, 10]` |
| `images` | `[T, H, W, 3]` | `[T, 4, H, W, 3]` |

## 四、6D 旋转表示法

### 4.1 数学原理

rot6d 是旋转矩阵的前两行：

```
旋转矩阵 R:
[ r00, r01, r02 ]   ← 第一行
[ r10, r11, r12 ]   ← 第二行
[ r20, r21, r22 ]   ← 第三行（叉乘计算）

rot6d = [r00, r01, r02, r10, r11, r12]
```

### 4.2 转换函数

**四元数 → rot6d**（采集时使用）：
```python
import torch
from core.math_utils import quat_to_rot6d

# IsaacLab 使用 [w, x, y, z] 四元数格式
quat = torch.randn(1, 4)
rot6d = quat_to_rot6d(quat)  # [1, 6]
```

## 五、采集实现

### 5.1 数据记录代码

```python
from core.ee6d_utils import pose_to_ee6d_proprio

def record_step():
    # 获取 TCP 位姿
    tcp_pos_w = ...       # [N, 3] 位置
    tcp_quat_w = ...      # [N, 4] 四元数 [w, x, y, z]
    
    # 夹爪状态 [0, 1]
    gripper_state = ...   # [N, 1] 0.0=打开, 1.0=闭合
    
    # 组合 proprio [N, 10] - 存储绝对位姿
    proprio = pose_to_ee6d_proprio(
        torch.cat([tcp_pos_w, tcp_quat_w], dim=-1),
        gripper_state
    )
    
    # 存储
    data_collector.record_step(obs, proprio, ...)
```

### 5.2 采集参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **采样频率** | 30 Hz | 每 4 个渲染帧采集 1 帧 |
| **未来窗口** | 1.0 秒 | 30 帧 @ 30Hz |
| **索引边距** | 15 帧 | 安全边距 |
| **domain_id** | 19 | agilebot |

## 六、训练标签生成流程

### 6.1 Action 定义

**核心定义**: `action_t = state_{t+1}` （action 是下一个时刻的完整状态）

```
时间轴 (共N个transition):
采集: state_0, state_1, state_2, ..., state_N

记录:
record_0: obs=state_0,      action=state_1
record_1: obs=state_1,      action=state_2
record_2: obs=state_2,      action=state_3
...
record_{N-1}: obs=state_{N-1}, action=state_N

注意: state_N 只作为 action 出现，不作为 obs 记录（因为没有下一个状态）
```

### 6.2 数据流程

```
采集: proprio [T, 10] rot6d 格式（绝对位姿）
       action [T, 10] rot6d 格式（绝对位姿）
  ↓ Handler 直接读取
left [T, 10] rot6d
  ↓ 线性插值
seq [total_points, 10] 插值结果（绝对位姿）
  ↓
训练样本: abs_trajectory [total_points, 10]（绝对位姿）
```

### 6.3 插值参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **fps** | 30 Hz | 采集帧率 |
| **qdur** | 1.0 秒 | 预测窗口时长 |
| **num_actions** | 10 | 预测动作步数 |
| **margin** | 15 帧 | 索引边距 |

## 七、索引边距说明

### 7.1 什么是索引边距

索引边距（margin）确保从采样起点有足够的未来数据用于生成训练标签。

```
轨迹长度 T = 300 采集帧（30Hz 下 10 秒）
索引边距 margin = 15 帧（30Hz 下 0.5 秒）
未来窗口 qdur = 1.0 秒 = 30 帧

有效采样起点：range(0, T - margin) = range(0, 285)
```

### 7.2 频率关系

```
渲染/推理: 120 Hz
    ↓ 每 4 帧采集 1 帧
采集:     30 Hz
    ↓ 索引边距按采集频率计算
边距:     15 帧 = 0.5 秒 @ 30Hz
```

## 八、Meta.json 配置

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

## 九、数据质量检查

### 9.1 采集后验证

```python
import h5py
import numpy as np

f = h5py.File("agilebot_dataset.hdf5", "r")
ep = f["episode_000001"]
steps = ep["steps"]

# 检查形状
assert steps["proprio"].shape[1] == 10, "proprio 必须是 10 维"
assert steps["action"].shape[1] == 10, "action 必须是 10 维"
assert len(steps["main_cam"].shape) == 4, "图像必须是 4D"

# 检查属性
assert ep.attrs["domain_id"] == 19
assert ep.attrs["fps"] == 30

print("数据格式验证通过！")
```

### 9.2 检查清单

- [ ] `proprio` 形状为 `[T, 10]`
- [ ] `action` 形状为 `[T, 10]`
- [ ] rot6d 部分是旋转矩阵前两行
- [ ] gripper 在 [0, 1] 范围
- [ ] `fps` 为 30 Hz
- [ ] `domain_id` 为 19

## 十、常见问题

### Q1: 为什么不用欧拉角？

**A**: rot6d 在数学上更优，无万向锁，插值平滑。

### Q2: 采集时转换会影响性能吗？

**A**: 影响很小，四元数到 rot6d 是简单矩阵操作。

### Q3: 推理时如何使用？

**A**: AgileBot 模型输出 **绝对位姿 action**，直接作为下一个状态使用。

### Q4: 为什么位置用线性插值？

**A**:
- **采样密度足够**：30Hz 采样下，相邻帧位置变化很小，线性插值足够平滑
- **实现简单**：线性插值计算高效，易于理解和维护

## 参考资料

- [X-VLA GitHub](https://github.com/2toINF/X-VLA)
- [Rotation Continuity 论文](https://zhouyisjtu.github.io/RotationContinuity/)
