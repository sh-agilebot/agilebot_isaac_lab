# 抓取放置数据采集发布包

该发布包仅包含 IsaacLab 抓取放置任务的数据采集代码与文档，可直接用于采集 HDF5 演示数据。

## 包含内容

```text
pick_place_data_collection_release_*/
├── scripts/collect/record.py
├── core/
├── controller/
├── common/
├── env/pick_place_env.py
├── assets/
│   ├── robots/
│   └── usd/
├── docs/
├── PACKAGE_MANIFEST.md
├── README.md
└── README_CN.md
```

## 环境准备

```bash
source ~/miniforge3/bin/activate isaaclab
```

## 快速开始

仅运行仿真（不写数据）：

```bash
mkdir -p datasets
python3 scripts/collect/record.py --num_envs 4
```

录制演示数据：

```bash
mkdir -p datasets
python3 scripts/collect/record.py \
  --record \
  --num_envs 4 \
  --num_demos 10 \
  --enable_cameras \
  --dataset_file ./datasets/agilebot_demo.hdf5
```

## 常用参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--num_envs` | 并行环境数量 | `32` |
| `--record` | 启用录制 | `False` |
| `--dataset_file` | HDF5 输出路径 | `./datasets/isaac_dataset.hdf5` |
| `--resume` | 在已有数据集后续写 | `False` |
| `--num_demos` | 录制条数（`0` 表示无限） | `0` |
| `--fps` | 数据采集帧率 | `30` |
| `--enable_cameras` | 启用相机观测 | `False` |
| `--headless` | 无界面运行 | `False` |
| `--enable_pinocchio` | 启用 Pinocchio IK | `False` |
| `--task` | 写入数据的语言指令 | `put tomato soup can into container` |

## 数据说明

- 动作格式为 EE6D absolute-local 10D。
- 采集结果按 episode 写入 HDF5，包含 proprio/action/joint_state 以及可选相机图像。
- 完整数据规范见 `docs/specs/data_spec_CN.md`

## 文档索引

- `docs/specs/data_spec_CN.md`：数据结构与字段规范
- `docs/guides/current_reset_and_perturbation_scheme.md`：重置与扰动逻辑说明
