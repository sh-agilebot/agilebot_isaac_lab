# Pick-Place Data Collection Release

This package contains the runnable code and documentation for IsaacLab-based pick-place data collection.

[中文说明 (Chinese Documentation)](README_CN.md)

## Included Content

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

## Environment

```bash
source ~/miniforge3/bin/activate isaaclab
```

## Quick Start

```bash
mkdir -p datasets
python3 scripts/collect/record.py --num_envs 4
```

Record demonstrations:

```bash
mkdir -p datasets
python3 scripts/collect/record.py \
  --record \
  --num_envs 4 \
  --num_demos 10 \
  --enable_cameras \
  --dataset_file ./datasets/agilebot_demo.hdf5
```

## Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--num_envs` | Number of parallel environments | `32` |
| `--record` | Enable recording | `False` |
| `--dataset_file` | Output HDF5 file path | `./datasets/isaac_dataset.hdf5` |
| `--resume` | Append to an existing dataset | `False` |
| `--num_demos` | Number of demos to record (`0` means infinite) | `0` |
| `--fps` | Data recording FPS | `30` |
| `--enable_cameras` | Enable camera streams | `False` |
| `--headless` | Run without GUI | `False` |
| `--enable_pinocchio` | Enable Pinocchio-based IK support | `False` |
| `--task` | Language instruction saved in data | `put tomato soup can into container` |

## Data Notes

- Recorded action format is EE6D absolute-local 10D.
- The script writes HDF5 episodes with proprio/action/joint_state and optional camera images.
- Detailed data spec: `docs/specs/data_spec_EN.md`

## Documentation

- `docs/specs/data_spec_EN.md`: dataset schema and conventions (English)
- `docs/guides/current_reset_and_perturbation_scheme_EN.md`: reset and perturbation logic

