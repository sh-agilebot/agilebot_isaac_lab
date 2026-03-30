# Pick-and-Place Data Collection Release Package

This package contains code and docs for **data collection** in the IsaacLab pick-place project.

## Included
- `scripts/collect/record.py` data collection entrypoint
- `core/` collection runtime, simulator loop, background HDF5 writer
- `controller/` pick-place controller and grasp state machine
- `common/` shared utilities
- `env/pick_place_env.py` scene definition used by collection script
- `assets/robots` and `assets/usd` robot configuration and USD assets
- `docs/` documentation related to collection/data format
- `README.md`, `README_CN.md`

## Quick Start
```bash
source ~/miniforge3/bin/activate isaaclab
python3 scripts/collect/record.py --num_envs 4 --robot Agilebot --record --num_demos 10 --enable_cameras
```
