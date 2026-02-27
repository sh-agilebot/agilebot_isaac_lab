# Agilebot Asset Setup

[中文版本](./README_zh.md)

### `agilebot.py` - IsaacLab Asset Config for Jebot Collaborative Robot

This file defines the Jebot robot asset configuration for **IsaacLab**, including USD asset paths and optional variants (for example, different grippers).

> 📦 **Digital asset repository**: Get Jebot digital assets from the official repository: `https://github.com/sh-agilebot/agilebot_isaac_lab`
>
> ⚠️ **Important**: You must update `USD_PATH` (and any other USD path variables) in `agilebot.py`, and all paths must be **absolute paths** to local USD files.

---

### Usage

#### ✅ Option 1 (Recommended): Integrate into IsaacLab official assets

1. Place `agilebot.py` in:
   ```
   ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/
   ```

2. Add this import in the sibling `__init__.py`:
   ```python
   from . import agilebot
   ```

3. Import it in your project like other official robot assets:
   ```python
   from isaaclab_assets.robots import agilebot
   ```

#### ✅ Option 2: Use as a standalone module

1. Put `agilebot.py` in your own project directory.
2. Make sure Python can find it (for example, via `sys.path.append()` or `PYTHONPATH`).
3. Import directly:
   ```python
   import agilebot
   ```

---

### Notes

- All USD paths in code must be **absolute paths** (for example, `D:/assets/robot/agilebot.usd`).
- Do not use relative paths (for example, `./assets/agilebot.usd`), or loading may fail when the working directory changes.
- If you need multiple end-effectors (for example, different grippers), define variants via the `variants` field and ensure each corresponding USD file exists.

