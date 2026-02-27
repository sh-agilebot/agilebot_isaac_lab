# Agilebot 资产配置

[English Version](./README.md)

### `agilebot.py` - 捷勃特协作机器人 IsaacLab 资产配置文件

该文件用于在 **IsaacLab** 中定义捷勃特机器人资产配置，包括 USD 资产路径和可选变体（例如不同夹爪）。

> 📦 **数字资产仓库**：请从官方仓库获取捷勃特数字资产：`https://github.com/sh-agilebot/agilebot_isaac_lab`
>
> ⚠️ **重要**：你必须修改 `agilebot.py` 中的 `USD_PATH`（以及其他 USD 路径变量），并且所有路径都必须使用指向本地 USD 文件的**绝对路径**。

---

### 使用方式

#### ✅ 方式一（推荐）：集成到 IsaacLab 官方资产库

1. 将 `agilebot.py` 放到以下目录：
   ```
   ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/
   ```

2. 在同级目录的 `__init__.py` 中添加导入：
   ```python
   from . import agilebot
   ```

3. 在项目中像官方机器人资产一样导入：
   ```python
   from isaaclab_assets.robots import agilebot
   ```

#### ✅ 方式二：作为独立模块使用

1. 将 `agilebot.py` 放到你的项目目录中。
2. 确保 Python 可以找到该模块（例如通过 `sys.path.append()` 或 `PYTHONPATH`）。
3. 直接导入：
   ```python
   import agilebot
   ```

---

### 注意事项

- 代码中的所有 USD 路径都必须是**绝对路径**（例如 `D:/assets/robot/agilebot.usd`）。
- 不要使用相对路径（例如 `./assets/agilebot.usd`），否则工作目录变化时可能加载失败。
- 若需支持多个末端执行器（例如不同夹爪），建议通过 `variants` 字段定义，并确保对应 USD 文件存在。
