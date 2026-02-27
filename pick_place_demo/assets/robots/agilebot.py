"""Agilebot robot config used by the pick-and-place demo."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WRIST_CAMERA_GRIPPER_USD_PATH = str(
    (
        PROJECT_ROOT
        / "assets"
        / "usd"
        / "gbt-c5a_wrist_camera_gripper"
        / "gbt-c5a_wrist_camera_gripper.usd"
    ).resolve()
)

if not Path(WRIST_CAMERA_GRIPPER_USD_PATH).is_file():
    raise FileNotFoundError(
        f"USD not found: {WRIST_CAMERA_GRIPPER_USD_PATH}. "
        "Please ensure assets/usd/gbt-c5a_wrist_camera_gripper is present."
    )


GBT_C5A_WRIST_CAMERA_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=WRIST_CAMERA_GRIPPER_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 1.57,
            "joint3": -1.57,
            "joint4": 1.57,
            "joint5": -1.57,
            "joint6": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            stiffness=500.0,
            damping=100,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint[3]"],
            stiffness=500.0,
            damping=100,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            stiffness=500.0,
            damping=100,
            friction=0.0,
            armature=0.0,
        ),
    },
)
GBT_C5A_WRIST_CAMERA_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
GBT_C5A_WRIST_CAMERA_GRIPPER_CFG.init_state.joint_pos["finger_joint"] = 0.0
GBT_C5A_WRIST_CAMERA_GRIPPER_CFG.init_state.joint_pos[".*_inner_finger_joint"] = 0.0
GBT_C5A_WRIST_CAMERA_GRIPPER_CFG.init_state.joint_pos[".*_inner_knuckle_joint"] = 0.0
GBT_C5A_WRIST_CAMERA_GRIPPER_CFG.init_state.joint_pos["right_outer_knuckle_joint"] = 0.0
GBT_C5A_WRIST_CAMERA_GRIPPER_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    effort_limit_sim=100.0,
    velocity_limit_sim=1.0,
    stiffness=25.0,
    damping=0.5,
    friction=0.0,
    armature=0.0,
)
