"""
Configuration for Agilebot GBT series robots.

Available configurations:
* GBT_CFG                     : GBT arm without gripper
* GBT_LONG_SUCTION_CFG        : GBT arm with long suction gripper
* GBT_SHORT_SUCTION_CFG       : GBT arm with short suction gripper
* GBT_ROBOTIQ_GRIPPER_CFG : GBT arm with Robotiq 2F-140 gripper
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# -----------------------------------------------------------------------------
# Asset path configuration
# -----------------------------------------------------------------------------

AGILEBOT_ASSETS_DIR = ""  # leave empty to use Nucleus

ROBOT_TYPE = "GBT-C5A"  # Options: GBT-C5A / GBT-C7A / GBT-C12A / GBT-C16A


USD_PATH = f"{AGILEBOT_ASSETS_DIR}/{ROBOT_TYPE.lower()}.usd"

# There is currently no such directory structure; please specify the absolute path directly.
USD_PATH = "/home/gbt/ws/usd/gbt_c5a.usd/gbt-c5a/gbt-c5a.usd"

# -----------------------------------------------------------------------------
# Base GBT arm (no gripper)
# -----------------------------------------------------------------------------

GBT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
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
        # Configure joint properties following the UR10 setup.
        # "arm": ImplicitActuatorCfg(
        #     joint_names_expr=["joint[1-6]"],
        #     effort_limit_sim=200.0,
        #     stiffness=800.0,
        #     damping=40.0,
        # ),
        # Configure joint properties following the Franka Panda setup.
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
GBT_CFG.spawn.variants = {"Gripper": "None"}

"""Configuration of GBT arm using implicit actuator models."""

# -----------------------------------------------------------------------------
# GBT + Long suction gripper
# -----------------------------------------------------------------------------

GBT_LONG_SUCTION_CFG = GBT_CFG.copy()
GBT_LONG_SUCTION_CFG.spawn.variants = {"Gripper": "Long_Suction_Gripper"}
GBT_LONG_SUCTION_CFG.spawn.rigid_props.disable_gravity = True
GBT_LONG_SUCTION_CFG.init_state.joint_pos = {
    "joint1": 0.0,
    "joint2": 1.57,
    "joint3": -1.57,
    "joint4": 1.57,
    "joint5": -1.57,
    "joint6": 0.0,
}

"""Configuration of GBT arm with long suction gripper."""

# -----------------------------------------------------------------------------
# GBT + Short suction gripper
# -----------------------------------------------------------------------------

GBT_SHORT_SUCTION_CFG = GBT_LONG_SUCTION_CFG.copy()
GBT_SHORT_SUCTION_CFG.spawn.variants = {"Gripper": "Short_Suction_Gripper"}

"""Configuration of GBT arm with short suction gripper."""

# -----------------------------------------------------------------------------
# GBT + Robotiq 2F-140 gripper
# -----------------------------------------------------------------------------
GBT_ROBOTIQ_GRIPPER_CFG = GBT_CFG.copy()
GBT_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2f_140"}
GBT_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
GBT_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos["finger_joint"] = 0.0
GBT_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_inner_finger_joint"] = 0.0
GBT_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_inner_knuckle_joint"] = 0.0
GBT_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos["right_outer_knuckle_joint"] = 0.0
# # the major actuator joint for gripper
GBT_ROBOTIQ_GRIPPER_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
)
# The settings cause abnormal gripper motion; therefore, they are not used for now.
# GBT_ROBOTIQ_GRIPPER_CFG.actuators = {
#     "arm": ImplicitActuatorCfg(
#         joint_names_expr=["joint[1-6]"],
#         effort_limit_sim=200.0,
#         velocity_limit_sim=2.175,
#         stiffness=1100.0,
#         damping=80.0,
#     ),
    
#     "gripper_drive": ImplicitActuatorCfg(
#         joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
#         effort_limit_sim=1,
#         velocity_limit_sim=10.0,
#         stiffness=17,
#         damping=0.02,
#     ),
    # # enable the gripper to grasp in a parallel manner
    # "gripper_finger": ImplicitActuatorCfg(
    #     joint_names_expr=[".*_inner_finger_joint"],
    #     effort_limit_sim=50,
    #     velocity_limit_sim=10.0,
    #     stiffness=0.2,
    #     damping=0.001,
    # ),
    # # set PD to zero for passive joints in close-loop gripper
    # "gripper_passive": ImplicitActuatorCfg(
    #     joint_names_expr=[".*_inner_knuckle_joint", "right_outer_knuckle_joint"],
    #     effort_limit_sim=1.0,
    #     velocity_limit_sim=10.0,
    #     stiffness=0.0,
    #     damping=0.0,
    # ),
# }

"""Configuration of GBT arm with Robotiq 2F-140 gripper."""
