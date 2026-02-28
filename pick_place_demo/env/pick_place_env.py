
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import RigidObjectCfg

##
# Pre-defined configs
##
from assets.robots.agilebot import GBT_C5A_WRIST_CAMERA_GRIPPER_CFG
from isaaclab.sensors import CameraCfg, ContactSensorCfg


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a tabletop scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    # mount
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    #     ),
    # )

    # Table (origin on the top surface)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.6, 0, 0), rot=(0.707, 0, 0, 0.707)
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )
    robot = GBT_C5A_WRIST_CAMERA_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Required by ContactSensor to report per-body contact forces.
    robot.spawn.activate_contact_sensors = True

    # sensors
    # NOTE:
    # ContactSensor cannot reliably monitor multiple top-level prim groups in one config.
    # Use separate sensors for robot/container/object and merge results in runtime logic.
    collision_sensor_robot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        # Only monitor robot contacts against container/object targets (exclude table).
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/small_KLT",
            "{ENV_REGEX_NS}/tomato_soup_can",
        ],
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )

    collision_sensor_container = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/small_KLT",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )

    collision_sensor_object = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can",
        update_period=0.0,
        history_length=2,
        debug_vis=False,
    )

    main_camera = CameraCfg(
        prim_path=f"{table.prim_path}/main_cam",
        update_period=0.0, # Update every simulation step
        height=224,
        width=224,
        # data_types=["rgb", "distance_to_image_plane"],
        data_types=["rgb",],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.25, focus_distance=150.0, horizontal_aperture=5.45, clipping_range=(0.01, 1000000.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0,-0.45 , 0.7), rot=(0.92388, 0.38268, 0.0, 0.0), convention="opengl"),
    )

    wrist_camera = CameraCfg(
        # prim_path="{ENV_REGEX_NS}/Robot/gbt_c5a_camera_gripper/link6/flange/camera_mount/Orbbec/camera_rgb/camera_rgb",
        prim_path="{ENV_REGEX_NS}/Robot/wrist_camera_link/camera_color_frame/wrist_camera",
        update_period=0.0,
        height=224,
        width=224,
        # data_types=["rgb", "distance_to_image_plane"],
        data_types=["rgb",],
        spawn=None
       
    )

    # Rigid body properties of each cube
    # cube_properties = RigidBodyPropertiesCfg(
    #     solver_position_iteration_count=16,
    #     solver_velocity_iteration_count=1,
    #     max_angular_velocity=1000.0,
    #     max_linear_velocity=1000.0,
    #     max_depenetration_velocity=5.0,
    #     disable_gravity=False,
    # )
    # Set each stacking cube deterministically
    # cube_1 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Cube_1",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.4, 0.0, 0.0203), rot=(1, 0, 0, 0)
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #         # mass_props=MassPropertiesCfg(mass=0.01),
    #         semantic_tags=[("class", "cube_1")],
    #     ),
    # )
    # cube_2 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Cube_2",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.55, 0.05, 0.0203), rot=(1, 0, 0, 0)
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #         # mass_props=MassPropertiesCfg(mass=0.01),
    #         semantic_tags=[("class", "cube_2")],
    #     ),
    # )
    # cube_3 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Cube_3",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.60, -0.1, 0.0203), rot=(1, 0, 0, 0)
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #         mass_props=MassPropertiesCfg(mass=0.01),
    #         semantic_tags=[("class", "cube_3")],
    #     ),
    # )

    # small_KLT:/home/gbt/isaac_assets/Assets/Isaac/5.1/Isaac/Props/KLT_Bin/small_KLT.usd
    small_KLT = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/small_KLT",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.30, 0.32, 0.076), rot=(0.70711, 0.0, 0.0, 0.70711)  # Move bin farther from robot base.
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
            scale=(1.0, 1.0, 1.0),
            # rigid_props=RigidBodyPropertiesCfg(
            #     solver_position_iteration_count=16,
            #     solver_velocity_iteration_count=1,
            #     max_angular_velocity=1000.0,
            #     max_linear_velocity=1000.0,
            #     max_depenetration_velocity=5.0,
            #     disable_gravity=False,
            # ),
            # mass_props=MassPropertiesCfg(mass=0.01),
            semantic_tags=[("class", "small_KLT")],
    
        )
    )
    # Required by ContactSensor so container contacts are available in net force reporting.
    small_KLT.spawn.activate_contact_sensors = True
    # /home/gbt/isaac_assets/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd
    tomato_soup_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/tomato_soup_can",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, -0.05, 0.03), rot=(0.70711, 0.0, 0.0, 0.70711) # 90-degree rotation around the Z axis
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            scale=(1.0, 1.0, 1.0),
            # rigid_props=RigidBodyPropertiesCfg(
            #     solver_position_iteration_count=16,
            #     solver_velocity_iteration_count=1,
            #     max_angular_velocity=1000.0,
            #     max_linear_velocity=1000.0,
            #     max_depenetration_velocity=5.0,
            #     disable_gravity=False,
            # ),
           
            semantic_tags=[("class", "tomato_soup_can")],
    
        )
    )
    # Required by ContactSensor so object contacts are available in net force reporting.
    tomato_soup_can.spawn.activate_contact_sensors = True

    
