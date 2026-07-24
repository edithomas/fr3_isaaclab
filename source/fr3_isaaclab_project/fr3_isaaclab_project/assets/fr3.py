"""Franka Research 3 (FR3) robot configuration.

Joint & body names VERIFIED against the actual FR3 USD from Nucleus:
  ISAAC_NUCLEUS_DIR/Robots/Franka/FR3/fr3.usd

Matches Isaac Lab v2.1.0 API (ArticulationCfg with sim_utils spawn settings).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ---------------------------------------------------------------------------
# FR3 USD asset path (VERIFIED on Nucleus, Isaac Sim 4.5.0)
# ---------------------------------------------------------------------------
FR3_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/FR3/fr3.usd"

# ---------------------------------------------------------------------------
# Joint name patterns – VERIFIED against the USD
# ---------------------------------------------------------------------------
FR3_ARM_JOINTS_REGEX = "fr3_joint.*"        # matches fr3_joint1..7 revolute DOFs
FR3_SHOULDER_PATTERN = "fr3_joint[1-4]"      # joints 1-4
FR3_FOREARM_PATTERN = "fr3_joint[5-7]"       # joints 5-7

# ---------------------------------------------------------------------------
# End-effector body name – VERIFIED against the USD
# ---------------------------------------------------------------------------
FR3_EE_BODY = "fr3_hand"

# ---------------------------------------------------------------------------
# Initial joint positions (conservative default pose)
# ---------------------------------------------------------------------------
FR3_INIT_JOINT_POS = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.569,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.810,
    "fr3_joint5": 0.0,
    "fr3_joint6": 3.037,
    "fr3_joint7": 0.741,
}

# ---------------------------------------------------------------------------
# FR3 Articulation configuration
# ---------------------------------------------------------------------------
FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FR3_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=FR3_INIT_JOINT_POS,
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[FR3_SHOULDER_PATTERN],
            effort_limit=87.0,
            velocity_limit=2.62,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=[FR3_FOREARM_PATTERN],
            effort_limit=12.0,
            velocity_limit=2.62,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
