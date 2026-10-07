"""Scene / experiment configuration for the contact-estimation experiments.

Mapping to the experiment draft (contact_estimation_experiment_draft.md):

* Hand            : Allegro hand (4 fingers x 4 joints), wrist fixed, palm up.
* Estimator input : fingertip 3-axis force (one sensor per distal link), joint pos q, joint target u.
* Ground truth    : object contact of every hand link (constants.GT_LINKS); the draft's 4 regions
                    (constants.DRAFT_REGIONS) are a subset. Labels only, never estimator input.
* Vision          : one RGB-D camera per env (optional; excluded from the estimator for now).
* Objects         : see objects.py (shape families x sizes, identical material).
* Initial grasps  : fingertip, palm-deep, index-side presets (+ randomization in sampling.py).

All numbers are pilot defaults meant to be tuned, not validated values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

from .constants import (  # noqa: F401  (re-exported)
    DRAFT_REGIONS,
    FINGERS,
    FINGERTIP_BODIES,
    GT_CONTACT_OFF_N,
    GT_CONTACT_ON_N,
    GT_LINKS,
)
from .objects import ObjectAsset


def _object_spawn_cfg(assets: list[ObjectAsset]) -> sim_utils.MultiAssetSpawnerCfg:
    return sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[a.spawn for a in assets],
        # env i gets assets[i % n]; deterministic so envs can be labelled
        random_choice=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
    )


##
# Initial grasp presets
##


@dataclass
class GraspPreset:
    """Object placement + closing posture for one initial-grasp condition.

    Placement is in the *palm frame* (see ``ContactProbeEnv._compute_hand_frame``): x wrist -> fingers,
    y towards the ring finger, z palm normal (up). Origin: midway between the palm link and the knuckle
    line; the palm surface is at z ~ 0, the knuckles at x ~ 0.02-0.03, the curled thumb walls off x ~ -0.06.

    ``object_xy``: object center (x, y). ``clearance``: height of the object's lowest point above z = 0,
    so the center height follows from the object's size and orientation.
    ``posture`` maps joint-name regexes to target angles [rad] reached at the end of the closing phase.
    """

    name: str
    object_xy: tuple[float, float]
    clearance: float
    posture: dict[str, float] = field(default_factory=dict)


# Allegro joints: <f>_joint_0 = ab/adduction, _1 = MCP flex, _2 = PIP flex, _3 = DIP flex.
# Thumb: thumb_joint_0 = opposition rotation, _1.._3 = flexion chain.
OPEN_POSTURE = {
    r"(index|middle|ring)_joint_0": 0.0,
    r"(index|middle|ring)_joint_[1-3]": 0.0,
    r"thumb_joint_0": 1.0,
    r"thumb_joint_[1-3]": 0.1,
}

GRASP_PRESETS: tuple[GraspPreset, ...] = (
    GraspPreset(
        name="fingertip",
        object_xy=(0.025, 0.0),
        clearance=0.032,
        posture={
            r"(index|middle|ring)_joint_0": 0.0,
            r"(index|middle|ring)_joint_1": 0.35,
            r"(index|middle|ring)_joint_2": 0.75,
            r"(index|middle|ring)_joint_3": 0.75,
            r"thumb_joint_0": 1.30,
            r"thumb_joint_1": 0.25,
            r"thumb_joint_2": 0.55,
            r"thumb_joint_3": 0.75,
        },
    ),
    GraspPreset(
        name="palm_deep",
        object_xy=(-0.008, 0.0),
        clearance=0.015,
        posture={
            r"(index|middle|ring)_joint_0": 0.0,
            r"(index|middle|ring)_joint_1": 0.95,
            r"(index|middle|ring)_joint_2": 0.85,
            r"(index|middle|ring)_joint_3": 0.60,
            r"thumb_joint_0": 1.35,
            r"thumb_joint_1": 0.35,
            r"thumb_joint_2": 0.70,
            r"thumb_joint_3": 0.70,
        },
    ),
    GraspPreset(
        name="index_side",
        object_xy=(0.000, -0.020),
        clearance=0.020,
        posture={
            r"index_joint_0": -0.10,
            r"(middle|ring)_joint_0": 0.0,
            r"index_joint_1": 0.90,
            r"index_joint_2": 0.95,
            r"index_joint_3": 0.70,
            r"(middle|ring)_joint_1": 0.55,
            r"(middle|ring)_joint_2": 0.80,
            r"(middle|ring)_joint_3": 0.70,
            r"thumb_joint_0": 1.30,
            r"thumb_joint_1": 0.30,
            r"thumb_joint_2": 0.60,
            r"thumb_joint_3": 0.70,
        },
    ),
)
PRESET_NAMES = tuple(p.name for p in GRASP_PRESETS)

##
# Timing
##

PHYSICS_DT = 1.0 / 120.0
DECIMATION = 2  # -> 60 Hz control / logging rate (draft: 50-100 Hz)
PLACE_S = 0.2  # object held (pinned) above the open hand
CLOSE_S = 1.0  # interpolate open -> grasp posture, object still pinned
HOLD_S = 0.8  # object released; grasp settles under gravity before t = 0

CAMERA_RES = (240, 240)  # (H, W) per-env RGB-D (default)

##
# Scene
##

# spawn pose of the Allegro hand (Isaac Lab in-hand task, roughly palm up). At start-up the env re-orients
# the fixed hand so the palm is level (see ContactProbeEnv._level_hand); these only seed that step.
HAND_ROOT_POS = (0.0, 0.0, 0.5)
HAND_ROOT_ROT = (0.257551, 0.283045, 0.683330, -0.621782)
PALM_HEIGHT = 0.5  # world z of the palm-frame origin [m]
# measured by releasing spheres on the open palm (rolling accel. ~ 5/7 g sin(slope)); the Allegro palm is
# curved, these make its center roughly level so a released sphere stays put
PALM_TILT_DEG = -8.0  # >0 raises the fingertips
PALM_ROLL_DEG = 5.0  # >0 raises the ring-finger side


def _gt_sensor_cfg(link: str) -> ContactSensorCfg:
    # filtering against the object only works when the sensor covers exactly one body
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=DECIMATION,
        update_period=0.0,
    )


@configclass
class ContactProbeSceneCfg(InteractiveSceneCfg):
    """Allegro hand + one object per env + contact sensors (+ RGB-D camera). Built by make_scene_cfg."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.28)))
    dome_light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    )

    robot: ArticulationCfg = ALLEGRO_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ALLEGRO_HAND_CFG.spawn.replace(activate_contact_sensors=True),
        init_state=ALLEGRO_HAND_CFG.init_state.replace(pos=HAND_ROOT_POS, rot=HAND_ROOT_ROT),
    )
    object: RigidObjectCfg | None = None

    # "tactile" input: unfiltered net force on each distal link (what a fingertip F/T sensor would see)
    fingertip_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_link_3",
        history_length=DECIMATION,
        update_period=0.0,
    )

    camera: TiledCameraCfg | None = None


def make_scene_cfg(
    num_envs: int,
    assets: list[ObjectAsset],
    enable_camera: bool = True,
    camera_res: tuple[int, int] = CAMERA_RES,
    env_spacing: float = 1.0,
) -> ContactProbeSceneCfg:
    # replicate_physics=False is required for a different object per env
    cfg = ContactProbeSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=False)
    cfg.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=_object_spawn_cfg(assets),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.35, 0.1)),
    )
    # one filtered sensor per hand link -> attributes contact_gt_<link>
    for link in GT_LINKS:
        setattr(cfg, f"contact_gt_{link}", _gt_sensor_cfg(link))
    if enable_camera:
        cfg.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=0.4, horizontal_aperture=20.955, clipping_range=(0.02, 5.0)
            ),
            width=camera_res[1],
            height=camera_res[0],
        )
    return cfg


__all__ = [
    "DRAFT_REGIONS", "FINGERS", "FINGERTIP_BODIES", "GT_CONTACT_OFF_N", "GT_CONTACT_ON_N", "GT_LINKS",
    "GRASP_PRESETS", "OPEN_POSTURE", "GraspPreset", "ContactProbeSceneCfg", "make_scene_cfg",
]
