"""Object library: shape families x sizes, with spawn configs and geometry for placement.

Every object has the same material (density / friction / color); only shape and size change, as the
draft asks. Each asset carries the vertices of its collision shape in its local frame so the sampler can
compute how low it reaches after an arbitrary rotation (to place it just above the palm).

Suggested evaluation splits (recorded as metadata only; nothing is excluded from collection):
    heldout_size  : scales 0.925 and 1.075 of the seen families
    heldout_shape : every ellipsoid
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

DENSITY = 300.0  # kg/m^3 (light plastic / wood); same for every object
FRICTION = (1.0, 0.8)  # static, dynamic
COLOR = (0.85, 0.45, 0.15)

SCALES = (0.85, 0.925, 1.0, 1.075, 1.15)
HELDOUT_SCALES = (0.925, 1.075)
HELDOUT_FAMILIES = ("ellipsoid",)

# nominal dimensions [m] at scale 1.0 (graspable width ~7-8 cm; the Allegro hand is ~1.5x a human hand)
NOMINAL = {
    "sphere": dict(radius=0.040),
    "cylinder": dict(radius=0.035, height=0.10),
    "box": dict(size=(0.07, 0.09, 0.07)),
    "capsule": dict(radius=0.030, height=0.06),  # height = straight part
    "rounded_box": dict(size=(0.07, 0.09, 0.07), corner_radius=0.012),
    "ellipsoid": dict(semi_axes=(0.045, 0.035, 0.030)),
}
FAMILIES = tuple(NOMINAL)

# rotation (w, x, y, z) taking the asset frame to the palm frame when orientation is not randomized:
# long axes across the palm (palm-frame y), like a handle in a power grasp
_Z_TO_Y = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0)
NOMINAL_QUAT = {
    "sphere": (1.0, 0.0, 0.0, 0.0),
    "cylinder": _Z_TO_Y,
    "box": (1.0, 0.0, 0.0, 0.0),
    "capsule": _Z_TO_Y,
    "rounded_box": (1.0, 0.0, 0.0, 0.0),
    "ellipsoid": (0.7071068, 0.0, 0.0, 0.7071068),  # long semi-axis across the palm
}


##
# Custom mesh spawner (convex-hull collision from a trimesh)
##


def _make_mesh(kind: str, dims: dict) -> trimesh.Trimesh:
    if kind == "rounded_box":
        half = np.asarray(dims["size"]) / 2 - dims["corner_radius"]
        ball = trimesh.creation.icosphere(subdivisions=2, radius=dims["corner_radius"])
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
        pts = np.concatenate([ball.vertices + c for c in corners])
        return trimesh.convex.convex_hull(pts)
    if kind == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(dims["semi_axes"])
        return mesh
    raise ValueError(kind)


@clone
def spawn_trimesh(prim_path, cfg: "TrimeshCfg", translation=None, orientation=None):
    _spawn_mesh_geom_from_mesh(prim_path, cfg, _make_mesh(cfg.kind, cfg.dims), translation, orientation)
    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TrimeshCfg(sim_utils.MeshCfg):
    """Rigid convex mesh generated procedurally (collision approximation: convex hull)."""

    func: Callable = spawn_trimesh
    kind: str = "rounded_box"
    dims: dict = None


##
# Library
##


@dataclass
class ObjectAsset:
    name: str
    family: str
    scale: float
    dims: dict
    split: str
    mass: float  # [kg], from the collision-shape volume and DENSITY
    vertices: np.ndarray  # (V, 3) collision-shape vertices in the asset frame
    nominal_quat: tuple[float, float, float, float]
    spawn: sim_utils.SpawnerCfg

    def meta(self) -> dict:
        dims = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.dims.items()}
        return dict(name=self.name, family=self.family, scale=self.scale, dims=dims, split=self.split,
                    mass=self.mass, nominal_quat=list(self.nominal_quat))


def _scaled(dims: dict, s: float) -> dict:
    return {k: (tuple(x * s for x in v) if isinstance(v, tuple) else v * s) for k, v in dims.items()}


def _geometry(family: str, d: dict) -> tuple[trimesh.Trimesh, sim_utils.SpawnerCfg]:
    common = dict(
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLOR, roughness=0.6),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION[0], dynamic_friction=FRICTION[1]),
        mass_props=sim_utils.MassPropertiesCfg(density=DENSITY),
    )
    if family == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=d["radius"]), sim_utils.SphereCfg(
            radius=d["radius"], **common)
    if family == "cylinder":
        return trimesh.creation.cylinder(radius=d["radius"], height=d["height"], sections=48), sim_utils.CylinderCfg(
            radius=d["radius"], height=d["height"], axis="Z", **common)
    if family == "box":
        return trimesh.creation.box(extents=d["size"]), sim_utils.CuboidCfg(size=d["size"], **common)
    if family == "capsule":
        return trimesh.creation.capsule(height=d["height"], radius=d["radius"], count=[24, 24]), sim_utils.CapsuleCfg(
            radius=d["radius"], height=d["height"], axis="Z", **common)
    mesh = _make_mesh(family, d)
    return mesh, TrimeshCfg(kind=family, dims=d, **common)


def make_asset(family: str, scale: float) -> ObjectAsset:
    d = _scaled(NOMINAL[family], scale)
    mesh, spawn = _geometry(family, d)
    if family in HELDOUT_FAMILIES:
        split = "heldout_shape"
    elif any(abs(scale - h) < 1e-6 for h in HELDOUT_SCALES):
        split = "heldout_size"
    else:
        split = "train"
    return ObjectAsset(
        name=f"{family}_s{scale:.3f}",
        family=family,
        scale=scale,
        dims=d,
        split=split,
        mass=float(mesh.volume * DENSITY),
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        nominal_quat=NOMINAL_QUAT[family],
        spawn=spawn,
    )


def build_library(families=FAMILIES, scales=SCALES) -> list[ObjectAsset]:
    return [make_asset(f, s) for f in families for s in scales]


PILOT_LIBRARY_ARGS = dict(families=("sphere", "cylinder", "box"), scales=(1.0,))
"""The 3 nominal objects of the first pilot / visualize_env.py."""
