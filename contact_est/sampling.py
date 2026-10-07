"""Per-episode randomization: initial grasp preset, object position / orientation, posture, empty hand, motion.

Positions and orientations are in the palm frame (x wrist -> fingers, y towards the ring finger, z palm
normal). The object's height is set from its rotated collision vertices so that its lowest point sits
``preset.clearance`` (+ noise) above the palm, whatever its size and orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import config as C
from .probes import MOTIONS


@dataclass
class SamplingCfg:
    randomize: bool = True
    """False: nominal placement (preset xy, nominal orientation, no noise) -- used for visualization."""
    xy_noise: float = 0.010
    """object center x/y offset ~ U(-xy_noise, xy_noise) [m]"""
    clearance_noise: float = 0.006
    """extra height of the lowest point ~ U(0, clearance_noise) [m]"""
    orientation: str = "full"
    """'nominal' | 'yaw' (random rotation about the palm normal) | 'full' (uniform over SO(3))"""
    posture_noise: float = 0.05
    """grasp posture joint offsets ~ N(0, posture_noise) [rad]"""
    empty_frac: float = 0.05
    """fraction of empty-hand episodes (draft: probe without object to measure motion-induced bias)"""
    motions: tuple[str, ...] = ("A", "B")


# ---------------------------------------------------------------- quaternion helpers (w, x, y, z)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform over SO(3) (Shoemake)."""
    u1, u2, u3 = rng.random(3)
    a, b = np.sqrt(1 - u1), np.sqrt(u1)
    q = np.array([a * np.sin(2 * np.pi * u2), a * np.cos(2 * np.pi * u2), b * np.sin(2 * np.pi * u3),
                  b * np.cos(2 * np.pi * u3)])
    return np.array([q[3], q[0], q[1], q[2]])


def yaw_quat(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


# ---------------------------------------------------------------- sampler


def sample_episodes(
    env, rng: np.random.Generator, cfg: SamplingCfg, preset_idx: np.ndarray | None = None,
    motion: list[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Returns (episode tensors for ContactProbeEnv.reset, per-episode metadata as numpy arrays)."""
    n = env.num_envs
    n_presets = len(C.GRASP_PRESETS)
    if preset_idx is None:
        preset_idx = rng.integers(0, n_presets, n)
    if motion is None:
        motion = list(rng.choice(cfg.motions, n))

    pos = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    for i in range(n):
        asset = env.assets[env.env_asset[i]]
        preset = C.GRASP_PRESETS[preset_idx[i]]
        q = np.asarray(asset.nominal_quat, dtype=np.float64)
        if cfg.randomize and cfg.orientation == "yaw":
            q = quat_mul(yaw_quat(rng.uniform(-np.pi, np.pi)), q)
        elif cfg.randomize and cfg.orientation == "full":
            q = random_quat(rng)
        lowest = (asset.vertices @ quat_to_matrix(q).T)[:, 2].min()
        xy = np.asarray(preset.object_xy)
        clearance = preset.clearance
        if cfg.randomize:
            xy = xy + rng.uniform(-cfg.xy_noise, cfg.xy_noise, 2)
            clearance += rng.uniform(0.0, cfg.clearance_noise)
        pos[i] = [xy[0], xy[1], clearance - lowest]
        quat[i] = q

    posture = env.preset_postures[torch.as_tensor(preset_idx, device=env.device)].clone()
    if cfg.randomize and cfg.posture_noise > 0:
        noise = rng.normal(0.0, cfg.posture_noise, tuple(posture.shape))
        posture += torch.as_tensor(noise, dtype=posture.dtype, device=env.device)
    posture = env.clamp_joints(posture)
    empty = rng.random(n) < (cfg.empty_frac if cfg.randomize else 0.0)

    episode = dict(
        object_pos=torch.as_tensor(pos, dtype=torch.float32, device=env.device),
        object_quat=torch.as_tensor(quat, dtype=torch.float32, device=env.device),
        grasp_posture=posture,
        empty=torch.as_tensor(empty, device=env.device),
    )
    meta = dict(
        asset_idx=np.asarray(env.env_asset, dtype=np.int32),
        preset_idx=np.asarray(preset_idx, dtype=np.int32),
        motion=np.asarray(motion),
        empty=empty,
        init_object_pos=pos.astype(np.float32),
        init_object_quat=quat.astype(np.float32),
        grasp_posture=posture.cpu().numpy().astype(np.float32),
    )
    assert all(m in MOTIONS for m in motion)
    return episode, meta
