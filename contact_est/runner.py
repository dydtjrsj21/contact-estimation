"""One episode = grasp phase (env.reset) + 10 s scripted probe; shared by visualization and collection."""

from __future__ import annotations

import numpy as np
import torch

from .dataset import DROP_DIST_M
from .probes import DURATION_S, ProbeMotion


def run_episode(env, episode: dict, motions: list[str], jitter: bool = True, render_every: int = 1):
    """Returns (frames, probe). ``render_every``: render cameras every k-th control step (0: never).

    The step counter runs over the whole episode, so frames with ``k % render_every == 0`` carry fresh
    images (k = index into the returned frame list).
    """
    frames = env.reset(episode, render_every=render_every)
    n_grasp = len(frames)
    probe = ProbeMotion(motions, env.joint_names, env.num_envs, env.device, jitter=jitter)
    base = env.grasp_posture.clone()
    for k in range(round(DURATION_S / env.dt)):
        t = k * env.dt
        u = env.clamp_joints(base + probe.delta(t))
        render = render_every > 0 and (n_grasp + k) % render_every == 0
        frames.append(env.step(u, t=t, render=render))
    return frames, probe


def stack_frames(frames: list[dict]) -> tuple[dict[str, np.ndarray], dict | None]:
    """List of per-step observations -> numpy arrays shaped (env, time, ...), plus the vision frames
    (only the steps that rendered carry images)."""
    series = {"t": np.asarray([f["t"] for f in frames], dtype=np.float32)}
    for k in frames[0]:
        if k in ("t", "rgb", "depth"):
            continue
        series[k] = torch.stack([f[k] for f in frames], 1).cpu().numpy()
    vision = None
    idx = [i for i, f in enumerate(frames) if "rgb" in f]
    if idx:
        vision = {
            "t": series["t"][idx],
            "rgb": torch.stack([frames[i]["rgb"] for i in idx], 1).cpu().numpy().astype(np.uint8),
            "depth": torch.stack([frames[i]["depth"] for i in idx], 1).to(torch.float16).cpu().numpy(),
        }
    return series, vision


def dropped_mask(series: dict, empty: np.ndarray) -> np.ndarray:
    """Object ended far from the palm (not counted for empty-hand episodes)."""
    dist = np.linalg.norm(series["object_pos"][:, -1], axis=-1)
    return (dist > DROP_DIST_M) & ~empty


def drop_time(series: dict, empty: np.ndarray) -> np.ndarray:
    """First time the object is farther than DROP_DIST_M from the palm (inf if never / empty hand).
    Data before this time is still usable for episodes that drop late in the probe."""
    far = np.linalg.norm(series["object_pos"], axis=-1) > DROP_DIST_M  # (E, T)
    t = np.where(far.any(1), series["t"][far.argmax(1)], np.inf).astype(np.float32)
    t[empty] = np.inf
    return t
