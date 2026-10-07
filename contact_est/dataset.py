"""On-disk dataset format (HDF5 shards + meta.json) and a loader. No Isaac imports.

Layout of ``<dataset>/``::

    meta.json            assets, presets, joint / link names, timing, palm frame, sampling + CLI config
    shard_00000.h5       one collection round = num_envs episodes
    shard_00001.h5 ...

Each shard holds, with E = episodes in the shard and T = time steps (grasp phase + probe, 60 Hz):

    time-series   t (T,)                         seconds; t < 0 is the grasp phase, t >= 0 the probe
                  tip_force     (E, T, 4, 3)     estimator input: fingertip force in link frame [N]
                  joint_pos     (E, T, J)        estimator input: measured joint angles [rad]
                  joint_target  (E, T, J)        estimator input: commanded joint targets [rad]
                  gt_force      (E, T, L)        LABEL: |force| hand link <-> object [N]
                  gt_contact    (E, T, L)        LABEL: contact after hysteresis (bool)
                  object_pos    (E, T, 3)        object position in the palm frame [m]
                  object_quat   (E, T, 4)        object orientation in the palm frame (w, x, y, z)
    per-episode   episode_id, asset_idx, preset_idx, motion, empty, dropped, drop_time (s, inf if held),
                  init_object_pos,
                  init_object_quat, grasp_posture (E, J), probe_start / probe_amp (E, 4)
    optional      rgb (E, Tv, H, W, 3) uint8, depth (E, Tv, H, W) float16 [m], vision_t (Tv,)

Joint order = meta["joint_names"], link order = meta["gt_links"], fingertip order = meta["fingertip_bodies"].
"""

from __future__ import annotations

import glob
import json
import os

import h5py
import numpy as np

SERIES = ("tip_force", "joint_pos", "joint_target", "gt_force", "gt_contact", "object_pos", "object_quat")
EPISODE_FIELDS = ("episode_id", "asset_idx", "preset_idx", "motion", "empty", "dropped", "drop_time",
                  "init_object_pos", "init_object_quat", "grasp_posture", "probe_start", "probe_amp")

DROP_DIST_M = 0.15  # object center farther than this from the palm-frame origin at the end -> dropped


def write_meta(path: str, meta: dict):
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def read_meta(path: str) -> dict:
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)


def write_shard(path: str, shard: int, series: dict[str, np.ndarray], episodes: dict[str, np.ndarray],
                vision: dict[str, np.ndarray] | None = None) -> str:
    fname = os.path.join(path, f"shard_{shard:05d}.h5")
    tmp = fname + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.attrs["shard"] = shard
        f.create_dataset("t", data=series["t"].astype(np.float32))
        for k in SERIES:
            v = series[k]
            v = v.astype(np.uint8) if v.dtype == bool else v.astype(np.float32)
            f.create_dataset(k, data=v, chunks=(1, *v.shape[1:]), compression="gzip", compression_opts=4)
        for k in EPISODE_FIELDS:
            v = episodes[k]
            if v.dtype.kind in "US":
                v = v.astype("S")
            f.create_dataset(k, data=v)
        if vision:
            f.create_dataset("vision_t", data=vision["t"].astype(np.float32))
            f.create_dataset("rgb", data=vision["rgb"], chunks=(1, 1, *vision["rgb"].shape[2:]),
                             compression="gzip", compression_opts=4)
            f.create_dataset("depth", data=vision["depth"].astype(np.float16),
                             chunks=(1, 1, *vision["depth"].shape[2:]), compression="gzip", compression_opts=4)
    os.replace(tmp, fname)  # atomic: a shard on disk is always complete
    return fname


def shard_files(path: str) -> list[str]:
    return sorted(glob.glob(os.path.join(path, "shard_*.h5")))


def load(path: str, fields=SERIES, include_dropped: bool = False, include_empty: bool = True) -> dict:
    """Concatenate all shards. Returns per-episode fields, the requested series and ``t``.

    Dropped episodes (object left the hand) are excluded unless ``include_dropped``.
    Strings (``motion``) are decoded; ``asset``/``family``/``split``/``preset`` name columns are added.
    """
    meta = read_meta(path)
    out: dict[str, list] = {k: [] for k in (*EPISODE_FIELDS, *fields)}
    t = None
    for fname in shard_files(path):
        with h5py.File(fname, "r") as f:
            keep = np.ones(len(f["episode_id"]), dtype=bool)
            if not include_dropped:
                keep &= ~f["dropped"][:].astype(bool)
            if not include_empty:
                keep &= ~f["empty"][:].astype(bool)
            idx = np.flatnonzero(keep)
            t = f["t"][:]
            for k in EPISODE_FIELDS:
                out[k].append(f[k][:][idx])
            for k in fields:
                out[k].append(f[k][idx] if len(idx) else np.zeros((0, *f[k].shape[1:]), f[k].dtype))
    data = {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
    data["t"] = t
    data["motion"] = data["motion"].astype(str)
    if "gt_contact" in data:
        data["gt_contact"] = data["gt_contact"].astype(bool)
    assets = meta["assets"]
    data["asset"] = np.array([assets[i]["name"] for i in data["asset_idx"]])
    data["family"] = np.array([assets[i]["family"] for i in data["asset_idx"]])
    data["split"] = np.array([assets[i]["split"] for i in data["asset_idx"]])
    data["preset"] = np.array([meta["presets"][i]["name"] for i in data["preset_idx"]])
    return data
