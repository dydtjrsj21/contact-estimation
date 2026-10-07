"""Regenerate the time-series figures without the simulator.

    python scripts/replot.py outputs/vis_B/episode.npz            # a visualize_env.py run
    python scripts/replot.py data/pilot_v1 --episodes 0 5 17       # episodes of a collected dataset
"""

import argparse
import os

import numpy as np

from contact_est import dataset as D
from contact_est.render import plot_timeseries


def probe_delta(joint_target, grasp_posture, t):
    d = joint_target - grasp_posture[:, None]
    d[:, t < 0] = np.nan
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="episode.npz or dataset directory")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="dataset episode_ids to plot")
    parser.add_argument("--out", default=None)
    a = parser.parse_args()

    if a.path.endswith(".npz"):
        d = dict(np.load(a.path))
        out = a.out or os.path.dirname(a.path)
        names, presets, motions = d["objects"], d["presets"], d["motion"]
        joint_names, regions = d["joint_names"], d["region_names"]
        ids = range(len(names))
    else:
        meta = D.read_meta(a.path)
        d = D.load(a.path, include_dropped=True)
        d["probe_delta"] = probe_delta(d["joint_target"], d["grasp_posture"], d["t"])
        out = a.out or os.path.join(a.path, "report", "episodes")
        names, presets, motions = d["asset"], d["preset"], d["motion"]
        joint_names, regions = meta["joint_names"], meta["gt_links"]
        wanted = a.episodes if a.episodes is not None else d["episode_id"][:6]
        ids = [int(np.flatnonzero(d["episode_id"] == e)[0]) for e in wanted]
    os.makedirs(out, exist_ok=True)
    for i in ids:
        ep = {k: d[k][i] for k in ("probe_delta", "tip_force", "gt_contact")}
        eid = int(d["episode_id"][i]) if "episode_id" in d else i
        title = f"episode {eid}: {names[i]} / {presets[i]} grasp / motion {motions[i]}"
        plot_timeseries(d["t"], ep, joint_names, regions, motions[i], title, os.path.join(out, f"timeseries_{eid}.png"))
    print(f"wrote {len(ids)} figures to {out}")
