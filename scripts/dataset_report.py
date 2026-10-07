"""Summarise a collected dataset (no simulator needed).

    python scripts/dataset_report.py data/contact_v1

Writes <dataset>/report/: summary.json, drop_rate.png, contact_frequency.png, transitions_by_motion.png
and one example episode per object family (episodes/timeseries_<id>.png).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from contact_est import dataset as D  # noqa: E402
from contact_est.constants import DRAFT_REGIONS, REGION_SHORT  # noqa: E402
from contact_est.render import INK, INK_2, INK_3, SURFACE, plot_timeseries  # noqa: E402

MOTION_COLORS = {"A": "#2a78d6", "B": "#eb6834"}  # categorical slots 1-2


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def heatmap(ax, values, rows, cols, fmt="{:.0%}", vmax=None):
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax or max(values.max(), 1e-6), aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=0)
    ax.set_yticks(range(len(rows)), rows)
    hi = im.norm.vmax
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 * hi else INK)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--examples", type=int, default=6)
    a = p.parse_args()
    meta = D.read_meta(a.path)
    out = os.path.join(a.path, "report")
    os.makedirs(os.path.join(out, "episodes"), exist_ok=True)

    d = D.load(a.path, fields=("gt_contact", "tip_force", "joint_target"), include_dropped=True)
    links = meta["gt_links"]
    presets = [pr["name"] for pr in meta["presets"]]
    families = list(dict.fromkeys(x["family"] for x in meta["assets"]))
    probe = d["t"] >= 0
    held = ~d["dropped"] & ~d["empty"]
    n = len(d["episode_id"])

    # ---------------------------------------------------------------- summary numbers
    def rate(mask):
        m = mask & ~d["empty"]
        return float(d["dropped"][m].mean()) if m.any() else float("nan")

    contact = d["gt_contact"][:, probe]  # (E, Tp, L)
    flips = (np.diff(contact.astype(np.int8), axis=1) != 0).sum(1)  # (E, L)
    summary = dict(
        episodes=n,
        held=int(held.sum()),
        dropped=int(d["dropped"].sum()),
        empty=int(d["empty"].sum()),
        held_hours=float(held.sum() * len(d["t"]) * meta["dt"] / 3600),
        drop_rate_by_family={f: rate(d["family"] == f) for f in families},
        drop_rate_by_preset={pr: rate(d["preset"] == pr) for pr in presets},
        drop_rate_by_scale={str(s): rate(np.isclose([meta["assets"][i]["scale"] for i in d["asset_idx"]], s))
                            for s in sorted({x["scale"] for x in meta["assets"]})},
        held_by_split={s: int((held & (d["split"] == s)).sum()) for s in np.unique(d["split"])},
        draft_region_positive_frac={r: float(contact[held][..., links.index(r)].mean()) for r in DRAFT_REGIONS},
        transitions_per_episode_by_motion={
            m: float(flips[held & (d["motion"] == m)].sum(1).mean()) for m in np.unique(d["motion"])},
        empty_episode_tip_force_mean_n=float(np.linalg.norm(d["tip_force"][d["empty"]][:, probe], axis=-1).mean())
        if d["empty"].any() else None,
    )
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    # ---------------------------------------------------------------- drop rate: family x preset
    vals = np.array([[rate((d["family"] == f) & (d["preset"] == pr)) for pr in presets] for f in families])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    fig.patch.set_facecolor(SURFACE)
    heatmap(ax, vals, families, presets, vmax=1.0)
    style(ax, "Drop rate (object left the hand) by family x initial grasp")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "drop_rate.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # ---------------------------------------------------------------- contact frequency: family x link
    freq = np.array([contact[held & (d["family"] == f)].mean((0, 1)) for f in families])
    fig, ax = plt.subplots(figsize=(12, 3.8))
    fig.patch.set_facecolor(SURFACE)
    heatmap(ax, freq, families, [REGION_SHORT.get(x, x) for x in links], fmt="{:.2f}")
    for lbl, x in zip(ax.get_xticklabels(), links):
        lbl.set_color(INK if x in DRAFT_REGIONS else INK_3)
    style(ax, "Fraction of probe time in contact with the object (held episodes; dark labels = draft regions)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "contact_frequency.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # ---------------------------------------------------------------- transitions per link, motion A vs B
    motions = [m for m in ("A", "B") if (d["motion"] == m).any()]
    fig, ax = plt.subplots(figsize=(12, 3.6))
    fig.patch.set_facecolor(SURFACE)
    w = 0.38
    x = np.arange(len(links))
    for k, m in enumerate(motions):
        v = flips[held & (d["motion"] == m)].mean(0)
        ax.bar(x + (k - (len(motions) - 1) / 2) * (w + 0.02), v, width=w, color=MOTION_COLORS[m],
               label=f"motion {m} ({'whole hand' if m == 'A' else 'per finger'})")
    ax.set_xticks(x, [REGION_SHORT.get(lk, lk) for lk in links])
    for lbl, lk in zip(ax.get_xticklabels(), links):
        lbl.set_color(INK if lk in DRAFT_REGIONS else INK_3)
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("contact on/off transitions\nper episode (probe)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    style(ax, "How often each link's contact changes during probing")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "transitions_by_motion.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # ---------------------------------------------------------------- example episodes (one per family)
    rng = np.random.default_rng(0)
    d["probe_delta"] = d["joint_target"] - d["grasp_posture"][:, None]
    d["probe_delta"][:, ~probe] = np.nan
    picks = []
    for f in families:
        idx = np.flatnonzero(held & (d["family"] == f))
        if len(idx):
            picks.append(int(rng.choice(idx)))
    for i in picks[: a.examples]:
        ep = {k: d[k][i] for k in ("probe_delta", "tip_force", "gt_contact")}
        q = ", ".join(f"{v:+.2f}" for v in d["init_object_quat"][i])
        title = f"episode {d['episode_id'][i]}: {d['asset'][i]} / {d['preset'][i]} / motion {d['motion'][i]} / quat ({q})"
        plot_timeseries(d["t"], ep, meta["joint_names"], links, d["motion"][i], title,
                        os.path.join(out, "episodes", f"timeseries_{d['episode_id'][i]}.png"))
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
