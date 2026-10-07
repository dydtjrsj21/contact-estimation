"""Data-scaling figure for scripts/data_scaling.sh: macro F1 vs. number of episodes, on contact_v1's own test /
held-out episodes (the same episodes in every run, because every dataset keeps its own split).

    python scripts/plot_scaling.py outputs/scaling      # -> outputs/scaling/scaling.png + scaling.md
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from contact_est.render import INK, INK_2, MODEL_COLORS, SURFACE  # noqa: E402

SETS = ("test", "heldout_size", "heldout_shape")
LABELS = {"A": "A: MLP q,u", "B": "B: MLP fn,q,u", "C": "C: LSTM+MLP q,u", "D": "D: LSTM+MLP fn,q,u"}


def v1_macro(metrics: dict, s: str, key: str) -> float:
    x = metrics[s]
    if "dataset" in x["by"]:
        x = x["by"]["dataset"]["contact_v1"]
    return x["macro"][key]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    a = p.parse_args()
    runs = sorted((d for d in os.listdir(a.root) if d.startswith("x") and os.path.exists(
        os.path.join(a.root, d, "metrics.json"))), key=lambda d: int(d[1:]))
    table = {}  # (model, factor) -> {set: [per seed]}
    for d in runs:
        factor = int(d[1:])
        for r in json.load(open(os.path.join(a.root, d, "metrics.json"))):
            cell = table.setdefault((r["model"], factor), {s: [] for s in SETS})
            for s in SETS:
                cell[s].append(v1_macro(r["metrics"], s, "f1"))
    models = sorted({m for m, _ in table})
    factors = sorted({f for _, f in table})

    fig, axes = plt.subplots(1, len(SETS), figsize=(4.4 * len(SETS), 3.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes, SETS):
        ends = []
        for m in models:
            fs = [f for f in factors if (m, f) in table]
            v = [np.mean(table[m, f][s]) for f in fs]
            e = [np.std(table[m, f][s], ddof=1) for f in fs]
            ax.errorbar(fs, v, yerr=e, color=MODEL_COLORS[m], lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                        capsize=0, label=LABELS[m])
            ends.append([v[-1], m, fs[-1]])
        ends.sort()
        for k in range(1, len(ends)):  # keep end labels at least 0.012 apart
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.012)
        for y, m, f in ends:
            ax.text(f * 1.08, y, m, color=INK_2, fontsize=8, va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(factors, [f"{f}×\n{7200 * f:,}" for f in factors])
        ax.set_facecolor(SURFACE)
        ax.set_title(f"contact_v1 {s} episodes", loc="left", color=INK, fontsize=10)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#d8d6d0")
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
        ax.set_xlabel("episodes collected", color=INK_2, fontsize=9)
    axes[0].set_ylabel("macro F1 (mean ± std, 3 seeds)", color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.suptitle("More data: every model improves, the history models (C, D) the most", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.root, "scaling.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)

    md = ["# Data scaling: macro F1 on contact_v1's own episodes (mean ± std over 3 seeds)\n",
          "| model | episodes | " + " | ".join(SETS) + " |", "|---|---|" + "---|" * len(SETS)]
    for m in models:
        for f in factors:
            if (m, f) in table:
                md.append(f"| {m} | {f}× ({7200 * f:,}) | " + " | ".join(
                    f"{np.mean(table[m, f][s]):.3f} ± {np.std(table[m, f][s], ddof=1):.3f}" for s in SETS) + " |")
    with open(os.path.join(a.root, "scaling.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
