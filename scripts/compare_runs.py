"""Compare several train_baselines.py runs on VALIDATION results only (test sets stay untouched for the final run).

    python scripts/compare_runs.py outputs/sweep/*/ --out outputs/sweep

Writes <out>/comparison.md and three figures:
    comparison_best_vs_final.png   best val AP (checkpoint) vs. val AP at the last step, per run and model
    comparison_curves.png          clean-train vs. val AP over training, for the baseline and the best run
    comparison_data_size.png       best val AP vs. fraction of train episodes (runs that differ only in it)
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from contact_est.render import FINGER_COLORS, INK, INK_2, MODEL_COLORS, SURFACE  # noqa: E402

REG_KEYS = ("weight_decay", "dropout", "force_noise", "force_bias", "joint_noise", "posture_shift", "lstm_hidden",
            "head_hidden", "mlp_hidden", "force", "window_s", "steps")
PAIR_COLORS = FINGER_COLORS[:2]  # two highlighted runs in the curve figure (categorical slots 1-2)


def load(path: str) -> dict:
    cfg = json.load(open(os.path.join(path, "config.json")))
    res = json.load(open(os.path.join(path, "metrics.json")))
    name = os.path.basename(os.path.normpath(path))
    models = sorted({r["model"] for r in res})
    per = {}
    for m in models:
        rs = sorted([r for r in res if r["model"] == m], key=lambda r: r["seed"])
        steps = [h["step"] for h in rs[0]["history"]]
        curve = lambda key: np.mean([[h[key] for h in r["history"]] for r in rs], 0)  # noqa: E731
        per[m] = dict(
            best=[r["best_val_ap"] for r in rs], best_step=[r["best_step"] for r in rs], steps=steps,
            val_ap=curve("val_ap"), train_ap=curve("train_clean_ap"), val_loss=curve("val_loss"),
            train_loss=curve("train_clean_loss"),
        )
    return dict(name=name, cfg=cfg, models=per)


def describe(cfg: dict) -> str:
    parts = []
    for key, unit in (("dropout", ""), ("weight_decay", ""), ("force_noise", " N"), ("force_bias", " N"),
                      ("joint_noise", " rad"), ("posture_shift", " rad")):
        if cfg.get(key) and not (key == "weight_decay" and cfg[key] == 1e-4):  # 1e-4 = default
            parts.append(f"{key}={cfg[key]:g}{unit}")
    if cfg.get("lstm_hidden", 64) != 64:
        parts.append(f"lstm={cfg['lstm_hidden']}")
    if cfg.get("train_frac", 1) < 1:
        parts.append(f"data={cfg['train_frac']:.0%}")
    return ", ".join(parts) or "none"


def ms(v) -> str:
    v = np.asarray(v, float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def style(ax, title=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d6d0")
    ax.grid(axis="x", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", default="r0_none", help="run name the others are compared with")
    a = p.parse_args()
    runs = [load(r) for r in a.runs if os.path.exists(os.path.join(r, "metrics.json"))]
    runs.sort(key=lambda r: r["name"])
    models = sorted({m for r in runs for m in r["models"]})
    os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------------------------ table
    md = ["# Run comparison (validation only)\n",
          "best = val macro AP at the kept checkpoint (mean ± std over seeds) · step = checkpoint steps · "
          "final = val AP at the last step · train = clean train-frame AP at the last step. A large best − final "
          "drop or train ≫ val means memorisation.\n"]
    for m in models:
        md += [f"## Model {m}\n", "| run | settings | best | step | final | train (final) | train − val (final) |",
               "|---|---|---|---|---|---|---|"]
        for r in runs:
            if m not in r["models"]:
                continue
            x = r["models"][m]
            md.append(f"| {r['name']} | {describe(r['cfg'])} | {ms(x['best'])} | {x['best_step']} | "
                      f"{x['val_ap'][-1]:.3f} | {x['train_ap'][-1]:.3f} | {x['train_ap'][-1] - x['val_ap'][-1]:.3f} |")
        md.append("")
    full = [r for r in runs if r["cfg"].get("train_frac", 1) >= 1]
    best_run = {}
    for m in models:
        cands = [r for r in full if m in r["models"]] or [r for r in runs if m in r["models"]]
        best_run[m] = max(cands, key=lambda r: np.mean(r["models"][m]["best"]))
    md.append("Best full-data run by val AP: " + ", ".join(f"{m}: `{best_run[m]['name']}`" for m in models))
    with open(os.path.join(a.out, "comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # ------------------------------------------------------------------ best vs final (dot plot)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 0.42 * len(runs) + 1.6), sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(runs))[::-1]
    for ax, m in zip(axes[0], models):
        c = MODEL_COLORS.get(m, INK_2)
        for yi, r in zip(y, runs):
            if m not in r["models"]:
                continue
            x = r["models"][m]
            b, s, fin = np.mean(x["best"]), np.std(x["best"], ddof=1) if len(x["best"]) > 1 else 0, x["val_ap"][-1]
            ax.plot([fin, b], [yi, yi], color="#d8d6d0", lw=2, zorder=1)
            ax.errorbar(b, yi, xerr=s, fmt="o", ms=8, color=c, mec=SURFACE, mew=2, ecolor=INK_2, elinewidth=1,
                        zorder=3)
            ax.plot(fin, yi, "o", ms=8, mfc=SURFACE, mec=c, mew=2, zorder=2)
            ax.text(b + max(s, 0.004) + 0.004, yi, f"{b:.3f}", va="center", fontsize=7, color=INK_2)
        style(ax, f"Model {m}")
        ax.set_xlabel("validation macro AP", color=INK_2, fontsize=9)
    axes[0][0].set_yticks(y, [r["name"] for r in runs], fontsize=8)
    fig.suptitle("Validation macro AP: ● best checkpoint, ○ last step (settings: comparison.md)", x=0.01,
                 ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "comparison_best_vs_final.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------------------------ curves: baseline vs best
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    base = next((r for r in runs if r["name"] == a.baseline), None)
    for ax, m in zip(axes[0], models):
        shown = [r for r in (base, best_run[m]) if r is not None and m in r["models"]]
        shown = list({r["name"]: r for r in shown}.values())
        for color, r in zip(PAIR_COLORS, shown):
            x = r["models"][m]
            ax.plot(x["steps"], x["train_ap"], color=color, lw=2, ls="--")
            ax.plot(x["steps"], x["val_ap"], color=color, lw=2, label=r["name"])
        style(ax, f"Model {m}")
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
        ax.set_xlabel("training step", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center right")
    axes[0][0].set_ylabel("macro AP", color=INK_2, fontsize=9)
    fig.suptitle("Clean train frames (dashed) vs. validation (solid)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "comparison_curves.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------------------------ data size (learning curve)
    groups: dict[str, list] = {}
    for r in runs:
        key = describe({k: v for k, v in r["cfg"].items() if k != "train_frac"})
        groups.setdefault(key, []).append(r)
    groups = {k: sorted(v, key=lambda r: r["cfg"].get("train_frac", 1)) for k, v in groups.items()
              if len({r["cfg"].get("train_frac", 1) for r in v}) > 1}
    if groups:
        fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.6), sharey=True, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax, m in zip(axes[0], models):
            for color, (key, rs) in zip(PAIR_COLORS, groups.items()):
                rs = [r for r in rs if m in r["models"]]
                if len(rs) < 2:
                    continue
                fr = [r["cfg"].get("train_frac", 1) for r in rs]
                v = [np.mean(r["models"][m]["best"]) for r in rs]
                e = [np.std(r["models"][m]["best"], ddof=1) for r in rs]
                full = next((r["name"] for r in rs if r["cfg"].get("train_frac", 1) >= 1), key)
                ax.errorbar(fr, v, yerr=e, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, capsize=0,
                            label=f"settings of {full}")
                ax.text(fr[-1], v[-1], f"  {v[-1]:.3f}", color=INK_2, fontsize=7, va="center")
            ticks = sorted({r["cfg"].get("train_frac", 1) for rs in groups.values() for r in rs})
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
            style(ax, f"Model {m}")
            ax.grid(axis="y", color="#ecebe7", lw=0.8)
            ax.set_xlabel("fraction of the train episodes", color=INK_2, fontsize=9)
            ax.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="lower right")
        axes[0][0].set_ylabel("best val macro AP", color=INK_2, fontsize=9)
        fig.suptitle("More train episodes → better? (same settings, fewer episodes)", x=0.01, ha="left", color=INK,
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "comparison_data_size.png"), dpi=120, facecolor=SURFACE)
        plt.close(fig)
    print(f"wrote {a.out}/comparison.md and figures")


if __name__ == "__main__":
    main()
