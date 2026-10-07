"""Train and evaluate the draft's baselines A-D (sections 7-9) on a collected dataset, on every idle GPU.

    python scripts/train_baselines.py data/contact_v1                           # A-D x seeds 0 1 2 (~8 min)
    python scripts/train_baselines.py data/contact_v1 --holdout_preset index_side   # + unseen initial grasp
    python scripts/train_baselines.py data/contact_v1 --force 3axis             # link-frame force vector
    python scripts/train_baselines.py data/contact_v1 --quick                   # smoke test (< 1 min)
    python scripts/train_baselines.py data/contact_v1 data/contact_v1_s*        # several datasets combined

No simulator needed: run it with the ``contactsim`` python directly (not through run.sh). Every
(model, seed) job runs in its own worker process pinned to one GPU; ``--gpus auto`` takes every GPU with
at most ``--max_used_mib`` memory in use and no load. The dataset is prepared once in this process and
shared with the workers (fork, copy-on-write); this process never initialises CUDA.

    A  MLP       current q, u           B  MLP       current f, q, u
    C  LSTM+MLP  last 1 s of q, u       D  LSTM+MLP  last 1 s of f, q, u

f is the fingertip force: by default (``--force normal``) one normal-force magnitude per fingertip (4 values);
``--force 3axis`` uses the link-frame vector (12 values). The simulator's fingertip force contains only normal
contact force (Isaac Lab ``net_forces_w``), so the 3 axes add the contact-normal direction, not friction.

Default regularisation (chosen on val, scripts/sweep_regularization.sh): per-window joint offset 0.06 rad on
q and u, fingertip force noise / bias 0.1 N, joint noise 0.005 rad, dropout 0.3, weight decay 0.01, 40k steps.

Split (by episode, never by window): objects of the dataset's ``train`` split -> train / val / test;
``heldout_size`` and ``heldout_shape`` are extra test sets. Episodes whose object fell out are used up to
``drop_time``. Thresholds are tuned per region on val (best F1) and then applied to every test set.

Writes <out>/: config.json, metrics.json, summary.md, f1_by_region.png, f1_by_motion.png, val_curves.png,
models/<model>_s<seed>.pt and episodes/pred_<set>_ep<id>.png.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from contact_est import dataset as D  # noqa: E402
from contact_est import learning as L  # noqa: E402
from contact_est import metrics as M  # noqa: E402
from contact_est.constants import REGION_SHORT  # noqa: E402
from contact_est.render import INK, INK_2, MODEL_COLORS, SURFACE, plot_predictions  # noqa: E402



def model_labels(force: str) -> dict[str, str]:
    f = {"normal": "fn", "3axis": "f"}[force]
    return {"A": "A: MLP q,u", "B": f"B: MLP {f},q,u", "C": "C: LSTM+MLP q,u", "D": f"D: LSTM+MLP {f},q,u"}

FIGURE_MODELS = ("C", "D")  # overlaid in the episode figures: history without vs. with force (adjacent hues)
GROUPS = ("motion", "family", "preset")

# shared with forked workers
_P: L.Prepared | None = None
_ARGS = None
_EXAMPLES: np.ndarray | None = None
# per worker
_DEVICE = None
_X = _Y = None


# ---------------------------------------------------------------------------------------------------- GPUs
def idle_gpus(max_used_mib: int, max_util: int = 10) -> list[int]:
    """Indices (as torch sees them) of GPUs with little memory in use and no load, honouring
    CUDA_VISIBLE_DEVICES. Uses nvidia-smi so this process never touches CUDA."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    idle = [int(i) for i, mem, util in (map(str.strip, ln.split(",")) for ln in out.strip().splitlines())
            if int(mem) <= max_used_mib and int(util) <= max_util]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        phys = [int(v) for v in visible.split(",") if v.strip()]
        return [phys.index(g) for g in idle if g in phys]
    return idle


def _init_worker(gpu_queue):
    global _DEVICE
    _DEVICE = torch.device(f"cuda:{gpu_queue.get()}")
    torch.cuda.set_device(_DEVICE)


# ---------------------------------------------------------------------------------------------------- one job
def run_job(name: str, seed: int) -> dict:
    global _X, _Y
    t0 = time.time()
    P, a = _P, _ARGS
    if _X is None:  # once per worker
        _X = torch.as_tensor(P.x, device=_DEVICE)
        _Y = torch.as_tensor(P.y, device=_DEVICE)
    tag = f"{name}_s{seed} cuda:{_DEVICE.index}"

    def log(msg):
        print(f"[{tag}] {msg}", flush=True)

    torch.manual_seed(seed)
    cfg = L.TrainConfig(steps=a.steps, batch=a.batch, lr=a.lr, weight_decay=a.weight_decay, eval_every=a.eval_every,
                        window_s=a.window_s, select=a.select, train_frac=a.train_frac, force_noise=a.force_noise,
                        force_bias=a.force_bias, joint_noise=a.joint_noise, posture_shift=a.posture_shift)
    est = L.Estimator(name, len(P.regions), round(a.window_s / P.dt), a.force, a.mlp_hidden, a.lstm_hidden,
                      a.head_hidden, a.dropout).to(_DEVICE)
    fit = L.train(est, _X, _Y, P, cfg, seed, log)

    val = np.flatnonzero(P.sets == "val")
    thr = M.best_thresholds(L.predict(est, _X, P.valid, val), P.y[val].astype(bool), P.valid[val])
    metrics = {}
    for s in L.EVAL_SETS:
        idx = np.flatnonzero(P.sets == s)
        if not len(idx):
            continue
        prob = L.predict(est, _X, P.valid, idx)
        groups = GROUPS + (("dataset",) if len(np.unique(P.info["dataset"])) > 1 else ())
        metrics[s] = M.evaluate(prob, P.y[idx].astype(bool), P.valid[idx], thr, P.regions, P.dt,
                                {g: P.info[g][idx] for g in groups})
        log(f"{s:14s} macro F1 {metrics[s]['macro']['f1']:.3f}  AP {metrics[s]['macro']['ap']:.3f}")

    torch.save(dict(model=name, seed=seed, spec=L.MODELS[name], force=a.force, inputs=est.inputs,
                    input_columns={g: (c.start, c.stop) for g, c in L.INPUT_COLUMNS.items()}, state_dict=est.state_dict(), window=est.window,
                    thresholds=thr, mean=P.mean, std=P.std, regions=P.regions, train_config=asdict(cfg),
                    hidden=dict(mlp=a.mlp_hidden, lstm=a.lstm_hidden, head=a.head_hidden), dropout=a.dropout),
               os.path.join(a.out, "models", f"{name}_s{seed}.pt"))
    examples = L.predict(est, _X, P.valid, _EXAMPLES) if seed == a.seeds[0] else None
    log(f"done in {time.time() - t0:.0f} s")
    return dict(model=name, seed=seed, gpu=_DEVICE.index, seconds=time.time() - t0, thresholds=thr.tolist(),
                metrics=metrics, examples=examples, **fit)


# ---------------------------------------------------------------------------------------------------- helpers
def pick_examples(P: L.Prepared, n: int, seed: int) -> np.ndarray:
    """Episodes for the prediction figures: round-robin over (eval set, motion), each with at least two
    contact changes in the target regions; plus one episode that drops mid-probe, if there is one."""
    rng = np.random.default_rng(seed)
    flips = np.zeros(len(P.sets), int)
    for e in range(len(P.sets)):
        g = P.y[e][P.valid[e]]
        flips[e] = int((np.diff(g.astype(np.int8), axis=0) != 0).sum())
    pools = []
    for s in L.EVAL_SETS:
        for m in ("B", "A"):
            idx = np.flatnonzero((P.sets == s) & (P.info["motion"] == m) & ~P.info["empty"] & (flips >= 2))
            if len(idx):
                pools.append(list(rng.permutation(idx)))
    picked = []
    late_drop = np.flatnonzero(np.isin(P.sets, L.EVAL_SETS) & P.info["dropped"] & (P.valid.sum(1) * P.dt >= 3))
    if len(late_drop) and n > 1:
        picked.append(int(rng.choice(late_drop)))
    while len(picked) < n and any(pools):
        for pool in pools:
            if pool and len(picked) < n:
                picked.append(int(pool.pop()))
    return np.asarray(picked, dtype=np.int64)


def mean_std(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0


def fmt(values, digits=3, scale=1.0) -> str:
    v = np.asarray(values, dtype=float) * scale
    if not np.isfinite(v).any():
        return "–"
    m, s = mean_std(v)
    return f"{m:.{digits}f} ± {s:.{digits}f}" if len(v) > 1 else f"{m:.{digits}f}"


def regularisation(a) -> str:
    parts = [f"weight decay {a.weight_decay:g}"]
    if a.dropout:
        parts.append(f"dropout {a.dropout:g}")
    for key, unit in (("force_noise", "N"), ("force_bias", "N"), ("joint_noise", "rad"), ("posture_shift", "rad")):
        if getattr(a, key):
            parts.append(f"{key.replace('_', ' ')} {getattr(a, key):g} {unit}")
    if a.train_frac < 1:
        parts.append(f"{a.train_frac:.0%} of train episodes")
    return ", ".join(parts)


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d6d0")
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax, groups: list[str], values: dict[str, list[list[float]]], models: list[str],
                 labels: dict[str, str]):
    """values[model][group] = per-seed numbers. Bars = models (slots 5-8), error bar = seed std."""
    x = np.arange(len(groups))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        stats = [mean_std(v) for v in values[m]]
        ax.bar(x + (k - (len(models) - 1) / 2) * w, [s[0] for s in stats], width=w, color=MODEL_COLORS[m],
               edgecolor=SURFACE, linewidth=1.5, yerr=[s[1] for s in stats], ecolor=INK_2, capsize=0,
               error_kw=dict(lw=1), label=labels[m])
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1)


# ---------------------------------------------------------------------------------------------------- report
def write_report(results: list[dict], P: L.Prepared, a, split_rows: list[str]):
    models = [m for m in a.models if any(r["model"] == m for r in results)]
    R = {m: sorted([r for r in results if r["model"] == m], key=lambda r: r["seed"]) for m in models}
    sets = [s for s in L.EVAL_SETS if s in results[0]["metrics"]]
    regions = P.regions
    short = [REGION_SHORT.get(r, r) for r in regions]
    labels = model_labels(a.force)

    def always_contact(s, r=None):
        """Reference: predict contact everywhere. F1 = 2p / (1 + p), AP = p (p = positive rate)."""
        ps = [results[0]["metrics"][s]["regions"][x]["pos_rate"] for x in ([r] if r else regions)]
        return np.mean([2 * p / (1 + p) for p in ps]), np.mean(ps)

    def collect(m, fn):
        return [fn(r["metrics"]) for r in R[m]]

    md = ["# Contact estimation baselines A–D\n",
          f"Dataset(s) {', '.join(f'`{x}`' for x in a.data)} · regions {', '.join(short)} · {len(a.seeds)} seed(s) {a.seeds} · "
          f"{a.steps} steps × batch {a.batch} · window {a.window_s:.1f} s · split seed {a.split_seed} · "
          f"fingertip force input: **{a.force}**"
          + (" (|F| per tip)" if a.force == "normal" else " (link-frame vector of the normal contact force)")
          + f" · regularisation: {regularisation(a)}"
          + (f" · held-out grasp `{a.holdout_preset}`" if a.holdout_preset else "") + "\n",
          "Values are mean ± std over seeds. Thresholds are tuned per region on `val` (best F1) and applied "
          "unchanged to every test set. Frames after an object fell out are not evaluated.\n",
          "## Data split (episodes)\n", *split_rows, ""]

    md += ["## Macro F1 (mean over regions)\n", "| model | " + " | ".join(sets) + " |",
           "|---|" + "---|" * len(sets)]
    for m in models:
        md.append(f"| {labels[m]} | " + " | ".join(fmt(collect(m, lambda x: x[s]["macro"]["f1"]))
                                                     for s in sets) + " |")
    md.append("| *always contact (reference)* | " + " | ".join(f"{always_contact(s)[0]:.3f}" for s in sets) + " |")
    md.append("\nBest val macro AP (used to pick checkpoints and settings): " + ", ".join(
        f"{m} {fmt([r['best_val_ap'] for r in R[m]])}" for m in models))
    md.append("\nCheckpoint step (per seed): " + ", ".join(
        f"{m} {[r['best_step'] for r in R[m]]}" for m in models) + f" of {a.steps}")
    md += ["", "## Macro average precision (threshold-free)\n", "| model | " + " | ".join(sets) + " |",
           "|---|" + "---|" * len(sets)]
    for m in models:
        md.append(f"| {labels[m]} | " + " | ".join(fmt(collect(m, lambda x: x[s]["macro"]["ap"]))
                                                     for s in sets) + " |")
    md.append("| *always contact (reference)* | " + " | ".join(f"{always_contact(s)[1]:.3f}" for s in sets) + " |")

    # the draft's decision table (section 7 / 10): paired differences per seed
    pairs = [("D", "B", "history helps (D > B)"), ("D", "C", "force adds to motion history (D > C)"),
             ("B", "A", "current force helps (B > A)"), ("C", "A", "motion history helps (C > A)")]
    md += ["", "## Draft comparisons: Δ macro F1 (paired by seed)\n", "| comparison | " + " | ".join(sets) + " |",
           "|---|" + "---|" * len(sets)]
    for hi, lo, text in pairs:
        if hi in R and lo in R:
            row = []
            for s in sets:
                d = [rh["metrics"][s]["macro"]["f1"] - rl["metrics"][s]["macro"]["f1"]
                     for rh, rl in zip(R[hi], R[lo]) if rh["seed"] == rl["seed"]]
                row.append(fmt(d))
            md.append(f"| {text} | " + " | ".join(row) + " |")

    for s in sets:
        md += ["", f"## `{s}`: per region\n",
               "| model | " + " | ".join(f"{r} F1" for r in short) + " | " + " | ".join(f"{r} AP" for r in short)
               + " |", "|---|" + "---|" * (2 * len(short))]
        for m in models:
            md.append(f"| {m} | " + " | ".join(fmt(collect(m, lambda x, r=r: x[s]["regions"][r]["f1"]))
                                              for r in regions) + " | "
                      + " | ".join(fmt(collect(m, lambda x, r=r: x[s]["regions"][r]["ap"])) for r in regions)
                      + " |")
        md.append("| *always contact* | " + " | ".join(f"{always_contact(s, r)[0]:.3f}" for r in regions) + " | "
                  + " | ".join(f"{always_contact(s, r)[1]:.3f}" for r in regions) + " |")
        md += ["", "False-positive rate (predicted contact while there is none):\n",
               "| model | " + " | ".join(short) + " |", "|---|" + "---|" * len(short)]
        for m in models:
            md.append(f"| {m} | " + " | ".join(fmt(collect(m, lambda x, r=r: x[s]["regions"][r]["fpr"]))
                                              for r in regions) + " |")
        md += ["", "Contact change detection: median latency [ms] (fraction of changes detected before the "
               "contact changes again). Onset = contact starts, offset = contact ends. Latency only counts "
               "changes the prediction had to switch for (it showed the old state just before).\n",
               "| model | " + " | ".join(f"{r} onset | {r} offset" for r in short) + " |",
               "|---|" + "---|" * (2 * len(short))]
        for m in models:
            cells = []
            for r in regions:
                for kind in ("onset", "offset"):
                    lat = fmt(collect(m, lambda x, r=r, k=kind: x[s]["regions"][r][f"{k}_latency_median_s"]), 0,
                              1000)
                    det = fmt(collect(m, lambda x, r=r, k=kind: x[s]["regions"][r][f"{k}_detect_rate"]), 2)
                    cells.append(f"{lat} ({det})")
            md.append(f"| {m} | " + " | ".join(cells) + " |")
        n_events = {r: results[0]["metrics"][s]["regions"][r] for r in regions}
        md.append("\nEvents (onset/offset): " + ", ".join(
            f"{REGION_SHORT.get(r, r)} {n_events[r]['onset_events']}/{n_events[r]['offset_events']}"
            for r in regions))

        for g in results[0]["metrics"][s]["by"]:
            values = sorted(results[0]["metrics"][s]["by"][g])
            if len(values) < 2:
                continue
            md += ["", f"Macro F1 by {g}:\n", "| model | " + " | ".join(values) + " |", "|---|" + "---|" * len(values)]
            for m in models:
                md.append(f"| {m} | " + " | ".join(fmt(collect(m, lambda x, v=v: x[s]["by"][g][v]["macro"]["f1"]))
                                                  for v in values) + " |")

    md += ["", "## Notes\n",
           "- Latency counts from the frame where the ground truth changes to the first frame where the "
           "thresholded prediction agrees, searched only until the ground truth changes again (else: missed).",
           "- Empty-hand episodes are in train/val/test (all labels 0); they show up as false positives only.",
           "- Checkpoint = best val macro AP." if a.select == "ap" else "- Checkpoint = lowest val loss.",
           "- Figures: `f1_by_region.png`, `f1_by_motion.png`, `val_curves.png`, `episodes/`."]
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ------------------------------------------------------------------ figures
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.6), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes[0], sets):
        vals = {m: [collect(m, lambda x, r=r: x[s]["regions"][r]["f1"]) for r in regions]
                + [collect(m, lambda x: x[s]["macro"]["f1"])] for m in models}
        grouped_bars(ax, short + ["macro"], vals, models, labels)
        style(ax, s)
    axes[0][0].set_ylabel("F1 (val-tuned threshold)", color=INK_2, fontsize=9)
    axes[0][-1].legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle("F1 per region and test set (bars: mean over seeds, whisker: std)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "f1_by_region.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(sets), figsize=(3.2 * len(sets), 3.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, s in zip(axes[0], sets):
        motions = sorted(results[0]["metrics"][s]["by"]["motion"])
        vals = {m: [collect(m, lambda x, v=v: x[s]["by"]["motion"][v]["macro"]["f1"]) for v in motions]
                for m in models}
        grouped_bars(ax, [f"motion {v}\n({'whole hand' if v == 'A' else 'per finger'})" for v in motions], vals,
                     models, labels)
        style(ax, s)
    axes[0][0].set_ylabel("macro F1", color=INK_2, fontsize=9)
    axes[0][-1].legend(loc="upper right", frameon=False, fontsize=7, labelcolor=INK_2)
    fig.suptitle("Whole-hand squeeze (A) vs. per-finger probing (B)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "f1_by_motion.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # training curves: one column per model; clean train frames (dashed) vs. val (solid)
    fig, axes = plt.subplots(2, len(models), figsize=(3.4 * len(models), 5.6), sharex=True, sharey="row",
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, m in enumerate(models):
        steps = [h["step"] for h in R[m][0]["history"]]
        for row, metric in enumerate(("loss", "ap")):
            ax = axes[row][j]
            for key, ls, name in ((f"train_clean_{metric}", "--", "train"), (f"val_{metric}", "-", "val")):
                v = np.mean([[h[key] for h in r["history"]] for r in R[m]], 0)
                ax.plot(steps, v, color=MODEL_COLORS[m], lw=2, ls=ls, label=name)
                ax.text(steps[-1], v[-1], f" {name}", color=INK_2, fontsize=7, va="center")
            if metric == "ap":
                v = np.mean([[h["val_ap"] for h in r["history"]] for r in R[m]], 0)
                for r in R[m]:
                    if r["best_step"] in steps:
                        i = steps.index(r["best_step"])
                        ax.plot(steps[i], v[i], "o", ms=8, color=MODEL_COLORS[m], mec=SURFACE, mew=2)
            style(ax, labels[m] if row == 0 else "")
            if row == 1:
                ax.set_xlabel("training step", color=INK_2, fontsize=9)
        axes[0][0].set_ylabel("BCE loss (class-weighted)", color=INK_2, fontsize=9)
        axes[1][0].set_ylabel("macro AP (● checkpoint)", color=INK_2, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle("Training curves, mean over seeds: train frames without noise (dashed) vs. validation (solid)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "val_curves.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)

    meta = D.read_meta(a.data[0])
    first = {r["model"]: r for r in results if r["seed"] == a.seeds[0]}
    os.makedirs(os.path.join(a.out, "episodes"), exist_ok=True)
    for j, e in enumerate(_EXAMPLES):
        info = {k: v[e] for k, v in P.info.items()}
        drop = f" / dropped at {info['drop_time']:.1f} s" if info["dropped"] else ""
        title = (f"{P.sets[e]} · {info['dataset']} episode {info['episode_id']}: {info['asset']} / "
                 f"{info['preset']} / motion {info['motion']}{drop}")
        ep = dict(probe_delta=P.raw["probe_delta"][e], tip_force=P.raw["tip_force"][e], gt_contact=P.y[e])
        probs = {m: first[m]["examples"][j] for m in FIGURE_MODELS if m in first}
        plot_predictions(P.t, ep, meta["joint_names"], regions, info["motion"], title,
                         os.path.join(a.out, "episodes",
                                      f"pred_{P.sets[e]}_{info['dataset']}_ep{info['episode_id']}.png"),
                         probs, P.valid[e], labels=labels)


# ---------------------------------------------------------------------------------------------------- main
def main():
    global _P, _ARGS, _EXAMPLES
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="+", help="one or more dataset directories (same collection setup)")
    p.add_argument("--out", default=None, help="default outputs/baselines_<time>")
    p.add_argument("--models", nargs="+", default=list(L.MODELS), choices=list(L.MODELS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--regions", nargs="+", default=None, help="gt links to estimate (default: draft regions)")
    p.add_argument("--steps", type=int, default=L.TrainConfig.steps)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=L.TrainConfig.weight_decay)
    T = L.TrainConfig
    p.add_argument("--dropout", type=float, default=T.dropout, help="MLP layers and the LSTM's MLP head")
    p.add_argument("--force_noise", type=float, default=T.force_noise, help="[N] per-frame fingertip force noise")
    p.add_argument("--force_bias", type=float, default=T.force_bias, help="[N] per-window fingertip force offset")
    p.add_argument("--joint_noise", type=float, default=T.joint_noise, help="[rad] per-frame q noise")
    p.add_argument("--posture_shift", type=float, default=T.posture_shift,
                   help="[rad] per-window joint offset added to q and u (stops memorising episodes by posture)")
    p.add_argument("--no_regularisation", action="store_true",
                   help="set dropout, every noise and weight decay 1e-4 back to the first-baseline setting")
    p.add_argument("--train_frac", type=float, default=1.0, help="fraction of train episodes used")
    p.add_argument("--eval_every", type=int, default=L.TrainConfig.eval_every)
    p.add_argument("--select", choices=("ap", "loss"), default="ap",
                   help="checkpoint kept: best val macro AP or lowest val loss")
    p.add_argument("--force", choices=list(L.FORCE_INPUTS), default="normal",
                   help="fingertip force input of B/D: normal = |F| per tip, 3axis = link-frame vector")
    p.add_argument("--window_s", type=float, default=1.0, help="LSTM history length")
    p.add_argument("--mlp_hidden", type=int, default=128)
    p.add_argument("--lstm_hidden", type=int, default=64)
    p.add_argument("--head_hidden", type=int, default=64, help="MLP head on top of the LSTM")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--holdout_preset", default=None, help="initial grasp kept out of training, e.g. index_side")
    p.add_argument("--gpus", default="auto", help="'auto' (every idle GPU) or a list like 1,2,5")
    p.add_argument("--max_used_mib", type=int, default=1000, help="'auto': a GPU counts as idle below this")
    p.add_argument("--jobs_per_gpu", type=int, default=1)
    p.add_argument("--examples", type=int, default=8, help="episodes plotted with predictions")
    p.add_argument("--quick", action="store_true", help="smoke test: 300 steps, one seed")
    a = p.parse_args()
    if a.no_regularisation:
        a.dropout = a.force_noise = a.force_bias = a.joint_noise = a.posture_shift = 0.0
        a.weight_decay = 1e-4
    if a.quick:
        a.steps, a.eval_every, a.seeds = 300, 50, a.seeds[:1]
    a.out = a.out or os.path.join("outputs", time.strftime(f"baselines_{a.force}_%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(a.out, "models"), exist_ok=True)

    gpus = idle_gpus(a.max_used_mib) if a.gpus == "auto" else [int(g) for g in a.gpus.split(",")]
    if not gpus:
        raise SystemExit("no idle GPU found (see --gpus / --max_used_mib)")
    jobs = [(m, s) for s in a.seeds for m in a.models]
    n_workers = min(len(jobs), len(gpus) * a.jobs_per_gpu)
    print(f"GPUs {gpus} -> {n_workers} worker(s) for {len(jobs)} job(s)", flush=True)

    t0 = time.time()
    _P = L.prepare(a.data, a.regions, a.val_frac, a.test_frac, a.split_seed, a.holdout_preset)
    _ARGS = a
    _EXAMPLES = pick_examples(_P, a.examples, a.split_seed)
    split_rows = ["| set | episodes | with drop (pre-drop part) | empty hand | target frames | "
                  + " | ".join(f"{REGION_SHORT.get(r, r)} pos." for r in _P.regions) + " |",
                  "|---|---|---|---|---|" + "---|" * len(_P.regions)]
    for s in ("train", "val", *L.EVAL_SETS):
        m = _P.sets == s
        if m.any():
            pos = [_P.y[m][..., k][_P.valid[m]].mean() for k in range(len(_P.regions))]
            split_rows.append(f"| {s} | {m.sum()} | {(_P.info['dropped'] & m).sum()} | {(_P.info['empty'] & m).sum()} "
                              f"| {_P.valid[m].sum():,} | " + " | ".join(f"{v:.2f}" for v in pos) + " |")
    print(f"data prepared in {time.time() - t0:.0f} s", *split_rows, sep="\n", flush=True)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(dict(vars(a), gpus_used=gpus, regions=_P.regions, mean=_P.mean.tolist(), std=_P.std.tolist()),
                  f, indent=1)

    ctx = mp.get_context("fork")  # workers inherit the prepared arrays without copying them
    gpu_queue = ctx.Queue()
    for g in gpus:
        for _ in range(a.jobs_per_gpu):
            gpu_queue.put(g)
    results = []
    with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init_worker, initargs=(gpu_queue,)) as ex:
        futures = {ex.submit(run_job, m, s): (m, s) for m, s in jobs}
        for fut in as_completed(futures):
            results.append(fut.result())
            print(f"[{len(results)}/{len(jobs)} jobs] {futures[fut][0]}_s{futures[fut][1]} finished "
                  f"({time.time() - t0:.0f} s elapsed)", flush=True)

    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "examples"} for r in results], f, indent=1)
    write_report(results, _P, a, split_rows)
    print(open(os.path.join(a.out, "summary.md")).read().split("## `")[0])
    print(f"wrote {a.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
