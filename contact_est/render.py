"""Visualisation helpers (no Isaac dependency): RGB mosaic with overlays and per-env time-series plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from .constants import DRAFT_REGIONS, REGION_SHORT  # noqa: E402
from .probes import SEGMENT_LABELS  # noqa: E402

# categorical slots 1-4 (validated light-mode palette), fixed order index/middle/ring/thumb
FINGER_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
FINGER_SHORT = ("I", "M", "R", "T")
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#9a9893"
SURFACE = "#fcfcfb"
CONTACT_ON = "#2a78d6"
FORCE_BAR_MAX_N = 3.0
# tiles show the draft's 4 regions; the time-series raster shows every recorded link
PANEL_H = 64


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _tile(rgb: np.ndarray, label: str, t: float, gt: np.ndarray, regions, tip_f: np.ndarray) -> Image.Image:
    h, w, _ = rgb.shape
    img = Image.new("RGB", (w, h + PANEL_H), SURFACE)
    img.paste(Image.fromarray(rgb), (0, 0))
    d = ImageDraw.Draw(img)
    f_small, f_big = _font(11), _font(13)
    d.text((6, 4), label, fill="white", font=f_big, stroke_width=2, stroke_fill="black")
    phase = "grasp" if t < 0 else "probe"
    d.text((w - 6, 4), f"{phase} t={t:+.2f}s", fill="white", font=f_small, anchor="ra", stroke_width=2,
           stroke_fill="black")

    # ground-truth contact dots
    y0 = h + 6
    col_w = w // len(regions)
    for j, r in enumerate(regions):
        x = j * col_w + 6
        on = bool(gt[j])
        d.ellipse((x, y0 + 1, x + 10, y0 + 11), fill=CONTACT_ON if on else SURFACE, outline=INK_2)
        d.text((x + 14, y0), REGION_SHORT.get(str(r), str(r)), fill=INK if on else INK_3, font=f_small)

    # fingertip |F| bars
    y1 = h + 26
    bar_w = (w - 12) // 4
    for j, (name, c) in enumerate(zip(FINGER_SHORT, FINGER_COLORS)):
        x = 6 + j * bar_w
        mag = float(np.linalg.norm(tip_f[j]))
        frac = min(mag / FORCE_BAR_MAX_N, 1.0)
        d.text((x, y1), name, fill=INK_2, font=f_small)
        bx0, bx1 = x + 12, x + bar_w - 8
        d.rounded_rectangle((bx0, y1 + 2, bx1, y1 + 12), radius=3, outline="#d8d6d0", fill=SURFACE)
        if frac > 0.01:
            d.rounded_rectangle((bx0, y1 + 2, bx0 + max(4, int(frac * (bx1 - bx0))), y1 + 12), radius=3, fill=c)
        d.text((x, y1 + 16), f"{mag:4.2f}N", fill=INK_3, font=f_small)
    return img


def draw_mosaic(rgb: np.ndarray, t: float, gt_contact: np.ndarray, tip_force: np.ndarray, labels: list[str],
                region_names, cols: int = 3) -> np.ndarray:
    """One video frame. rgb (N, H, W, 3), gt_contact (N, L), tip_force (N, 4, 3) at the same instant."""
    region_names = [str(r) for r in region_names]
    shown = [r for r in DRAFT_REGIONS if r in region_names]
    idx = [region_names.index(r) for r in shown]
    tiles = [_tile(rgb[i], labels[i], t, gt_contact[i, idx], shown, tip_force[i]) for i in range(len(rgb))]
    tw, th = tiles[0].size
    n = len(tiles)
    rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tw + (cols - 1) * 2, rows * th + (rows - 1) * 2), "#d8d6d0")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % cols) * (tw + 2), (i // cols) * (th + 2)))
    return np.asarray(canvas)


FINGERS = ("index", "middle", "ring", "thumb")


def _style_axis(ax, t: np.ndarray, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d6d0")
    ax.grid(axis="y", color="#ecebe7", lw=0.8)
    ax.axvspan(t[0], 0.0, color="#f1f0ec", zorder=0)


def _plot_command(ax, t: np.ndarray, probe_delta: np.ndarray, joint_names, motion: str):
    """Commanded probe: sum of joint-target offsets per finger, with the motion's segment labels."""
    _style_axis(ax, t, "probe cmd\nΣΔu [rad]")
    for f, c in zip(FINGERS, FINGER_COLORS):
        idx = [i for i, n in enumerate(joint_names) if n.startswith(f + "_joint_")]
        ax.plot(t, probe_delta[:, idx].sum(-1), color=c, lw=2, label=f)
    ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=8, labelcolor=INK_2)
    ax.text(t[0] + 0.05, ax.get_ylim()[0], "grasping", va="bottom", fontsize=8, color=INK_3)
    for j, label in enumerate(SEGMENT_LABELS.get(str(motion), [])):
        ax.text(2.0 * (j + 1) + 1.0, ax.get_ylim()[1], label, ha="center", va="top", fontsize=8, color=INK_2)


def _plot_tip_norm(ax, t: np.ndarray, tip: np.ndarray):
    """Fingertip |F| per finger; ``tip`` (T, 4, 3)."""
    _style_axis(ax, t, "fingertip |F| [N]\n(estimator input)")
    for j, (f, c) in enumerate(zip(FINGERS, FINGER_COLORS)):
        ax.plot(t, np.linalg.norm(tip[:, j], axis=-1), color=c, lw=2, label=f)
    ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=8, labelcolor=INK_2)


def plot_timeseries(t: np.ndarray, ep: dict, joint_names, region_names, motion: str, title: str, path: str):
    """The draft's key figure: command -> fingertip force -> hidden contact, on one time axis.

    ``ep``: one episode, ``probe_delta`` (T, J) (NaN during the grasp phase), ``tip_force`` (T, 4, 3),
    ``gt_contact`` (T, L).
    """
    joint_names = [str(j) for j in joint_names]
    fingers = FINGERS
    regions = [str(r) for r in region_names]
    n_reg = len(regions)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9 + 0.18 * max(0, n_reg - 4)), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.3, 1.3, max(1.0, 0.25 * n_reg)]))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)

    def style(ax, ylabel):
        _style_axis(ax, t, ylabel)

    # 1) commanded probe (sum of flexion offsets per finger)
    _plot_command(axes[0], t, ep["probe_delta"], joint_names, motion)

    # 2) fingertip |F|
    tip = ep["tip_force"]  # (T, 4, 3)
    _plot_tip_norm(axes[1], t, tip)

    # 3) 3-axis force of the most loaded fingertip, in its link frame
    probe = t >= 0
    j_max = int(np.linalg.norm(tip[probe], axis=-1).mean(0).argmax())
    ax = axes[2]
    style(ax, f"{fingers[j_max]} tip F\n(link frame) [N]")
    for a, (name, ls) in enumerate(zip("xyz", ("-", "--", ":"))):
        ax.plot(t, tip[:, j_max, a], color=FINGER_COLORS[j_max], lw=2, ls=ls, label=f"f{name}")
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8, labelcolor=INK_2)

    # 4) ground-truth contact raster (labels, never input)
    ax = axes[3]
    style(ax, "GT contact\n(label only)")
    gt = ep["gt_contact"]
    for j, r in enumerate(regions):
        y = len(regions) - 1 - j
        ax.fill_between(t, y + 0.12, y + 0.88, where=gt[:, j], step="mid", color=CONTACT_ON, lw=0)
        ax.axhline(y + 0.5, color="#ecebe7", lw=0.8, zorder=0)
    ax.set_yticks(np.arange(len(regions)) + 0.5, [REGION_SHORT.get(r, r) for r in regions[::-1]], fontsize=7)
    for lbl, r in zip(ax.get_yticklabels(), regions[::-1]):  # draft regions in dark ink
        lbl.set_color(INK if r in DRAFT_REGIONS else INK_3)
    ax.set_ylim(0, len(regions))
    ax.grid(False)
    ax.set_xlabel("time [s]  (shaded: grasp closing, t < 0)", color=INK_2, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


# estimator models A-D take categorical slots 5-8, so they never share a hue with the fingers (slots 1-4);
# figures that overlay lines show only adjacent slots (C/D)
MODEL_COLORS = {"A": "#e87ba4", "B": "#008300", "C": "#4a3aa7", "D": "#e34948"}
GT_FILL = "#dcdad4"


def plot_predictions(t: np.ndarray, ep: dict, joint_names, regions, motion: str, title: str, path: str,
                     probs: dict[str, np.ndarray], valid: np.ndarray, labels: dict[str, str] | None = None):
    """Draft section 9's figure: command -> fingertip force -> hidden contact vs. predicted probability.

    ``ep``: ``probe_delta`` (T, J), ``tip_force`` (T, 4, 3), ``gt_contact`` (T, K) for the K ``regions``.
    ``probs``: model name -> (T, K) contact probability (NaN where not evaluated). ``valid``: (T,) target frames
    (probe phase, before a drop); frames after a drop are hatched.
    """
    joint_names = [str(j) for j in joint_names]
    regions = [str(r) for r in regions]
    labels = labels or {}
    k = len(regions)
    fig, axes = plt.subplots(2 + k, 1, figsize=(10, 4.2 + 1.15 * k), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1.3] + [0.85] * k))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    _plot_command(axes[0], t, ep["probe_delta"], joint_names, motion)
    _plot_tip_norm(axes[1], t, ep["tip_force"])

    after_drop = (t >= 0) & ~valid
    for j, r in enumerate(regions):
        ax = axes[2 + j]
        _style_axis(ax, t, f"{REGION_SHORT.get(r, r)}\nP(contact)")
        ax.fill_between(t, 0, 1, where=ep["gt_contact"][:, j].astype(bool), step="mid", color=GT_FILL, lw=0,
                        label="ground truth contact")
        if after_drop.any():
            ax.fill_between(t, 0, 1, where=after_drop, step="mid", facecolor="none", edgecolor="#c9c7c0",
                            hatch="///", lw=0, label="object dropped (not evaluated)")
        for name, p in probs.items():
            ax.plot(t, p[:, j], color=MODEL_COLORS.get(name, INK_2), lw=2, label=labels.get(name, name))
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        if j == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1.32), ncol=4, frameon=False, fontsize=8,
                      labelcolor=INK_2)
    axes[-1].set_xlabel("time [s]  (shaded: grasp closing, t < 0)", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
