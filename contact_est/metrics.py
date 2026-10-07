"""Metrics for per-region contact estimation (draft section 9). Numpy only, no Isaac / torch imports.

Every function takes per-episode arrays: ``prob`` (E, T, K) contact probability, ``gt`` (E, T, K) bool and
``valid`` (E, T) bool marking the target frames (probe phase, before a drop; one contiguous run per episode).
Accuracy is deliberately absent: with imbalanced regions "never in contact" already scores high.
"""

from __future__ import annotations

import warnings

import numpy as np

THRESHOLD_GRID = np.round(np.arange(0.02, 0.99, 0.02), 2)


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Precision / recall / F1 / false-positive rate of boolean predictions (NaN when undefined)."""
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    nan = float("nan")
    return dict(
        precision=tp / (tp + fp) if tp + fp else nan,
        recall=tp / (tp + fn) if tp + fn else nan,
        f1=2 * tp / (2 * tp + fp + fn) if tp + fn else nan,
        fpr=fp / (fp + tn) if fp + tn else nan,
    )


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (threshold-free; NaN without positives)."""
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    hit = y[np.argsort(-score, kind="stable")]
    precision = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float(precision[hit].sum() / n_pos)


def best_thresholds(prob: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-region threshold maximising F1 (tune on the validation set, then apply to the test sets)."""
    out = np.full(prob.shape[-1], 0.5)
    for k in range(prob.shape[-1]):
        y, s = gt[..., k][valid], prob[..., k][valid]
        f1 = [binary_metrics(y, s >= th)["f1"] for th in THRESHOLD_GRID]
        if not np.all(np.isnan(f1)):
            out[k] = THRESHOLD_GRID[int(np.nanargmax(f1))]
    return out


def transition_latency(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, dt: float) -> dict:
    """How well the prediction follows ground-truth contact changes, for one region.

    ``gt``, ``pred``, ``valid``: (E, T). For every onset (0 -> 1) and offset (1 -> 0) inside the valid run:
    - detected: the prediction shows the new state at some frame while the ground truth stays in it.
    - latency: for changes the prediction was tracking (it showed the OLD state on the frame before), the
      time until it first shows the new state. A prediction that sat in the new state all along (e.g. "always
      in contact") detects the change but contributes no latency, so it cannot score 0 s for free.
    Returns ``onset_*`` / ``offset_*``: events, detection rate, tracked events, median / p90 latency [s].
    """
    lat: dict[str, list[float]] = {"onset": [], "offset": []}
    n = {"onset": 0, "offset": 0}
    detected = {"onset": 0, "offset": 0}
    for e in range(len(gt)):
        idx = np.flatnonzero(valid[e])
        if len(idx) < 2:
            continue
        g, p = gt[e, idx[0]: idx[-1] + 1], pred[e, idx[0]: idx[-1] + 1]
        change = np.flatnonzero(g[1:] != g[:-1]) + 1
        for c, end in zip(change, np.append(change[1:], len(g))):
            kind = "onset" if g[c] else "offset"
            n[kind] += 1
            hit = np.flatnonzero(p[c:end] == g[c])
            if hit.size:
                detected[kind] += 1
                if p[c - 1] != g[c]:
                    lat[kind].append(hit[0] * dt)
    out = {}
    for kind in ("onset", "offset"):
        v = np.asarray(lat[kind])
        out[f"{kind}_events"] = n[kind]
        out[f"{kind}_detect_rate"] = detected[kind] / n[kind] if n[kind] else float("nan")
        out[f"{kind}_tracked"] = len(v)
        out[f"{kind}_latency_median_s"] = float(np.median(v)) if len(v) else float("nan")
        out[f"{kind}_latency_p90_s"] = float(np.percentile(v, 90)) if len(v) else float("nan")
    return out


def _summary(prob, gt, valid, thresholds, regions, dt, latency: bool) -> dict:
    per = {}
    for k, r in enumerate(regions):
        y, s = gt[..., k][valid], prob[..., k][valid]
        m = binary_metrics(y, s >= thresholds[k])
        m.update(ap=average_precision(y, s), pos_rate=float(y.mean()) if len(y) else float("nan"),
                 threshold=float(thresholds[k]))
        if latency:
            m.update(transition_latency(gt[..., k], prob[..., k] >= thresholds[k], valid, dt))
        per[r] = m
    keys = ("f1", "ap", "precision", "recall", "fpr") + (
        ("onset_latency_median_s", "offset_latency_median_s", "onset_detect_rate", "offset_detect_rate")
        if latency else ())
    with warnings.catch_warnings():  # 'Mean of empty slice' when a group has no positives at all
        warnings.simplefilter("ignore", RuntimeWarning)
        macro = {key: float(np.nanmean([per[r][key] for r in regions])) for key in keys}
    return dict(regions=per, macro=macro, episodes=int(valid.any(1).sum()), frames=int(valid.sum()))


def evaluate(prob: np.ndarray, gt: np.ndarray, valid: np.ndarray, thresholds: np.ndarray, regions,
             dt: float, groups: dict[str, np.ndarray] | None = None) -> dict:
    """Full metrics for one evaluation set, plus the same without latencies per group value.

    ``groups``: name -> (E,) labels (e.g. motion, family, preset); each value gets its own entry under ``by``.
    """
    out = _summary(prob, gt, valid, thresholds, list(regions), dt, latency=True)
    out["by"] = {}
    for name, lab in (groups or {}).items():
        out["by"][name] = {
            str(v): _summary(prob[lab == v], gt[lab == v], valid[lab == v], thresholds, list(regions), dt,
                             latency=False)
            for v in np.unique(lab)
        }
    return out
