"""Contact estimators of draft section 7 (models A-D) and the data pipeline that feeds them. No Isaac imports.

Target frames ("valid"): the probe phase (t >= 0) and, for episodes whose object fell out of the hand, only
the frames before ``drop_time``. A window may reach back before t = 0 (the hold phase) for its history.
Everything up to ``prepare`` is numpy; ``train`` / ``predict`` run on one torch device.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from . import dataset as D
from .metrics import average_precision

INPUT_COLUMNS = {"f": slice(0, 12), "fn": slice(12, 16), "q": slice(16, 32), "u": slice(32, 48)}
N_FEATURES = 48
"""Feature layout (index, middle, ring, thumb for the tips; joint order = meta["joint_names"]):
f  = fingertip force vector in the link frame, 4 x xyz [N]. Isaac Lab's ``net_forces_w`` sums only the
     NORMAL contact forces, so these 3 axes say where the contact normal points, not friction.
fn = fingertip normal force |f|, one scalar per tip [N] (what a 1-axis pressure sensor reads)
q  = measured joint angles, u = commanded joint targets [rad]."""

FORCE_INPUTS = {"normal": "fn", "3axis": "f"}
"""``--force`` choice -> the column group that replaces "f" in a model's inputs."""

MODELS = {
    "A": dict(arch="mlp", inputs=("q", "u")),  # "f" = fingertip force, resolved by FORCE_INPUTS
    "B": dict(arch="mlp", inputs=("f", "q", "u")),
    "C": dict(arch="lstm_mlp", inputs=("q", "u")),
    "D": dict(arch="lstm_mlp", inputs=("f", "q", "u")),
}
"""Draft section 7. D > B: history helps. D > C: force adds information beyond the motion history."""

EVAL_SETS = ("test", "heldout_size", "heldout_shape", "heldout_preset")
STD_FLOOR = 1e-2  # nearly constant inputs (an unloaded fingertip) must not be blown up by normalisation


@dataclass
class Prepared:
    x: np.ndarray        # (E, T, 48) normalised features, layout INPUT_COLUMNS
    y: np.ndarray        # (E, T, K) uint8 contact labels of ``regions``
    valid: np.ndarray    # (E, T) bool target frames
    sets: np.ndarray     # (E,) train / val / test / heldout_size / heldout_shape / heldout_preset
    regions: list[str]
    t: np.ndarray
    dt: float
    mean: np.ndarray     # (48,) normalisation, from train target frames only
    std: np.ndarray
    info: dict[str, np.ndarray]  # per episode: dataset, episode_id, asset, family, preset, motion, empty, dropped,
    #                              drop_time (episode_id is unique within a dataset only)
    raw: dict[str, np.ndarray]   # un-normalised tip_force, probe_delta (for figures)


def assign_sets(d: dict, val_frac: float, test_frac: float, seed: int, holdout_preset: str | None) -> np.ndarray:
    """Episode-level split. Objects of the dataset's ``train`` split (seen shapes and sizes) are divided
    at random into train / val / test. ``heldout_size`` / ``heldout_shape`` stay as evaluation sets.
    Empty-hand episodes join the train pool (there is no object to hold out). ``holdout_preset`` moves one
    initial grasp out of the train pool into ``heldout_preset`` (draft section 8, unseen initial grasp)."""
    sets = d["split"].astype(object)
    sets[d["empty"]] = "train"
    if holdout_preset:
        sets[(sets == "train") & (d["preset"] == holdout_preset)] = "heldout_preset"
    pool = np.random.default_rng(seed).permutation(np.flatnonzero(sets == "train"))
    n_test, n_val = round(test_frac * len(pool)), round(val_frac * len(pool))
    sets[pool[:n_test]] = "test"
    sets[pool[n_test: n_test + n_val]] = "val"
    return sets.astype(str)


def _load_one(path: str, regions: list[str], ref: dict | None, val_frac, test_frac, split_seed, holdout_preset):
    """One dataset -> un-normalised arrays. Each dataset is split on its own, so adding datasets never changes
    the split of the ones already there."""
    meta = D.read_meta(path)
    if ref is not None:
        for k in ("joint_names", "gt_links", "dt", "fingertip_bodies", "presets"):
            if meta[k] != ref[k]:
                raise ValueError(f"{path}: '{k}' differs from {ref['_path']}; datasets cannot be combined")
        if [(x["name"], x["split"]) for x in meta["assets"]] != [(x["name"], x["split"]) for x in ref["assets"]]:
            raise ValueError(f"{path}: object library differs from {ref['_path']}")
    d = D.load(path, fields=("tip_force", "joint_pos", "joint_target", "gt_contact"), include_dropped=True)
    t = d["t"]
    valid = (t[None] >= 0) & (t[None] < d["drop_time"][:, None])
    keep = valid.any(1)  # objects that fell out before the probe started contribute nothing
    sets = assign_sets(d, val_frac, test_frac, split_seed, holdout_preset)[keep]
    e, n_t = int(keep.sum()), len(t)
    tip = d["tip_force"][keep]
    x = np.concatenate([tip.reshape(e, n_t, 12), np.linalg.norm(tip, axis=-1), d["joint_pos"][keep],
                        d["joint_target"][keep]], -1).astype(np.float32)
    link = [meta["gt_links"].index(r) for r in regions]
    probe_delta = d["joint_target"][keep] - d["grasp_posture"][keep][:, None]
    probe_delta[:, t < 0] = np.nan
    info = {k: d[k][keep] for k in ("episode_id", "asset", "family", "preset", "motion", "empty", "dropped",
                                    "drop_time")}
    info["dataset"] = np.full(e, os.path.basename(os.path.normpath(path)))
    return meta, t, dict(x=x, y=d["gt_contact"][keep][..., link].astype(np.uint8), valid=valid[keep], sets=sets,
                         info=info, tip_force=tip, probe_delta=probe_delta)


def prepare(paths: str | list[str], regions=None, val_frac: float = 0.15, test_frac: float = 0.15,
            split_seed: int = 0, holdout_preset: str | None = None) -> Prepared:
    """Load one or several datasets (same hand, objects and timing; e.g. the same collection with other
    seeds), split each by episode, normalise with the train frames of all of them."""
    paths = [paths] if isinstance(paths, str) else list(paths)
    ref, parts, t = None, [], None
    for path in paths:
        meta, t, part = _load_one(path, [str(r) for r in (regions or D.read_meta(path)["draft_regions"])], ref,
                                  val_frac, test_frac, split_seed, holdout_preset)
        if ref is None:
            ref = dict(meta, _path=path)
        parts.append(part)
    regions = [str(r) for r in (regions or ref["draft_regions"])]

    def cat(key, sub=None):
        return np.concatenate([p[key][sub] if sub else p[key] for p in parts]) if len(parts) > 1 else (
            parts[0][key][sub] if sub else parts[0][key])

    x, sets, valid = cat("x"), cat("sets"), cat("valid")
    for p in parts:  # free the per-dataset copies as soon as they are concatenated
        p.pop("x")
    train_frames = x[valid & (sets == "train")[:, None]]
    # float64: a float32 sum over millions of frames drifts by up to ~0.2 std
    mean = train_frames.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(train_frames.std(0, dtype=np.float64), STD_FLOOR).astype(np.float32)
    del train_frames
    x -= mean
    x /= std
    info = {k: cat("info", k) for k in parts[0]["info"]}
    return Prepared(x=x, y=cat("y"), valid=valid, sets=sets, regions=regions, t=t, dt=float(ref["dt"]), mean=mean,
                    std=std, info=info, raw=dict(tip_force=cat("tip_force"), probe_delta=cat("probe_delta")))


# ---------------------------------------------------------------------------------------------------- models
class MLP(nn.Module):
    """Current frame only (the last step of the window)."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 128, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers, nn.Linear(d, n_out))

    def forward(self, x):  # (B, W, F)
        return self.net(x[:, -1])


class LSTMMLP(nn.Module):
    """LSTM over the window (past -> current frame), MLP head on its last hidden state."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 64, head_hidden: int = 64, layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(head_hidden, n_out))

    def forward(self, x):  # (B, W, F)
        h, _ = self.lstm(x)
        return self.head(h[:, -1])


class Estimator(nn.Module):
    """Selects the model's input columns from the full feature vector, returns per-region logits.
    ``force``: "normal" (one |F| per fingertip) or "3axis" (link-frame vector) for models that use force."""

    def __init__(self, name: str, n_out: int, window: int, force: str = "normal", mlp_hidden: int = 128,
                 lstm_hidden: int = 64, head_hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        spec = MODELS[name]
        self.inputs = tuple(FORCE_INPUTS[force] if g == "f" else g for g in spec["inputs"])
        cols = np.concatenate([np.arange(N_FEATURES)[INPUT_COLUMNS[g]] for g in self.inputs])
        self.register_buffer("cols", torch.as_tensor(cols, dtype=torch.long))
        self.name = name
        self.n_out = n_out
        if spec["arch"] == "mlp":
            self.window = 1
            self.net = MLP(len(cols), n_out, mlp_hidden, dropout=dropout)
        else:
            self.window = window
            self.net = LSTMMLP(len(cols), n_out, lstm_hidden, head_hidden, dropout=dropout)

    def forward(self, xw):  # (B, W, N_FEATURES)
        return self.net(xw[..., self.cols])


# ---------------------------------------------------------------------------------------------------- training
@dataclass
class TrainConfig:
    steps: int = 40000
    batch: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup: int = 200
    eval_every: int = 500
    val_stride: int = 3  # validation (model selection) on every 3rd frame
    window_s: float = 1.0
    select: str = "ap"  # keep the weights with the best val macro AP ("ap") or the lowest val loss ("loss")
    train_frac: float = 1.0  # use this fraction of the train episodes (learning curve: does more data help?)
    # input noise, train time only, in raw units. Per frame = sensor noise; per window = a constant offset over
    # the window (sensor bias / joint calibration error), which also stops a model from recognising an episode
    # by its exact posture. The joint offset is shared by q and u, so the tracking error u - q is unchanged.
    # Defaults = the setting chosen on val in outputs/sweep (without posture_shift, val AP peaks within ~1 epoch:
    # every episode has its own grasp posture, and models memorise episodes by it).
    force_noise: float = 0.1     # [N] per frame, fingertip force (f and fn)
    force_bias: float = 0.1      # [N] per window
    joint_noise: float = 0.005   # [rad] per frame, measured q only
    posture_shift: float = 0.06  # [rad] per window and joint, q and u together
    dropout: float = 0.3         # MLP layers / the LSTM's MLP head (used when building the Estimator)

    @property
    def augments(self) -> bool:
        return any((self.force_noise, self.force_bias, self.joint_noise, self.posture_shift))


class Augment:
    """Adds the TrainConfig input noise to normalised windows (B, W, N_FEATURES)."""

    def __init__(self, cfg: TrainConfig, std: np.ndarray, device):
        frame, window = np.zeros(N_FEATURES, np.float32), np.zeros(N_FEATURES, np.float32)
        for g in ("f", "fn"):
            frame[INPUT_COLUMNS[g]], window[INPUT_COLUMNS[g]] = cfg.force_noise, cfg.force_bias
        frame[INPUT_COLUMNS["q"]] = cfg.joint_noise
        self.frame = torch.as_tensor(frame / std, device=device)
        self.window = torch.as_tensor(window / std, device=device)
        inv = 1.0 / std
        self.shift_q = torch.as_tensor(cfg.posture_shift * inv[INPUT_COLUMNS["q"]], device=device)
        self.shift_u = torch.as_tensor(cfg.posture_shift * inv[INPUT_COLUMNS["u"]], device=device)

    def __call__(self, xw: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        b, w, f = xw.shape
        noise = torch.randn(xw.shape, device=xw.device, generator=gen) * self.frame
        noise = noise + torch.randn((b, 1, f), device=xw.device, generator=gen) * self.window
        shift = torch.randn((b, 1, self.shift_q.numel()), device=xw.device, generator=gen)
        noise[..., INPUT_COLUMNS["q"]] += shift * self.shift_q
        noise[..., INPUT_COLUMNS["u"]] += shift * self.shift_u
        return xw + noise


def frame_index(valid: np.ndarray, episodes: np.ndarray, device, stride: int = 1):
    """(episode, time) indices of the target frames of ``episodes`` (bool mask over episodes)."""
    ep, ti = np.nonzero(valid & episodes[:, None])
    if stride > 1:
        ep, ti = ep[ti % stride == 0], ti[ti % stride == 0]
    return torch.as_tensor(ep, device=device), torch.as_tensor(ti, device=device)


def gather(x: torch.Tensor, ep: torch.Tensor, ti: torch.Tensor, window: int) -> torch.Tensor:
    """Windows ending at (and including) frame ``ti``: (B, window, F). Never looks ahead."""
    offs = torch.arange(1 - window, 1, device=x.device)
    return x[ep[:, None], ti[:, None] + offs]


@torch.no_grad()
def _validate(est, x, y, ep, ti, loss_fn, batch=16384) -> tuple[float, float]:
    """Class-weighted BCE and macro average precision on the given frames."""
    est.eval()
    total, probs = 0.0, []
    for s in range(0, len(ep), batch):
        e, t = ep[s: s + batch], ti[s: s + batch]
        logits = est(gather(x, e, t, est.window))
        total += loss_fn(logits, y[e, t].float()).item() * len(e)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    est.train()
    p, g = np.concatenate(probs), y[ep, ti].bool().cpu().numpy()
    ap = np.nanmean([average_precision(g[:, k], p[:, k]) for k in range(p.shape[1])])
    return total / max(len(ep), 1), float(ap)


def train(est: Estimator, x: torch.Tensor, y: torch.Tensor, P: Prepared, cfg: TrainConfig, seed: int,
          log=print) -> dict:
    """Class-weighted BCE on random (episode, frame) windows of the train set; keeps the weights that score
    best on val (``cfg.select``: macro AP or loss)."""
    device = x.device
    train_eps = P.sets == "train"
    if cfg.train_frac < 1:  # the same subset for every model and seed
        idx = np.flatnonzero(train_eps)
        keep = np.random.default_rng(0).permutation(idx)[: round(cfg.train_frac * len(idx))]
        train_eps = np.isin(np.arange(len(P.sets)), keep)
    tr_ep, tr_t = frame_index(P.valid, train_eps, device)
    va_ep, va_t = frame_index(P.valid, P.sets == "val", device, cfg.val_stride)
    # clean (no noise, eval mode) train frames, as many as val: the train-val gap shows memorisation
    sub = torch.randperm(len(tr_ep), generator=torch.Generator().manual_seed(0))[: len(va_ep)].to(device)
    tc_ep, tc_t = tr_ep[sub], tr_t[sub]
    augment = Augment(cfg, P.std, device) if cfg.augments else None
    assert int(tr_t.min()) - est.window + 1 >= 0, "window reaches before the first recorded frame"
    pos = y[tr_ep, tr_t].float().mean(0)
    pos_weight = (1 - pos) / pos.clamp_min(1e-4)  # draft: BCE weighted for the contact frequency
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.AdamW(est.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)))
    best, best_state, best_step, history, running = float("inf"), None, 0, [], []
    for step in range(1, cfg.steps + 1):
        i = torch.randint(len(tr_ep), (cfg.batch,), device=device, generator=gen)
        e, t = tr_ep[i], tr_t[i]
        xw = gather(x, e, t, est.window)
        if augment is not None:
            xw = augment(xw, gen)
        loss = loss_fn(est(xw), y[e, t].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(est.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())
        if step % cfg.eval_every == 0 or step == cfg.steps:
            val_loss, val_ap = _validate(est, x, y, va_ep, va_t, loss_fn)
            clean_loss, clean_ap = _validate(est, x, y, tc_ep, tc_t, loss_fn)
            history.append(dict(step=step, train_loss=float(np.mean(running)), train_clean_loss=clean_loss,
                                train_clean_ap=clean_ap, val_loss=val_loss, val_ap=val_ap))
            running = []
            score = -val_ap if cfg.select == "ap" else val_loss
            if score < best:
                best, best_state, best_step = score, copy.deepcopy(est.state_dict()), step
            log(f"step {step:6d}  train loss {clean_loss:.4f} AP {clean_ap:.4f}  |  val loss {val_loss:.4f} "
                f"AP {val_ap:.4f}")
    est.load_state_dict(best_state)
    best_ap = max(h["val_ap"] for h in history if h["step"] == best_step)
    return dict(history=history, best_step=best_step, best_val_ap=best_ap, pos_weight=pos_weight.tolist(),
                train_episodes=int(train_eps.sum()))


@torch.no_grad()
def predict(est: Estimator, x: torch.Tensor, valid: np.ndarray, episodes: np.ndarray, batch: int = 16384
            ) -> np.ndarray:
    """Contact probability (len(episodes), T, K) for the given episode indices, NaN outside target frames."""
    est.eval()
    le, ti = np.nonzero(valid[episodes])
    out = np.full((len(episodes), valid.shape[1], est.n_out), np.nan, np.float32)
    ep_t = torch.as_tensor(episodes[le], device=x.device)
    ti_t = torch.as_tensor(ti, device=x.device)
    for s in range(0, len(le), batch):
        p = torch.sigmoid(est(gather(x, ep_t[s: s + batch], ti_t[s: s + batch], est.window)))
        out[le[s: s + batch], ti[s: s + batch]] = p.cpu().numpy()
    return out
