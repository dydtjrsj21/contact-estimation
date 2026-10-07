# contact_estimation — hidden-contact estimation pilot (Isaac Sim 4.5 / Isaac Lab 2.1.1)

> 📑 Experiment report (Korean): `report/README.md`. It is kept **local only** (git-ignored), so it is not in the repository.

Simulation environment for the first experiment in
[`../dexmachina/contact_estimation_experiment_draft.md`](../dexmachina/contact_estimation_experiment_draft.md):
*can fingertip force responses to small probing motions tell us whether the palm / proximal / middle
phalanges are touching the object?*

![motion B at t = 5 s](outputs/vis_motionB/mosaic_t+5.0s.png)

## What the environment contains

| Draft item | Implementation |
|---|---|
| Hand, wrist fixed | Allegro hand (16 DoF, Isaac Lab `ALLEGRO_HAND_CFG`), fixed base, palm up and leveled |
| Estimator input: fingertip force | net **normal** contact force on each distal link (`*_link_3`), in the link frame, averaged over physics sub-steps. Isaac Lab's `net_forces_w` sums normal forces only (no friction), so the 3 axes encode the contact-normal direction. The baselines use its magnitude (one normal force per tip) by default |
| Estimator input: `q`, `u` | measured joint angles, commanded joint targets |
| Ground-truth patches (labels only) | one contact sensor per hand link (17), **filtered to the object**; the draft's 4 regions (palm, index proximal/middle, middle proximal) are a subset. On/off hysteresis 0.20 N / 0.05 N |
| Vision (not an input yet) | one RGB-D camera per env (240² in visualization, optional 128² in datasets) |
| Objects | pilot: sphere Ø8 cm, cylinder Ø7×10 cm, box 7×9×7 cm. Dataset: 6 families × 5 sizes ([objects.py](contact_est/objects.py)). Same density (300 kg/m³) and friction |
| Initial grasps | `fingertip`, `palm_deep`, `index_side` |
| Probing motions (10 s) | **B**: hold / all close / index close / middle close / ring open. **A**: whole-hand close ×3 / open. Start time ±0.15 s and amplitude ±20 % jittered per env; the actual command is logged |
| Logging rate | 60 Hz (physics 120 Hz, decimation 2) |

`visualize_env.py` with defaults runs every pilot object × initial grasp pair once in parallel
(env *i*: object `i % 3`, grasp `i // 3`). With `--library full --randomize` it previews one random dataset
episode for each of the 30 objects (see `outputs/vis_full_random/`).

An episode works like a person placing the object in the hand. The object is held fixed above the
open hand while the fingers close into the preset posture, then released. The grasp settles under
gravity for 0.8 s, and the probe starts at t = 0. As the draft says, the preset name is **not** a label:
contacts are recorded at every step.

## Setup (already done on this server)

```bash
bash scripts/install_env.sh        # creates conda env `contactsim`: Isaac Sim 4.5 + Isaac Lab 2.1.1
```

Isaac Lab is cloned to `../IsaacLab` (tag v2.1.1). This server needs three workarounds, all handled by
`install_env.sh` and `run.sh`:

1. **glibc 2.31 (Ubuntu 20.04)**: the Isaac Sim 4.5 wheels are tagged `manylinux_2_34`. They are
   downloaded for that platform and their file names re-tagged before installing. They run fine here.
2. **No system `libGLU`**: without it the RTX material backend fails and Kit segfaults. `libglu` comes
   from conda-forge and only that one library is put on `LD_LIBRARY_PATH`.
3. **Driver 535.104 < 535.129** (the RTX minimum): the driver check is turned off with
   `--/rtx/verifyDriverVersion/enabled=false`. Rendering works.

Also: don't use the shared `/home/nas/kyh/conda_envs/isaacsim` env. Its PyTorch 2.12 brings CUDA 13
libraries, which make Kit report "CUDA in bad state" with this driver.

## Run / visualize

Always launch through `scripts/run.sh`. It applies the runtime fixes above.

```bash
cd contact_estimation
bash scripts/run.sh scripts/visualize_env.py --headless --enable_cameras --motion B   # ~1.5 min
bash scripts/run.sh scripts/visualize_env.py --headless --enable_cameras --motion A
bash scripts/run.sh scripts/visualize_env.py --headless --motion B                    # no cameras, faster
python scripts/replot.py outputs/<run>/episode.npz                                   # redo figures only
```

Options: `--num_envs`, `--seed`, `--no_jitter`, `--palm_tilt`, `--out`, plus the standard Isaac Lab
launcher flags (`--device cuda:3`, …). The server has no display, so run headless. With a display or
livestream, drop `--headless` to watch it live.

Each run writes to `outputs/vis_<motion>_<time>/` (or `--out`):

| File | Content |
|---|---|
| `mosaic.mp4` | 3×3 camera views (12 s). Each tile shows its condition, phase/time, GT contact dots and fingertip-force bars (I/M/R/T) |
| `mosaic_t*.png` | stills: end of the pinned close, t = 0, 3, 5, 7, 9 s |
| `timeseries_env*.png` | the draft's key figure: probe command → fingertip \|F\| → 3-axis force of the most loaded tip → GT contact raster |
| `episode.npz` | every logged signal, shape `(env, time, …)`: `tip_force`, `joint_pos`, `joint_target`, `probe_delta`, `gt_force`, `gt_contact`, `object_pos`, plus metadata (`objects`, `presets`, `region_names`, `dropped`, …). RGB-D is not saved yet |

Ready-made results: [`outputs/vis_motionB/`](outputs/vis_motionB) and [`outputs/vis_motionA/`](outputs/vis_motionA).

## Dataset: `data/contact_v1`

Collected with
```bash
bash scripts/run.sh scripts/collect_dataset.py --headless --out data/contact_v1 --episodes 7200 --envs_per_object 8 --seed 1
python scripts/dataset_report.py data/contact_v1          # -> data/contact_v1/report/
```
30 rounds × 240 parallel envs took about 27 s per round (≈14 min on one A6000), 529 MB on disk.
Re-running the same command resumes and only collects missing shards.

**What varies per episode**

| Factor | Values |
|---|---|
| Object mesh | 6 families (sphere, cylinder, box, capsule, rounded box, ellipsoid) × 5 sizes (0.85, 0.925, 1.0, 1.075, 1.15 × nominal ≈ 7–8 cm). Same density (300 kg/m³) and friction |
| Initial grasp | `fingertip` / `palm_deep` / `index_side` preset (uniform), joint posture + N(0, 0.05 rad) |
| Object position | preset xy + U(±1 cm); height chosen so the rotated object's lowest point is 1.5–3.8 cm above the palm |
| Object orientation | uniform over SO(3) (`--orientation yaw` / `nominal` also available) |
| Probe | motion A or B (50/50), bump start ±0.15 s, amplitude ±20 % |
| Empty hand | 5 % of episodes (the draft's no-object baseline) |

Each object is pinned in place while the hand closes, released at t = −0.8 s, then probed from 0–10 s.

**Format** ([contact_est/dataset.py](contact_est/dataset.py)): `meta.json` (objects, presets, joint and link
names, palm frame, sampling config) plus `shard_XXXXX.h5` (240 episodes each), 720 steps at 60 Hz
(t = −2 … 10 s).

| Field | Shape | Role |
|---|---|---|
| `tip_force` | (E, T, 4, 3) | **input**: fingertip normal contact force (vector sum) in the link frame (index, middle, ring, thumb) |
| `joint_pos`, `joint_target` | (E, T, 16) | **input**: q, u |
| `gt_contact`, `gt_force` | (E, T, 17) | **label**: object contact (bool, hysteresis 0.2/0.05 N) and \|force\| for every hand link |
| `object_pos`, `object_quat` | (E, T, 3/4) | object pose in the palm frame (for vision / analysis) |
| per episode | (E, …) | `asset_idx`, `preset_idx`, `motion`, `empty`, `dropped`, `drop_time`, initial pose, grasp posture, probe start/amp |

```python
from contact_est import dataset as D
d = D.load("data/contact_v1")            # held + empty episodes (dropped ones excluded)
d["tip_force"].shape                       # (5872, 720, 4, 3)
meta = D.read_meta("data/contact_v1"); draft = [meta["gt_links"].index(r) for r in meta["draft_regions"]]
y = d["gt_contact"][..., draft]            # the draft's 4 targets: palm, index prox/mid, middle prox
train = d["split"] == "train"              # also "heldout_size", "heldout_shape"
```

**What came out** ([report/summary.json](data/contact_v1/report/summary.json)):

- 7,200 episodes: **5,534 held** (18.4 h of data), 1,328 dropped (flagged, excluded by default), 338 empty.
- Held episodes by suggested split: train 2,671 · heldout_size (scales 0.925 / 1.075) 1,861 ·
  heldout_shape (ellipsoid) 1,002. Split by episode, never by window.
- Draft-region labels are imbalanced: in contact for 77 % (palm), 11 % (index proximal), 7 % (index middle)
  and 20 % (middle proximal) of probe time. Use the class-weighted loss the draft suggests.
- Motion A (whole hand) changes contacts about 23.5 times per episode, motion B (per finger) about 13.6 times.
- Drop rate depends on shape × grasp ([report/drop_rate.png](data/contact_v1/report/drop_rate.png)):
  boxes and rounded boxes 3–11 %, but **sphere + fingertip 78 %** and capsule + fingertip 41 %. The held
  set therefore under-represents round objects in fingertip grasps.
- Data checks: no NaNs; fingertip \|F\| is 8.3 N at the 99th percentile; 2 episodes have a brief
  penetration spike above 50 N.

### More of the same: `data/contact_v1_s2` … `contact_v1_s8`

Seven more collections with the exact `contact_v1` setup and seeds 2–8 (one per GPU, in parallel, about 16 min):
```bash
for g in 1 2 3 4 5 6 7; do s=$((g + 1))
  CUDA_VISIBLE_DEVICES=$g bash scripts/run.sh scripts/collect_dataset.py --headless --device cuda:0 \
      --out data/contact_v1_s$s --episodes 7200 --envs_per_object 8 --seed $s &
  sleep 20; done; wait
```
Together that is 57,600 episodes (44,147 held), 4.2 GB. Each collection matches `contact_v1` closely: drop rate
18–19 %, draft-region contact palm 0.76–0.77 / I-prox 0.10–0.12 / I-mid 0.07 / M-prox 0.20–0.21, fingertip
|F| 99th percentile 8.2–8.4 N, no NaNs. 24 episodes (0.04 %) contain a brief penetration spike above 50 N (up to 508 N).
`train_baselines.py` takes several dataset directories. Each one is split on its own, so `contact_v1`'s
train/val/test episodes never change when data is added.

## Baselines A–D (draft sections 7–9)

```bash
python scripts/train_baselines.py data/contact_v1                    # A-D x seeds 0 1 2 on every idle GPU (~5 min)
python scripts/train_baselines.py data/contact_v1 --force 3axis       # link-frame force vector instead of |F|
python scripts/train_baselines.py data/contact_v1 --holdout_preset index_side   # adds an unseen-grasp test set
python scripts/train_baselines.py data/contact_v1 --quick             # smoke test
```

No simulator: run it with the `contactsim` python directly. Every (model, seed) job runs in a worker pinned to
one GPU; `--gpus auto` (default) takes every GPU with < 1 GB in use and no load.

| Model | Input |
|---|---|
| A | MLP, current `q, u` |
| B | MLP, current `f, q, u` |
| C | LSTM (64) + MLP head, last 1 s of `q, u` |
| D | LSTM (64) + MLP head, last 1 s of `f, q, u` |

- `f` = fingertip force. Default `--force normal`: one normal-force magnitude per tip (4 values); `3axis` uses the
  link-frame vector (12 values).
- Targets: the 4 draft regions, class-weighted BCE. Episodes whose object fell out are used **up to `drop_time`**.
- Split by episode: the `train` objects are split 70/15/15 into train / val / test; `heldout_size` and
  `heldout_shape` are extra test sets. Normalisation uses train frames only. Thresholds are tuned per region on val.
- Training: 40k steps, checkpoint = best val macro AP. Default regularisation (chosen on val, see below): a
  per-window joint offset of 0.06 rad added to `q` and `u` (like a calibration error), fingertip force noise and
  bias of 0.1 N, joint noise 0.005 rad, dropout 0.3, weight decay 0.01. `--no_regularisation` turns these off.
- Outputs (`outputs/baselines_<force>/`): `summary.md` (macro / per-region F1 and AP, false-positive rate,
  contact start/end latency, breakdowns by motion, family and grasp, the draft's Δ comparisons, an
  "always contact" reference), `metrics.json`, `f1_by_region.png`, `f1_by_motion.png`, `val_curves.png`
  (clean-train vs. val), `episodes/pred_*.png` (command → fingertip force → GT contact vs. C/D probabilities),
  `models/*.pt`.

**Overfitting and the regularisation sweep** (`bash scripts/sweep_regularization.sh` → `outputs/sweep/`,
compared on val only with `scripts/compare_runs.py`). Without regularisation every model memorises episodes
within about one epoch: train AP goes to 0.97–0.99 and val AP peaks at 500–1,000 steps, then falls. Each episode
has its own grasp posture, so `q`/`u` identify it. The per-window joint offset removes this. Val AP went from
0.58 to 0.70–0.74, and the curves now converge. Sensor noise alone, or dropout plus weight decay alone, is not
enough. Halving or quartering the train episodes costs about 0.06 val AP per halving and has not levelled off,
so **more episodes is the most promising way to gain accuracy**. Doubling the training steps (20k → 40k) adds only about +0.01.

**Results** (`outputs/baselines_final_normal`, default settings, macro F1, mean ± std over 3 seeds):

| model | test | heldout_size | heldout_shape |
|---|---|---|---|
| A: MLP q,u | 0.632 ± 0.005 | 0.650 ± 0.004 | 0.504 ± 0.002 |
| B: MLP fn,q,u | 0.683 ± 0.009 | **0.699 ± 0.006** | **0.539 ± 0.006** |
| C: LSTM+MLP q,u | 0.631 ± 0.013 | 0.661 ± 0.003 | 0.487 ± 0.014 |
| D: LSTM+MLP fn,q,u | **0.693 ± 0.006** | 0.698 ± 0.010 | **0.539 ± 0.013** |
| always contact | 0.358 | 0.396 | 0.341 |

- Force helps: B − A = +0.05, D − C = +0.04 to +0.06 on every test set. Most of the gain is on the index middle
  phalanx (F1 0.56 → 0.74).
- History does not add much beyond the current frame: D − B = +0.011 ± 0.015 (test), 0.00 on the held-out sets.
  The LSTMs also react more slowly to contact changes (median 180–300 ms vs. 60–120 ms for B), and their val
  loss keeps rising after val AP has levelled off, so they still overfit somewhat.
- In the draft's decision table (section 10), this is "current force helps, force history does not yet". The next
  candidates are more data, more informative probing motions, and stronger regularisation for the LSTMs.
- The palm is still close to "always contact", with a false-positive rate of about 0.6. Unseen shape (ellipsoid)
  stays at about 0.54.
- An earlier run without regularisation is kept in `outputs/baselines_normal`, `outputs/baselines_3axis` (normal
  and 3-axis force were equal there) and `outputs/baselines_normal_20k_selectloss`.

**More data** (`bash scripts/data_scaling.sh` → `outputs/scaling/`, figure `outputs/scaling/scaling.png`). The
models train on 1× / 2× / 4× / 8× the episodes (`contact_v1` + `contact_v1_s2…s8`), 80k steps, default settings.
Scores below are on `contact_v1`'s own test episodes, which are the same episodes in every run (macro F1, mean ± std, 3 seeds):

| model | 1× (7,200) | 2× | 4× | 8× (57,600) |
|---|---|---|---|---|
| A: MLP q,u | 0.628 ± 0.009 | | | 0.682 ± 0.002 |
| B: MLP fn,q,u | 0.692 ± 0.006 | 0.707 ± 0.003 | 0.723 ± 0.005 | 0.735 ± 0.004 |
| C: LSTM+MLP q,u | 0.631 ± 0.007 | | | 0.737 ± 0.001 |
| D: LSTM+MLP fn,q,u | 0.692 ± 0.014 | 0.736 ± 0.004 | 0.772 ± 0.005 | **0.800 ± 0.001** |

- With 8× the data, every comparison in the draft comes out positive, on every test set (Δ macro F1 on the pooled
  test set): history helps (D − B = +0.057), force adds to motion history (D − C = +0.050), current force helps
  (B − A = +0.044), and motion history helps (C − A = +0.051). D is the best model everywhere: test 0.789 pooled,
  unseen sizes 0.805, unseen shape 0.634.
- The history models gain most from data: about +0.035 per doubling for D, +0.015 for B. With 1× data, history
  looked useless (D ≈ B) only because there was too little data.
- At 8×, every model's val loss still falls at 80k steps and the train–val gap is small, so overfitting is gone.
  D's best checkpoints sit at 74–78k of 80k: longer training or a larger LSTM could still help.
- D at 8×, pooled test, F1 per region: palm 0.84, I-prox 0.73, I-mid 0.82, M-prox 0.77. The palm is still the
  weak spot (false-positive rate 0.51).

## Code map

```
contact_est/constants.py  hand links, draft regions, thresholds (no Isaac imports)
contact_est/objects.py    object library: families x sizes, procedural meshes, splits
contact_est/config.py     grasp presets / timing / palm leveling / scene config
contact_est/env.py        ContactProbeEnv: calibrate (palm frame + leveling), reset(episode), step, observe
contact_est/sampling.py   per-episode randomization (preset, pose, orientation, posture, empty, motion)
contact_est/probes.py     motions A and B (raised-cosine bumps, per-env motion + jitter)
contact_est/runner.py     one episode = grasp + probe; stacking; drop detection
contact_est/dataset.py    HDF5 shard writer + loader (no Isaac imports)
contact_est/render.py     mosaic overlay, time-series and prediction figures (no Isaac imports)
contact_est/learning.py   baselines A-D: data prep (pre-drop frames, splits, normalisation), MLP / LSTM+MLP, training
contact_est/metrics.py    per-region F1 / AP / false-positive rate, contact start/end latency, group breakdowns
scripts/visualize_env.py  one episode per env -> video, stills, plots (pilot grid or --library full --randomize)
scripts/collect_dataset.py  dataset collection (resumable)
scripts/dataset_report.py   statistics + figures for a dataset
scripts/replot.py         time-series figures from an episode.npz or dataset episodes
scripts/train_baselines.py  train + evaluate A-D on every idle GPU -> outputs/baselines_<force>/
scripts/sweep_regularization.sh  regularisation / data-size sweep for B and D -> outputs/sweep/
scripts/compare_runs.py   compare runs on validation results (table + figures)
scripts/data_scaling.sh   1x/2x/4x/8x data -> outputs/scaling/; scripts/plot_scaling.py draws scaling.png
scripts/install_env.sh, scripts/run.sh
```

**Palm frame** (used for object poses and camera placement): x points from the wrist to the fingers,
z is the palm normal (up), and y points toward the ring finger. The origin is midway between
`palm_link` and the knuckle line.

## Known limitations / next steps

- **Round objects in fingertip grasps mostly fall out** (see the drop rates above). The pilot grid's
  sphere + fingertip condition fails the same way.
- Simulated sensors have no noise, bias or latency. Empty-hand episodes read only finger-on-finger
  contact (0.78 N mean). Add sensor noise and bias at training time before comparing with hardware.
- Ground truth is per link, not per sensor patch. "palm" means the whole palm link.
- The fingertip force is the net contact force on the whole distal link, including finger–finger
  contact, like a real fingertip sensor (not an object-only force). It holds **normal force only**:
  Isaac Lab's `net_forces_w` leaves out friction, so it is not the full force a 3-axis F/T sensor reads.
  Its magnitude is what a 1-axis normal-force (pressure) sensor reads.
- Only the object's pose varies; the hand is always palm-up with the wrist fixed. Varying the hand
  orientation (gravity direction) would be the next axis.
- Vision is not stored in `contact_v1`. `--enable_cameras --vision` adds RGB + depth (128², 15 Hz,
  camera fixed relative to the hand; depth is `inf` where nothing is hit). That measured **6.1 MB per
  episode** (vs. 75 KB without), so 1,000 vision episodes ≈ 6 GB, and the disk has about 48 GB free.
  Env spacing is 3 m with vision so neighbouring hands stay out of view.
- Palm leveling (`PALM_TILT_DEG = -8`, `PALM_ROLL_DEG = 5`) was measured empirically by rolling
  spheres on the curved palm.
- Next: with 8× data D is still improving at the end of training. Train longer or use a larger model, collect
  more data, try better probing motions, and look at the palm (false-positive rate about 0.5).
