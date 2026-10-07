"""Collect a contact-estimation dataset: many objects (shape x size) x random pose / orientation / grasp.

Every env holds one fixed object (env i -> object i % #objects); each round resets all envs into a newly
sampled episode (grasp preset, object position + orientation in the palm, posture noise, empty hand,
motion A/B with jitter), runs the 2 s grasp + 10 s probe, and writes one HDF5 shard. See
contact_est/dataset.py for the format and contact_est/objects.py for the library and suggested splits.

Re-running with the same --out resumes: existing shards are kept and only missing rounds are collected.

    bash scripts/run.sh scripts/collect_dataset.py --headless --out data/v1 --episodes 2400
    bash scripts/run.sh scripts/collect_dataset.py --headless --enable_cameras --vision --out data/v1_vis ...
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", required=True, help="dataset directory")
parser.add_argument("--episodes", type=int, default=2400, help="total episodes (rounded up to full rounds)")
parser.add_argument("--envs_per_object", type=int, default=8, help="parallel envs = #objects x this")
parser.add_argument("--families", nargs="*", default=None, help="subset of object families (default: all)")
parser.add_argument("--scales", type=float, nargs="*", default=None, help="subset of size scales (default: all)")
parser.add_argument("--orientation", choices=["nominal", "yaw", "full"], default="full")
parser.add_argument("--xy_noise", type=float, default=0.010)
parser.add_argument("--posture_noise", type=float, default=0.05)
parser.add_argument("--empty_frac", type=float, default=0.05)
parser.add_argument("--motions", nargs="*", default=["A", "B"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--vision", action="store_true", help="also store RGB-D (needs --enable_cameras)")
parser.add_argument("--vision_every", type=int, default=4, help="store every k-th control step (60 Hz / k)")
parser.add_argument("--camera_res", type=int, default=128)
parser.add_argument("--env_spacing", type=float, default=None,
                    help="[m] default 1.0, or 3.0 with --vision so neighbouring hands stay out of view")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
ENABLE_CAMERAS, DEVICE = args.enable_cameras, args.device
if args.vision and not ENABLE_CAMERAS:
    sys.exit("--vision needs --enable_cameras")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from contact_est import config as C  # noqa: E402
from contact_est import dataset as D  # noqa: E402
from contact_est.env import ContactProbeEnv  # noqa: E402
from contact_est.objects import DENSITY, FAMILIES, FRICTION, SCALES, build_library  # noqa: E402
from contact_est.probes import MOTIONS, SEGMENT_LABELS  # noqa: E402
from contact_est.runner import drop_time, dropped_mask, run_episode, stack_frames  # noqa: E402
from contact_est.sampling import SamplingCfg, sample_episodes  # noqa: E402

CAM_EYE_HAND = (0.02, 0.24, 0.17)
CAM_TARGET_HAND = (0.01, 0.0, 0.04)


def log(msg: str, out_dir: str):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(f"[collect] {line}", flush=True)
    with open(os.path.join(out_dir, "collect.log"), "a") as f:
        f.write(line + "\n")


def build_meta(env, assets, sampling: SamplingCfg) -> dict:
    origin, R = env.hand_frame
    return dict(
        format_version=1,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        args={k: v for k, v in vars(args).items() if k in (
            "episodes", "envs_per_object", "families", "scales", "orientation", "xy_noise", "posture_noise",
            "empty_frac", "motions", "seed", "vision", "vision_every", "camera_res", "env_spacing")},
        num_envs=env.num_envs,
        dt=env.dt,
        phases=dict(place_s=C.PLACE_S, close_s=C.CLOSE_S, hold_s=C.HOLD_S, probe_s=10.0,
                    note="object pinned during place+close, released at t=-hold_s; probe starts at t=0"),
        joint_names=env.joint_names,
        gt_links=env.region_names,
        draft_regions=list(C.DRAFT_REGIONS),
        fingertip_bodies=list(C.FINGERTIP_BODIES),
        gt_contact_threshold_n=dict(on=C.GT_CONTACT_ON_N, off=C.GT_CONTACT_OFF_N),
        drop_dist_m=D.DROP_DIST_M,
        palm_frame=dict(origin_env=origin.tolist(), axes_env_columns=R.tolist(), tilt_deg=C.PALM_TILT_DEG,
                        roll_deg=C.PALM_ROLL_DEG,
                        note="x wrist->fingers, y towards ring finger, z palm normal; object poses use this"),
        material=dict(density=DENSITY, friction=list(FRICTION)),
        assets=[a.meta() for a in assets],
        presets=[dict(name=p.name, object_xy=list(p.object_xy), clearance=p.clearance, posture=p.posture)
                 for p in C.GRASP_PRESETS],
        motions={m: dict(segments=[dict(t0=s.t0, fingers=list(s.fingers), sign=s.sign) for s in segs],
                         labels=SEGMENT_LABELS[m]) for m, segs in MOTIONS.items()},
        sampling=sampling.__dict__,
    )


def main():
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    assets = build_library(args.families or FAMILIES, args.scales or SCALES)
    num_envs = len(assets) * args.envs_per_object
    rounds = math.ceil(args.episodes / num_envs)
    sampling = SamplingCfg(randomize=True, orientation=args.orientation, xy_noise=args.xy_noise,
                           posture_noise=args.posture_noise, empty_frac=args.empty_frac, motions=tuple(args.motions))

    spacing = args.env_spacing or (3.0 if args.vision else 1.0)
    env = ContactProbeEnv(num_envs, assets, device=DEVICE, enable_cameras=args.vision,
                          camera_res=(args.camera_res, args.camera_res), env_spacing=spacing)
    env.calibrate()
    if env.camera is not None:
        env.set_camera_view(CAM_EYE_HAND, CAM_TARGET_HAND)

    meta = build_meta(env, assets, sampling)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):  # resuming: the configuration must not change
        old = D.read_meta(out_dir)
        for k in ("num_envs", "assets", "presets", "joint_names", "gt_links", "sampling"):
            if json.loads(json.dumps(meta[k])) != old[k]:
                sys.exit(f"{meta_path} exists with a different '{k}'; use a new --out")
        meta["created"] = old["created"]
    D.write_meta(out_dir, meta)
    log(f"{len(assets)} objects x {args.envs_per_object} envs = {num_envs} envs/round, {rounds} rounds "
        f"-> {rounds * num_envs} episodes, device {DEVICE}", out_dir)

    render_every = args.vision_every if args.vision else 0
    for r in range(rounds):
        fname = os.path.join(out_dir, f"shard_{r:05d}.h5")
        if os.path.exists(fname):
            continue
        t0 = time.time()
        rng = np.random.default_rng([args.seed, r])
        torch.manual_seed(args.seed * 100003 + r)
        episode, ep = sample_episodes(env, rng, sampling)
        frames, probe = run_episode(env, episode, list(ep["motion"]), jitter=True, render_every=render_every)
        series, vision = stack_frames(frames)
        del frames

        ep["episode_id"] = (r * num_envs + np.arange(num_envs)).astype(np.int64)
        ep["dropped"] = dropped_mask(series, ep["empty"])
        ep["drop_time"] = drop_time(series, ep["empty"])
        ep["probe_start"] = probe.start.cpu().numpy().astype(np.float32)
        ep["probe_amp"] = probe.amp.cpu().numpy().astype(np.float32)
        D.write_shard(out_dir, r, series, ep, vision)

        held = ~ep["dropped"] & ~ep["empty"]
        probe_t = series["t"] >= 0
        contact = series["gt_contact"][held][:, probe_t]  # (E, Tp, L)
        flips = (np.diff(contact.astype(np.int8), axis=1) != 0).sum((1, 2))
        draft = [env.region_names.index(x) for x in C.DRAFT_REGIONS]
        log(f"round {r + 1}/{rounds}: {time.time() - t0:5.1f}s | dropped {ep['dropped'].mean():.0%} | "
            f"empty {ep['empty'].mean():.0%} | held: draft-region contact {contact[..., draft].mean():.2f}, "
            f"contact flips/episode {flips.mean():.1f}", out_dir)
    log("done", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(0)  # SimulationApp.close() hangs on this server; shards are written atomically
