"""Run one probing episode per env and visualise it.

Default: the pilot grid (sphere / cylinder / box x 3 initial grasps, nominal placement), as in the draft.
``--library full --randomize`` previews what the dataset collector samples (every object once, random
position / orientation / preset).

Outputs (in --out):
    mosaic.mp4            per-env RGB camera, with GT contact (draft regions) + fingertip force overlay
    mosaic_t*.png         stills
    timeseries_env*.png   command / fingertip force / GT contact (all links) over time, one figure per env
    episode.npz           all logged signals

Example (headless server):
    bash scripts/run.sh scripts/visualize_env.py --headless --enable_cameras --motion B
    bash scripts/run.sh scripts/visualize_env.py --headless --enable_cameras --library full --randomize
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--library", choices=["pilot", "full"], default="pilot")
parser.add_argument("--num_envs", type=int, default=None, help="default: pilot 9 (3x3 grid), full = #objects")
parser.add_argument("--motion", choices=["A", "B", "mixed"], default="B", help="A: whole hand, B: per finger")
parser.add_argument("--randomize", action="store_true", help="random preset / position / orientation / posture")
parser.add_argument("--orientation", choices=["nominal", "yaw", "full"], default="full")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_jitter", action="store_true", help="disable timing/amplitude jitter of the probe")
parser.add_argument("--out", type=str, default=None, help="output dir (default: outputs/vis_<tag>_<time>)")
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument("--cols", type=int, default=None, help="mosaic columns")
parser.add_argument("--palm_tilt", type=float, default=None, help="override config.PALM_TILT_DEG [deg]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# AppLauncher strips its own options from the namespace; keep the ones we need
ENABLE_CAMERAS, DEVICE = args.enable_cameras, args.device
app = AppLauncher(args).app

# ---- everything below needs the running app
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from contact_est import config as C  # noqa: E402
from contact_est.env import ContactProbeEnv  # noqa: E402
from contact_est.objects import PILOT_LIBRARY_ARGS, build_library  # noqa: E402
from contact_est.render import draw_mosaic, plot_timeseries  # noqa: E402
from contact_est.runner import dropped_mask, run_episode, stack_frames  # noqa: E402
from contact_est.sampling import SamplingCfg, sample_episodes  # noqa: E402

CAM_EYE_HAND = (0.02, 0.24, 0.17)  # palm frame: x wrist->fingers, y towards ring finger, z palm normal
CAM_TARGET_HAND = (0.01, 0.0, 0.04)  # looking in from the open ring-finger side


def main():
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pilot = args.library == "pilot"
    tag = args.motion if pilot else f"{args.library}_{args.motion}"
    out_dir = args.out or os.path.join(
        os.path.dirname(__file__), "..", "outputs", f"vis_{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    assets = build_library(**PILOT_LIBRARY_ARGS) if pilot else build_library()
    num_envs = args.num_envs or (9 if pilot else len(assets))
    if args.palm_tilt is not None:
        C.PALM_TILT_DEG = args.palm_tilt
    env = ContactProbeEnv(num_envs, assets, device=DEVICE, enable_cameras=ENABLE_CAMERAS)
    env.calibrate()
    if env.camera is not None:
        env.set_camera_view(CAM_EYE_HAND, CAM_TARGET_HAND)

    # without --randomize: env i -> preset (i // #objects), i.e. the objects x presets grid
    preset_idx = None if args.randomize else np.arange(num_envs) // len(assets) % len(C.GRASP_PRESETS)
    motions = ("A", "B") if args.motion == "mixed" else (args.motion,)
    cfg = SamplingCfg(randomize=args.randomize, orientation=args.orientation, motions=motions, empty_frac=0.0)
    episode, ep_meta = sample_episodes(env, rng, cfg, preset_idx=preset_idx)
    ep_assets = [assets[a] for a in ep_meta["asset_idx"]]
    preset_names = [C.GRASP_PRESETS[p].name for p in ep_meta["preset_idx"]]
    for i in range(num_envs):
        print(f"[contact_est] env {i}: object={ep_assets[i].name:20s} grasp={preset_names[i]:10s} "
              f"motion={ep_meta['motion'][i]}")

    frames, _ = run_episode(env, episode, list(ep_meta["motion"]), jitter=not args.no_jitter, render_every=1)
    series, vision = stack_frames(frames)
    n_grasp = int((series["t"] < 0).sum())
    series["probe_delta"] = series["joint_target"] - env.grasp_posture.cpu().numpy()[:, None]
    series["probe_delta"][:, :n_grasp] = np.nan  # closing ramp is not part of the probe
    dropped = dropped_mask(series, ep_meta["empty"])

    np.savez_compressed(
        os.path.join(out_dir, "episode.npz"), **series, **ep_meta, dropped=dropped,
        joint_names=np.array(env.joint_names), region_names=np.array(env.region_names),
        objects=np.array([a.name for a in ep_assets]), presets=np.array(preset_names), dt=env.dt,
    )

    for i in np.flatnonzero(dropped):
        k = int(np.argmax(np.linalg.norm(series["object_pos"][i], axis=-1) > 0.15))
        print(f"[contact_est] WARNING env {i} ({ep_assets[i].name}, {preset_names[i]}) dropped the object at "
              f"t={series['t'][k]:+.2f}s")
    draft = [env.region_names.index(r) for r in C.DRAFT_REGIONS]
    frac = series["gt_contact"][:, n_grasp:][..., draft].mean(1)
    for i in range(num_envs):
        print(f"[contact_est] env {i} draft-region contact fraction: "
              + ", ".join(f"{r}={f:.2f}" for r, f in zip(C.DRAFT_REGIONS, frac[i])))

    # ---- plots
    for i in range(num_envs):
        ep = {k: series[k][i] for k in ("probe_delta", "tip_force", "gt_contact")}
        title = f"env {i}: {ep_assets[i].name} / {preset_names[i]} grasp / motion {ep_meta['motion'][i]}"
        plot_timeseries(series["t"], ep, env.joint_names, env.region_names, ep_meta["motion"][i], title,
                        os.path.join(out_dir, f"timeseries_env{i}.png"))

    # ---- video
    if vision is not None:
        cols = args.cols or (3 if num_envs <= 9 else 6)
        labels = [
            f"{a.family} / {p}" if pilot else f"{a.family} x{a.scale:.2f} / {p}" for a, p in zip(ep_assets, preset_names)
        ]

        def frame(k):
            return draw_mosaic(vision["rgb"][:, k], float(vision["t"][k]), series["gt_contact"][:, k],
                               series["tip_force"][:, k], labels, env.region_names, cols)

        stride = max(1, round(1.0 / (env.dt * args.video_fps)))
        writer = imageio.get_writer(os.path.join(out_dir, "mosaic.mp4"), fps=args.video_fps, quality=8)
        for k in range(0, len(vision["t"]), stride):
            writer.append_data(frame(k))
        writer.close()
        for t_snap in (-C.HOLD_S - 0.02, -0.02, 3.0, 5.0, 7.0, 9.0):
            k = int(np.argmin(np.abs(vision["t"] - t_snap)))
            imageio.imwrite(os.path.join(out_dir, f"mosaic_t{t_snap:+.1f}s.png"), frame(k))

    print(f"[contact_est] wrote {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        # Isaac Sim 4.5 SimulationApp.close() hangs on this headless server; all outputs are already
        # written and closed at this point, so terminate the process directly.
        sys.stdout.flush()
        os._exit(0)
