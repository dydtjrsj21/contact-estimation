"""Contact-probe environment: Allegro hand holding an object, scripted probing, sensor read-out.

Must be imported *after* the Isaac Sim app has been launched (``isaaclab.app.AppLauncher``).

Env ``i`` always holds ``assets[i % len(assets)]`` (fixed when the scene is built). Everything else about
an episode -- object position and orientation in the hand, grasp posture, empty hand -- is passed to
:meth:`ContactProbeEnv.reset` and may change every episode (see :mod:`contact_est.sampling`).
"""

from __future__ import annotations

import math
import re

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.scene import InteractiveScene

from . import config as C
from .objects import ObjectAsset

# where the object waits in "empty hand" episodes (env frame, on the ground beside the hand)
PARK_POS = (0.35, 0.35, 0.10)


class ContactProbeEnv:
    """Thin stepping wrapper around an :class:`InteractiveScene`.

    Control / logging runs at ``1 / (PHYSICS_DT * DECIMATION)`` Hz. :meth:`step` takes absolute joint
    position targets ``u`` (num_envs, num_joints) and returns an observation dict:

    estimator inputs
        ``tip_force``    (N, 4, 3) fingertip net contact force in the fingertip link frame [N]
        ``joint_pos``    (N, J)    measured joint angles q [rad]
        ``joint_target`` (N, J)    commanded joint targets u [rad]
    labels (never feed to the estimator)
        ``gt_force``     (N, L)    |force| between each hand link (constants.GT_LINKS) and the object [N]
        ``gt_contact``   (N, L)    contact label after on/off hysteresis
    vision / diagnostics
        ``object_pos`` (N, 3), ``object_quat`` (N, 4): object pose in the palm frame
        ``rgb`` (N, H, W, 3) uint8, ``depth`` (N, H, W) [m] when cameras are enabled
    """

    def __init__(
        self,
        num_envs: int,
        assets: list[ObjectAsset],
        device: str = "cuda:0",
        enable_cameras: bool = True,
        camera_res: tuple[int, int] = C.CAMERA_RES,
        env_spacing: float = 1.0,
    ):
        sim_cfg = sim_utils.SimulationCfg(dt=C.PHYSICS_DT, device=device, render_interval=C.DECIMATION)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.scene = InteractiveScene(C.make_scene_cfg(num_envs, assets, enable_cameras, camera_res, env_spacing))
        self.sim.reset()

        self.num_envs = num_envs
        self.device = device
        self.dt = C.PHYSICS_DT * C.DECIMATION
        self.assets = assets
        self.env_asset = [i % len(assets) for i in range(num_envs)]
        self.robot = self.scene["robot"]
        self.object = self.scene["object"]
        self.tip_sensor = self.scene["fingertip_contact"]
        self.gt_sensors = [self.scene[f"contact_gt_{link}"] for link in C.GT_LINKS]
        self.camera = self.scene["camera"] if enable_cameras else None

        self.joint_names: list[str] = list(self.robot.joint_names)
        self.region_names = list(C.GT_LINKS)
        # fingertip order inside the multi-body sensor follows prim discovery; map to C.FINGERTIP_BODIES
        self.tip_sensor_ids = [self.tip_sensor.body_names.index(b) for b in C.FINGERTIP_BODIES]
        self.tip_body_ids = [self.robot.body_names.index(b) for b in C.FINGERTIP_BODIES]

        self.joint_lo = self.robot.data.soft_joint_pos_limits[0, :, 0]
        self.joint_hi = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.open_posture = self.posture_vec(C.OPEN_POSTURE).expand(num_envs, -1).clone()
        self.preset_postures = torch.cat([self.posture_vec(p.posture) for p in C.GRASP_PRESETS])  # (P, J)

        self.joint_target = self.open_posture.clone()
        self.grasp_posture = self.open_posture.clone()
        self.gt_state = torch.zeros(num_envs, len(self.region_names), dtype=torch.bool, device=device)
        self.hand_frame = None  # (origin, R) in env-local coordinates; R columns = palm-frame axes
        self.hand_quat = None
        self._flex_dir = None  # fingertip displacement when flexing; disambiguates the palm normal

    # ------------------------------------------------------------------ helpers

    def posture_vec(self, posture: dict[str, float]) -> torch.Tensor:
        """Joint-regex dict -> (1, J) tensor; unspecified joints default to 0."""
        q = torch.zeros(1, len(self.robot.joint_names), device=self.device)
        for i, name in enumerate(self.robot.joint_names):
            for pattern, value in posture.items():
                if re.fullmatch(pattern, name):
                    q[0, i] = value
        return self.clamp_joints(q)

    def clamp_joints(self, q: torch.Tensor) -> torch.Tensor:
        return torch.clamp(q, self.joint_lo, self.joint_hi)

    def _compute_hand_frame(self):
        """Palm frame from link positions (open hand): x palm->fingers, z palm normal, y = z cross x
        (towards the ring finger for this right hand)."""
        names = self.robot.body_names
        pos = self.robot.data.body_pos_w[0] - self.scene.env_origins[0]
        palm = pos[names.index("palm_link")]
        knuckles = {f: pos[names.index(f"{f}_link_0")] for f in ("index", "middle", "ring")}
        k_center = torch.stack(list(knuckles.values())).mean(0)
        x = torch.nn.functional.normalize(k_center - palm, dim=0)
        y = knuckles["index"] - knuckles["ring"]
        y = torch.nn.functional.normalize(y - (y @ x) * x, dim=0)
        z = torch.linalg.cross(x, y)
        # z must point to the side the fingers curl towards (the palm side); flip y too to stay right-handed
        if (self._flex_dir @ z) < 0:
            z, y = -z, -y
        origin = palm + 0.5 * (k_center - palm)
        R = torch.stack([x, y, z], dim=1)
        self.hand_frame = (origin, R)
        self.hand_quat = math_utils.quat_from_matrix(R)

    def hand_to_env(self, offset: torch.Tensor) -> torch.Tensor:
        origin, R = self.hand_frame
        return origin + offset @ R.T

    def env_to_hand(self, pos_env: torch.Tensor) -> torch.Tensor:
        origin, R = self.hand_frame
        return (pos_env - origin) @ R

    # ------------------------------------------------------------------ simulation

    def _sim_step(self, render: bool):
        self.robot.set_joint_position_target(self.joint_target)
        self.scene.write_data_to_sim()
        for k in range(C.DECIMATION):
            self.sim.step(render=render and k == C.DECIMATION - 1)
            self.scene.update(C.PHYSICS_DT)

    def _calibrate_palm_normal(self):
        """Flex all fingers briefly (no object nearby) to find which side of the palm they curl to."""
        tips0 = self.robot.data.body_pos_w[0, self.tip_body_ids].clone()
        flexed = self.open_posture.clone()
        for i, n in enumerate(self.joint_names):
            if re.fullmatch(r"(index|middle|ring)_joint_[12]", n):
                flexed[:, i] = 0.8
        self.joint_target = flexed
        for _ in range(30):
            self._sim_step(render=False)
        tips1 = self.robot.data.body_pos_w[0, self.tip_body_ids]
        self._flex_dir = (tips1 - tips0)[:3].mean(0)

    def _level_hand(self):
        """Re-orient the fixed hand: palm frame x along +X pitched up by PALM_TILT_DEG, z up, then rolled by
        PALM_ROLL_DEG (ring-finger side up).

        Round objects have no rolling resistance in PhysX, so a tilted palm lets spheres/cylinders roll
        out before the fingers close. The palm-frame origin is placed at (0, 0, PALM_HEIGHT) per env.
        """
        origin, R = self.hand_frame
        c, s = math.cos(math.radians(C.PALM_TILT_DEG)), math.sin(math.radians(C.PALM_TILT_DEG))
        pitch = torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], device=self.device)
        c, s = math.cos(math.radians(C.PALM_ROLL_DEG)), math.sin(math.radians(C.PALM_ROLL_DEG))
        roll = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], device=self.device)
        R_target = pitch @ roll
        D = R_target @ R.T
        q_old = self.robot.data.root_quat_w[0]
        p_old = self.robot.data.root_pos_w[0] - self.scene.env_origins[0]
        q_new = math_utils.quat_mul(math_utils.quat_from_matrix(D), q_old)
        p_new = torch.tensor([0.0, 0.0, C.PALM_HEIGHT], device=self.device) + D @ (p_old - origin)
        pose = torch.cat([self.scene.env_origins + p_new, q_new.expand(self.num_envs, -1)], dim=-1)
        self.robot.write_root_pose_to_sim(pose)

    def _open_and_measure(self):
        n = self.num_envs
        all_ids = torch.arange(n, device=self.device)
        far = self.scene.env_origins + torch.tensor([0.0, 0.0, -5.0], device=self.device)
        pose = torch.cat([far, torch.tensor([[1.0, 0, 0, 0]], device=self.device).expand(n, -1)], -1)
        self.object.write_root_pose_to_sim(pose, env_ids=all_ids)
        self._flex_dir = None
        for _ in range(2):
            q = self.open_posture
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=all_ids)
            self.joint_target = q.clone()
            self._sim_step(render=False)
            if self._flex_dir is None:
                self._calibrate_palm_normal()
        self._compute_hand_frame()

    def calibrate(self):
        """Level the palm and compute the palm frame (object parked away). Called by :meth:`reset` if needed."""
        self._open_and_measure()
        self._level_hand()
        self._open_and_measure()

    def reset(self, episode: dict[str, torch.Tensor], render_every: int = 1) -> list[dict]:
        """Place the object in the open hand, close into the grasp posture, then let go of the object.

        ``episode`` (all per env, see sampling.py):
            ``object_pos`` (N, 3), ``object_quat`` (N, 4): object pose in the palm frame while pinned
            ``grasp_posture`` (N, J): joint targets at the end of the closing phase
            ``empty`` (N,) bool: no object in the hand (it waits on the ground at PARK_POS)

        Like an operator placing the object, the object is held fixed (pinned) while the hand opens and
        closes; it is released at the start of the hold phase and settles into the grasp under gravity.
        Returns the observations recorded during this grasping phase (t < 0). Cameras (if any) render on
        every ``render_every``-th step (0: never).
        """
        n = self.num_envs
        all_ids = torch.arange(n, device=self.device)
        if self.hand_frame is None:
            self.calibrate()

        pos_env = self.hand_to_env(episode["object_pos"])
        quat_env = math_utils.quat_mul(self.hand_quat.expand(n, -1), episode["object_quat"])
        empty = episode["empty"].to(self.device)
        pos_env[empty] = torch.tensor(PARK_POS, device=self.device)
        quat_env[empty] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        pin_pose = torch.cat([self.scene.env_origins + pos_env, quat_env], dim=-1)
        self.grasp_posture = self.clamp_joints(episode["grasp_posture"].to(self.device))

        q0 = self.open_posture
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=all_ids)
        self.gt_state.zero_()

        frames = []
        t = -(C.PLACE_S + C.CLOSE_S + C.HOLD_S)
        n_place, n_close, n_hold = (round(s / self.dt) for s in (C.PLACE_S, C.CLOSE_S, C.HOLD_S))
        for k in range(n_place + n_close + n_hold):
            if k < n_place + n_close:  # object pinned
                self.object.write_root_pose_to_sim(pin_pose, env_ids=all_ids)
                self.object.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=all_ids)
            if k < n_place:
                u = self.open_posture
            elif k < n_place + n_close:
                a = (k - n_place + 1) / n_close
                a = 0.5 - 0.5 * math.cos(a * math.pi)  # smooth ramp
                u = (1 - a) * self.open_posture + a * self.grasp_posture
            else:
                u = self.grasp_posture
            frames.append(self.step(u, t=t, render=render_every > 0 and k % render_every == 0))
            t += self.dt
        return frames

    def step(self, joint_target: torch.Tensor, t: float = 0.0, render: bool = True) -> dict:
        self.joint_target = joint_target.clone()
        render = render and self.camera is not None
        self._sim_step(render=render)
        return self.observe(t, vision=render)

    # ------------------------------------------------------------------ sensing

    def observe(self, t: float = 0.0, vision: bool = True) -> dict:
        # fingertip: average net force over the physics sub-steps of this control step, in link frame
        f_w = self.tip_sensor.data.net_forces_w_history[:, :, self.tip_sensor_ids].mean(1)  # (N, 4, 3)
        q_tip = self.robot.data.body_quat_w[:, self.tip_body_ids]  # (N, 4, 4)
        tip_force = math_utils.quat_apply_inverse(q_tip.reshape(-1, 4), f_w.reshape(-1, 3)).reshape(f_w.shape)

        # ground truth: |force| between each hand link and the object (filtered), with hysteresis
        gt_force = torch.stack(
            [s.data.force_matrix_w_history[:, :, 0, 0].mean(1).norm(dim=-1) for s in self.gt_sensors], dim=-1
        )
        self.gt_state = torch.where(self.gt_state, gt_force > C.GT_CONTACT_OFF_N, gt_force > C.GT_CONTACT_ON_N)

        n = self.num_envs
        obj_pos = self.env_to_hand(self.object.data.root_pos_w - self.scene.env_origins)
        obj_quat = math_utils.quat_mul(math_utils.quat_conjugate(self.hand_quat).expand(n, -1),
                                       self.object.data.root_quat_w)
        obs = {
            "t": t,
            "tip_force": tip_force,
            "joint_pos": self.robot.data.joint_pos.clone(),
            "joint_target": self.joint_target.clone(),
            "gt_force": gt_force,
            "gt_contact": self.gt_state.clone(),
            "object_pos": obj_pos,
            "object_quat": obj_quat,
        }
        if vision and self.camera is not None:
            obs["rgb"] = self.camera.data.output["rgb"][..., :3].clone()
            obs["depth"] = self.camera.data.output["distance_to_image_plane"][..., 0].clone()
        return obs

    def set_camera_view(self, eye_hand: tuple[float, float, float], target_hand=(0.0, 0.0, 0.03)):
        """Place every env camera; eye/target given in the palm frame."""
        eye = self.hand_to_env(torch.tensor(eye_hand, device=self.device))
        target = self.hand_to_env(torch.tensor(target_hand, device=self.device))
        origins = self.scene.env_origins
        self.camera.set_world_poses_from_view(origins + eye, origins + target)
