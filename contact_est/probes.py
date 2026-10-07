"""Scripted exploratory motions (draft section 4/5), 10 s each, added on top of the grasp posture.

Motion "B" (per-finger probing)          Motion "A" (whole-hand squeeze, same length)
  0-2 s  hold                              0-2 s  hold
  2-4 s  all fingers close a bit           2-4 s  all fingers close a bit
  4-6 s  index only closes                 4-6 s  all fingers close a bit
  6-8 s  middle only closes                6-8 s  all fingers close a bit
  8-10 s one finger opens a bit            8-10 s all fingers open a bit

Each bump is a raised cosine. Its start time and amplitude are jittered per env and per episode so
that a model cannot memorise "contact always happens at t = 4.3 s"; the actual command is logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

DURATION_S = 10.0
SEGMENT_S = 2.0
BUMP_S = 1.4  # bump length inside each 2 s segment
START_JITTER_S = 0.15
AMP_JITTER = 0.2  # amplitude scaled by U(1 - j, 1 + j)

# rad added per flexion joint at the peak of a "close" bump (MCP, PIP, DIP / thumb chain)
FLEX_DELTA = {"1": 0.18, "2": 0.18, "3": 0.10}
THUMB_FLEX_DELTA = {"2": 0.12, "3": 0.10}


@dataclass
class Segment:
    t0: float
    fingers: tuple[str, ...]
    sign: float  # +1 close, -1 open


ALL = ("index", "middle", "ring", "thumb")

MOTIONS: dict[str, list[Segment]] = {
    "B": [
        Segment(2.0, ALL, +1),
        Segment(4.0, ("index",), +1),
        Segment(6.0, ("middle",), +1),
        Segment(8.0, ("ring",), -1),
    ],
    "A": [
        Segment(2.0, ALL, +1),
        Segment(4.0, ALL, +1),
        Segment(6.0, ALL, +1),
        Segment(8.0, ALL, -1),
    ],
}


SEGMENT_LABELS: dict[str, list[str]] = {
    "B": ["all close", "index close", "middle close", "ring open"],
    "A": ["all close", "all close", "all close", "all open"],
}


class ProbeMotion:
    """Vectorised scripted probe: ``delta(t)`` -> (num_envs, num_joints) joint-target offsets.

    ``motion`` is one motion name for all envs or one per env. All motions share the same segment
    timing, so only the per-segment joint pattern differs. ``start`` / ``amp`` (num_envs, segments) hold
    the jittered bump start times [s] and amplitude scales (logged with the data).
    """

    def __init__(
        self, motion: str | list[str], joint_names: list[str], num_envs: int, device: str, jitter: bool = True
    ):
        motions = [motion] * num_envs if isinstance(motion, str) else list(motion)
        self.num_envs = num_envs
        self.device = device
        self.segments = MOTIONS[motions[0]]  # timing reference (identical across motions)
        # per-env, per-segment joint delta pattern (N, S, J)
        table = {
            name: torch.stack([self._pattern(seg.fingers, joint_names) * seg.sign for seg in segs])
            for name, segs in MOTIONS.items()
        }
        self.patterns = torch.stack([table[m] for m in motions]).to(device)
        n_seg = len(self.segments)
        t0 = torch.tensor([s.t0 for s in self.segments], device=device)
        if jitter:
            self.start = t0 + 0.2 + START_JITTER_S * (2 * torch.rand(num_envs, n_seg, device=device) - 1)
            self.amp = 1.0 + AMP_JITTER * (2 * torch.rand(num_envs, n_seg, device=device) - 1)
        else:
            self.start = (t0 + 0.3).expand(num_envs, -1).clone()
            self.amp = torch.ones(num_envs, n_seg, device=device)

    @staticmethod
    def _pattern(fingers: tuple[str, ...], joint_names: list[str]) -> torch.Tensor:
        p = torch.zeros(len(joint_names))
        for i, name in enumerate(joint_names):
            m = re.fullmatch(r"(index|middle|ring|thumb)_joint_(\d)", name)
            if m is None or m.group(1) not in fingers:
                continue
            table = THUMB_FLEX_DELTA if m.group(1) == "thumb" else FLEX_DELTA
            p[i] = table.get(m.group(2), 0.0)
        return p

    def delta(self, t: float) -> torch.Tensor:
        phase = (t - self.start) / BUMP_S  # (N, S)
        inside = (phase >= 0) & (phase <= 1)
        w = torch.where(inside, 0.5 - 0.5 * torch.cos(2 * torch.pi * phase), torch.zeros_like(phase)) * self.amp
        return torch.einsum("ns,nsj->nj", w, self.patterns)

    def active_segment(self, t: float) -> torch.Tensor:
        """Index of the segment whose bump is active (-1 if holding), per env; handy for plots."""
        phase = (t - self.start) / BUMP_S
        inside = (phase >= 0) & (phase <= 1)
        idx = torch.where(inside.any(-1), inside.float().argmax(-1), torch.full_like(inside[:, 0], -1, dtype=torch.long))
        return idx
