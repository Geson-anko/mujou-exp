"""The evaluation of PLAN.md, section 9, on copies of the modules in plain
PyTorch: the rollout, the linear probe and the effective rank on the clips of
``balls.eval_clips``, and the correlation of the gradients in time."""

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from exp.lib.metrics import effective_rank, probe_r2
from exp.video.balls import Frame
from exp.video.model import Network

__all__ = [
    "Snapshot",
    "evaluate",
    "rollout_errors",
]

HORIZONS = (1, 5, 10, 20)


@dataclass
class Snapshot:
    """Copies of the modules of a ``Network``, taken between two steps."""

    encoder: nn.Module  # o -> z
    step: nn.Module  # (z_{k-1}, s_{k-1}) -> s_k
    readout: nn.Module  # s -> z_hat

    @classmethod
    def of(cls, net: Network) -> Snapshot:
        enc = nn.Sequential(net.g1.module, net.z.module)
        return cls(
            copy.deepcopy(enc),
            copy.deepcopy(net.dynamics.module),
            copy.deepcopy(net.compare.module),
        )


Step = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Readout = Callable[[torch.Tensor], torch.Tensor]


def run_states(
    z: torch.Tensor,
    step: Step,
    readout: Readout,
    d_s: int,
    observed: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Observe ``z`` (``(n, T, D)``) for ``observed`` frames from $s = 0$, then
    run the model open loop for ``steps`` frames with its own prediction as the
    next input. Returns the states ``(n, observed + steps, d_s)`` ($s_k$, with
    $s_0 = 0$) and the predictions ``(n, steps, D)`` of the frames
    ``observed, ..., observed + steps - 1``."""
    n = z.shape[0]
    s = z.new_zeros(n, d_s)
    states = [s]
    for k in range(1, observed):
        s = step(z[:, k - 1], s)
        states.append(s)
    preds = []
    z_in = z[:, observed - 1]
    for _ in range(steps):
        s = step(z_in, s)
        z_in = readout(s)
        states.append(s)
        preds.append(z_in)
    preds_ = torch.stack(preds, 1) if preds else z.new_zeros(n, 0, z.shape[-1])
    return torch.stack(states, 1), preds_


def rollout_errors(
    z: torch.Tensor,
    step: Step,
    readout: Readout,
    d_s: int,
    observed: int = 16,
    horizons: Sequence[int] = HORIZONS,
) -> dict[int, float]:
    r"""The mean of $\frac1D \lVert \hat z - z \rVert^2$ at ``n`` frames past
    the ``observed`` ones (``n = 1``: the frame right after), for each ``n`` of
    ``horizons``."""
    _, preds = run_states(z, step, readout, d_s, observed, max(horizons))
    out = {}
    for h in horizons:
        target = z[:, observed - 1 + h]
        out[h] = float(((preds[:, h - 1] - target) ** 2).mean())
    return out


@torch.no_grad()
def evaluate(snap: Snapshot, clips: Frame, d_s: int) -> dict[str, float]:
    """The metrics of section 9 on ``clips`` (``(P, C, T, ...)``): the rollout
    errors relative to the variance of $z$, the probe $R^2$ of the positions,
    velocities and physics from $z$ and from $s$, and the effective rank of $z$."""
    p, c, t = clips.image.shape[:3]
    z = snap.encoder(clips.image.reshape(-1, *clips.image.shape[3:])).reshape(
        p * c, t, -1
    )
    var = float(z.var(0).mean())  # per dimension, over the clips
    out: dict[str, float] = {
        "z_var": var,
        "rank": effective_rank(z.reshape(p * c * t, -1)),
    }
    for h, err in rollout_errors(z, snap.step, snap.readout, d_s).items():
        out[f"rollout{h}"] = err / var if var > 0 else math.nan  # z collapsed
    states, _ = run_states(z, snap.step, snap.readout, d_s, t, 0)
    half = c // 2  # train on the first half of the clips of each phase
    targets = torch.cat([clips.state, clips.params], -1).reshape(p, c, t, -1)
    feats = {"z": z.reshape(p, c, t, -1), "s": states.reshape(p, c, t, -1)}
    balls = clips.state.shape[-1] // 4
    idx = torch.arange(4 * balls).reshape(balls, 4)
    names = {
        "pos": idx[:, :2].flatten(),
        "vel": idx[:, 2:].flatten(),
        "phys": torch.tensor([4 * balls, 4 * balls + 1]),
    }
    for fname, f in feats.items():
        r2 = probe_r2(
            f[:, :half].reshape(-1, f.shape[-1]),
            targets[:, :half].reshape(-1, targets.shape[-1]),
            f[:, half:].reshape(-1, f.shape[-1]),
            targets[:, half:].reshape(-1, targets.shape[-1]),
        )
        for tname, cols in names.items():
            out[f"probe_{fname}_{tname}"] = float(r2[cols].mean())
    return out
