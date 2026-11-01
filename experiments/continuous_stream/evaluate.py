"""The evaluation of PLAN.md, section 9, on copies of the modules in plain
PyTorch: the rollout, the linear probe and the effective rank on the clips of
``video.eval_clips``, and the correlation of the gradients in time."""

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from experiments.continuous_stream.model import Network
from experiments.continuous_stream.video import Frame

__all__ = [
    "Snapshot",
    "effective_rank",
    "evaluate",
    "probe_r2",
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


def effective_rank(z: torch.Tensor) -> float:
    """The exponential of the entropy of the normalized eigenvalues of the
    covariance of ``z`` (``(n, D)``)."""
    cov = torch.cov(z.T.double())
    values = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = values / values.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def probe_r2(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """The $R^2$ on the test set of each target of the ridge regression from
    ``x`` (``(n, F)``) to ``y`` (``(n, K)``), the features standardized on the
    train set. Returns ``(K,)``."""
    mean, std = train_x.mean(0), train_x.std(0).clamp(min=1e-8)
    a = torch.cat(
        [(train_x - mean) / std, train_x.new_ones(len(train_x), 1)], 1
    ).double()
    b = torch.cat([(test_x - mean) / std, test_x.new_ones(len(test_x), 1)], 1).double()
    eye = torch.eye(a.shape[1], dtype=a.dtype, device=a.device)
    eye[-1, -1] = 0  # no penalty on the intercept
    w = torch.linalg.solve(a.T @ a + ridge * len(a) * eye, a.T @ train_y.double())
    resid = ((b @ w - test_y.double()) ** 2).sum(0)
    total = ((test_y.double() - test_y.double().mean(0)) ** 2).sum(0)
    return (1 - resid / total).float()


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
