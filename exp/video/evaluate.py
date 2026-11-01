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
from exp.video.model import DynamicsStack, Network, Top

__all__ = [
    "Snapshot",
    "evaluate",
    "halves",
    "mean_errors",
    "persistence_errors",
    "rollout_errors",
    "rollout_shape",
    "shrink_errors",
    "shrink_predictions",
]

HORIZONS = (1, 5, 10, 20)


@dataclass
class Snapshot:
    """Copies of the modules of a ``Network``, taken between two steps."""

    encoder: nn.Module  # o -> z
    step: nn.Module  # (z_{k-1}, s_{k-1}) -> s_k
    readout: nn.Module  # s -> z_hat (or the change of z, with residual)
    residual: bool = False  # z_hat = z_{k-1} + readout(s)

    @classmethod
    def of(cls, net: Network) -> Snapshot:
        enc = nn.Sequential(*(getattr(net, n).module for n in net.encoder_names))
        layers = [getattr(net, n).module for n in net.dynamics_names[1:]]
        step = DynamicsStack(net.dynamics.module, layers)
        readout = nn.Sequential(Top(step.d_s), net.compare.module)
        return cls(
            copy.deepcopy(enc),
            copy.deepcopy(step),
            copy.deepcopy(readout),
            net.residual,
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
    residual: bool = False,
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
        z_in = z_in + readout(s) if residual else readout(s)
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
    residual: bool = False,
) -> dict[int, float]:
    r"""The mean of $\frac1D \lVert \hat z - z \rVert^2$ at ``n`` frames past
    the ``observed`` ones (``n = 1``: the frame right after), for each ``n`` of
    ``horizons``."""
    _, preds = run_states(z, step, readout, d_s, observed, max(horizons), residual)
    out = {}
    for h in horizons:
        target = z[:, observed - 1 + h]
        out[h] = float(((preds[:, h - 1] - target) ** 2).mean())
    return out


def persistence_errors(
    z: torch.Tensor, observed: int = 16, horizons: Sequence[int] = HORIZONS
) -> dict[int, float]:
    r"""The errors of ``rollout_errors`` for the model that says the last
    observed $z$ again: how much $z$ changes in ``n`` frames, the bar a
    rollout has to clear to have learned any dynamics."""
    last = z[:, observed - 1]
    return {h: float(((last - z[:, observed - 1 + h]) ** 2).mean()) for h in horizons}


def mean_errors(
    z: torch.Tensor, observed: int = 16, horizons: Sequence[int] = HORIZONS
) -> dict[int, float]:
    r"""The errors of ``rollout_errors`` for the model that says the mean of
    $z$ over all the clips and frames: a constant, the bar for a rollout that
    knows nothing. It is not 1 (relative to the variance of $z$ across the
    clips), as the mean moves within the clips."""
    mean = z.mean((0, 1))
    return {h: float(((mean - z[:, observed - 1 + h]) ** 2).mean()) for h in horizons}


def halves(phases: int, clips: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """The two halves of the clips ``(phases * clips,)`` (phase-major, as the
    clips of ``evaluate``): ``True`` for the first half of the clips of each
    phase."""
    return (torch.arange(clips, device=device) < clips // 2).repeat(phases)


def shrink_errors(
    z: torch.Tensor,
    half: torch.Tensor,
    observed: int = 16,
    horizons: Sequence[int] = HORIZONS,
) -> dict[int, float]:
    r"""The errors of ``rollout_errors`` for the model that says the last
    observed $z$ shrunk toward the mean $m$ of ``mean_errors``,
    $\hat z = m + \lambda_n (z_\text{last} - m)$ (``shrink_predictions``),
    over all the clips. It knows no dynamics, only how fast $z$ forgets: the
    bar a rollout has to clear to have learned more than that."""
    preds = shrink_predictions(z, half, observed, horizons)
    return {
        h: float(((preds[h] - z[:, observed - 1 + h]) ** 2).mean()) for h in horizons
    }


def shrink_predictions(
    z: torch.Tensor,
    half: torch.Tensor,
    observed: int = 16,
    horizons: Sequence[int] = HORIZONS,
) -> dict[int, torch.Tensor]:
    r"""The predictions $m + \lambda_n (z_\text{last} - m)$ ``(n, D)`` of the
    clips ``z`` ``(n, T, D)`` at each horizon, out of sample: $\lambda_n$ is
    fitted by least squares on the clips where ``half`` (``(n,)`` bool) holds
    for the others, and on the others for those. $m$ is the mean over all the
    clips and frames, as in ``mean_errors``."""
    m = z.mean((0, 1))
    a = z[:, observed - 1] - m  # (n, D)
    out = {}
    for h in horizons:
        b = z[:, observed - 1 + h] - m
        pred = torch.empty_like(a)
        for fit in (half, ~half):
            lam = (a[fit] * b[fit]).sum() / (a[fit] ** 2).sum()
            pred[~fit] = m + lam * a[~fit]
        out[h] = pred
    return out


def rollout_shape(
    z: torch.Tensor,
    step: Step,
    readout: Readout,
    d_s: int,
    observed: int = 16,
    horizons: Sequence[int] = HORIZONS,
    residual: bool = False,
) -> dict[str, float]:
    r"""How the rollout of ``rollout_errors`` misses at each horizon ``n``,
    across the clips: ``amp{n}``, the variance of the predictions over that of
    the targets, $\sum_d \text{Var}(\hat z_d) / \sum_d \text{Var}(z_d)$
    (below 1 they shrink toward their mean, above 1 they overshoot), and
    ``corr{n}``, their correlation, $\sum_d \text{Cov}(\hat z_d, z_d)$ over
    the square root of the product of the two sums (0 for predictions that
    know nothing of the clip)."""
    _, preds = run_states(z, step, readout, d_s, observed, max(horizons), residual)
    out = {}
    for h in horizons:
        p = preds[:, h - 1] - preds[:, h - 1].mean(0)
        t = z[:, observed - 1 + h] - z[:, observed - 1 + h].mean(0)
        vp, vt = (p**2).sum(), (t**2).sum()
        out[f"amp{h}"] = float(vp / vt)
        out[f"corr{h}"] = float((p * t).sum() / (vp * vt).sqrt().clamp(min=1e-30))
    return out


@torch.no_grad()
def evaluate(snap: Snapshot, clips: Frame, d_s: int) -> dict[str, float]:
    """The metrics of section 9 on ``clips`` (``(P, C, T, ...)``): the rollout
    errors relative to the variance of $z$ (and those of
    ``persistence_errors``, ``mean_errors`` and ``shrink_errors``, fitted on
    the ``halves`` of the clips), the ``rollout_shape``, the probe $R^2$ of the positions,
    velocities and physics from $z$ and from $s$ (when the clips have them),
    and the effective rank of $z$. ``d_s`` is the size of the state of
    ``snap.step``."""
    p, c, t = clips.image.shape[:3]
    images = clips.image.reshape(-1, *clips.image.shape[3:])
    # In chunks, to keep the memory of the activations small on a small GPU.
    z = torch.cat([snap.encoder(x) for x in images.split(1024)]).reshape(p * c, t, -1)
    var = float(z.var(0).mean())  # per dimension, over the clips
    out: dict[str, float] = {
        "z_var": var,
        "rank": effective_rank(z.reshape(p * c * t, -1)),
    }
    rollouts = rollout_errors(z, snap.step, snap.readout, d_s, residual=snap.residual)
    for h, err in rollouts.items():
        out[f"rollout{h}"] = err / var if var > 0 else math.nan  # z collapsed
    for h, err in persistence_errors(z).items():
        out[f"persist{h}"] = err / var if var > 0 else math.nan
    for h, err in mean_errors(z).items():
        out[f"mean{h}"] = err / var if var > 0 else math.nan
    for h, err in shrink_errors(z, halves(p, c, z.device)).items():
        out[f"shrink{h}"] = err / var if var > 0 else math.nan
    out.update(rollout_shape(z, snap.step, snap.readout, d_s, residual=snap.residual))
    if clips.state.shape[-1] == 0:  # no latent state to probe (EgoDex)
        return out
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
