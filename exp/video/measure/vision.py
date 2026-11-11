r"""The probes of a vision's representation $z$ (M15, design 8.4), and the
baselines that make them mean something.

A probe is a ridge regression from features ``(n, F)`` to targets ``(n, K)``,
fit on the samples where ``half`` holds and scored ($R^2$) on the others. The
features and the targets are centered on the fit samples (an intercept
without penalty) and the features divided by one scale, the root mean
variance of a feature there, so that the probe does not change when $z$ is
rotated or scaled as a whole; the penalty $\alpha n$ on the weights is chosen
for each target from ``RIDGES`` by ``FOLDS``-fold cross-validation within the
fit samples (contiguous folds, so that the windows of one clip stay together
when the samples are clip-major), the largest within one standard error of
the least error (``_one_se``). The choice keeps a probe of many features
(the raw pixels) from overfitting, and one of a few exact features at the
smallest penalty. The scale is common, not one per feature: a feature that
hardly varies on the fit samples (a pixel the balls there almost never lit,
a dimension of $z$ that died) stays small, so the penalty keeps it from
fitting a few samples' residuals, which cross-validation does not see (the
folds without those samples hold it constant) and which a feature that
varies on the scored samples would carry there as error.

- ``ball_probe``: $R^2$ of each ball's position (and velocity) from $z$;
- ``thumbnail_probe``: $R^2$ of the ``size`` x ``size`` grey thumbnail of
  the frame, pooled over its pixels ($1 - \sum \mathrm{SSE} / \sum
  \mathrm{SST}$, so that a pixel of little variance does not weigh as much
  as one of much);
- ``vision_metrics``: the effective rank of $z$, the least standard
  deviation of a dimension (0 on a collapse), and the probes, of an encoder
  on a ``ProbeSet``;
- ``baselines``: the same of the vision before it learns (``init_*``) and the
  ball probes from the raw pixels of the chunks (``pixels_*``). A vision that
  beats neither has not learned. There is no pixel baseline of the thumbnail:
  the thumbnail is a linear map of the pixels, so its pixel probe is 1.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from exp.lib.metrics import effective_rank
from exp.video.jepa.vit import build_vision
from exp.video.schema.run import RunConf

__all__ = [
    "BATCH",
    "FOLDS",
    "RIDGES",
    "THUMB",
    "ProbeSet",
    "ball_probe",
    "baselines",
    "thumbnail_probe",
    "vision_metrics",
]

RIDGES = tuple(10.0**e for e in range(-6, 3))  # alpha: the penalty is alpha n
FOLDS = 5
THUMB = 8  # the side of the thumbnail
BATCH = 256  # chunks encoded at once
_STATE = ("x", "y", "vx", "vy")  # the columns of a ball in the states


def _ridge_predict(x: Tensor, y: Tensor, tx: Tensor, alphas: Tensor) -> Tensor:
    r"""The predictions of the ridge regression from ``x`` ``(n, F)`` to ``y``
    ``(n, K)`` at ``tx`` ``(m, F)``, for each penalty ``alphas`` ``(R,)``:
    ``(R, m, K)`` float64.

    With the scaled features $A = (X - \bar X) / \sigma = U S V^\top$ ($\sigma$
    the root mean variance of a feature, module docstring; thin SVD, so it
    costs $O(n F \min(n, F))$ whichever of $n$ and $F$ is larger), the
    weights are $W = V\,\mathrm{diag}(s / (s^2 + \alpha n))\,U^\top (Y -
    \bar Y)$.
    """
    x, y, tx = x.double(), y.double(), tx.double()
    mean = x.mean(0)
    scale = float(x.var(0).mean().sqrt())
    scale = scale if scale > 0 else 1.0  # constant features
    a, b = (x - mean) / scale, (tx - mean) / scale
    y_mean = y.mean(0)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)  # (n, r), (r,), (r, F)
    uy = u.T @ (y - y_mean)  # (r, K)
    bv = b @ vh.T  # (m, r)
    lam = alphas.to(s) * len(x)  # (R,)
    f = s / (s * s + lam[:, None])  # (R, r)
    return torch.einsum("mr,Rr,rk->Rmk", bv, f, uy) + y_mean


def _one_se(err: Tensor) -> Tensor:
    r"""The index ``(K,)`` into ``RIDGES`` of each target's penalty, from the
    cross-validated squared errors ``err`` ``(R, n, K)`` of each penalty and
    sample: the largest whose mean error is within one standard error (of
    the mean over the samples) of the least (the one-standard-error rule).

    Where the error hardly changes with the penalty, this takes the largest
    such: a feature that is constant in the other folds (seen at a few fit
    samples only) leaves the cross-validation blind to what fitting it costs
    on the scored samples, and only the penalty keeps it from being fit.
    Where the targets are exact, the error grows with the penalty much more
    than its standard error, and the least is kept.
    """
    mean = err.mean(1)  # (R, K)
    least = mean.argmin(0, keepdim=True)  # (1, K)
    at_least = err.gather(0, least[:, None].expand(1, err.shape[1], -1))[0]
    bar = mean.gather(0, least)[0] + at_least.std(0) / err.shape[1] ** 0.5  # (K,)
    within = mean <= bar  # (R, K); the least is within
    index = torch.arange(len(err), device=err.device)[:, None]
    return torch.where(within, index, -1).amax(0)


def _probe(x: Tensor, y: Tensor, half: Tensor) -> tuple[Tensor, Tensor]:
    """The predictions ``(m, K)`` float64 of the probe from ``x`` ``(n, F)`` to
    ``y`` ``(n, K)`` (module docstring) at the ``m`` samples where ``half``
    ``(n,)`` is false, and their targets ``(m, K)`` float64.

    Raises:
        ValueError: Fewer than ``FOLDS`` samples to fit on, or none to score.
    """
    fit_x, fit_y, test_x = x[half], y[half].double(), x[~half]
    n = len(fit_x)
    if n < FOLDS or len(test_x) == 0:
        raise ValueError(f"a probe needs {FOLDS} samples to fit and 1 to score")
    alphas = torch.tensor(RIDGES, dtype=torch.float64)
    bounds = [n * j // FOLDS for j in range(FOLDS + 1)]
    errs = []  # (R, hi - lo, K) each: the squared errors of each held-out fold
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        held = torch.zeros(n, dtype=torch.bool, device=x.device)
        held[lo:hi] = True
        pred = _ridge_predict(fit_x[~held], fit_y[~held], fit_x[held], alphas)
        errs.append((pred - fit_y[held]) ** 2)
    err = torch.cat(errs, 1)  # (R, n, K)
    best = _one_se(err)  # (K,): the penalty of each target
    pred = _ridge_predict(fit_x, fit_y, test_x, alphas)  # (R, m, K)
    chosen = pred.gather(0, best.view(1, 1, -1).expand(1, len(test_x), -1))[0]
    return chosen, y[~half].double()


def _sums(pred: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """The squared errors of ``pred`` and the squares about the mean of ``y``,
    summed over the samples: ``(K,)`` each."""
    return ((pred - y) ** 2).sum(0), ((y - y.mean(0)) ** 2).sum(0)


def ball_probe(
    z: Tensor, states: Tensor, balls: int, half: Tensor, *, velocity: bool = False
) -> dict[str, float]:
    r"""The $R^2$ of the probe from ``z`` ``(n, D)`` to each ball's position
    (and, with ``velocity``, velocity), fit where ``half`` ``(n,)`` holds and
    scored on the others (module docstring).

    Args:
        states: ``(n, 4 balls)``: x, y, vx, vy of each ball in turn.

    Returns:
        ``probe_x{k}``, ``probe_y{k}`` for each ball ``k`` and their least
        ``probe_pos_min``; with ``velocity`` also ``probe_vx{k}``,
        ``probe_vy{k}`` and ``probe_vel_min``.
    """
    if states.shape[1] != 4 * balls:
        raise ValueError(f"states of {states.shape[1]} columns, not 4 x {balls}")
    names = _STATE if velocity else _STATE[:2]
    cols = [4 * k + j for k in range(balls) for j in range(len(names))]
    sse, sst = _sums(*_probe(z, states[:, cols], half))
    r2 = (1 - sse / sst).tolist()
    out = {
        f"probe_{names[j]}{k}": r2[k * len(names) + j]
        for k in range(balls)
        for j in range(len(names))
    }
    out["probe_pos_min"] = min(out[f"probe_{a}{k}"] for k in range(balls) for a in "xy")
    if velocity:
        out["probe_vel_min"] = min(
            out[f"probe_{a}{k}"] for k in range(balls) for a in ("vx", "vy")
        )
    return out


def _thumbnail(images: Tensor, size: int) -> Tensor:
    """The grey (the mean of the channels) of ``images`` ``(n, C, H, W)``
    shrunk to ``(n, size * size)`` by the mean of each block."""
    grey = images.mean(1, keepdim=True)
    return nn.functional.adaptive_avg_pool2d(grey, size).flatten(1)


def thumbnail_probe(
    z: Tensor, images: Tensor, half: Tensor, size: int = THUMB
) -> float:
    r"""The $R^2$, pooled over the pixels (module docstring), of the probe
    from ``z`` ``(n, D)`` to the ``size`` x ``size`` grey thumbnail of
    ``images`` ``(n, C, H, W)``, fit where ``half`` ``(n,)`` holds."""
    sse, sst = _sums(*_probe(z, _thumbnail(images, size), half))
    return float(1 - sse.sum() / sst.sum())


@dataclass(frozen=True)
class ProbeSet:
    """The chunks a vision is probed on.

    ``chunks`` ``(n, N, C, H, W)`` in ``[0, 1]``; ``half`` ``(n,)`` bool,
    true for the chunks the probes fit on; ``states`` ``(n, 4K)`` the balls'
    states at each chunk's last frame, or ``None`` (Walking Tours).
    """

    chunks: Tensor
    half: Tensor
    states: Tensor | None = None

    def __post_init__(self) -> None:
        n = len(self.chunks)
        if self.chunks.dim() != 5 or self.half.shape != (n,):
            raise ValueError(
                f"chunks (n, N, C, H, W) and half (n,), not "
                f"{tuple(self.chunks.shape)} and {tuple(self.half.shape)}"
            )
        if self.states is not None and (
            len(self.states) != n or self.states.shape[1] % 4
        ):
            raise ValueError(f"states (n, 4K), not {tuple(self.states.shape)}")

    @classmethod
    def halved(cls, chunks: Tensor, states: Tensor | None = None) -> ProbeSet:
        """The probes fit on the first half of the chunks (by order) and scored
        on the rest."""
        half = torch.arange(len(chunks), device=chunks.device) < len(chunks) // 2
        return cls(chunks, half, states)

    @property
    def balls(self) -> int:
        """The balls of ``states`` (0 without states)."""
        return 0 if self.states is None else self.states.shape[1] // 4


@torch.no_grad()
def vision_metrics(
    encode: Callable[[Tensor], Tensor], clips: ProbeSet, *, batch: int = BATCH
) -> dict[str, float]:
    """The metrics of ``encode`` (chunks ``(b, N, C, H, W)`` to $z$ ``(b, D)``)
    on ``clips``, encoded ``batch`` chunks at a time: ``rank`` (the effective
    rank of $z$), ``z_std_min`` (the least standard deviation of a dimension),
    ``thumb`` (``thumbnail_probe`` of the chunks' last frames) and, with
    states, ``ball_probe`` (with the velocity when $N > 1$)."""
    z = torch.cat([encode(c) for c in clips.chunks.split(batch)])
    row = {
        "rank": effective_rank(z),
        "z_std_min": float(z.std(0).min()),
        "thumb": thumbnail_probe(z, clips.chunks[:, -1], clips.half),
    }
    if clips.states is not None:
        velocity = clips.chunks.shape[1] > 1
        row |= ball_probe(z, clips.states, clips.balls, clips.half, velocity=velocity)
    return row


def baselines(conf: RunConf, clips: ProbeSet) -> dict[str, float]:
    """The bars of ``vision_metrics`` on ``clips`` (on ``conf.device``):
    ``init_*``, the metrics of the vision of ``conf`` at its initial weights
    (``build_vision`` of ``conf.seed``, as the run starts), and, with states,
    ``pixels_*``, the ball probes from the raw pixels of the chunks."""
    init = build_vision(conf, seed=conf.seed)
    out = {f"init_{k}": v for k, v in vision_metrics(init.encode, clips).items()}
    if clips.states is not None:
        velocity = clips.chunks.shape[1] > 1
        pixels = ball_probe(
            clips.chunks.flatten(1),
            clips.states,
            clips.balls,
            clips.half,
            velocity=velocity,
        )
        out |= {f"pixels_{k}": v for k, v in pixels.items()}
    return out
