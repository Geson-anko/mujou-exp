r"""The baselines of M14: what a rollout has to beat, on the windows of the
evaluation clips.

Every baseline predicts in the raw observation space and is then
standardized; an error is the mean over the windows and the dimensions of the
squared error in the standardized space, at ``h`` frames past the last
observed one (``h = 1``: the frame right after). With $p$ the positions of the
last three observed frames ($p_2$ the last) and $\Delta t$ the frame:

- ``mean``: the mean of the calibration clips, 0 in the standardized space
- ``copy``: the last observation again
- ``cv``: constant velocity, $p_h = p_2 + h v$, $v = p_2 - p_1$
- ``cvobs`` (``"pos_vel"`` only): constant velocity with the observed one,
  $p_h = p_2 + h \Delta t\,v_\text{obs}$
- ``ca``: constant acceleration, $p_h = p_2 + h v + h (h + 1) a / 2$,
  $a = p_2 - 2 p_1 + p_0$
- ``true``: the physics itself from the last observed state (0 by
  construction: the bar of a perfect model)

``cv`` below ``copy`` is motion; a rollout below ``cv`` has learned gravity,
below ``ca`` on the windows with a contact it has learned the bounces. A
contact counts from the first of the three frames ``ca`` reads
(``clips.STENCIL``), so on the windows without one (``@free``) ``ca`` is exact
under a constant gravity without drag.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from exp.video.data.clips import (
    STENCIL,
    ClipSet,
    contact_mask,
    simulate,
    window_states,
)
from exp.video.data.observe import PIXELS, Standardizer, observe, position_columns
from exp.video.data.physics import BallState
from exp.video.schema.data import BallsConf
from exp.video.schema.measure import MeasureConf

__all__ = [
    "baselines",
    "const_accel",
    "const_velocity",
    "const_velocity_observed",
    "copy_pred",
    "errors",
    "mean_pred",
    "true_physics",
]


def mean_pred(n: int, horizon: int, d: int) -> Tensor:
    """The prediction of the mean: ``(n, horizon, d)`` zeros in the
    standardized space."""
    return torch.zeros(n, horizon, d, dtype=torch.float64)


def copy_pred(obs: Tensor, observed: int, horizon: int) -> Tensor:
    """The last observed frame of ``obs`` ``(n, T, d)`` again:
    ``(n, horizon, d)`` (in either space: standardizing commutes with it)."""
    return obs[:, observed - 1 : observed].expand(-1, horizon, -1).clone()


def _steps(horizon: int, like: Tensor) -> Tensor:
    """``h`` = 1..horizon as ``(1, horizon, 1, 1)``."""
    return torch.arange(1, horizon + 1, dtype=like.dtype).view(1, -1, 1, 1)


def _assemble(obs: str, pos: Tensor, vel: Tensor) -> Tensor:
    """The observations ``(n, horizon, d)`` of positions ``(n, horizon, K, 2)``
    and, for ``"pos_vel"``, velocities ``(n, horizon, K, 2)``."""
    if obs == "pos":
        return pos.flatten(-2)
    return torch.cat([pos, vel], -1).flatten(-2)


def _positions(raw: Tensor, obs: str, balls: int) -> Tensor:
    """``(n, T, K, 2)`` of observations ``(n, T, d)``."""
    return raw[..., position_columns(obs, balls)]


def const_velocity(
    raw: Tensor, obs: str, balls: int, observed: int, horizon: int, frame: float
) -> Tensor:
    r"""$p_h = p_2 + h (p_2 - p_1)$ from the raw observations ``(n, T, d)``; for
    ``"pos_vel"`` the velocity columns are $(p_2 - p_1) / \Delta t$.

    Returns:
        ``(n, horizon, d)`` raw.
    """
    p = _positions(raw, obs, balls)
    p1, p2 = p[:, observed - 2 : observed - 1], p[:, observed - 1 : observed]
    v = p2 - p1
    pos = p2 + _steps(horizon, raw) * v
    return _assemble(obs, pos, (v / frame).expand_as(pos))


def const_velocity_observed(
    raw: Tensor, balls: int, observed: int, horizon: int, frame: float
) -> Tensor:
    r"""Stage 0 only: $p_h = p_2 + h \Delta t\,v_\text{obs}$ with the observed
    velocity kept, from the raw ``"pos_vel"`` observations ``(n, T, 4K)``.

    Returns:
        ``(n, horizon, 4K)`` raw.
    """
    s = raw[:, observed - 1 : observed].unflatten(-1, (balls, 4))  # (n, 1, K, 4)
    last, v = s[..., :2], s[..., 2:]
    pos = last + _steps(horizon, raw) * frame * v
    return _assemble("pos_vel", pos, v.expand_as(pos))


def const_accel(
    raw: Tensor, obs: str, balls: int, observed: int, horizon: int, frame: float
) -> Tensor:
    r"""$p_h = p_2 + h v + h (h + 1) a / 2$ with $v = p_2 - p_1$ and
    $a = p_2 - 2 p_1 + p_0$, from the raw observations ``(n, T, d)``; for
    ``"pos_vel"`` the velocity columns are $(p_h - p_{h-1}) / \Delta t =
    (v + h a) / \Delta t$.

    Returns:
        ``(n, horizon, d)`` raw.
    """
    p = _positions(raw, obs, balls)
    p0, p1, p2 = (p[:, observed - 3 + i : observed - 2 + i] for i in range(3))
    v, a = p2 - p1, p2 - 2 * p1 + p0
    h = _steps(horizon, raw)
    pos = p2 + h * v + h * (h + 1) / 2 * a
    return _assemble(obs, pos, (v + h * a) / frame)


def true_physics(
    clips: ClipSet, balls: BallsConf, observed: int, horizon: int
) -> Tensor:
    """The physics run from the last observed true state of each window of
    ``clips``, on the clock of that frame: the same doubles as the clips.
    ``balls`` must have the radii and the physics of the clips
    (``ValueError`` otherwise).

    Returns:
        ``(n_windows, horizon, 4K)`` float64 states.
    """
    clips.check(balls)
    states = clips.states
    n, k = len(clips.starts), len(balls.radii)
    out = states.new_empty(n, horizon, 4 * k)
    for first in clips.starts[:, 1].unique().tolist():
        rows = (clips.starts[:, 1] == first).nonzero().squeeze(1)
        j = first + observed - 1  # the last observed frame
        s = states[clips.starts[rows, 0], j].unflatten(-1, (k, 4))
        start = BallState(s[..., :2].clone(), s[..., 2:].clone())
        future = simulate(start, balls, frames=horizon + 1, frame=clips.frame, first=j)
        out[rows] = future[:, 1:]
    return out


def errors(
    pred: Tensor, target: Tensor, horizons: Sequence[int], mask: Tensor | None = None
) -> dict[int, float]:
    """The mean squared error at each horizon ``h``: of ``pred[:, h - 1]``
    against ``target[:, h - 1]``, over the windows (where ``mask`` holds) and
    the dimensions; NaN when ``mask`` keeps no window.

    Args:
        pred: ``(n, H, d)``, ``H >= max(horizons)``.
        target: ``(n, H, d)``.
        mask: ``(n, len(horizons))`` bool, the windows to count at each horizon.
    """
    out = {}
    for i, h in enumerate(horizons):
        sq = ((pred[:, h - 1] - target[:, h - 1]) ** 2).mean(-1)  # (n,)
        out[h] = float(sq.mean() if mask is None else sq[mask[:, i]].mean())
    return out


def baselines(
    clips: ClipSet, balls: BallsConf, std: Standardizer, m: MeasureConf
) -> dict[str, float]:
    """The errors of every baseline on the windows of ``clips``, keyed
    ``{name}{h}`` for ``name`` in mean, copy, cv, cvobs (``"pos_vel"`` only),
    ca, true and ``h`` in ``m.horizons``, each also over the windows with a
    contact up to ``h`` (``@contact``, any ball), those without (``@free``),
    and over the columns of each ball ``i`` (``@ball{i}``).

    A contact counts from frame ``m.observed - STENCIL`` (see
    ``contact_mask``), so ``m.observed >= STENCIL``."""
    if balls.obs == PIXELS:
        raise ValueError('"pixels" has its baselines in measure.pixel_baselines')
    if m.observed < STENCIL:
        raise ValueError(f"observed={m.observed} is fewer than STENCIL={STENCIL}")
    if m.observed != clips.observed or max(m.horizons) > clips.horizon:
        raise ValueError(
            f"clips of {clips.observed} + {clips.horizon} frames do not fit "
            f"observed={m.observed}, horizons={m.horizons}"
        )
    obs, k, frame = balls.obs, len(balls.radii), clips.frame
    clips.check(balls)
    states = window_states(clips)
    raw = observe(states, obs, k)  # (n, span, d) float64
    o, horizon = m.observed, max(m.horizons)
    target = std.apply(raw[:, o : o + horizon])
    preds = {
        "mean": mean_pred(len(raw), horizon, raw.shape[-1]),
        "copy": std.apply(copy_pred(raw, o, horizon)),
        "cv": std.apply(const_velocity(raw, obs, k, o, horizon, frame)),
        "ca": std.apply(const_accel(raw, obs, k, o, horizon, frame)),
        "true": std.apply(observe(true_physics(clips, balls, o, horizon), obs, k)),
    }
    if obs == "pos_vel":
        preds["cvobs"] = std.apply(const_velocity_observed(raw, k, o, horizon, frame))
    contact = contact_mask(states, o, m.horizons, radii=balls.radii, frame=frame)
    contact = contact.any(-1)  # (n, len(horizons))
    width = raw.shape[-1] // k
    out: dict[str, float] = {}
    for name, pred in preds.items():
        variants = {
            "": errors(pred, target, m.horizons),
            "@contact": errors(pred, target, m.horizons, contact),
            "@free": errors(pred, target, m.horizons, ~contact),
        }
        for i in range(k):
            cols = slice(width * i, width * (i + 1))
            variants[f"@ball{i}"] = errors(
                pred[..., cols], target[..., cols], m.horizons
            )
        for suffix, errs in variants.items():
            for h, err in errs.items():
                out[f"{name}{h}{suffix}"] = err
    return out
