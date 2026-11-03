r"""The evaluation of the dynamics of M14: rollouts on the windows of the
evaluation clips (``evaluate_dynamics``) and from the live state of the
training streams (``fork_eval``), in the standardized observation space.

A rollout runs the modules as the nodes do: $s_k = \mathrm{step}(x_{k-1},
s_{k-1})$ from $s_0 = 0$, $\hat x_k = W_o\,\mathrm{Top}(s_k)$ (plus $x_{k-1}$
with ``residual``). On a window it is teacher forced for the ``observed``
frames $x_0 .. x_{O-1}$, then predicts open loop from its own predictions:
$\hat x_O$ is ``h = 1`` (the frame right after the last observed one, as in
``measure.baselines``), $\hat x_{O+1}$ from $\hat x_O$ is ``h = 2``, and so on.

The keys of a row (an error is the mean over the windows and the dimensions
of the squared error):

- ``roll{h}``; ``roll{h}@contact`` / ``roll{h}@free`` on the windows with a
  wall contact up to ``h`` (any ball; ``clips.contact_mask``, as
  ``baselines`` counts it) / without; ``roll{h}@contactball`` on the columns
  of the balls that touch a wall up to ``h`` only, over the pairs (window,
  ball) of such a contact (with one ball the same as ``@contact``; with
  several, ``@contact`` also counts the balls in free flight of the window);
  ``roll{h}@ball{i}`` on the columns of ball ``i`` (for ``"pixels"``, on the
  pixels covered by the true disc of ball ``i`` at the target frame,
  ``pixel_baselines.disc_masks``; ``@contactball`` likewise); ``roll{h}_se`` the
  bootstrap standard error of ``roll{h}`` over the windows
- the baselines of the same windows (``dynamics_baselines``): ``mean{h}``,
  ``copy{h}``, ``cv{h}``, ``ca{h}``, each also ``@contact``, ``@free`` and
  ``@contactball``
- ``roll{h}/cv{h}`` and ``roll{h}/ca{h}``
- ``bounce{h}`` $=$ ``roll{h}@contact`` $/ \min$ of ``mean``, ``copy``,
  ``cv``, ``ca`` ``{h}@contact``, and ``bounceball{h}`` the same of
  ``@contactball``: below 1 the rollout has learned the bounces, as no
  predictor that does not model them is as good on the contacts (``mean``
  and ``copy`` count: a prediction that stands still or fades beats the
  extrapolations at a bounce, ``cv10@contact`` 2.57 against ``mean10@contact``
  1.29 for one ball)
- ``gravity{h}`` $=$ ``roll{h}@free`` $/ \min$ of ``mean``, ``copy``, ``cv``
  ``{h}@free``: below 1 the rollout has learned gravity (``ca`` is left out:
  without a contact it is the physics but the drag)
- ``amp{h}``: the variance of the predictions over the windows relative to
  that of the truth (summed over the dimensions); ``motion{h}``: the mean
  length of the predicted step from ``h - 1`` to ``h`` relative to the true
  one. Below 1 the prediction stops and fades.
- ``nonfinite``: the windows with a prediction that is not finite

``fork_eval`` adds ``fork{h}``, the baselines ``forkcopy{h}``, ``forkcv{h}``,
``forkca{h}`` of the same futures and ``fork{h}/forkcv{h}`` $=$ ``fork{h}``
$/$ ``forkcv{h}`` (against constant velocity on the same futures, not on the
clips).
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from exp.lib.metrics import bootstrap_se
from exp.video.data.clips import ClipSet, contact_mask, window_states, windows
from exp.video.data.observe import PIXELS, Standardizer, observe
from exp.video.data.physics import BallStreams
from exp.video.dyn.modules import DynamicsModules, Top
from exp.video.measure.baselines import (
    baselines,
    const_accel,
    const_velocity,
    copy_pred,
    mean_pred,
)
from exp.video.measure.pixel_baselines import (
    baselines as pixel_baselines,
    disc_masks,
    region_sq,
)
from exp.video.schema.data import BallsConf
from exp.video.schema.measure import MeasureConf

__all__ = [
    "BASELINE_KEYS",
    "ball_contact_errors",
    "dynamics_baselines",
    "evaluate_dynamics",
    "fork_eval",
    "open_loop",
    "ratio",
    "rollout",
]

# The baselines of dynamics_baselines a row carries, with these suffixes.
_SUFFIXES = ("", "@contact", "@free", "@contactball")
BASELINE_KEYS = tuple((name, _SUFFIXES) for name in ("mean", "copy", "cv", "ca"))
_BOUNCE = ("mean", "copy", "cv", "ca")  # the bar of bounce{h}, bounceball{h}
_GRAVITY = ("mean", "copy", "cv")  # the bar of gravity{h}
_SE_SEED = 0  # of the bootstrap: the same resamples at every evaluation


@torch.no_grad()
def open_loop(
    mods: DynamicsModules, x: Tensor, s: Tensor, *, residual: bool, horizon: int
) -> Tensor:
    r"""Predict ``horizon`` frames open loop from the last input ``x``
    ``(n, d)`` and the state ``s`` ``(n, L d_s)`` it was not fed yet:
    $s \leftarrow \mathrm{step}(x, s)$, $x \leftarrow W_o\,\mathrm{Top}(s)$
    (plus $x$ with ``residual``), ``horizon`` times.

    Returns:
        ``(n, horizon, d)``.
    """
    top = Top(mods.step.d_s)
    out = []
    for _ in range(horizon):
        s = mods.step(x, s)
        y = mods.readout(top(s))
        x = x + y if residual else y
        out.append(x)
    return torch.stack(out, 1)


@torch.no_grad()
def rollout(
    mods: DynamicsModules,
    obs: Tensor,
    *,
    residual: bool,
    observed: int,
    horizon: int,
) -> Tensor:
    """Teacher forced for the ``observed`` first frames of ``obs`` ``(n, T,
    d)`` (``T >= observed``) from the state 0, then open loop: the predictions
    of frames ``observed .. observed + horizon - 1``, ``(n, horizon, d)``."""
    s = obs.new_zeros(obs.shape[0], mods.step.d_state)
    for j in range(1, observed):
        s = mods.step(obs[:, j - 1], s)
    return open_loop(mods, obs[:, observed - 1], s, residual=residual, horizon=horizon)


def _sq(pred: Tensor, target: Tensor) -> Tensor:
    """The squared error ``(n, H)`` of each window and step, over the last
    axis."""
    return ((pred - target) ** 2).mean(-1)


def ratio(a: float, b: float) -> float:
    """``a / b``, NaN or infinite rather than an error when ``b`` is 0."""
    if b == 0:
        return float("nan") if a == 0 or a != a else float("inf")
    return a / b


def _mean(x: Tensor, mask: Tensor | None = None) -> float:
    """The mean of ``x`` (where ``mask``), NaN when the mask keeps nothing."""
    x = x if mask is None else x[mask]
    return float(x.double().mean()) if x.numel() else float("nan")


def _least(values: Sequence[float]) -> float:
    """The smallest of ``values``, NaN when one is NaN (no window of the
    kind)."""
    return float("nan") if any(v != v for v in values) else min(values)


def _per_ball(sq: Tensor, k: int, regions: Tensor | None) -> Tensor:
    """The error of each ball ``(n, K)`` from the squared error ``sq`` ``(n,
    d)``: the mean over its block of ``d // K`` columns, or over its pixels of
    ``regions`` ``(n, K, d)``."""
    if regions is None:
        return sq.reshape(len(sq), k, -1).mean(-1)
    return region_sq(sq, regions).to(sq.dtype)


def ball_contact_errors(
    pred: Tensor,
    target: Tensor,
    contact: Tensor,
    horizons: Sequence[int],
    regions: Tensor | None = None,
) -> dict[int, float]:
    """The mean squared error at each horizon ``h`` over the pairs (window,
    ball) where the ball touches a wall up to ``h``, each over the columns of
    that ball; NaN when there is no such pair.

    Args:
        pred: ``(n, H, K w)``, the ``w`` columns of each ball together, as
            ``observe`` lays them out; ``H >= max(horizons)``.
        target: ``(n, H, K w)``.
        contact: ``(n, len(horizons), K)`` bool, ``clips.contact_mask``.
        regions: ``(n, len(horizons), K, d)`` bool, the columns of each ball
            when they are not a block (``disc_masks`` of the target frames).
    """
    k = contact.shape[-1]
    out = {}
    for i, h in enumerate(horizons):
        sq = (pred[:, h - 1] - target[:, h - 1]) ** 2
        per_ball = _per_ball(sq, k, None if regions is None else regions[:, i])
        keep = contact[:, i].to(sq.device) & ~per_ball.isnan()  # no empty region
        out[h] = _mean(per_ball, keep)  # over (n, K)
    return out


def dynamics_baselines(
    clips: ClipSet,
    balls: BallsConf,
    std: Standardizer,
    m: MeasureConf,
    *,
    mean_image: Tensor | None = None,
) -> dict[str, float]:
    """``baselines.baselines`` of ``clips`` and the ``{name}{h}@contactball``
    (``ball_contact_errors``) of ``mean``, ``copy``, ``cv`` and ``ca``, from
    the same windows and the same ``contact_mask``: what ``evaluate_dynamics``
    takes as ``base``. For ``"pixels"`` it is ``pixel_baselines.baselines``
    (``mean_image`` is its ``mean_image``)."""
    if balls.obs == PIXELS:
        return pixel_baselines(clips, balls, std, m, mean_image=mean_image)
    out = baselines(clips, balls, std, m)
    obs, k, frame = balls.obs, len(balls.radii), clips.frame
    states = window_states(clips)
    raw = observe(states, obs, k)  # (n, span, d) float64
    o, horizon = m.observed, max(m.horizons)
    target = std.apply(raw[:, o : o + horizon])
    preds = {
        "mean": mean_pred(len(raw), horizon, raw.shape[-1]),
        "copy": std.apply(copy_pred(raw, o, horizon)),
        "cv": std.apply(const_velocity(raw, obs, k, o, horizon, frame)),
        "ca": std.apply(const_accel(raw, obs, k, o, horizon, frame)),
    }
    contact = contact_mask(states, o, m.horizons, radii=balls.radii, frame=frame)
    for name, pred in preds.items():
        for h, err in ball_contact_errors(pred, target, contact, m.horizons).items():
            out[f"{name}{h}@contactball"] = err
    return out


def _shape(
    pred: Tensor, target: Tensor, last: Tensor, horizons: Sequence[int]
) -> dict[str, float]:
    """``amp{h}`` and ``motion{h}`` of predictions ``(n, H, d)`` against the
    target, both following the last observed frame ``last`` ``(n, d)``."""
    out: dict[str, float] = {}
    for h in horizons:
        p, t = pred[:, h - 1].double(), target[:, h - 1].double()
        spread = float(p.var(0, correction=0).sum())
        out[f"amp{h}"] = ratio(spread, float(t.var(0, correction=0).sum()))
        p0 = last if h == 1 else pred[:, h - 2]
        t0 = last if h == 1 else target[:, h - 2]
        moved = float((p - p0.double()).norm(dim=-1).mean())
        out[f"motion{h}"] = ratio(moved, float((t - t0.double()).norm(dim=-1).mean()))
    return out


@torch.no_grad()
def evaluate_dynamics(
    mods: DynamicsModules,
    clips: ClipSet,
    std: Standardizer,
    base: dict[str, float],
    m: MeasureConf,
    *,
    obs: str,
    balls: int,
    residual: bool,
    size: int = 32,
) -> dict[str, float]:
    """The keys of the module docstring but ``fork*``, of a rollout of
    ``mods`` (on their device, read between steps) on the windows of
    ``clips``; ``base`` is ``dynamics_baselines`` of the same clips; ``size``
    is the side of the images of ``"pixels"``.

    Raises:
        ValueError: The clips do not fit ``m`` (``observed``, the horizons).
    """
    o, horizon = m.observed, max(m.horizons)
    if o != clips.observed or horizon > clips.horizon:
        raise ValueError(
            f"clips of {clips.observed} + {clips.horizon} frames do not fit "
            f"observed={o}, horizons={m.horizons}"
        )
    device = mods.readout.weight.device
    x, states = windows(clips, obs, std, size=size)
    x = x.to(device)
    pred = rollout(mods, x, residual=residual, observed=o, horizon=horizon)
    target = x[:, o : o + horizon]
    per_ball = contact_mask(
        states, o, m.horizons, radii=clips.radii, frame=clips.frame
    )  # (n, len(horizons), K)
    contact = per_ball.any(-1).to(device)  # (n, len(horizons))
    regions = None
    if obs == PIXELS:  # (n, len(horizons), K, size^2)
        regions = torch.stack(
            [disc_masks(states[:, o + h - 1], clips.radii, size) for h in m.horizons],
            1,
        ).to(device)
    by_ball = ball_contact_errors(pred, target, per_ball, m.horizons, regions)
    sq = _sq(pred, target)  # (n, H)
    gen = torch.Generator(device).manual_seed(_SE_SEED)
    out: dict[str, float] = {}
    for i, h in enumerate(m.horizons):
        e = sq[:, h - 1]
        out[f"roll{h}"] = _mean(e)
        out[f"roll{h}@contact"] = _mean(e, contact[:, i])
        out[f"roll{h}@free"] = _mean(e, ~contact[:, i])
        out[f"roll{h}@contactball"] = by_ball[h]
        ball = _per_ball(
            (pred[:, h - 1] - target[:, h - 1]) ** 2,
            balls,
            None if regions is None else regions[:, i],
        )  # (n, K)
        for b in range(balls):
            out[f"roll{h}@ball{b}"] = _mean(ball[:, b], ~ball[:, b].isnan())
        out[f"roll{h}_se"] = bootstrap_se(e, gen=gen)
        for name, suffixes in BASELINE_KEYS:
            for suffix in suffixes:
                out[f"{name}{h}{suffix}"] = base[f"{name}{h}{suffix}"]
        out[f"roll{h}/cv{h}"] = ratio(out[f"roll{h}"], base[f"cv{h}"])
        out[f"roll{h}/ca{h}"] = ratio(out[f"roll{h}"], base[f"ca{h}"])
        for key, suffix, bar in (
            ("bounce", "@contact", _BOUNCE),
            ("bounceball", "@contactball", _BOUNCE),
            ("gravity", "@free", _GRAVITY),
        ):
            best = _least([base[f"{name}{h}{suffix}"] for name in bar])
            out[f"{key}{h}"] = ratio(out[f"roll{h}{suffix}"], best)
    out |= _shape(pred, target, x[:, o - 1], m.horizons)
    out["nonfinite"] = int((~torch.isfinite(pred).flatten(1).all(1)).sum())
    return out


@torch.no_grad()
def fork_eval(
    mods: DynamicsModules,
    s: Tensor,
    streams: BallStreams,
    history: Tensor,
    std: Standardizer,
    m: MeasureConf,
    *,
    obs: str,
    balls: int,
    residual: bool,
    frame: float,
    radii: Sequence[float] | None = None,
    size: int = 32,
) -> dict[str, float]:
    r"""The rollout of each training stream from where it is: from the live
    state ``s`` ``(B, L d_s)`` (``nets.live_states``: after the last frame, at
    the weights as they are) and the last observation, open loop up to the
    largest horizon, against the stream's future (``streams.fork()``, which
    leaves ``streams`` as it is), in the standardized space; with
    ``forkcopy{h}``, ``forkcv{h}`` and ``forkca{h}``, the baselines of the
    same futures from the last frames.

    Args:
        mods: The live modules, on the device of ``s``.
        history: The raw observations ``(B, STENCIL, d)`` of the last frames
            shown (the last one last), float32 as the host made them; for
            ``"pixels"`` the raw *states* ``(B, STENCIL, 4K)`` of those frames
            (the extrapolations move them and render the result).
        radii: The radii of the balls and the side of the images ``size``,
            for ``"pixels"``.
    """
    horizon = max(m.horizons)
    device = s.device
    future = streams.fork()
    truth = torch.stack(
        [
            observe(future.next(), obs, balls, radii=radii, size=size)
            for _ in range(horizon)
        ],
        1,
    )  # (B, H, d) raw float32
    target = std.apply(truth).to(device)
    n = history.shape[1]
    raw = history.double()
    if obs == PIXELS:  # from the states: move them, then render
        views = {
            "copy": observe(raw[:, -1:], obs, balls, radii=radii, size=size),
            "cv": const_velocity(raw, "pos_vel", balls, n, horizon, frame),
            "ca": const_accel(raw, "pos_vel", balls, n, horizon, frame),
        }
        for name in ("cv", "ca"):
            views[name] = observe(views[name], obs, balls, radii=radii, size=size)
        views["copy"] = views["copy"].expand(-1, horizon, -1)
        last = observe(history[:, -1], obs, balls, radii=radii, size=size)
    else:
        views = {
            "copy": copy_pred(raw, n, horizon),
            "cv": const_velocity(raw, obs, balls, n, horizon, frame),
            "ca": const_accel(raw, obs, balls, n, horizon, frame),
        }
        last = history[:, -1]
    last = std.apply(last).to(device)  # what the nodes saw last
    pred = open_loop(mods, last, s, residual=residual, horizon=horizon)
    preds = {"fork": pred} | {f"fork{k}": std.apply(v) for k, v in views.items()}
    out: dict[str, float] = {}
    for name, p in preds.items():
        sq = _sq(p.to(device, target.dtype), target)
        for h in m.horizons:
            out[f"{name}{h}"] = _mean(sq[:, h - 1])
    for h in m.horizons:
        out[f"fork{h}/forkcv{h}"] = ratio(out[f"fork{h}"], out[f"forkcv{h}"])
    return out
