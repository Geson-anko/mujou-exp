r"""The baselines of stage 3 of M14: what a rollout of images has to beat, on
the windows of the evaluation clips.

The observation is the image of the balls (``observe`` with ``"pixels"``),
standardized with one mean and one standard deviation over all pixels, and an
error is the mean over the windows and the pixels of the squared error in that
space. The extrapolations move the *states*, from the true positions of the
last observed frames, and render the result:

- ``mean``: the mean image of the calibration clips, per pixel (``mean_image``;
  without it the uniform image of the one mean of all pixels, 0 standardized)
- ``copy``: the last observed image again
- ``cv``, ``ca``: constant velocity / acceleration of the positions
  (``baselines.const_velocity``, ``baselines.const_accel``), rendered
- ``true``: the physics itself, rendered (0 by construction)

Keys as ``baselines.baselines`` for the same names and horizons, plus
``@contactball`` (as ``dynamics.ball_contact_errors``), but ``@ball{i}`` is
the error over the pixels covered by the true disc of ball ``i`` at the
target frame (``disc_masks``) rather than over a block of columns: the error
of a small ball that a prediction lets vanish is not drowned by the pixels of
the background. Regions of balls that overlap share their pixels.
"""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from exp.video.data.clips import (
    STENCIL,
    ClipSet,
    contact_mask,
    window_states,
)
from exp.video.data.observe import Standardizer, render_flat
from exp.video.measure.baselines import (
    const_accel,
    const_velocity,
    errors,
    mean_pred,
    true_physics,
)
from exp.video.schema.data import BallsConf
from exp.video.schema.measure import MeasureConf

__all__ = ["baselines", "disc_masks", "region_sq"]


def disc_masks(states: Tensor, radii: Sequence[float], size: int) -> Tensor:
    """The pixels of the image (``physics.render``: the centers of the pixels
    at ``(i + 1/2) / size``, row 0 at the top where y is 1) within the radius
    of each ball from its center.

    Args:
        states: ``(n, 4K)``.

    Returns:
        ``(n, K, size * size)`` bool.
    """
    xy = states.reshape(len(states), len(radii), 4)[..., :2].double()  # (n, K, 2)
    centers = (torch.arange(size, dtype=torch.float64) + 0.5) / size
    x = centers.view(1, 1, 1, size)
    y = (1 - centers).view(1, 1, size, 1)
    r = torch.tensor(radii, dtype=torch.float64).view(1, -1, 1, 1)
    dist2 = (x - xy[..., 0, None, None]) ** 2 + (y - xy[..., 1, None, None]) ** 2
    return (dist2 <= r**2).flatten(2)


def region_sq(sq: Tensor, masks: Tensor) -> Tensor:
    """The mean of ``sq`` ``(n, d)`` over the pixels of each region of
    ``masks`` ``(n, K, d)`` bool: ``(n, K)`` float64, NaN for a region with no
    pixel."""
    total = torch.where(masks, sq[:, None].double(), 0.0).sum(-1)
    return total / masks.sum(-1)


def _predictors(
    clips: ClipSet, balls: BallsConf, states: Tensor, o: int, horizon: int
) -> dict[str, Callable[[], Tensor]]:
    """The raw images ``(n, horizon, size^2)`` float32 of each baseline but the
    mean, from ``states`` ``(n, span, 4K)``, each made when called (they are
    big)."""
    k, frame, size = len(balls.radii), clips.frame, balls.size

    def image(s: Tensor) -> Tensor:
        return render_flat(s, balls.radii, size)

    def extrapolate(f: Callable[..., Tensor]) -> Callable[[], Tensor]:
        return lambda: image(f(states, "pos_vel", k, o, horizon, frame))

    return {
        "copy": lambda: image(states[:, o - 1 : o]).expand(-1, horizon, -1),
        "cv": extrapolate(const_velocity),
        "ca": extrapolate(const_accel),
        "true": lambda: image(true_physics(clips, balls, o, horizon)),
    }


def baselines(
    clips: ClipSet,
    balls: BallsConf,
    std: Standardizer,
    m: MeasureConf,
    *,
    mean_image: Tensor | None = None,
) -> dict[str, float]:
    """The errors of every baseline on the windows of ``clips``, keyed
    ``{name}{h}{suffix}`` for ``name`` in mean, copy, cv, ca, true, ``h`` in
    ``m.horizons`` and ``suffix`` in ``""``, ``@contact``, ``@free``,
    ``@contactball`` and ``@ball{i}`` (module docstring). NaN where no window
    or ball is of the kind.

    ``balls`` must have the radii and the physics of the clips and the
    ``std`` of ``balls.obs = "pixels"``; ``m.observed >= STENCIL``.
    ``mean_image`` ``(size^2,)`` is the raw mean image of the ``mean``
    baseline (``clips.fit_mean_image``)."""
    if m.observed < STENCIL:
        raise ValueError(f"observed={m.observed} is fewer than STENCIL={STENCIL}")
    if m.observed != clips.observed or max(m.horizons) > clips.horizon:
        raise ValueError(
            f"clips of {clips.observed} + {clips.horizon} frames do not fit "
            f"observed={m.observed}, horizons={m.horizons}"
        )
    clips.check(balls)
    k, frame = len(balls.radii), clips.frame
    states = window_states(clips)  # (n, span, 4K) float64
    o, horizon = m.observed, max(m.horizons)
    target = std.apply(render_flat(states[:, o : o + horizon], balls.radii, balls.size))
    masks = torch.stack(  # (n, len(horizons), K, size^2): the discs of the target
        [disc_masks(states[:, o + h - 1], balls.radii, balls.size) for h in m.horizons],
        1,
    )
    per_ball = contact_mask(states, o, m.horizons, radii=balls.radii, frame=frame)
    contact = per_ball.any(-1)  # (n, len(horizons))
    out: dict[str, float] = {}
    if mean_image is None:
        mean = mean_pred(len(states), horizon, target.shape[-1]).float()
    else:  # standardized, the same image for every window and horizon
        mean = std.apply(mean_image.float()).expand(len(states), horizon, -1)
    predictors = {"mean": lambda: mean}
    predictors |= _predictors(clips, balls, states, o, horizon)
    for name, make in predictors.items():
        pred = make()
        if name != "mean":
            pred = std.apply(pred)
        variants = {
            "": errors(pred, target, m.horizons),
            "@contact": errors(pred, target, m.horizons, contact),
            "@free": errors(pred, target, m.horizons, ~contact),
        }
        for suffix, errs in variants.items():
            for h, err in errs.items():
                out[f"{name}{h}{suffix}"] = err
        for i, h in enumerate(m.horizons):
            sq = (pred[:, h - 1] - target[:, h - 1]) ** 2  # (n, size^2)
            region = region_sq(sq, masks[:, i])  # (n, K)
            for b in range(k):
                out[f"{name}{h}@ball{b}"] = float(region[:, b].nanmean())
            touching = region[per_ball[:, i] & ~region.isnan()]
            out[f"{name}{h}@contactball"] = (
                float(touching.mean()) if touching.numel() else float("nan")
            )
    return out
