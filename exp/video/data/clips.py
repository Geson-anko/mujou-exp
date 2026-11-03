"""The fixed clips of the evaluation: new balls drawn from the seed of a split,
cut into windows of ``observed`` frames to show and ``horizon`` frames to
predict.

The splits keep the search and the report apart: the standardization is fitted
on ``"calib"`` only, the sweeps look at ``"val"``, the README reports
``"test"``. Every clip starts on the clock of the physics at 0 (with a
physics that changes in time, the clips see its first phase only).
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from exp.video.data.observe import PIXELS, Standardizer, observe
from exp.video.data.physics import BallState, Physics, advance_, draw
from exp.video.schema.data import BallsConf, PhysicsConf

__all__ = [
    "SPLIT_SEEDS",
    "STENCIL",
    "ClipSet",
    "contact_mask",
    "fit_mean_image",
    "fit_standardizer",
    "make_clips",
    "simulate",
    "window_states",
    "windows",
]

SPLIT_SEEDS = {"calib": 777, "val": 1001, "test": 12345}

# The last observed frames the extrapolation baselines read (the three points
# of constant acceleration): a bounce among them spoils the extrapolation as
# much as one ahead, so ``contact_mask`` counts it.
STENCIL = 3


@dataclass(frozen=True)
class ClipSet:
    """The clips of a split and their windows."""

    states: Tensor  # (n_clips, L, 4K) float64: the true states
    starts: Tensor  # (n_windows, 2) int64: (clip, first frame)
    observed: int
    horizon: int
    frame: float  # seconds between two frames
    radii: tuple[float, ...]  # the balls the clips were made of
    physics: PhysicsConf  # and their physics

    def check(self, balls: BallsConf) -> None:
        """Raise ``ValueError`` unless ``balls`` has the radii and the physics
        of the clips (its observation and image size may differ)."""
        if tuple(balls.radii) != self.radii or balls.physics != self.physics:
            raise ValueError(
                f"the clips were made of radii={list(self.radii)} and "
                f"{self.physics}, not radii={balls.radii} and {balls.physics}"
            )

    @property
    def span(self) -> int:
        """The frames of a window, ``observed + horizon``."""
        return self.observed + self.horizon


def simulate(
    s: BallState, balls: BallsConf, *, frames: int, frame: float, first: int = 0
) -> Tensor:
    """The states of ``frames`` frames from ``s``, the state of frame ``first``
    (the clock at ``first * frame``), moving ``s`` in place; the frames advance
    as those of ``BallStreams``, so a simulation from any frame of a clip
    gives the same doubles as the clip.

    Args:
        s: ``(n, K, 2)``.

    Returns:
        ``(n, frames, 4K)`` float64; frame 0 is ``s`` itself.
    """
    radii = torch.tensor(balls.radii, dtype=torch.float64)
    physics = Physics.from_conf(balls.physics)
    out = [torch.cat([s.pos, s.vel], -1).flatten(1)]
    for k in range(first + 1, first + frames):
        t = k * frame
        advance_(s, radii, physics, t - frame, frame, balls.physics.substeps)
        out.append(torch.cat([s.pos, s.vel], -1).flatten(1))
    return torch.stack(out, 1)


def make_clips(
    balls: BallsConf,
    *,
    split: str,
    clips: int = 256,
    length: int = 128,
    observed: int = 16,
    horizon: int = 20,
    stride: int = 30,
    frame: float,
) -> ClipSet:
    """``clips`` clips of ``length`` frames of new balls drawn one clip after the
    other from the seed of ``split``, and their windows of
    ``observed + horizon`` frames every ``stride`` frames (4 per clip with the
    defaults: 1024 windows)."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"split must be one of {tuple(SPLIT_SEEDS)}, not {split!r}")
    span = observed + horizon
    if span > length:
        raise ValueError(f"a window of {span} frames does not fit {length}")
    gen = torch.Generator().manual_seed(SPLIT_SEEDS[split])
    s = draw(clips, balls.radii, gen)
    states = simulate(s, balls, frames=length, frame=frame)
    firsts = torch.arange(0, length - span + 1, stride)
    starts = torch.cartesian_prod(torch.arange(clips), firsts)
    return ClipSet(
        states,
        starts,
        observed,
        horizon,
        frame,
        tuple(balls.radii),
        copy.deepcopy(balls.physics),
    )


def windows(
    c: ClipSet, obs: str, std: Standardizer, *, size: int = 32
) -> tuple[Tensor, Tensor]:
    """The windows of ``c`` (``size``: the side of the images of ``"pixels"``).

    Returns:
        The standardized observations ``(n_windows, observed + horizon, d)``
        float32, and the true states ``(n_windows, observed + horizon, 4K)``
        float64.
    """
    states = window_states(c)
    balls = states.shape[-1] // 4
    raw = observe(states, obs, balls, radii=c.radii, size=size)
    return std.apply(raw).float(), states


def window_states(c: ClipSet) -> Tensor:
    """The true states ``(n_windows, observed + horizon, 4K)`` float64 of the
    windows of ``c``."""
    frames = c.starts[:, 1:] + torch.arange(c.span)  # (n_windows, span)
    return c.states[c.starts[:, :1], frames]


def fit_standardizer(
    balls: BallsConf, *, frame: float, clips: int = 256, length: int = 128
) -> Standardizer:
    """The standardization of the observation ``balls.obs``, fitted on every
    frame of the clips of ``"calib"`` (never on ``"val"`` or ``"test"``): per
    dimension, but for ``"pixels"`` one mean and one standard deviation over
    all pixels."""
    c = make_clips(balls, split="calib", clips=clips, length=length, frame=frame)
    k = len(balls.radii)
    raw = observe(c.states, balls.obs, k, radii=balls.radii, size=balls.size)
    return Standardizer.fit(raw, pooled=balls.obs == PIXELS)


def fit_mean_image(
    balls: BallsConf, *, frame: float, clips: int = 256, length: int = 128
) -> Tensor:
    """The mean image ``(size * size,)`` float64 of ``balls`` (``obs =
    "pixels"``) over every frame of the clips of ``"calib"``, per pixel."""
    c = make_clips(balls, split="calib", clips=clips, length=length, frame=frame)
    raw = observe(
        c.states, balls.obs, len(balls.radii), radii=balls.radii, size=balls.size
    )
    flat = raw.reshape(-1, raw.shape[-1])
    total = torch.zeros(flat.shape[-1], dtype=torch.float64)
    for rows in flat.split(2048):
        total += rows.sum(0, dtype=torch.float64)
    return total / len(flat)


def contact_mask(
    states: Tensor,
    observed: int,
    horizons: Sequence[int],
    *,
    radii: Sequence[float],
    frame: float,
) -> Tensor:
    r"""Whether ball ``k`` touches a wall between frame ``observed - STENCIL``
    and frame ``observed - 1 + h`` of each window, for each ``h`` of
    ``horizons``: from the first frame the extrapolations read (a bounce there
    makes ``ca`` miss just as one ahead does) to the predicted one.

    A bounce between frames ``j - 1`` and ``j`` flips the sign of a velocity
    component. Along $x$ (no gravity; a drag only slows) a flip is always a
    bounce, and so is a flip from down to up along $y$ (gravity pulls down).
    A flip from up to down is also the top of a flight, so it counts only when
    the ball would cross the ceiling at its speed of frame ``j - 1``,
    $y + v_y\,\Delta t > 1 - r$. A top within $\gamma \Delta t^2 / 2$ below
    the ceiling is thus a false contact.

    With $e = 1$ every bounce flips a velocity between two frames, so none is
    missed. With $e < 1$ (and a drag) a ball coming to rest on the floor can
    bounce in a frame with no flip between the frames (two bounces in one
    frame, or a ball sliding on the floor), and that contact is missed: about
    2% of the contacts of $e \approx 0.6$.

    Args:
        states: ``(n_windows, T, 4K)``, ``T >= observed + max(horizons)``.

    Returns:
        ``(n_windows, len(horizons), K)`` bool.
    """
    first = observed - STENCIL
    if first < 0:
        raise ValueError(f"observed={observed} is fewer than STENCIL={STENCIL}")
    s = states.unflatten(-1, (len(radii), 4))  # (n, T, K, 4)
    last = observed - 1 + max(horizons)
    before, after = s[:, first:last], s[:, first + 1 : last + 1]
    flip = before[..., 2:] * after[..., 2:] < 0  # (n, STENCIL - 1 + H, K, 2)
    ceiling = 1 - torch.tensor(radii, dtype=states.dtype)
    rising = before[..., 3] > 0
    past = before[..., 1] + before[..., 3] * frame > ceiling
    y_bounce = flip[..., 1] & (~rising | past)
    touch = flip[..., 0] | y_bounce  # interval i: frame first + i to first + i + 1
    reached = touch.cummax(1).values
    # frame observed - 1 + h ends the interval h + STENCIL - 2
    return reached[:, [h + STENCIL - 2 for h in horizons]]
