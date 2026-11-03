"""What a model sees of the states of the balls, and the standardization of it.

A state is x, y, vx, vy of each ball, ``(..., 4K)``. Stage 0 of M14 observes
it whole (``"pos_vel"``); stages 1 and 2 observe the positions only
(``"pos"``: x, y of each ball, ``(..., 2K)``). Stage 3 observes the grey
image of the balls (``"pixels"``, ``physics.render``) flattened to
``(..., size * size)`` float32; it cannot be turned back to states, and its
standardization is one mean and one standard deviation over all pixels.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from exp.video.data.physics import render

__all__ = [
    "PIXELS",
    "STATE_OBSERVATIONS",
    "Standardizer",
    "observe",
    "position_columns",
    "to_states",
]

STATE_OBSERVATIONS = ("pos_vel", "pos")
PIXELS = "pixels"
_CHUNK = 2048  # states rendered at a time: the blobs are (chunk, K, size, size)


def _check(obs: str) -> None:
    if obs not in STATE_OBSERVATIONS:
        raise ValueError(f"obs must be one of {STATE_OBSERVATIONS}, not {obs!r}")


def render_flat(states: Tensor, radii: Sequence[float], size: int) -> Tensor:
    """The images of ``states`` ``(..., 4K)`` (``physics.render``, in float64
    whatever the dtype of ``states``) as float32 ``(..., size * size)``, row
    after row from the top."""
    flat = states.reshape(-1, states.shape[-1])
    out = torch.empty(len(flat), size * size)
    for i in range(0, len(flat), _CHUNK):
        image = render(flat[i : i + _CHUNK].double(), size, radii)
        out[i : i + _CHUNK] = image.flatten(1).float()
    return out.reshape(*states.shape[:-1], size * size)


def observe(
    states: Tensor,
    obs: str,
    balls: int,
    *,
    radii: Sequence[float] | None = None,
    size: int = 32,
) -> Tensor:
    """The observation of ``states`` ``(..., 4K)``, a new tensor: ``(..., 4K)``
    the same for ``"pos_vel"``, ``(..., 2K)`` x, y of each ball for ``"pos"``,
    float32 ``(..., size * size)`` for ``"pixels"`` (``radii`` is then
    required, one per ball).

    Raises:
        ValueError: ``obs`` is unknown, or ``"pixels"`` without ``radii``.
    """
    if obs == PIXELS:
        if radii is None or len(radii) != balls:
            raise ValueError(f"pixels need the {balls} radii of the balls, not {radii}")
        return render_flat(states, radii, size)
    _check(obs)
    if obs == "pos_vel":
        return states.clone()
    return states.unflatten(-1, (balls, 4))[..., :2].contiguous().flatten(-2)


def to_states(obs_raw: Tensor, obs: str, balls: int) -> Tensor:
    """Back from ``observe`` to ``(..., 4K)``, a new tensor; for ``"pos"`` the
    velocities are zero."""
    _check(obs)
    if obs == "pos_vel":
        return obs_raw.clone()
    xy = obs_raw.unflatten(-1, (balls, 2))
    return torch.cat([xy, torch.zeros_like(xy)], -1).flatten(-2)


def position_columns(obs: str, balls: int) -> Tensor:
    """The columns ``(K, 2)`` of x, y of each ball in an observation."""
    _check(obs)
    width = 4 if obs == "pos_vel" else 2
    return torch.arange(balls * width).view(balls, width)[:, :2]


@dataclass
class Standardizer:
    r"""$y = (x - \mu) / \sigma$ per dimension of an observation."""

    mean: Tensor  # (d,) float64
    std: Tensor  # (d,) float64

    @classmethod
    def fit(cls, x: Tensor, *, pooled: bool = False) -> Standardizer:
        """The mean and the population standard deviation of ``x``
        ``(n, T, d)`` over all its clips and frames: per dimension, or with
        ``pooled`` one mean and one standard deviation over all the values
        (the pixels of an image), repeated ``d`` times."""
        flat = x.reshape(-1, x.shape[-1])
        if pooled:
            total = torch.zeros((), dtype=torch.float64)
            squares = torch.zeros((), dtype=torch.float64)
            for rows in flat.split(_CHUNK):
                total += rows.sum(dtype=torch.float64)
                squares += rows.double().square().sum()
            mean = total / flat.numel()
            std = (squares / flat.numel() - mean**2).clamp(min=0).sqrt()
            mean, std = mean.expand(flat.shape[-1]), std.expand(flat.shape[-1])
        else:
            mean = flat.double().mean(0)
            std = flat.double().var(0, correction=0).sqrt()
        if not bool((std > 0).all()):
            raise ValueError(f"a constant dimension cannot be standardized: {std}")
        return cls(mean.clone(), std.clone())

    def apply(self, x: Tensor) -> Tensor:
        """``(..., d)`` to the standardized space, in the dtype of ``x``."""
        return (x - self.mean.to(x)) / self.std.to(x)

    def invert(self, y: Tensor) -> Tensor:
        """``(..., d)`` back from the standardized space, in the dtype of ``y``."""
        return y * self.std.to(y) + self.mean.to(y)
