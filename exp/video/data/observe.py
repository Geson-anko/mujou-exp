"""What a model sees of the states of the balls, and the standardization of it.

A state is x, y, vx, vy of each ball, ``(..., 4K)``. Stage 0 of M14 observes
it whole (``"pos_vel"``); stages 1 and 2 observe the positions only
(``"pos"``: x, y of each ball, ``(..., 2K)``).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "STATE_OBSERVATIONS",
    "Standardizer",
    "observe",
    "position_columns",
    "to_states",
]

STATE_OBSERVATIONS = ("pos_vel", "pos")


def _check(obs: str) -> None:
    if obs not in STATE_OBSERVATIONS:
        raise ValueError(f"obs must be one of {STATE_OBSERVATIONS}, not {obs!r}")


def observe(states: Tensor, obs: str, balls: int) -> Tensor:
    """The observation of ``states`` ``(..., 4K)``, a new tensor: ``(..., 4K)``
    the same for ``"pos_vel"``, ``(..., 2K)`` x, y of each ball for ``"pos"``."""
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
    def fit(cls, x: Tensor) -> Standardizer:
        """The mean and the population standard deviation of ``x``
        ``(n, T, d)`` over all its clips and frames."""
        flat = x.reshape(-1, x.shape[-1]).double()
        std = flat.var(0, correction=0).sqrt()
        if not bool((std > 0).all()):
            raise ValueError(f"a constant dimension cannot be standardized: {std}")
        return cls(flat.mean(0), std)

    def apply(self, x: Tensor) -> Tensor:
        """``(..., d)`` to the standardized space, in the dtype of ``x``."""
        return (x - self.mean.to(x)) / self.std.to(x)

    def invert(self, y: Tensor) -> Tensor:
        """``(..., d)`` back from the standardized space, in the dtype of ``y``."""
        return y * self.std.to(y) + self.mean.to(y)
