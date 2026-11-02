"""The minGRU cell, and its residual form for stacking."""

from typing import override

import torch
from torch import nn

__all__ = ["MinGRU", "ResidualMinGRU"]


class MinGRU(nn.Module):
    r"""$u = \sigma(W_u z)$, $\tilde s = W_s z$,
    $s = (1 - u) \odot s_\text{prev} + u \odot \tilde s$ (Feng et al. 2024, the
    sequential form)."""

    def __init__(self, d_z: int, d_s: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_z, d_s)  # W_u
        self.candidate = nn.Linear(d_z, d_s)  # W_s

    @override
    def forward(self, z: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        u = torch.sigmoid(self.gate(z))
        return (1 - u) * s_prev + u * self.candidate(z)


class ResidualMinGRU(nn.Module):
    r"""A layer of minGRU on top of another: $y = x + h$ with
    $h = \mathrm{minGRU}(\mathrm{GELU}(x), h_\text{prev})$, both of width ``d``.
    Its state is $h = y - x$."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.cell = MinGRU(d, d)

    @override
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return x + self.cell(nn.functional.gelu(x), h_prev)
