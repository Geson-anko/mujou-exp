"""The minGRU cell, its residual form for stacking, and the eligibility traces
that learn it through time online."""

from collections.abc import Mapping
from typing import override

import torch
from torch import nn

from mujou import Stateful

__all__ = ["MinGRU", "MinGRUTrace", "ResidualMinGRU"]


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


class MinGRUTrace(Stateful):
    r"""The eligibility traces $e_k = \partial s_k / \partial \theta$ of a
    ``MinGRU`` through its own state, with its inputs held: real-time recurrent
    learning, exact as the recurrence is diagonal in $s$.

    With $a = W_u x + b_u$, $u = \sigma(a)$, $c = W_s x + b_s$ and
    $s_k = (1 - u) \odot s_{k-1} + u \odot c$, each row $i$ of a weight moves
    only $s_i$:

    $$
    e^{W_u}_{k,ij} = (1 - u_i)\, e^{W_u}_{k-1,ij} + u_i (1 - u_i)(c_i - s_{k-1,i})\, x_j,
    \qquad
    e^{W_s}_{k,ij} = (1 - u_i)\, e^{W_s}_{k-1,ij} + u_i\, x_j,
    $$

    and the biases with $x_j = 1$. Each sample of the ``batch`` has its own
    traces, of the size of the weights. ``grads(delta)`` is
    $\sum_i \delta_i e_{k,i\cdot}$ summed over the batch: the gradient of
    $\delta \cdot s_k$ through all the steps since the last reset.

    It is a ``mujou.Stateful``: a node that holds it keeps the traces in its
    state, so a run resumed from ``run.state_dict()`` goes on with them.
    """

    def __init__(self, cell: MinGRU, batch: tuple[int, ...]) -> None:
        self._cell = cell
        w = cell.gate.weight
        d_s, d_x = w.shape
        self._e_wu = w.new_zeros(*batch, d_s, d_x)
        self._e_bu = w.new_zeros(*batch, d_s)
        self._e_ws = w.new_zeros(*batch, d_s, d_x)
        self._e_bs = w.new_zeros(*batch, d_s)

    @torch.no_grad()
    def step_(self, x: torch.Tensor, s_prev: torch.Tensor) -> None:
        """Move the traces to $s_k$ = ``cell(x, s_prev)``, at the cell's
        current weights."""
        u = torch.sigmoid(self._cell.gate(x))
        c = self._cell.candidate(x)
        keep = 1 - u
        du = u * keep * (c - s_prev)  # ds/da
        outer = x.unsqueeze(-2)  # (..., 1, d_x)
        self._e_wu.mul_(keep.unsqueeze(-1)).add_(du.unsqueeze(-1) * outer)
        self._e_bu.mul_(keep).add_(du)
        self._e_ws.mul_(keep.unsqueeze(-1)).add_(u.unsqueeze(-1) * outer)
        self._e_bs.mul_(keep).add_(u)

    @torch.no_grad()
    def reset_(self, which: torch.Tensor) -> None:
        """Zero the traces of the samples where ``which`` (``batch``) holds:
        their state starts again from 0."""
        for e in (self._e_wu, self._e_ws):
            e.masked_fill_(which[..., None, None], 0.0)
        for e in (self._e_bu, self._e_bs):
            e.masked_fill_(which[..., None], 0.0)

    @torch.no_grad()
    def grads(self, delta: torch.Tensor) -> list[torch.Tensor]:
        r"""The gradients of $\delta \cdot s_k$ (``delta``: ``(*batch, d_s)``)
        for the parameters of the cell, in their order, summed over the batch."""
        d = delta.reshape(-1, delta.shape[-1])
        n = d.shape[0]
        e_wu = self._e_wu.reshape(n, *self._e_wu.shape[-2:])
        e_ws = self._e_ws.reshape(n, *self._e_ws.shape[-2:])
        return [
            torch.einsum("bi,bij->ij", d, e_wu),
            (d * self._e_bu.reshape(n, -1)).sum(0),
            torch.einsum("bi,bij->ij", d, e_ws),
            (d * self._e_bs.reshape(n, -1)).sum(0),
        ]

    @override
    def state_dict(self) -> dict[str, torch.Tensor]:
        """The traces (sharing memory with the trace's)."""
        return {
            "e_wu": self._e_wu,
            "e_bu": self._e_bu,
            "e_ws": self._e_ws,
            "e_bs": self._e_bs,
        }

    @override
    def load_state_dict(self, state: Mapping[str, torch.Tensor], /) -> None:
        """Copy the traces of ``state`` into this trace's (of the same shapes;
        any device)."""
        with torch.no_grad():
            for name, mine in self.state_dict().items():
                mine.copy_(state[name])
