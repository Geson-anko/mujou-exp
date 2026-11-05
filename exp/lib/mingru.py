"""The minGRU cell, its residual form for stacking, and the eligibility traces
that learn it through time online. ``MinGRU.scan`` and ``ResidualMinGRU.scan``
run a cell over a whole sequence of inputs at once (the matmuls over all the
steps together, the elementwise recurrence step by step), for a learner that
replays a window of frames."""

from collections.abc import Mapping
from typing import Any, override

import torch
from torch import nn

from mujou import Stateful

__all__ = ["MinGRU", "MinGRUTrace", "ResidualMinGRU", "linear_scan"]


class _LinearScan(torch.autograd.Function):
    r"""$s_t = a_t \odot s_{t-1} + b_t$ over the time axis 0, with the
    backward of its adjoint recurrence $\bar s_t = G_t + a_{t+1} \odot
    \bar s_{t+1}$: $\bar b_t = \bar s_t$, $\bar a_t = \bar s_t \odot
    s_{t-1}$, $\bar s_0 = a_1 \odot \bar s_1$ (one fused kernel per step
    each way, instead of the graph of every step)."""

    @staticmethod
    @override
    def forward(ctx: Any, a: torch.Tensor, b: torch.Tensor, s0: torch.Tensor) -> Any:
        out = torch.empty_like(b)
        s = s0
        for t in range(a.shape[0]):
            s = torch.addcmul(b[t], a[t], s, out=out[t])
        ctx.save_for_backward(a, out, s0)
        return out

    @staticmethod
    @override
    def backward(ctx: Any, *grads: torch.Tensor) -> Any:
        (grad,) = grads
        a, out, s0 = ctx.saved_tensors
        adj = torch.empty_like(grad)  # the adjoint of each s_t
        adj[-1] = grad[-1]
        for t in range(a.shape[0] - 2, -1, -1):
            # its own gradient and that through s_{t+1}
            torch.addcmul(grad[t], a[t + 1], adj[t + 1], out=adj[t])
        prev = torch.cat([s0.unsqueeze(0), out[:-1]])  # s_{t-1}
        return adj * prev, adj, a[0] * adj[0]


def linear_scan(a: torch.Tensor, b: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
    r"""$s_t = a_t \odot s_{t-1} + b_t$ for $t = 1..T$ from $s_0$ = ``s0``
    ``(*batch, d)``, with ``a`` and ``b`` ``(*batch, T, d)``: the states
    $s_1..s_T$ ``(*batch, T, d)``, differentiable in ``a``, ``b`` and
    ``s0``."""
    out = _LinearScan.apply(
        a.movedim(-2, 0).contiguous(), b.movedim(-2, 0).contiguous(), s0
    )
    return out.movedim(0, -2)


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

    def scan(self, z: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        r"""The states $s_1..s_T$ ``(*batch, T, d_s)`` of the inputs $z_0..
        z_{T-1}$ ``(*batch, T, d_z)`` from $s_0$ = ``s0`` ``(*batch, d_s)``:
        ``forward`` step after step, with $u$ and $\tilde s$ of all the steps
        in one matmul each."""
        u = torch.sigmoid(self.gate(z))
        return linear_scan(1 - u, u * self.candidate(z), s0)


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

    def scan(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        r"""The outputs $y_1..y_T$ ``(*batch, T, d)`` of the inputs ``x``
        ``(*batch, T, d)`` from the state $h_0$ = ``h0`` ``(*batch, d)``:
        ``forward`` step after step (``MinGRU.scan``)."""
        return x + self.cell.scan(nn.functional.gelu(x), h0)


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
