r"""The modules of the dynamics, shared by the PC-ALM nodes, their BP baseline,
the evaluation and the offline bound.

The dynamics is a ``MinGRU`` $s_k = \mathrm{minGRU}(x_{k-1}, s_{k-1})$ and
``nodes - 1`` layers $y_k = x_k + \mathrm{minGRU}(\mathrm{GELU}(x_k),
h_{k-1})$ on top (``ResidualMinGRU``), read out by a linear map $W_o$ to the
next observation: $\hat x_k = W_o y_L$, or $\hat x_k = x_{k-1} + W_o y_L$ with
``residual``. ``build_dynamics`` is the one place that draws their initial
weights.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch
from mujou.types import Device
from torch import nn

from exp.lib.mingru import MinGRU, ResidualMinGRU
from exp.video.schema.dyn import DynamicsConf

__all__ = ["DynamicsModules", "DynamicsStack", "Top", "build_dynamics"]


class DynamicsStack(nn.Module):
    r"""The nodes of the dynamics as one module: from $x_{k-1}$ and the state
    $(s_1, h_2, \dots, h_L)_{k-1}$ (concatenated, ``L d_s``) of the ``Dynamics``
    and the ``DynamicsLayer`` nodes, the new state. With one node, it is the
    minGRU (the state is $s$)."""

    def __init__(self, first: MinGRU, rest: Sequence[ResidualMinGRU]) -> None:
        super().__init__()
        self.first = first
        self.rest = nn.ModuleList(rest)
        self.d_s = first.gate.out_features

    @property
    def d_state(self) -> int:
        """The width of the state, ``L d_s``."""
        return self.d_s * (1 + len(self.rest))

    @override
    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """The state ``(*batch, L d_s)`` after ``state`` ``(*batch, L d_s)``
        with the input ``x`` ``(*batch, d_in)``."""
        parts = state.split(self.d_s, -1)
        y = self.first(x, parts[0])
        out = [y]
        for layer, h in zip(self.rest, parts[1:], strict=True):
            y_next = layer(y, h)
            out.append(y_next - y)
            y = y_next
        return torch.cat(out, -1)


class Top(nn.Module):
    r"""The output $y_L = s_1 + h_2 + \dots + h_L$ of the top node of the
    dynamics, ``(*batch, d_s)``, from the state ``(*batch, L d_s)`` of
    ``DynamicsStack``."""

    def __init__(self, d_s: int) -> None:
        super().__init__()
        self.d_s = d_s

    @override
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.stack(state.split(self.d_s, -1)).sum(0)


@dataclass
class DynamicsModules:
    r"""The modules of the dynamics: ``step`` moves the state, ``readout``
    ($W_o$, ``d_s -> d_out``) reads the prediction out of ``Top`` of it."""

    step: DynamicsStack
    readout: nn.Linear


def build_dynamics(
    conf: DynamicsConf,
    d_in: int,
    d_out: int,
    *,
    seed: int,
    device: Device = "cpu",
) -> DynamicsModules:
    """The modules of ``conf`` with inputs of ``d_in`` and predictions of
    ``d_out``: after ``torch.manual_seed(seed)``, the minGRU, the
    ``conf.nodes - 1`` layers and the readout, in this order, on the CPU (so
    that the initial weights do not depend on the device), then moved to
    ``device``. The CPU's global RNG is restored afterwards, so what is drawn
    after it does not depend on the dynamics. With ``conf.residual``, the readout starts at 0: the
    prediction starts as the input before.

    Raises:
        ValueError: ``conf.residual`` with ``d_in != d_out`` (the input before
            is the base of the prediction).
    """
    if conf.residual and d_in != d_out:
        raise ValueError(
            f"residual predicts x_(k-1) + W_o s_k: d_in {d_in} must equal d_out {d_out}"
        )
    with torch.random.fork_rng(devices=[]):  # the global RNG is left as it was
        torch.manual_seed(seed)
        first = MinGRU(d_in, conf.d_s)
        rest = [ResidualMinGRU(conf.d_s) for _ in range(conf.nodes - 1)]
        readout = nn.Linear(conf.d_s, d_out)
    if conf.residual:  # the prediction starts as x_{k-1}, the change as 0
        nn.init.zeros_(readout.weight)
        nn.init.zeros_(readout.bias)
    return DynamicsModules(
        step=DynamicsStack(first, rest).to(device), readout=readout.to(device)
    )
