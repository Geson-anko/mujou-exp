r"""``Angular``: an optimizer that sets how fast each matrix moves in time.

V12 found that the decoder reads its code $z$ only when its weights move
slowly against the scenes: a decoder that turns by a radian in 120 seconds
or less tracks the present scene with its weights and ignores $z$, whatever
the optimizer. ``Angular`` makes that speed the setting itself. A matrix
moves along its momentum by the share $\delta/\tau$ of its norm,

$$\Delta\theta = -\frac{\delta}{\tau}\,\lVert\theta\rVert\,
\frac{\hat m}{\lVert\hat m\rVert},$$

$m$ the gradient's moving average of time constant $\tau_m$ ($m \leftarrow m
+ a (g - m)$ from 0, $a = 1 - e^{-\delta/\tau_m}$; $\hat m = m / c$ with its
mass $c$, which cancels in $\hat m / \lVert\hat m\rVert$). The relative step is $\delta/\tau$ exactly, whatever
the gradient's or the loss's scale: $\tau$ is the seconds the matrix takes to
turn by about a radian (when the step is across $\theta$). A matrix of zero
norm, or a zero momentum, does not move.

A parameter of 2 or more dimensions is a matrix (as ``optim_zoo.Muon``). The
rest has no scale to keep (a bias or a norm's offset may start at 0) and moves
by ``pc.optim.RMSProp``: the parameters of one value (the likelihood's
$\ln\sigma$) of ``tau_scalar``, which does not follow ``tau``, the others of
``vec_scale * tau``. With ``vec_scale`` 10, $\tau$ = 1000 s gives those the
RMSProp of $10^4$ s of V8.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import torch
from mujou.pc.optim import BoundOptimizer, Optimizer, RMSProp

__all__ = ["Angular"]


@dataclass(frozen=True, kw_only=True)
class Angular(Optimizer):
    """The matrices turn by ``elapsed / tau`` of their norm (module
    docstring); the vectors and scalars by RMSProp.

    The state holds the matrices' ``"m"`` and the RMSProps'
    ``"vec"`` and ``"scalar"``.
    """

    tau: float
    tau_m: float
    tau_v: float
    eps: float = 1e-8
    vec_scale: float = 10.0
    tau_scalar: float = 1e4

    def __post_init__(self) -> None:
        for name in ("tau", "tau_m", "tau_v", "eps", "vec_scale", "tau_scalar"):
            value = getattr(self, name)
            if not 0 < value < math.inf:
                raise ValueError(
                    f"Angular.{name} must be finite and > 0, got {value!r}"
                )

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        return _BoundAngular(list(params), self)


class _BoundAngular(BoundOptimizer):
    def __init__(self, params: list[torch.Tensor], config: Angular) -> None:
        self._config = config
        self._kind = [
            "matrix" if p.ndim >= 2 else "scalar" if p.numel() == 1 else "vec"
            for p in params
        ]
        self._matrices = [p for p, k in zip(params, self._kind) if k == "matrix"]
        self._m = [torch.zeros_like(p) for p in self._matrices]
        rms = {"tau_v": config.tau_v, "eps": config.eps}
        self._vec = RMSProp(tau=config.vec_scale * config.tau, **rms).bind(
            self._of(params, "vec")
        )
        self._scalar = RMSProp(tau=config.tau_scalar, **rms).bind(
            self._of(params, "scalar")
        )

    def _of(self, xs: Sequence[torch.Tensor], kind: str) -> list[torch.Tensor]:
        return [x for x, k in zip(xs, self._kind, strict=True) if k == kind]

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if len(grads) != len(self._kind):
            raise ValueError(f"{len(grads)} gradients for {len(self._kind)} params")
        if elapsed == 0:
            return
        self._vec.step_(self._of(grads, "vec"), elapsed)
        self._scalar.step_(self._of(grads, "scalar"), elapsed)
        a = -math.expm1(-elapsed / self._config.tau_m)
        eta = elapsed / self._config.tau
        with torch.no_grad():
            for p, g, m in zip(self._matrices, self._of(grads, "matrix"), self._m):
                m.lerp_(g, a)
                norm_m = torch.linalg.vector_norm(m)
                scale = -eta * torch.linalg.vector_norm(p) / norm_m
                p.add_(torch.where(norm_m > 0, scale, 0.0) * m)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "m": list(self._m),
            "vec": self._vec.state_dict(),
            "scalar": self._scalar.state_dict(),
        }

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        with torch.no_grad():
            for mine, saved in zip(self._m, state["m"], strict=True):
                mine.copy_(saved)
        self._vec.load_state_dict(state["vec"])
        self._scalar.load_state_dict(state["scalar"])
