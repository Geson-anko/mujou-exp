r"""The optimizers of PLAN.md, section 5: directions that do not expect i.i.d.
gradients, as ``pc.optim.Optimizer``.

Each optimizer here is first a direction $D$ computed from the gradients
(``DirectionOptimizer``), which a node steps along as SGD does,
$\Delta W = -\frac{\text{elapsed}}{\tau} D$, or with the step size of SGD
(``Grafted``), $\Delta W = -\frac{\text{elapsed}}{\tau} \lVert G \rVert_F
\frac{D}{\lVert D \rVert_F}$.

A weight of two or more dimensions is a matrix $(d_\text{out}, d_\text{in})$
(a convolution's $(c_\text{out}, c_\text{in} k^2)$); the other parameters (the
biases) take $D = G$.
"""

import math
from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import torch
from mujou.pc import optim as pc_optim
from mujou.pc.optim import BoundOptimizer, Optimizer

from experiments.continuous_stream.config import OptimConfig
from mujou import Stateful

__all__ = [
    "AdamDirection",
    "BoundDirection",
    "DirectionOptimizer",
    "Grafted",
    "Projected",
    "Whitened",
    "from_config",
]


class BoundDirection(Stateful):
    """The direction of an optimizer bound to the parameters of one node."""

    @abstractmethod
    def direction(
        self, grads: Sequence[torch.Tensor], elapsed: float
    ) -> list[torch.Tensor]:
        """Move the state by ``elapsed`` seconds with ``grads`` and return the
        direction of each parameter (of its shape)."""


@dataclass(frozen=True, kw_only=True)
class DirectionOptimizer(Optimizer):
    """A direction, stepped along with the learning rate ``elapsed / tau``.

    Args:
        tau: The time constant of the parameters, in seconds, as in
            ``pc.optim.SGD``.
    """

    tau: float

    @abstractmethod
    def bind_direction(self, params: Sequence[torch.Tensor]) -> BoundDirection:
        """Return the direction bound to ``params``, with a fresh state."""

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        params = list(params)
        return _Stepped(params, self.bind_direction(params), self.tau, graft=False)


@dataclass(frozen=True, kw_only=True)
class Grafted(Optimizer):
    """The direction of ``inner`` with the step size of SGD, layer by layer:
    ``||ΔW||_F = (elapsed / inner.tau) ||G||_F``."""

    inner: DirectionOptimizer

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        params = list(params)
        inner = self.inner
        return _Stepped(params, inner.bind_direction(params), inner.tau, graft=True)


class _Stepped(BoundOptimizer):
    def __init__(
        self,
        params: list[torch.Tensor],
        direction: BoundDirection,
        tau: float,
        *,
        graft: bool,
    ) -> None:
        self._params = params
        self._direction = direction
        self._tau = tau
        self._graft = graft

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if len(grads) != len(self._params):
            raise ValueError(f"{len(grads)} gradients for {len(self._params)} params")
        if elapsed == 0:
            return
        eta = elapsed / self._tau
        with torch.no_grad():
            ds = self._direction.direction(grads, elapsed)
            for p, g, d in zip(self._params, grads, ds, strict=True):
                if self._graft:
                    norm = torch.linalg.vector_norm(d)
                    if norm == 0:
                        continue
                    d = d * (torch.linalg.vector_norm(g) / norm)
                p.sub_(d, alpha=eta)

    @override
    def state_dict(self) -> Any:
        return self._direction.state_dict()

    @override
    def load_state_dict(self, state: Any, /) -> None:
        self._direction.load_state_dict(state)


def _matrix(t: torch.Tensor) -> torch.Tensor:
    """``t`` as the matrix ``(d_out, d_in)`` (a view)."""
    return t.reshape(t.shape[0], -1)


def _ema_(avg: torch.Tensor, target: torch.Tensor, elapsed: float, tau: float) -> None:
    """``tau d(avg)/dt = target - avg`` over ``elapsed`` with ``target`` held."""
    avg.lerp_(target, -math.expm1(-elapsed / tau))


def _empty() -> torch.Tensor:
    return torch.zeros(0)


class _Factors(Stateful):
    """The moving averages ``L = EMA(G G^T)`` and ``R = EMA(G^T G)`` of each
    matrix, and a matrix computed from each, refreshed every ``refresh``
    seconds (an empty tensor where a side is not used)."""

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        tau_l: float | None,
        tau_r: float | None,
        refresh: float,
    ) -> None:
        self._tau_l, self._tau_r, self._refresh = tau_l, tau_r, refresh
        self.matrix = [p.ndim >= 2 for p in params]

        def square(p: torch.Tensor, tau: float | None, side: int) -> torch.Tensor:
            if p.ndim < 2 or tau is None:
                return _empty()
            n = _matrix(p).shape[side]
            return torch.zeros(n, n, dtype=p.dtype, device=p.device)

        self.l = [square(p, tau_l, 0) for p in params]
        self.r = [square(p, tau_r, 1) for p in params]
        self.of_l = [x.clone() for x in self.l]
        self.of_r = [x.clone() for x in self.r]
        self.since = math.inf  # seconds since of_l, of_r were computed

    def update_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        for i, g in enumerate(grads):
            if not self.matrix[i]:
                continue
            m = _matrix(g)
            if self._tau_l is not None:
                _ema_(self.l[i], m @ m.T, elapsed, self._tau_l)
            if self._tau_r is not None:
                _ema_(self.r[i], m.T @ m, elapsed, self._tau_r)

    def due(self, elapsed: float) -> bool:
        """Count ``elapsed`` and say whether to compute the matrices anew."""
        self.since += elapsed
        if self.since >= self._refresh:
            self.since = 0.0
            return True
        return False

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "l": list(self.l),
            "r": list(self.r),
            "of_l": list(self.of_l),
            "of_r": list(self.of_r),
            "since": self.since,
        }

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        with torch.no_grad():
            for key in ("l", "r", "of_l", "of_r"):
                for mine, saved in zip(getattr(self, key), state[key], strict=True):
                    mine.copy_(saved)
        self.since = float(state["since"])


def _eigh(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The eigenvalues (ascending) and eigenvectors of the symmetric ``a``, in
    float64."""
    return torch.linalg.eigh(a.double())


@dataclass(frozen=True, kw_only=True)
class Whitened(DirectionOptimizer):
    r"""Whitened in time: $D = (L + \epsilon I)^{-a}\, G\, (R + \epsilon I)^{-b}$.

    $L$ and $R$ follow $G G^\top$ and $G^\top G$ with the time constants
    ``tau_l`` and ``tau_r`` (``None``: that side is not whitened), exactly in
    time (``L += (1 - exp(-elapsed / tau_l)) (G G^T - L)``), the current $G$
    included. The roots are computed by an eigendecomposition every
    ``refresh`` seconds (and in the first step); between, the last ones are
    used. $\epsilon$ is ``eps`` times the largest eigenvalue.

    $a = b = 0$ is SGD; with $a = b = 1/4$ and $\tau \to 0$, $D = U V^\top$ for
    $G = U \Sigma V^\top$ (an orthogonalized step, as Muon).
    """

    tau_l: float | None = None
    tau_r: float | None = None
    a: float = 0.25
    b: float = 0.25
    eps: float = 1e-4
    refresh: float = 1.0

    @override
    def bind_direction(self, params: Sequence[torch.Tensor]) -> BoundDirection:
        return _BoundWhitened(params, self)


def _inverse_root(a: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    """``(a + eps * max_eig I)^-power`` of the symmetric ``a``."""
    values, vectors = _eigh(a)
    top = values[-1].clamp(min=0)
    if top == 0:  # no gradient yet
        return torch.eye(len(a), dtype=a.dtype, device=a.device)
    scaled = (values.clamp(min=0) + eps * top).pow(-power)
    return ((vectors * scaled) @ vectors.T).to(a.dtype)


class _BoundWhitened(BoundDirection):
    def __init__(self, params: Sequence[torch.Tensor], config: Whitened) -> None:
        self._config = config
        self._factors = _Factors(params, config.tau_l, config.tau_r, config.refresh)

    @override
    def direction(
        self, grads: Sequence[torch.Tensor], elapsed: float
    ) -> list[torch.Tensor]:
        c, f = self._config, self._factors
        f.update_(grads, elapsed)
        if f.due(elapsed):
            for i in range(len(grads)):
                if f.l[i].numel():
                    f.of_l[i].copy_(_inverse_root(f.l[i], c.a, c.eps))
                if f.r[i].numel():
                    f.of_r[i].copy_(_inverse_root(f.r[i], c.b, c.eps))
        ds = []
        for i, g in enumerate(grads):
            d = _matrix(g)
            if f.of_l[i].numel():
                d = f.of_l[i] @ d
            if f.of_r[i].numel():
                d = d @ f.of_r[i]
            ds.append(d.reshape(g.shape))
        return ds

    @override
    def state_dict(self) -> Any:
        return self._factors.state_dict()

    @override
    def load_state_dict(self, state: Any, /) -> None:
        self._factors.load_state_dict(state)


@dataclass(frozen=True, kw_only=True)
class Projected(DirectionOptimizer):
    r"""A hard projection: $D = (I - U_L U_L^\top)\, G\, (I - U_R U_R^\top)$.

    $U_L$ and $U_R$ are the eigenvectors of the ``rank`` largest eigenvalues
    of $L$ and $R$ (as in ``Whitened``) before the current $G$ is added, those
    of eigenvalues above ``eps`` times the largest only. They are computed
    every ``refresh`` seconds (and in the first step).
    """

    tau_l: float | None = None
    tau_r: float | None = None
    rank: int = 4
    eps: float = 1e-4
    refresh: float = 1.0

    @override
    def bind_direction(self, params: Sequence[torch.Tensor]) -> BoundDirection:
        return _BoundProjected(params, self)


def _projector(a: torch.Tensor, rank: int, eps: float) -> torch.Tensor:
    """``I - U U^T`` with ``U`` the top ``rank`` eigenvectors of ``a`` (those
    of eigenvalues above ``eps`` times the largest)."""
    values, vectors = _eigh(a)
    top = values[-1]
    keep = values > eps * top if top > 0 else torch.zeros_like(values, dtype=torch.bool)
    keep[: len(values) - rank] = False
    u = vectors[:, keep]
    eye = torch.eye(len(a), dtype=u.dtype, device=u.device)
    return (eye - u @ u.T).to(a.dtype)


class _BoundProjected(BoundDirection):
    def __init__(self, params: Sequence[torch.Tensor], config: Projected) -> None:
        self._config = config
        self._factors = _Factors(params, config.tau_l, config.tau_r, config.refresh)

    @override
    def direction(
        self, grads: Sequence[torch.Tensor], elapsed: float
    ) -> list[torch.Tensor]:
        c, f = self._config, self._factors
        if f.due(elapsed):  # from L and R before the current G
            for i in range(len(grads)):
                if f.l[i].numel():
                    f.of_l[i].copy_(_projector(f.l[i], c.rank, c.eps))
                if f.r[i].numel():
                    f.of_r[i].copy_(_projector(f.r[i], c.rank, c.eps))
        ds = []
        for i, g in enumerate(grads):
            d = _matrix(g)
            if f.of_l[i].numel():
                d = f.of_l[i] @ d
            if f.of_r[i].numel():
                d = d @ f.of_r[i]
            ds.append(d.reshape(g.shape))
        f.update_(grads, elapsed)
        return ds

    @override
    def state_dict(self) -> Any:
        return self._factors.state_dict()

    @override
    def load_state_dict(self, state: Any, /) -> None:
        self._factors.load_state_dict(state)


@dataclass(frozen=True, kw_only=True)
class AdamDirection(DirectionOptimizer):
    r"""The direction of ``pc.optim.AdamW``, $\hat m / (\sqrt{\hat v} + \varepsilon)$,
    with the moments and masses moved exactly in time as there. Stepped by
    itself, it is ``pc.optim.AdamW`` without weight decay."""

    tau_m: float
    tau_v: float
    eps: float = 1e-8

    @override
    def bind_direction(self, params: Sequence[torch.Tensor]) -> BoundDirection:
        return _BoundAdamDirection(params, self)


class _BoundAdamDirection(BoundDirection):
    def __init__(self, params: Sequence[torch.Tensor], config: AdamDirection) -> None:
        self._config = config
        self._m = [torch.zeros_like(p) for p in params]
        self._v = [torch.zeros_like(p) for p in params]
        self._c_m = 0.0
        self._c_v = 0.0

    @override
    def direction(
        self, grads: Sequence[torch.Tensor], elapsed: float
    ) -> list[torch.Tensor]:
        c = self._config
        a_m = -math.expm1(-elapsed / c.tau_m)
        a_v = -math.expm1(-elapsed / c.tau_v)
        self._c_m += a_m * (1 - self._c_m)
        self._c_v += a_v * (1 - self._c_v)
        ds = []
        for g, m, v in zip(grads, self._m, self._v, strict=True):
            m.lerp_(g, a_m)
            v.lerp_(g * g, a_v)
            denom = (v.sqrt() / math.sqrt(self._c_v)).add_(c.eps)
            ds.append(m / self._c_m / denom)
        return ds

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "m": list(self._m),
            "v": list(self._v),
            "c_m": self._c_m,
            "c_v": self._c_v,
        }

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        with torch.no_grad():
            for mine, saved in zip(self._m, state["m"], strict=True):
                mine.copy_(saved)
            for mine, saved in zip(self._v, state["v"], strict=True):
                mine.copy_(saved)
        self._c_m = float(state["c_m"])
        self._c_v = float(state["c_v"])


def from_config(c: OptimConfig) -> Optimizer:
    """The optimizer of the settings ``c``."""
    if c.kind == "sgd":
        return pc_optim.SGD(tau=c.tau)
    if c.kind == "adamw":
        return pc_optim.AdamW(
            tau=c.tau, tau_m=c.tau_m, tau_v=c.tau_v, weight_decay=c.weight_decay
        )
    inner: DirectionOptimizer
    if c.kind == "adam_direction":
        inner = AdamDirection(tau=c.tau, tau_m=c.tau_m, tau_v=c.tau_v)
    elif c.kind == "whitened":
        inner = Whitened(
            tau=c.tau,
            tau_l=c.tau_l,
            tau_r=c.tau_r,
            a=c.a,
            b=c.b,
            eps=c.eps,
            refresh=c.refresh,
        )
    elif c.kind == "projected":
        inner = Projected(
            tau=c.tau,
            tau_l=c.tau_l,
            tau_r=c.tau_r,
            rank=c.rank,
            eps=c.eps,
            refresh=c.refresh,
        )
    else:
        raise ValueError(f"unknown optimizer {c.kind!r}")
    return Grafted(inner=inner) if c.graft else inner
