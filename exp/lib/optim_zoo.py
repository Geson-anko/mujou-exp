r"""Basic optimizers of ``torch.optim`` in time, as ``pc.optim.Optimizer``:
Nesterov's momentum (``Nesterov``), centered RMSProp (``RMSPropCentered``),
``Adadelta``, ``Rprop`` and ``Muon`` (with AdamW on what is not a matrix).

As in ``mujou.pc.optim``, a step of ``elapsed`` $= \delta$ seconds moves the
moments exactly as the flow would with the gradient held over $\delta$
($x \leftarrow x + a (y - x)$, $a = 1 - e^{-\delta/\tau_x}$), and the
parameters by $\eta = \delta / \tau$ times a direction. A moment that starts
at zero has its mass $c \leftarrow c + a (1 - c)$ (the share of it made of
gradients) to correct its bias, as ``pc.optim.AdamW``. With a constant
$\delta$, each is the ``torch.optim`` optimizer of the learning rate
$\delta / \tau$ and the decays $\beta = e^{-\delta / \tau_x}$, up to that
correction (torch has none, except AdamW's): the two agree from the same
moments once $c = 1$ (each docstring says how; ``tests/lib/test_optim_zoo.py``
checks it). A step of ``elapsed == 0`` moves nothing.

They do not use ``torch.optim``, which brings the GIL back into a
free-threaded Python.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import torch
from mujou.pc.optim import AdamW, BoundOptimizer, Optimizer

__all__ = [
    "NS_COEFFICIENTS",
    "Adadelta",
    "Muon",
    "Nesterov",
    "RMSPropCentered",
    "Rprop",
    "newton_schulz",
]


def _check_positive(owner: str, name: str, value: float) -> None:
    """As ``pc.optim.AdamW`` checks its time constants and ``eps``."""
    if not 0 < value < math.inf:
        raise ValueError(f"{owner}.{name} must be finite and > 0, got {value!r}")


def _decay(elapsed: float, tau: float) -> float:
    r"""$a = 1 - e^{-\text{elapsed}/\tau}$, the share a moment moves to its
    target in ``elapsed``."""
    return -math.expm1(-elapsed / tau)


def _copy_into(mine: Sequence[torch.Tensor], saved: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for m, s in zip(mine, saved, strict=True):
            m.copy_(s)


@dataclass(frozen=True, kw_only=True)
class Nesterov(Optimizer):
    r"""Nesterov's momentum in time: the momentum of ``pc.optim.SGD(tau,
    tau_m)`` (the gradient's moving average $m$, $\hat m = m / c_m$) looked
    ahead,

    $$\Delta\theta = -\frac{\delta}{\tau}\big((1 - \beta)\, g + \beta\,
    \hat m\big),\qquad \beta = e^{-\delta/\tau_m},$$

    with $m$ moved first. A constant $g$ gives $\hat m = g$, so the step is
    SGD's: the momentum averages the gradient, the learning rate stays.

    torch: ``SGD(lr=(δ/τ)(1 − β), momentum=β, nesterov=True)``. Its buffer is
    the sum $b \leftarrow \beta b + g$, and $(1 - \beta) b$ is $m$ from zero,
    so the steps agree once $c_m = 1$ (they differ by the bias correction
    before, as ``pc.optim.SGD``'s momentum).

    The state holds ``"m"`` (tensors of the parameters') and ``"c_m"``.
    """

    tau: float
    tau_m: float

    def __post_init__(self) -> None:
        _check_positive("Nesterov", "tau", self.tau)
        _check_positive("Nesterov", "tau_m", self.tau_m)

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        return _BoundNesterov(list(params), self)


class _BoundNesterov(BoundOptimizer):
    def __init__(self, params: list[torch.Tensor], config: Nesterov) -> None:
        self._params = params
        self._config = config
        self._m = [torch.zeros_like(p) for p in params]
        self._c_m = 0.0

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if elapsed == 0:
            return
        c = self._config
        a = _decay(elapsed, c.tau_m)
        self._c_m += a * (1 - self._c_m)
        eta = elapsed / c.tau
        with torch.no_grad():
            for p, g, m in zip(self._params, grads, self._m, strict=True):
                m.lerp_(g, a)  # tau_m dm/dt = g - m
                # (1 - beta) g + beta m_hat, beta = 1 - a
                p.sub_(g.lerp(m / self._c_m, 1 - a), alpha=eta)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"m": list(self._m), "c_m": self._c_m}

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        _copy_into(self._m, state["m"])
        self._c_m = float(state["c_m"])


@dataclass(frozen=True, kw_only=True)
class RMSPropCentered(Optimizer):
    r"""Centered RMSProp in time: the gradient divided by its standard
    deviation over about $\tau_v$,

    $$\tau_v \dot v = g^2 - v,\quad \tau_v \dot m = g - m,\quad
    \Delta\theta = -\frac{\delta}{\tau}\,\frac{g}{\sqrt{s} + \varepsilon},
    \quad s = \frac{v - m^2}{c},$$

    $c$ the mass of both moments (one time constant). $s = \hat v - c\,\hat
    m^2$ is torch's $v - m^2$ with the zero start's bias removed as
    ``pc.optim.RMSProp`` removes it from $v$: at the first step $s = (1 - a)
    g^2$, about RMSProp's $\hat v = g^2$. (The bias-corrected variance $\hat
    v - \hat m^2$ is 0 there, a step of $g / \varepsilon$.) $v - m^2 \ge 0$,
    as $m^2 \le c\, v$ for averages of the same weights summing to $c$; it is
    clamped at 0 against rounding.

    torch: ``RMSprop(lr=δ/τ, alpha=e^{−δ/τ_v}, eps=eps, centered=True)``,
    the same steps once $c = 1$.

    The state holds ``"v"``, ``"m"`` and ``"c"``.
    """

    tau: float
    tau_v: float
    eps: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("tau", "tau_v", "eps"):
            _check_positive("RMSPropCentered", name, getattr(self, name))

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        return _BoundRMSPropCentered(list(params), self)


class _BoundRMSPropCentered(BoundOptimizer):
    def __init__(self, params: list[torch.Tensor], config: RMSPropCentered) -> None:
        self._params = params
        self._config = config
        self._v = [torch.zeros_like(p) for p in params]
        self._m = [torch.zeros_like(p) for p in params]
        self._c = 0.0

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if elapsed == 0:
            return
        conf = self._config
        a = _decay(elapsed, conf.tau_v)
        self._c += a * (1 - self._c)
        eta = elapsed / conf.tau
        with torch.no_grad():
            for p, g, v, m in zip(self._params, grads, self._v, self._m, strict=True):
                v.lerp_(g * g, a)  # tau_v dv/dt = g^2 - v
                m.lerp_(g, a)  # tau_v dm/dt = g - m
                s = v.addcmul(m, m, value=-1).clamp_(min=0) / self._c
                p.addcdiv_(g, s.sqrt_().add_(conf.eps), value=-eta)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"v": list(self._v), "m": list(self._m), "c": self._c}

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        _copy_into(self._v, state["v"])
        _copy_into(self._m, state["m"])
        self._c = float(state["c"])


@dataclass(frozen=True, kw_only=True)
class Adadelta(Optimizer):
    r"""Adadelta in time: the gradient scaled by the RMS of the past steps'
    directions over the RMS of the gradient,

    $$D = \frac{\sqrt{\hat u + \varepsilon}}{\sqrt{\hat v + \varepsilon}}\,
    g,\qquad \Delta\theta = -\frac{\delta}{\tau}\, D,$$

    $v$ the moving average of $g^2$ (moved first) and $u$ that of $D^2$
    (moved after; $\hat u$ in $D$ is the one before the step, 0 at the first
    step), both of the time constant $\tau_v$, each divided by its own mass
    ($c_v$, and $c_u$ of $u$, one step behind). $\tau$ of 0.1 s at frames
    of 0.1 s is the learning rate 1 of the original Adadelta.

    torch: ``Adadelta(lr=δ/τ, rho=e^{−δ/τ_v}, eps=eps)`` (its ``acc_delta``
    is $u$, of the direction before the learning rate), the same steps once
    $c_v = c_u = 1$.

    The state holds ``"v"``, ``"u"``, ``"c_v"`` and ``"c_u"``.
    """

    tau: float
    tau_v: float
    eps: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("tau", "tau_v", "eps"):
            _check_positive("Adadelta", name, getattr(self, name))

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        return _BoundAdadelta(list(params), self)


class _BoundAdadelta(BoundOptimizer):
    def __init__(self, params: list[torch.Tensor], config: Adadelta) -> None:
        self._params = params
        self._config = config
        self._v = [torch.zeros_like(p) for p in params]
        self._u = [torch.zeros_like(p) for p in params]
        self._c_v = 0.0
        self._c_u = 0.0

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if elapsed == 0:
            return
        conf = self._config
        a = _decay(elapsed, conf.tau_v)
        self._c_v += a * (1 - self._c_v)
        c_u = self._c_u  # u's mass before this step (0: no u yet)
        eta = elapsed / conf.tau
        with torch.no_grad():
            for p, g, v, u in zip(self._params, grads, self._v, self._u, strict=True):
                v.lerp_(g * g, a)  # tau_v dv/dt = g^2 - v
                u_hat = u / c_u if c_u > 0 else torch.zeros_like(u)
                rms_v = (v / self._c_v).add_(conf.eps).sqrt_()
                d = u_hat.add_(conf.eps).sqrt_().div_(rms_v).mul_(g)
                u.lerp_(d * d, a)  # tau_v du/dt = D^2 - u
                p.sub_(d, alpha=eta)
        self._c_u += a * (1 - self._c_u)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "v": list(self._v),
            "u": list(self._u),
            "c_v": self._c_v,
            "c_u": self._c_u,
        }

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        _copy_into(self._v, state["v"])
        _copy_into(self._u, state["u"])
        self._c_v = float(state["c_v"])
        self._c_u = float(state["c_u"])


RPROP_UP, RPROP_DOWN = 1.2, 0.5  # torch's etas
RPROP_FLOOR, RPROP_START = 1e-3, 0.1  # of s_max: the least and the first rate


@dataclass(frozen=True, kw_only=True)
class Rprop(Optimizer):
    r"""Rprop in time: each element moves by its own rate $s$ (per second)
    along the sign of its gradient, $\Delta\theta = -\delta\, s\,
    \mathrm{sign}(g)$.

    At each step, before it moves, $s$ grows by 1.2 where $g$ has the sign
    of the last gradient and shrinks by 0.5 where it has the other (that
    element then does not move, and its last gradient is taken as 0, so the
    next step leaves $s$ as it is), and is clamped to $[10^{-3} s_{\max},
    s_{\max}]$, $s_{\max} = 1 / \tau$; it starts at $s_{\max} / 10$. The rate
    adapts per step, not per second: only the move is in time.

    torch: ``Rprop(lr=δ s_max / 10, etas=(0.5, 1.2), step_sizes=(δ s_max
    10⁻³, δ s_max))``, its step sizes $\delta s$; the same steps from the
    start.

    The state holds ``"rate"`` ($s$) and ``"prev"`` (the last gradient, 0
    where it flipped).
    """

    tau: float

    def __post_init__(self) -> None:
        _check_positive("Rprop", "tau", self.tau)

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        return _BoundRprop(list(params), self)


class _BoundRprop(BoundOptimizer):
    def __init__(self, params: list[torch.Tensor], config: Rprop) -> None:
        self._params = params
        s_max = 1 / config.tau
        self._max, self._min = s_max, RPROP_FLOOR * s_max
        self._rate = [torch.full_like(p, RPROP_START * s_max) for p in params]
        self._prev = [torch.zeros_like(p) for p in params]

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if elapsed == 0:
            return
        with torch.no_grad():
            for p, g, s, prev in zip(
                self._params, grads, self._rate, self._prev, strict=True
            ):
                agree = (g * prev).sign()  # +1 the same sign, -1 flipped, 0 none
                factor = torch.ones_like(s).masked_fill_(agree > 0, RPROP_UP)
                s.mul_(factor.masked_fill_(agree < 0, RPROP_DOWN))
                s.clamp_(self._min, self._max)
                g = torch.where(agree < 0, 0.0, g)  # a flip does not move
                p.addcmul_(g.sign(), s, value=-elapsed)
                prev.copy_(g)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"rate": list(self._rate), "prev": list(self._prev)}

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        _copy_into(self._rate, state["rate"])
        _copy_into(self._prev, state["prev"])


# Keller Jordan's quintic (torch.optim.Muon's defaults), 5 steps
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-7


def newton_schulz(x: torch.Tensor, steps: int = 5) -> torch.Tensor:
    r"""The orthogonalization of the matrix ``x`` ``(A, B)`` by ``steps``
    Newton-Schulz steps of the quintic ``NS_COEFFICIENTS`` $(a, b, c)$, in
    bfloat16, as ``torch.optim.Muon``: $X = x / \lVert x \rVert_F$ (the
    transpose if $A > B$), then $X \leftarrow a X + (b\, XX^\top + c\,
    (XX^\top)^2) X$. Its singular values end in about (0.5, 1.5), not at 1;
    returns bfloat16 of ``x``'s shape."""
    a, b, c = NS_COEFFICIENTS
    out = x.to(dtype=torch.bfloat16, copy=True)
    tall = x.size(0) > x.size(1)
    if tall:
        out = out.T
    out.div_(out.norm().clamp(min=NS_EPS))
    for _ in range(steps):
        gram = out @ out.T
        poly = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        out = torch.addmm(out, poly, out, beta=a)
    return out.T if tall else out


@dataclass(frozen=True, kw_only=True)
class Muon(Optimizer):
    r"""Muon in time on the matrices, AdamW on the rest.

    A parameter of 2 or more dimensions is a matrix: an ``nn.Linear``'s
    weight ``(A, B)``, an ``nn.Conv2d``'s ``(A, C, k, k)`` taken as ``(A, C k
    k)``. It moves along the orthogonalized look-ahead of its momentum,

    $$\Delta\theta = -\frac{\delta}{\tau}\big(w\,\theta + 0.2
    \sqrt{\max(A, B)}\; \mathrm{NS}_5\big((1 - \beta)\, g + \beta\, \hat
    m\big)\big),$$

    the momentum $m$ and $\beta$ as ``Nesterov``'s (moved first), the decay
    on $\theta$ before the step, $\mathrm{NS}_5$ ``newton_schulz``. The
    factor $0.2\sqrt{\max(A, B)}$ (Moonshot's ``match_rms_adamw``) gives the
    step the RMS of AdamW's, so ``tau`` is AdamW's. Every other parameter
    (biases, the norms' gains, $\ln\sigma$, learned tokens) moves by
    ``pc.optim.AdamW(tau, tau_m, tau_v, eps, weight_decay)``.

    torch: ``Muon(lr=δ/τ, momentum=β, nesterov=True, weight_decay=w,
    adjust_lr_fn="match_rms_adamw")`` (its buffer is $m$ from zero, so the
    same steps once $c_m = 1$) and ``AdamW(lr=δ/τ, betas=(β, e^{−δ/τ_v}),
    eps=eps, weight_decay=w)`` (the same steps from the start).

    The state holds the matrices' ``"m"`` and ``"c_m"`` and the AdamW's
    state ``"adamw"``.
    """

    tau: float
    tau_m: float
    tau_v: float
    eps: float = 1e-8
    weight_decay: float = 1e-2

    def __post_init__(self) -> None:
        for name in ("tau", "tau_m", "tau_v", "eps"):
            _check_positive("Muon", name, getattr(self, name))
        if not 0 <= self.weight_decay < math.inf:
            raise ValueError(
                f"Muon.weight_decay must be finite and >= 0, got {self.weight_decay!r}"
            )

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        return _BoundMuon(list(params), self)


class _BoundMuon(BoundOptimizer):
    def __init__(self, params: list[torch.Tensor], config: Muon) -> None:
        self._config = config
        self._is_matrix = [p.ndim >= 2 for p in params]
        self._matrices = [p for p, mat in zip(params, self._is_matrix) if mat]
        self._m = [torch.zeros_like(p) for p in self._matrices]
        self._c_m = 0.0
        self._adamw = AdamW(
            tau=config.tau,
            tau_m=config.tau_m,
            tau_v=config.tau_v,
            eps=config.eps,
            weight_decay=config.weight_decay,
        ).bind(p for p, mat in zip(params, self._is_matrix) if not mat)

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if len(grads) != len(self._is_matrix):
            raise ValueError(
                f"{len(grads)} gradients for {len(self._is_matrix)} params"
            )
        if elapsed == 0:
            return
        conf = self._config
        mats = [g for g, mat in zip(grads, self._is_matrix) if mat]
        self._adamw.step_(
            [g for g, mat in zip(grads, self._is_matrix) if not mat], elapsed
        )
        a = _decay(elapsed, conf.tau_m)
        self._c_m += a * (1 - self._c_m)
        eta = elapsed / conf.tau
        with torch.no_grad():
            for p, g, m in zip(self._matrices, mats, self._m, strict=True):
                m.lerp_(g, a)
                look = g.lerp(m / self._c_m, 1 - a).reshape(len(p), -1)
                rows, cols = look.shape
                o = newton_schulz(look).reshape(p.shape)
                p.mul_(1 - eta * conf.weight_decay)
                p.add_(o, alpha=-eta * 0.2 * math.sqrt(max(rows, cols)))

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"m": list(self._m), "c_m": self._c_m, "adamw": self._adamw.state_dict()}

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        _copy_into(self._m, state["m"])
        self._c_m = float(state["c_m"])
        self._adamw.load_state_dict(state["adamw"])
