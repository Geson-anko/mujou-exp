r"""Optimizers in time for the nodes, as ``pc.optim.Optimizer``: the direction
of Adam with the step size of SGD (``Grafted(AdamDirection)``), RMSProp (the
direction of Adam without the first moment, ``RMSPropDirection``), a warm-up of
any optimizer (``Warmup``), and ``adam``, Adam of a learning rate per step as a
``pc.optim.AdamW`` (for fits by steps, such as the offline bound).

``StepAdamW`` counts steps rather than time: ``torch.optim.AdamW`` with the
schedules of the JEPA recipes (a linear warm-up of the learning rate over
steps, then constant or a cosine decay; a linear schedule of the weight
decay), the decay only of the parameters of two or more dimensions, and the
clipping of the norm of all the node's gradients. Its warm-up ramps the
learning rate only, where ``Warmup`` reparametrizes the time of its inner
optimizer (and so also stretches the time constants of Adam's moments).

A direction optimizer computes a direction $D$ from the gradients
(``DirectionOptimizer``), which a node steps along as SGD does,
$\Delta W = -\frac{\text{elapsed}}{\tau} D$, or with the step size of SGD
(``Grafted``), $\Delta W = -\frac{\text{elapsed}}{\tau} \lVert G \rVert_F
\frac{D}{\lVert D \rVert_F}$, tensor by tensor. With ``fan_in`` (of
``fan_ins``) each tensor's step is divided by its fan-in $n$: the learning rate
$\eta / n$ of a matrix and $\eta$ of a vector, the per-layer rate of Adam under
the spectral condition (muP; Yang, Simon & Bernstein 2023).
"""

import math
from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, override

import torch
from mujou.pc.optim import AdamW, BoundOptimizer, Optimizer
from torch import nn

from mujou import Stateful

__all__ = [
    "AdamDirection",
    "BoundDirection",
    "DirectionOptimizer",
    "DECAYS",
    "Grafted",
    "RMSPropDirection",
    "StepAdamW",
    "Warmup",
    "adam",
    "fan_ins",
]


def _check_positive(owner: str, name: str, value: float) -> None:
    """As ``pc.optim.AdamW`` checks its time constants and ``eps``."""
    if not 0 < value < math.inf:
        raise ValueError(f"{owner}.{name} must be finite and > 0, got {value!r}")


class BoundDirection(Stateful):
    """The direction of an optimizer bound to the parameters of one node."""

    @abstractmethod
    def direction(
        self, grads: Sequence[torch.Tensor], elapsed: float
    ) -> list[torch.Tensor]:
        """Move the state by ``elapsed`` seconds with ``grads`` and return the
        direction of each parameter (of its shape)."""


def fan_ins(*modules: nn.Module) -> dict[torch.Tensor, int]:
    r"""The fan-in $n$ of each parameter of ``modules``, by its role in the
    module that holds it (not by its shape):

    - the weight of an ``nn.Linear``: ``in_features``, the inputs its sum runs
      over. A fused map is one ``nn.Linear`` and keeps it: the attention's
      ``qkv`` of width $d$ to $3d$ has $n = d$;
    - the bias of an ``nn.Linear``, the gain and the bias of an
      ``nn.LayerNorm``: 1 (vectors, added or multiplied element by element: no
      sum over inputs);
    - a parameter a module of the model holds itself, not through a torch
      layer (a learned token: the mask token, the extra tokens): 1 (added to
      the tokens: no input).

    The keys are the parameters themselves (as the state of
    ``torch.optim``), for ``DirectionOptimizer.fan_in``.

    Raises:
        ValueError: A parameter of another torch layer (``nn.Conv2d``,
            ``nn.Embedding``, ``nn.MultiheadAttention``, ...), whose role
            this rule does not state, or a parameter shared by two modules
            with two fan-ins.
    """
    out: dict[torch.Tensor, int] = {}
    for module in modules:
        for m in module.modules():
            for name, p in m.named_parameters(recurse=False):
                n = _fan_in(m, name)
                if out.setdefault(p, n) != n:
                    raise ValueError(f"the parameter {name!r} has two fan-ins")
    return out


def _fan_in(m: nn.Module, name: str) -> int:
    """The fan-in of the parameter ``name`` held by ``m`` itself (``fan_ins``)."""
    if isinstance(m, nn.Linear):
        return m.in_features if name == "weight" else 1
    if isinstance(m, nn.LayerNorm):
        return 1
    if type(m).__module__.startswith("torch."):
        raise ValueError(
            f"no fan-in rule for the parameter {name!r} of {type(m).__name__}"
        )
    return 1


@dataclass(frozen=True, kw_only=True)
class DirectionOptimizer(Optimizer):
    r"""A direction, stepped along with the learning rate ``elapsed / tau``,
    divided per tensor by its fan-in with ``fan_in``.

    Args:
        tau: The time constant of the parameters, in seconds, as in
            ``pc.optim.SGD``.
        fan_in: The fan-in $n$ of each parameter (``fan_ins``); a step of a
            tensor is then $-\frac{\text{elapsed}}{\tau n} D$. ``None``: 1
            for every one (the step without it, bit for bit). Binding a
            parameter it does not hold raises ``ValueError``.
    """

    tau: float
    fan_in: Mapping[torch.Tensor, int] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _check_positive(type(self).__name__, "tau", self.tau)

    @abstractmethod
    def bind_direction(self, params: Sequence[torch.Tensor]) -> BoundDirection:
        """Return the direction bound to ``params``, with a fresh state."""

    def fan_in_of(self, params: Sequence[torch.Tensor]) -> list[int]:
        """$n$ of each of ``params`` (1 for all without ``fan_in``).

        Raises:
            ValueError: ``fan_in`` does not hold one of ``params``.
        """
        if self.fan_in is None:
            return [1] * len(params)
        missing = [tuple(p.shape) for p in params if p not in self.fan_in]
        if missing:
            raise ValueError(f"no fan-in for the parameters of shapes {missing}")
        return [self.fan_in[p] for p in params]

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        params = list(params)
        return _Stepped(
            params,
            self.bind_direction(params),
            self.tau,
            self.fan_in_of(params),
            graft=False,
        )


@dataclass(frozen=True, kw_only=True)
class Grafted(Optimizer):
    """The direction of ``inner`` with the step size of SGD, layer by layer:
    ``||ΔW||_F = (elapsed / inner.tau) ||G||_F``."""

    inner: DirectionOptimizer

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        params = list(params)
        inner = self.inner
        return _Stepped(
            params,
            inner.bind_direction(params),
            inner.tau,
            inner.fan_in_of(params),
            graft=True,
        )


class _Stepped(BoundOptimizer):
    def __init__(
        self,
        params: list[torch.Tensor],
        direction: BoundDirection,
        tau: float,
        fan_in: list[int],
        *,
        graft: bool,
    ) -> None:
        self._params = params
        self._direction = direction
        self._tau = tau
        self._fan_in = fan_in
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
            for p, g, d, n in zip(self._params, grads, ds, self._fan_in, strict=True):
                if self._graft:
                    norm = torch.linalg.vector_norm(d)
                    if norm == 0:
                        continue
                    d = d * (torch.linalg.vector_norm(g) / norm)
                p.sub_(d, alpha=eta / n)

    @override
    def state_dict(self) -> Any:
        return self._direction.state_dict()

    @override
    def load_state_dict(self, state: Any, /) -> None:
        self._direction.load_state_dict(state)


@dataclass(frozen=True, kw_only=True)
class AdamDirection(DirectionOptimizer):
    r"""The direction of ``pc.optim.AdamW``, $\hat m / (\sqrt{\hat v} + \varepsilon)$,
    with the moments and masses moved exactly in time as there. Stepped by
    itself, it is ``pc.optim.AdamW`` without weight decay.

    Raises:
        ValueError: ``tau``, ``tau_m``, ``tau_v`` or ``eps`` is not finite and
            > 0 (as in ``pc.optim.AdamW``).
    """

    tau_m: float
    tau_v: float
    eps: float = 1e-8

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("tau_m", "tau_v", "eps"):
            _check_positive("AdamDirection", name, getattr(self, name))

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


@dataclass(frozen=True, kw_only=True)
class RMSPropDirection(DirectionOptimizer):
    r"""The direction of RMSProp, $g / (\sqrt{\hat v} + \varepsilon)$: that of
    ``AdamDirection`` with no first moment (the current gradient $g$ in place
    of $\hat m$), the second moment $v$ and its mass $c_v$ moved exactly in
    time as there, $v \leftarrow v + a_v (g^2 - v)$, $c_v \leftarrow c_v +
    a_v (1 - c_v)$, $a_v = 1 - e^{-\text{elapsed}/\tau_v}$, $\hat v = v /
    c_v$. Stepped by itself with steps of ``elapsed`` $\delta$, it is Adam of
    $\beta_1 = 0$, $\beta_2 = e^{-\delta/\tau_v}$ and the learning rate
    $\delta / \tau$.

    Raises:
        ValueError: ``tau``, ``tau_v`` or ``eps`` is not finite and > 0 (as in
            ``pc.optim.AdamW``).
    """

    tau_v: float
    eps: float = 1e-8

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("tau_v", "eps"):
            _check_positive("RMSPropDirection", name, getattr(self, name))

    @override
    def bind_direction(self, params: Sequence[torch.Tensor]) -> BoundDirection:
        return _BoundRMSPropDirection(params, self)


class _BoundRMSPropDirection(BoundDirection):
    def __init__(
        self, params: Sequence[torch.Tensor], config: RMSPropDirection
    ) -> None:
        self._config = config
        self._v = [torch.zeros_like(p) for p in params]
        self._c_v = 0.0

    @override
    def direction(
        self, grads: Sequence[torch.Tensor], elapsed: float
    ) -> list[torch.Tensor]:
        c = self._config
        a_v = -math.expm1(-elapsed / c.tau_v)
        self._c_v += a_v * (1 - self._c_v)
        ds = []
        for g, v in zip(grads, self._v, strict=True):
            v.lerp_(g * g, a_v)
            denom = (v.sqrt() / math.sqrt(self._c_v)).add_(c.eps)
            ds.append(g / denom)
        return ds

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"v": list(self._v), "c_v": self._c_v}

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        with torch.no_grad():
            for mine, saved in zip(self._v, state["v"], strict=True):
                mine.copy_(saved)
        self._c_v = float(state["c_v"])


@dataclass(frozen=True, kw_only=True)
class Warmup(Optimizer):
    r"""``inner`` with its time weighed by the ramp $r(t) = \min(1, t / T)$ over
    the first ``seconds`` $T$ of the steps of one node: a step of ``elapsed``
    seconds from $t$ is a step of $\int_t^{t + \text{elapsed}} r(s)\,ds$ of
    ``inner`` (for SGD, the learning rate rises linearly from 0 over $T$). The
    time $t$ is the sum of the steps' ``elapsed``, kept in the state, so the
    ramp is the same however the time is cut into steps.

    All the time of ``inner`` is reparametrized, not only its step size: an
    inner with a state (the moments of AdamW and of ``AdamDirection``) also
    sees its time constants stretched by $1 / r(t)$ within the ramp.
    """

    inner: Optimizer
    seconds: float

    def __post_init__(self) -> None:
        if not 0 < self.seconds < math.inf:
            raise ValueError(f"seconds must be finite and > 0, got {self.seconds!r}")

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        return _BoundWarmup(self.inner.bind(params), self.seconds)


class _BoundWarmup(BoundOptimizer):
    def __init__(self, inner: BoundOptimizer, seconds: float) -> None:
        self._inner = inner
        self._seconds = seconds
        self._t = torch.zeros((), dtype=torch.float64)

    def _ramped(self, t: float) -> float:
        r"""$\int_0^t r(s)\,ds$."""
        T = self._seconds
        return t * t / (2 * T) if t <= T else t - T / 2

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        t = float(self._t)
        self._t.fill_(t + elapsed)
        self._inner.step_(grads, self._ramped(t + elapsed) - self._ramped(t))

    @override
    def state_dict(self) -> Any:
        return {"t": self._t, "inner": self._inner.state_dict()}

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        self._t.copy_(state["t"])
        self._inner.load_state_dict(state["inner"])


def adam(
    lr: float, beta1: float = 0.9, beta2: float = 0.999, *, eps: float = 1e-8
) -> AdamW:
    r"""Adam (Kingma & Ba 2015, no weight decay) of the learning rate ``lr``
    and the decays ``beta1``, ``beta2`` per step, as a ``pc.optim.AdamW``
    stepped with ``elapsed=1.0`` per step: ``tau = 1 / lr``,
    ``tau_m = -1 / ln(beta1)``, ``tau_v = -1 / ln(beta2)``.

    Stepped so, $a_m = 1 - e^{-1/\tau_m} = 1 - \beta_1$ and the mass after
    $t$ steps is $1 - \beta_1^t$, so each step is that of Adam:
    $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ (and $v_t$ with $g_t^2$),
    $\theta_t = \theta_{t-1} - \mathrm{lr}\,\hat m_t / (\sqrt{\hat v_t} + \epsilon)$
    with $\hat m_t = m_t / (1 - \beta_1^t)$, $\hat v_t = v_t / (1 - \beta_2^t)$.
    It does not use ``torch.optim``, which brings the GIL back.

    Raises:
        ValueError: ``lr`` is not finite and > 0, or a beta is not in (0, 1).
    """
    if not 0 < lr < math.inf:
        raise ValueError(f"lr must be finite and > 0, got {lr!r}")
    for name, beta in (("beta1", beta1), ("beta2", beta2)):
        if not 0 < beta < 1:
            raise ValueError(f"{name} must be in (0, 1), got {beta!r}")
    return AdamW(
        tau=1 / lr,
        tau_m=-1 / math.log(beta1),
        tau_v=-1 / math.log(beta2),
        eps=eps,
        weight_decay=0.0,
    )


DECAYS = ("constant", "cosine")  # the learning rate after the warm-up


@dataclass(frozen=True, kw_only=True)
class StepAdamW(Optimizer):
    r"""AdamW by steps, as ``torch.optim.AdamW``, with schedules: each
    ``step_`` of ``elapsed`` > 0 is one step $t = 1, 2, \dots$ whatever its
    ``elapsed`` (a step of 0 moves nothing), of the learning rate
    $\eta_t$ (``lr_at``) and the weight decay $w_t$ (``wd_at``):

    1. with ``clip`` $c > 0$, the gradients $g$ of all the parameters are
       scaled by $\min(1, c / (\lVert g \rVert + 10^{-6}))$, $\lVert g \rVert$
       the norm of all of them together (``torch.nn.utils.clip_grad_norm_``);
    2. a parameter of two or more dimensions decays, $\theta \leftarrow
       \theta (1 - \eta_t w_t)$ (the biases, the norms' gains and the mask
       token, of one dimension, do not);
    3. $m \leftarrow \beta_1 m + (1 - \beta_1) g$, $v \leftarrow \beta_2 v +
       (1 - \beta_2) g^2$, $\theta \leftarrow \theta - \eta_t \hat m /
       (\sqrt{\hat v} + \varepsilon)$ with $\hat m = m / (1 - \beta_1^t)$,
       $\hat v = v / (1 - \beta_2^t)$.

    $\eta_t$ rises linearly to ``lr`` over the ``warmup_steps`` $W$
    ($\eta_t = \mathrm{lr}\, t / W$ for $t \le W$), then stays (``decay``
    ``"constant"``) or falls by a half cosine to ``min_lr`` times ``lr`` at
    step ``decay_steps`` $S$ and stays there (``"cosine"``). $w_t$ is
    ``weight_decay``, or with ``weight_decay_end`` it moves linearly from
    ``weight_decay`` at step 0 to ``weight_decay_end`` at $S$ (I-JEPA's 0.04 to
    0.4). The state is the moments and the count of steps.

    Raises:
        ValueError: ``lr`` is not finite and > 0, a beta is not in [0, 1),
            ``decay`` is not of ``DECAYS``, or a schedule needs ``decay_steps``
            beyond ``warmup_steps``.
    """

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    weight_decay_end: float | None = None
    warmup_steps: int = 0
    decay: str = "constant"
    decay_steps: int = 0
    min_lr: float = 0.0
    clip: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.lr < math.inf:
            raise ValueError(f"lr must be finite and > 0, got {self.lr!r}")
        if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
            raise ValueError(f"betas must be two in [0, 1), got {self.betas!r}")
        if self.decay not in DECAYS:
            raise ValueError(f"decay must be one of {DECAYS}, got {self.decay!r}")
        if self.warmup_steps < 0 or self.clip < 0 or self.weight_decay < 0:
            raise ValueError("warmup_steps, clip and weight_decay must be >= 0")
        scheduled = self.decay == "cosine" or self.weight_decay_end is not None
        if scheduled and not self.decay_steps > self.warmup_steps:
            raise ValueError(
                f"a schedule needs decay_steps ({self.decay_steps}) beyond "
                f"warmup_steps ({self.warmup_steps})"
            )

    def lr_at(self, t: int) -> float:
        r"""$\eta_t$ of step ``t`` (from 1)."""
        w = self.warmup_steps
        if t <= w:
            return self.lr * t / w
        if self.decay == "constant":
            return self.lr
        progress = min(1.0, (t - w) / (self.decay_steps - w))
        low = self.min_lr * self.lr
        return low + (self.lr - low) * 0.5 * (1 + math.cos(math.pi * progress))

    def wd_at(self, t: int) -> float:
        """$w_t$ of step ``t``."""
        if self.weight_decay_end is None:
            return self.weight_decay
        a = min(1.0, t / self.decay_steps)
        return self.weight_decay + (self.weight_decay_end - self.weight_decay) * a

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        return _BoundStepAdamW(list(params), self)


class _BoundStepAdamW(BoundOptimizer):
    def __init__(self, params: list[torch.Tensor], config: StepAdamW) -> None:
        self._params = params
        self._config = config
        self._m = [torch.zeros_like(p) for p in params]
        self._v = [torch.zeros_like(p) for p in params]
        self._t = torch.zeros((), dtype=torch.int64)

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if len(grads) != len(self._params):
            raise ValueError(f"{len(grads)} gradients for {len(self._params)} params")
        if elapsed == 0:
            return
        c = self._config
        t = int(self._t) + 1
        self._t.fill_(t)
        lr, wd = c.lr_at(t), c.wd_at(t)
        b1, b2 = c.betas
        c1, c2 = 1 - b1**t, 1 - b2**t
        with torch.no_grad():
            if c.clip > 0 and grads:
                norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(g) for g in grads])
                )
                coef = (c.clip / (norm + 1e-6)).clamp(max=1.0)
                grads = [g * coef.to(g) for g in grads]
            for p, g, m, v in zip(self._params, grads, self._m, self._v, strict=True):
                if p.dim() >= 2 and wd:
                    p.mul_(1 - lr * wd)
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v.sqrt() / math.sqrt(c2)).add_(c.eps)
                p.addcdiv_(m, denom, value=-lr / c1)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"m": list(self._m), "v": list(self._v), "t": self._t}

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        with torch.no_grad():
            for mine, saved in zip(self._m, state["m"], strict=True):
                mine.copy_(saved)
            for mine, saved in zip(self._v, state["v"], strict=True):
                mine.copy_(saved)
        self._t.copy_(state["t"])
