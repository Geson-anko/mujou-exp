r"""Orthogonal gradient descent in time (``OGD``): SGD, RMSProp or AdamW of
``mujou.pc.optim`` whose step is kept orthogonal to the node's own past
gradients, in the metric of its preconditioner.

A node's parameters are taken as one vector $\theta$ of $D$ values. Every
``every`` seconds of the node's own time it keeps its raw gradient $g$; the
last ``n`` kept are the columns of $G$ ``(D, k)``. With the preconditioner
$P$ of the base ($I$ for SGD, $\mathrm{diag}(1 / (\sqrt{\hat v} +
\varepsilon))$ for RMSProp and AdamW) and the base's direction $d$ ($g$ for
SGD and RMSProp, $\hat m$ for AdamW), the step solves

$$\min_\Delta\; d^\top \Delta + \frac{1}{2\eta} \Delta^\top P^{-1} \Delta
\quad \text{s.t.}\quad G^\top \Delta = 0
\;\Rightarrow\; \Delta = -\eta \big(P - PG (G^\top P G)^{+} G^\top P\big)\, d,$$

$\eta = \text{elapsed} / \tau$. Without the constraint it is the base's step
$-\eta P d$. The moments ($m$, $v$) and their masses move as the base's,
with the unprojected $g$; AdamW's weight decay $\theta \leftarrow \theta (1 -
\eta w)$ is applied outside the projection, first, as there. The
pseudo-inverse drops the eigendirections of $G^\top P G$ whose eigenvalues are
below ``RANK_FLOOR`` times the largest (consecutive gradients are alike), as
``exp.lib.grad_stats.residual_share`` does; the $k \times k$ eigenproblem is
solved on the CPU in float64.

While nothing is kept (the first step) the step is the base's, bit for bit.

The time is the node's own, the sum of the steps' ``elapsed``. A gradient is
kept at the first step, then at the step that ends nearest to each multiple
$j \cdot$ ``every``: the first whose time $t$ has $t + \text{elapsed}/2 > j
\cdot$ ``every`` (as ``GradStats`` keeps its gradients), so the times do not
depend on how the time is cut into steps. A step uses the gradients kept
before it, then keeps its own.

Locality: everything is of the node's own gradients, moments and parameters.
If every node's step has $g_{\mathrm{old},i}^\top \Delta_i = 0$, so has the
sum over the nodes: OGD's guarantee of no first-order interference with the
kept gradients holds for the whole network without any node seeing another's.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import torch
from mujou.pc.optim import SGD, AdamW, BoundOptimizer, Optimizer, RMSProp

__all__ = ["RANK_FLOOR", "OGD", "BoundOGD"]

RANK_FLOOR = 1e-6  # of the largest eigenvalue of G^T P G: a direction to drop


@dataclass(frozen=True, kw_only=True)
class OGD(Optimizer):
    """``base`` with its step projected off the last ``n`` gradients kept
    every ``every`` seconds (module docstring).

    Args:
        base: ``SGD`` (without momentum), ``RMSProp`` or ``AdamW`` of
            ``mujou.pc.optim``, whose settings it uses.
        n: How many kept gradients the step is orthogonal to.
        every: The seconds between two kept gradients.

    Raises:
        TypeError: ``base`` is of another kind.
        ValueError: ``n`` < 1, ``every`` is not finite and > 0, or ``base`` is
            SGD with momentum.
    """

    base: SGD | RMSProp | AdamW
    n: int
    every: float

    def __post_init__(self) -> None:
        if not isinstance(self.base, SGD | RMSProp | AdamW):
            raise TypeError(
                f"OGD.base is SGD, RMSProp or AdamW, not {type(self.base).__name__}"
            )
        if isinstance(self.base, SGD) and self.base.tau_m is not None:
            raise ValueError("OGD.base is SGD without momentum (tau_m None)")
        if self.n < 1:
            raise ValueError(f"OGD.n must be >= 1, got {self.n!r}")
        if not 0 < self.every < math.inf:
            raise ValueError(f"OGD.every must be finite and > 0, got {self.every!r}")

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOGD:
        return BoundOGD(list(params), self)


class BoundOGD(BoundOptimizer):
    """``OGD`` bound to the parameters of one node. Its state is the base's
    (the same keys) with ``"kept"`` ``(n, D)``, ``"times"`` and ``"count"``."""

    def __init__(self, params: list[torch.Tensor], config: OGD) -> None:
        self._params = params
        self._config = config
        base = config.base
        moments = isinstance(base, RMSProp | AdamW)
        self._m = (
            [torch.zeros_like(p) for p in params] if isinstance(base, AdamW) else []
        )
        self._v = [torch.zeros_like(p) for p in params] if moments else []
        self._c_m = 0.0
        self._c_v = 0.0
        size = sum(p.numel() for p in params)
        like = params[0] if params else torch.zeros(())
        self._kept = torch.zeros(config.n, size, dtype=like.dtype, device=like.device)
        self._t = 0.0
        self._due = 0.0  # the first step keeps
        self._times = [math.nan] * config.n
        self._count = 0  # kept so far (the next slot: % n)

    def _order(self) -> list[int]:
        """The slots of the kept gradients, oldest first."""
        n, k = self._config.n, min(self._count, self._config.n)
        return [(self._count - k + i) % n for i in range(k)]

    def kept_times(self) -> list[float]:
        """The node's time of each kept gradient, oldest first."""
        return [self._times[i] for i in self._order()]

    def kept_gradients(self) -> torch.Tensor:
        """The kept gradients as rows ``(k, D)``, oldest first (a copy)."""
        return self._kept[self._order()].clone()

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if len(grads) != len(self._params):
            raise ValueError(f"{len(grads)} gradients for {len(self._params)} params")
        if elapsed == 0:
            return  # the flow of no time
        base = self._config.base
        eta = elapsed / base.tau
        with torch.no_grad():
            if self._count == 0:
                self._base_step(grads, elapsed, eta)
            else:
                self._projected_step(grads, elapsed, eta)
            self._keep(grads, elapsed)

    def _base_step(
        self, grads: Sequence[torch.Tensor], elapsed: float, eta: float
    ) -> None:
        """The base's step, in the order of ``mujou.pc.optim``'s (bit for bit)."""
        base = self._config.base
        if isinstance(base, SGD):
            for p, g in zip(self._params, grads, strict=True):
                p.sub_(g, alpha=eta)
            return
        a_m, a_v, sqrt_c_v = self._move_masses(elapsed)
        if isinstance(base, RMSProp):
            for p, g, v in zip(self._params, grads, self._v, strict=True):
                v.lerp_(g * g, a_v)
                denom = (v.sqrt() / sqrt_c_v).add_(base.eps)
                p.addcdiv_(g, denom, value=-eta)
            return
        for p, g, m, v in zip(self._params, grads, self._m, self._v, strict=True):
            p.mul_(1 - eta * base.weight_decay)
            m.lerp_(g, a_m)
            v.lerp_(g * g, a_v)
            denom = (v.sqrt() / sqrt_c_v).add_(base.eps)
            p.addcdiv_(m, denom, value=-eta / self._c_m)

    def _projected_step(
        self, grads: Sequence[torch.Tensor], elapsed: float, eta: float
    ) -> None:
        r"""$\theta \leftarrow \theta - \eta\, P (d - G c)$, $c = (G^\top P
        G)^{+} G^\top P d$, after the moments and AdamW's decay."""
        base = self._config.base
        if isinstance(base, SGD):
            d, metric = _flat(grads), None
        else:
            a_m, a_v, sqrt_c_v = self._move_masses(elapsed)
            ds, denoms = [], []
            for i, (p, g, v) in enumerate(
                zip(self._params, grads, self._v, strict=True)
            ):
                if isinstance(base, AdamW):
                    p.mul_(1 - eta * base.weight_decay)
                    m = self._m[i]
                    m.lerp_(g, a_m)
                    ds.append(m / self._c_m)
                else:
                    ds.append(g)
                v.lerp_(g * g, a_v)
                denoms.append((v.sqrt() / sqrt_c_v).add_(base.eps))
            d, metric = _flat(ds), 1 / _flat(denoms)
        kept = self._kept[self._order()]  # (k, D)
        pd = d if metric is None else metric * d
        pk = kept if metric is None else kept * metric
        coef = _pinv_solve(pk @ kept.T, kept @ pd).to(d)
        step = pd - (kept.T @ coef if metric is None else pk.T @ coef)
        for p, s in zip(self._params, _split(step, self._params), strict=True):
            p.sub_(s, alpha=eta)

    def _move_masses(self, elapsed: float) -> tuple[float, float, float]:
        """Move $c_m$, $c_v$ by ``elapsed``; ``(a_m, a_v, sqrt(c_v))``."""
        base = self._config.base
        assert isinstance(base, RMSProp | AdamW)
        a_m = -math.expm1(-elapsed / base.tau_m) if isinstance(base, AdamW) else 0.0
        a_v = -math.expm1(-elapsed / base.tau_v)
        self._c_m += a_m * (1 - self._c_m)
        self._c_v += a_v * (1 - self._c_v)
        return a_m, a_v, math.sqrt(self._c_v)

    def _keep(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        """Keep ``grads`` when a multiple of ``every`` is due (module docstring)."""
        every = self._config.every
        t, half = self._t + elapsed, elapsed / 2
        if t + half > self._due:
            slot = self._count % self._config.n
            self._kept[slot].copy_(_flat(grads))
            self._times[slot] = t
            self._count += 1
            self._due = (math.floor((t + half) / every) + 1) * every
        self._t = t

    @override
    def state_dict(self) -> dict[str, Any]:
        base = self._config.base
        out: dict[str, Any] = {}
        if isinstance(base, AdamW):
            out |= {"m": list(self._m), "c_m": self._c_m}
        if isinstance(base, RMSProp | AdamW):
            out |= {"v": list(self._v), "c_v": self._c_v}
        return out | {
            "kept": self._kept,
            "times": torch.tensor(
                [self._t, self._due, *self._times], dtype=torch.float64
            ),
            "count": torch.tensor(self._count, dtype=torch.int64),
        }

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        with torch.no_grad():
            for mine, saved in zip(self._m, state.get("m", []), strict=True):
                mine.copy_(saved)
            for mine, saved in zip(self._v, state.get("v", []), strict=True):
                mine.copy_(saved)
            self._kept.copy_(state["kept"])
        self._c_m = float(state.get("c_m", 0.0))
        self._c_v = float(state.get("c_v", 0.0))
        times = [float(x) for x in state["times"]]
        self._t, self._due, self._times = times[0], times[1], times[2:]
        self._count = int(state["count"])


def _flat(ts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts])


def _split(flat: torch.Tensor, like: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """``flat`` cut into tensors of the shapes of ``like``."""
    sizes = [t.numel() for t in like]
    return [x.view_as(t) for x, t in zip(flat.split(sizes), like, strict=True)]


def _pinv_solve(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    r"""$M^{+} b$ of the symmetric $M$ ``gram`` ``(k, k)`` and ``rhs`` ``(k,)``
    in float64 on the CPU, dropping the eigenvalues below ``RANK_FLOOR``
    times the largest (0 when $M$ has none > 0)."""
    lam, vec = torch.linalg.eigh(gram.double().cpu())
    b = rhs.double().cpu()
    top = float(lam[-1])
    if top <= 0:
        return torch.zeros_like(b)
    keep = lam > RANK_FLOOR * top
    v = vec[:, keep]
    return v @ ((v.T @ b) / lam[keep])
