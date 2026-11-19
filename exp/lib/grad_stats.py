r"""What the gradients of one node do over time, measured around any
optimizer without changing it (``GradStats``).

A node's parameters are taken as one vector $\theta$ (the gradients $g$ of
each step concatenated in the bound order). At each step the wrapper hands
the gradients to the inner optimizer unchanged, so the learning is the
same bit for bit; it reads the gradients and the parameters, and keeps
copies of past gradients (in ``store``, bfloat16 by default):

- the previous gradient;
- an anchor per span $L$ of ``anchors``: the gradient of the step at about
  every $L$ seconds, taken again each $L$;
- the last ``keep`` gradients kept at about every $s$ seconds, per $s$ of
  ``spans``.

About every ``every`` seconds it makes a row ``{"t", "node", "g", "theta",
"dtheta", "cos_prev", "age_prev", "cos_<L>", "age_<L>", ..., "rho_<s>",
...}``: $\lVert g\rVert$, $\lVert\theta\rVert$ before the step and
$\lVert\Delta\theta\rVert$ of the step, the cosine of $g$ with the previous
gradient and with each anchor and their ages (seconds since they were
taken), and $\rho_s = \lVert (I - QQ^\top) g\rVert / \lVert g \rVert$, the
share of $g$ off the span of the kept gradients (``residual_share``; $Q$ an
orthonormal basis), the share an orthogonal step against them would keep.
About every ``tensors_every`` seconds the row also holds ``"rel"``,
$\lVert\Delta\theta_i\rVert / \lVert\theta_i\rVert$ of each tensor by its
name. A value with nothing to compare against yet (no previous gradient,
anchor or kept one, a zero norm), or not finite, is ``None``. The stats of a step use the
copies from before it; it then takes its own.

The time is the node's own, the sum of the steps' ``elapsed``. An event
of a period $P$ (a row, an anchor, a kept gradient) is due at $kP$, and
falls on the step that ends nearest to it: the first whose time $t$ has
$t + \text{elapsed}/2 > kP$; the next is then due at the first $kP$ past
that. Anchors and kept gradients are first taken at the first step.

Locality: everything is of the node's own gradients and parameters; the
host only collects the rows (``drain``) between steps.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, override

import torch
from mujou.pc.optim import BoundOptimizer, Optimizer

__all__ = ["GradStats", "residual_share"]

RANK_FLOOR = 1e-6  # of the largest eigenvalue of G G^T: a direction to drop


def residual_share(G: torch.Tensor, g: torch.Tensor) -> float:
    r"""$\lVert (I - QQ^\top) g\rVert / \lVert g\rVert$ for the rows of
    ``G`` ``(k, D)`` spanning the space of the orthonormal $Q$.

    The span is that of the eigenvectors of $GG^\top$ whose eigenvalues are
    at least ``RANK_FLOOR`` times the largest (in float64): parallel or
    nearly parallel rows add no direction. Returns 1.0 when ``G`` spans
    nothing and NaN when $g = 0$.

    The $k \times k$ eigenproblem is solved on the CPU, whatever the device
    of ``G``: the first ``eigh`` on CUDA loads torch's CUDA linalg lazily,
    which raises when two node threads do it at once.
    """
    gram = (G @ G.T).double().cpu()
    lam, vec = torch.linalg.eigh(gram)
    top = float(lam[-1]) if len(lam) else 0.0
    norm = float(torch.linalg.vector_norm(g))
    if top <= 0:
        return 1.0 if norm > 0 else math.nan
    keep = lam > RANK_FLOOR * top
    v, lam = vec[:, keep], lam[keep]
    # P g = G^T V diag(1/lam) V^T G g, the projection on the rows' span
    coef = v @ ((v.T @ (G @ g).double().cpu()) / lam)
    r = g - G.T @ coef.to(g)
    return float(torch.linalg.vector_norm(r)) / norm if norm > 0 else math.nan


@dataclass(frozen=True, kw_only=True)
class GradStats(Optimizer):
    """``inner`` with the gradient statistics of the module docstring.

    Args:
        inner: The optimizer that moves the parameters.
        node: The name of the node, in each row.
        names: The name of each parameter for ``"rel"`` (``p<i>``, its place
            in the bound order, for one it does not hold).
        store: The dtype of the kept copies: ``(1 + len(anchors) +
            len(spans) * keep) * D`` values in all.
        every: Seconds between two rows.
        tensors_every: Seconds between two ``"rel"``.
        anchors: The spans $L$ of the anchors, in seconds.
        spans: The intervals $s$ of the kept gradients, in seconds.
        keep: How many gradients each $s$ keeps ($n$).
    """

    inner: Optimizer
    node: str
    names: Mapping[torch.Tensor, str] | None = field(
        default=None, repr=False, compare=False
    )
    store: torch.dtype = torch.bfloat16
    every: float = 1.0
    tensors_every: float = 60.0
    anchors: tuple[float, ...] = (1.0, 10.0, 100.0)
    spans: tuple[float, ...] = (0.1, 1.0, 10.0)
    keep: int = 16

    def __post_init__(self) -> None:
        periods = [self.every, self.tensors_every, *self.anchors, *self.spans]
        if not all(0 < p < math.inf for p in periods) or self.keep < 1:
            raise ValueError(
                "GradStats: every period must be finite and > 0, keep >= 1"
            )

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundGradStats:
        params = list(params)
        names = self.names or {}
        labels = [names.get(p, f"p{i}") for i, p in enumerate(params)]
        return BoundGradStats(params, self.inner.bind(params), self, labels)


def _due(t: float, half: float, period: float) -> float:
    """The next due time past ``t + half`` of the events of ``period``."""
    return (math.floor((t + half) / period) + 1) * period


class BoundGradStats(BoundOptimizer):
    """``GradStats`` bound to the parameters of one node; ``drain()`` hands
    over its rows."""

    def __init__(
        self,
        params: list[torch.Tensor],
        inner: BoundOptimizer,
        config: GradStats,
        labels: list[str],
    ) -> None:
        self._params = params
        self._inner = inner
        self._config = config
        self._labels = labels
        c = config
        size = sum(p.numel() for p in params)
        dev = params[0].device if params else torch.device("cpu")
        # the dtype of the stats: the parameters' at least float32
        self._dtype = torch.promote_types(c.store, torch.float32)
        if params:
            self._dtype = torch.promote_types(self._dtype, params[0].dtype)
        self._prev = torch.zeros(size, dtype=c.store, device=dev)
        self._anchor = torch.zeros(len(c.anchors), size, dtype=c.store, device=dev)
        self._kept = torch.zeros(len(c.spans), c.keep, size, dtype=c.store, device=dev)
        self._t = 0.0
        self._t_prev = math.nan  # none yet
        self._t_anchor = [math.nan] * len(c.anchors)
        self._count = [0] * len(c.spans)  # kept so far (the next slot: % keep)
        self._due_row, self._due_rel = c.every, c.tensors_every
        self._due_anchor = [0.0] * len(c.anchors)
        self._due_kept = [0.0] * len(c.spans)
        self._rows: list[dict[str, Any]] = []

    def drain(self) -> list[dict[str, Any]]:
        """The rows made since the last call (and forget them)."""
        rows, self._rows = self._rows, []
        return rows

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if elapsed == 0:
            self._inner.step_(grads, elapsed)
            return
        c = self._config
        t, half = self._t + elapsed, elapsed / 2
        row = t + half > self._due_row
        rel = t + half > self._due_rel
        with torch.no_grad():
            g = torch.cat([x.reshape(-1).to(self._dtype) for x in grads])
            before = [p.detach().clone() for p in self._params] if row else []
            out = self._stats(g, t, elapsed) if row else {}
        self._inner.step_(grads, elapsed)
        with torch.no_grad():
            if row:
                out |= self._moves(before, rel)
                self._rows.append(_clean({"t": t, "node": c.node, **out}))
                self._due_row = _due(t, half, c.every)
                if rel:
                    self._due_rel = _due(t, half, c.tensors_every)
            self._keep(g, t, half)
        self._t = t

    def _stats(self, g: torch.Tensor, t: float, elapsed: float) -> dict[str, Any]:
        """The cosines and the shares of ``g`` against the copies before it."""
        c = self._config
        out: dict[str, Any] = {"g": float(torch.linalg.vector_norm(g))}
        has_prev = not math.isnan(self._t_prev)
        out["cos_prev"] = _cos(g, self._prev) if has_prev else None
        out["age_prev"] = elapsed if has_prev else None
        for i, span in enumerate(c.anchors):
            taken = not math.isnan(self._t_anchor[i])
            out[f"cos_{span:g}"] = _cos(g, self._anchor[i]) if taken else None
            out[f"age_{span:g}"] = t - self._t_anchor[i] if taken else None
        for i, s in enumerate(c.spans):
            k = min(self._count[i], c.keep)
            share = residual_share(self._kept[i, :k].to(g.dtype), g) if k else None
            out[f"rho_{s:g}"] = share
        return out

    def _moves(self, before: list[torch.Tensor], rel: bool) -> dict[str, Any]:
        r"""$\lVert\theta\rVert$, $\lVert\Delta\theta\rVert$ and, with
        ``rel``, each tensor's $\lVert\Delta\theta_i\rVert / \lVert\theta_i\rVert$."""
        sq_theta, sq_step, ratios = 0.0, 0.0, {}
        for name, p, b in zip(self._labels, self._params, before, strict=True):
            b = b.to(self._dtype)
            n_theta = float(torch.linalg.vector_norm(b))
            n_step = float(torch.linalg.vector_norm(p.to(self._dtype) - b))
            sq_theta, sq_step = sq_theta + n_theta**2, sq_step + n_step**2
            ratios[name] = n_step / n_theta if n_theta > 0 else None
        out: dict[str, Any] = {
            "theta": math.sqrt(sq_theta),
            "dtheta": math.sqrt(sq_step),
        }
        if rel:
            out["rel"] = ratios
        return out

    def _keep(self, g: torch.Tensor, t: float, half: float) -> None:
        """Take ``g`` as the previous gradient, and as an anchor and a kept
        one where due."""
        c = self._config
        self._prev.copy_(g)
        self._t_prev = t
        for i, span in enumerate(c.anchors):
            if t + half > self._due_anchor[i]:
                self._anchor[i].copy_(g)
                self._t_anchor[i] = t
                self._due_anchor[i] = _due(t, half, span)
        for i, s in enumerate(c.spans):
            if t + half > self._due_kept[i]:
                self._kept[i, self._count[i] % c.keep].copy_(g)
                self._count[i] += 1
                self._due_kept[i] = _due(t, half, s)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "inner": self._inner.state_dict(),
            "prev": self._prev,
            "anchor": self._anchor,
            "kept": self._kept,
            "times": torch.tensor(
                [
                    self._t,
                    self._t_prev,
                    self._due_row,
                    self._due_rel,
                    *self._t_anchor,
                    *self._due_anchor,
                    *self._due_kept,
                ],
                dtype=torch.float64,
            ),
            "count": torch.tensor(self._count, dtype=torch.int64),
        }

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        self._inner.load_state_dict(state["inner"])
        with torch.no_grad():
            self._prev.copy_(state["prev"])
            self._anchor.copy_(state["anchor"])
            self._kept.copy_(state["kept"])
        times = [float(x) for x in state["times"]]
        a = len(self._t_anchor)
        self._t, self._t_prev, self._due_row, self._due_rel = times[:4]
        self._t_anchor = times[4 : 4 + a]
        self._due_anchor = times[4 + a : 4 + 2 * a]
        self._due_kept = times[4 + 2 * a :]
        self._count = [int(x) for x in state["count"]]


def _finite(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    """``row`` with every float that is not finite as ``None`` (strict JSON)."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            value = _clean(value)
        elif isinstance(value, float):
            value = _finite(value)
        out[key] = value
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    b = b.to(a.dtype)
    den = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return float(a @ b) / den if den > 0 else None
