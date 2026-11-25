r"""The TwoNN estimate of the intrinsic dimension (Facco et al. 2017), and
the thinned point sets it is taken on (``exp/vision/plans/jepa.md``,
「情報の幅: TwoNN」).

Of points $x_i$ ``(N, D)``: duplicates first, the points within ``floor``
$\cdot R$ of an earlier kept point (0 included) dropped (``n_dup``; of them
``n_exact`` at the float64 distance 0, the exact duplicates), $R$ the
scale (by default $\sqrt{2\,\mathrm{tr}\,C}$ of the points, the RMS distance
of two random points). Of the $N'$ kept, each point's nearest and second
distances $r_1 \le r_2$ (pairs of ``exclude`` not allowed as neighbours),
$\mu_i = r_2 / r_1$. Sorted ascending, $F_i = i / N'$, and the fit through
the origin of $y_i = -\log(1 - F_i)$ on $x_i = \log \mu_i$ over $i \le
\lfloor (1 - \text{discard}) N' \rfloor$ (and below $N'$):

$$\hat d = \frac{\sum x_i y_i}{\sum x_i^2}, \qquad \hat d_{\mathrm{ML}} =
\frac{N' - 1}{\sum_i \log \mu_i} \ \ \text{(all } N'\text{)}.$$

Also ``tie_share`` (the share of $\mu_i < 1 + 10^{-6}$) and ``rho`` =
$\mathrm{median}(r_2) / R$. Of fewer than 3 kept points (a constant code)
the estimate is not defined (``d`` None).
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import torch
from torch import Tensor

__all__ = ["TIE", "TwoNN", "thinned", "twonn"]

TIE = 1e-6  # mu < 1 + TIE counts as r1 ~ r2


@dataclass(frozen=True)
class TwoNN:
    """The estimate (module docstring): ``d``, ``d_ml`` (None if not
    defined), the points ``n`` given and ``n_kept``, ``n_dup`` (of them
    ``n_exact`` at the distance 0), ``n_fit``
    (the points of the fit), ``tie_share``, ``rho``, and ``kept`` ``(N',)``
    the given indices of the kept points with ``nn`` ``(N', 2)`` the given
    indices of their two neighbours."""

    d: float | None
    d_ml: float | None
    n: int
    n_kept: int
    n_dup: int
    n_exact: int
    n_fit: int
    tie_share: float | None
    rho: float | None
    kept: Tensor
    nn: Tensor


def _fit_count(n: int, discard: float) -> int:
    """$\\lfloor (1 - \\text{discard}) n \\rfloor$ in exact arithmetic,
    below $n$ (whose $F = 1$)."""
    keep = Fraction(1) - Fraction(discard).limit_denominator(10**6)
    return min(math.floor(keep * n), n - 1)


def twonn(
    x: Tensor,
    *,
    radius: float | None = None,
    discard: float = 0.1,
    floor: float = 1e-6,
    exclude: Tensor | None = None,
) -> TwoNN:
    """The TwoNN of the points ``x`` ``(N, D)`` (module docstring): the
    scale ``radius`` $R$ (default of ``x``), ``exclude`` ``(N, N)`` bool the
    pairs that may not be neighbours."""
    p = x.to(torch.float64)
    n = len(p)
    if radius is None:
        radius = math.sqrt(2 * float((p - p.mean(0)).square().sum(1).mean()))
    dist = torch.cdist(p, p, compute_mode="donot_use_mm_for_euclid_dist")
    near = dist <= floor * radius
    near.fill_diagonal_(False)
    kept: list[int] = []
    exact = 0
    if not bool(near.any()):
        kept = list(range(n))
    else:
        for i in range(n):  # greedy: drop a point near an earlier kept one
            if not kept or not bool(near[i, kept].any()):
                kept.append(i)
            else:
                exact += bool((dist[i, kept] == 0).any())
    idx = torch.tensor(kept, dtype=torch.long)
    none = TwoNN(None, None, n, len(kept), n - len(kept), exact, 0, None, None,
                 idx, torch.empty(len(kept), 0, dtype=torch.long))  # fmt: skip
    if len(kept) < 3:
        return none
    sub = dist[idx][:, idx].clone()
    sub.fill_diagonal_(math.inf)
    if exclude is not None:
        sub[exclude[idx][:, idx]] = math.inf
    r, nn = sub.topk(2, dim=1, largest=False)
    if not bool(torch.isfinite(r).all()):
        return none
    mu = r[:, 1] / r[:, 0]
    m = len(mu)
    logs = mu.log().sort().values
    n_fit = _fit_count(m, discard)
    f = torch.arange(1, n_fit + 1, dtype=torch.float64) / m
    xs, ys = logs[:n_fit], -torch.log1p(-f)
    sxx, total = float(xs.square().sum()), float(logs.sum())
    if not (sxx > 0 and total > 0):  # every mu = 1: no slope
        return none
    d = float((xs * ys).sum()) / sxx
    d_ml = (m - 1) / total
    return TwoNN(
        d=d,
        d_ml=d_ml,
        n=n,
        n_kept=m,
        n_dup=n - m,
        n_exact=exact,
        n_fit=n_fit,
        tie_share=float((mu < 1 + TIE).double().mean()),
        rho=float(r[:, 1].median()) / radius if radius > 0 else None,
        kept=idx,
        nn=idx[nn],
    )


def thinned(n: int, spacing: int) -> list[Tensor]:
    """The ``spacing`` disjoint sets ``arange(o, n, spacing)`` of the
    offsets $o = 0 .. \\text{spacing} - 1$ of a grid of ``n`` points."""
    return [torch.arange(o, n, spacing) for o in range(spacing)]
