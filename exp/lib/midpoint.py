r"""Time linearity of codes: the midpoint's error of triples in whitened
coordinates (``exp/vision/plans/jepa.md``, 「V18c: 時間の線形性と情報の幅」).

**Whitening** (``Whitener``): of the reference codes $Z_{\mathrm{ref}}$
``(m, D)``, the mean $\mu$ and the covariance $C = U \Lambda U^\top$
(float64, over $m$; eigenvalues descending). The PCA coordinates $y =
U_k^\top (z - \mu)$ and the whitened $w_i = y_i / \sqrt{\lambda_i +
\epsilon}$, $\epsilon = c_\epsilon \mathrm{tr}(C) / D$. A code is not
scoreable at $k$ when $D < k$ or $\lambda_k \le 10^{-12} \lambda_1$ (a
constant included): its column is missing, $k$ is never lowered for it.

**Triples** of a lag $n$ (steps): centres $u$ give $a = w_u - w_{u-n}$, $b =
w_{u+n} - w_u$, $m = \tfrac12 (w_{u-n} + w_{u+n})$, and per triple the
numerator $\nu = \lVert w_u - m \rVert^2 = \tfrac14 \lVert a - b \rVert^2$
and the denominator $\delta = \tfrac12 (\lVert a \rVert^2 + \lVert b
\rVert^2)$ (``triple_terms``). Pooled over a set of triples,

$$G_{\mathrm{mid}} = \frac{\sum \nu}{\sum \delta}, \qquad s = \frac{\bar
D}{2V}, \quad \bar D = \frac1n \sum \delta,$$

$V$ the mean squared distance of the evaluation's codes from their mean
(``spread``). The extrapolation's error is $4 G_{\mathrm{mid}}$ triple by
triple ($w_{u+n} - (2 w_u - w_{u-n}) = -2 (w_u - m)$).

**Closed forms** (stationary, per unit-variance dimension of
autocorrelation $\rho$; ``stationary_terms``): $E\nu = \tfrac32 +
\tfrac12 \rho(2n) - 2\rho(n)$, $E\delta = 2(1 - \rho(n))$, so with $s(n) =
1 - \rho(n)$, $G_{\mathrm{mid}} = 1 - s(2n) / (4 s(n))$; OU ($\rho(n) =
r^n$): $G = (2 + s)/4$; white noise $3/4$; a random walk of unit increments
$E\nu = n/2$, $E\delta = n$ (``random_walk_terms``); a uniform permutation
of a fixed series of variance $V$ (over its $N$ values) $E\nu = \tfrac32 V
\frac{N}{N - 1}$, $E\delta = 2V \frac{N}{N - 1}$ (``permutation_terms``).
For a spectral measure $\mu$ of the whitened code, $\rho(n) =
E_\mu[\cos \omega n]$ and (Fable's identity) $G_{\mathrm{ext}} = 4
G_{\mathrm{mid}} = 2 E_\mu[(1 - c)^2] / E_\mu[1 - c] = 2s + 2
\mathrm{Var}_\mu[c] / s$, $c = \cos \omega n$: at least $2s$ (the line
$G_{\mathrm{mid}} = s/2$).

**Injection** (V-d2): isotropic white noise of total variance $\nu V$ added
to every code adds $\tfrac32 \nu V$ to $E\nu$ and $2\nu V$ to $E\delta$,
so $G_\nu = (sG + \tfrac34 \nu)/(s + \nu)$, $s_\nu = (s + \nu)/(1 + \nu)$,
and a difference $\delta$ is made by $\nu = \delta s / (\tfrac34 - G -
\delta)$ (``injected``, ``injection_for``).

**Per triple** (``plans/jepa.md``, 「線形の幅（2026-11-26）」): $\nu_t =
\tfrac12 \delta_t - \tfrac12 \langle a, b \rangle$, so $\nu_t / \delta_t =
(1 - \rho_t)/2$ with $\rho_t = \langle a, b \rangle / \delta_t$ (``rho``).
A triple **wins** at the bound $q$ where $q\,\delta_t - \nu_t > 0$, i.e.
$\rho_t > 1 - 2q$, judged without a division: a triple of $\delta_t = 0$
(no motion) loses (``wins``). Literally, the extrapolation $2 w_u - w_{u-n}$
beats the copy $w_u$ of the future where $\lVert b - a \rVert^2 = 4\nu_t
< \lVert b \rVert^2$ (``asym_wins``; not symmetric in time). ``win_sums``
holds the blocks' sums of these; ``pair_sq`` the squared displacements
$\lVert w_{u+n} - w_u \rVert^2$ of which every pooled measure is a
function.

**Directions** (``mid_covariances``, ``slow_directions``): $C_\nu = \sum_t
(w_t - m_t)(w_t - m_t)^\top$ and $C_\delta = \sum_t \tfrac12 (a a^\top +
b b^\top)$ over a set of triples; the generalized eigenproblem $C_\nu v =
\lambda C_\delta v$ (ascending $\lambda$) gives the directions whose
pooled $G$ on that set is $\lambda$ (the Rayleigh quotient $v^\top C_\nu v
/ v^\top C_\delta v$).
"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

__all__ = [
    "DEGENERATE",
    "TripleTerms",
    "Whitener",
    "asym_wins",
    "block_sums",
    "g_mid_of",
    "injected",
    "injection_for",
    "mid_covariances",
    "pair_sq",
    "permutation_terms",
    "random_walk_terms",
    "rho",
    "slow_directions",
    "spectral_rho",
    "spread",
    "stationary_terms",
    "triple_terms",
    "win_sums",
    "wins",
]

DEGENERATE = 1e-12  # lambda_k <= DEGENERATE * lambda_1: not scoreable
_F64 = torch.float64


@dataclass(frozen=True)
class Whitener:
    r"""The whitening of a reference (module docstring): ``mean`` ``(D,)``,
    ``basis`` $U_k$ ``(D, k)``, ``values`` the top $k$ eigenvalues ``(k,)``,
    ``eps`` $\epsilon$, ``total`` $\mathrm{tr}(C)$, and ``why`` the reason
    it is not scoreable (None if it is). Build it with ``fit``."""

    mean: Tensor
    basis: Tensor
    values: Tensor
    eps: float
    total: float
    k: int
    why: str | None

    @classmethod
    def fit(cls, ref: Tensor, k: int, *, c_eps: float = 1e-6) -> Self:
        """The whitening of ``k`` components of the reference codes ``ref``
        ``(m, D)``, $\\epsilon$ = ``c_eps`` $\\mathrm{tr}(C)/D$. Of $D < k$
        it holds no basis (``why`` says so)."""
        z = ref.to(_F64)
        d = z.shape[1]
        mean = z.mean(0)
        if d < k:
            empty = z.new_zeros(d, 0)
            return cls(mean, empty, z.new_zeros(0), 0.0, 0.0, k, f"D = {d} < k = {k}")
        x = z - mean
        c = x.T @ x / len(z)
        values, vectors = torch.linalg.eigh(c)
        values, vectors = values.flip(0), vectors.flip(1)
        total = float(values.sum())
        top = values[:k]
        why = None
        if not float(top[-1]) > DEGENERATE * float(top[0]):
            why = f"lambda_k = {float(top[-1]):.3g} <= 1e-12 lambda_1 = {float(top[0]):.3g}"
        return cls(mean, vectors[:, :k], top, c_eps * total / d, total, k, why)

    @property
    def scoreable(self) -> bool:
        return self.why is None

    def pca(self, z: Tensor) -> Tensor:
        """$y = U_k^\\top (z - \\mu)$ ``(n, k)`` float64 of ``z`` ``(n, D)``."""
        return (z.to(_F64) - self.mean) @ self.basis

    def white(self, z: Tensor) -> Tensor:
        """$w = y / \\sqrt{\\lambda + \\epsilon}$ ``(n, k)`` float64."""
        return self.pca(z) / (self.values + self.eps).sqrt()

    def spec(self) -> dict[str, object]:
        """The record of the spectrum: the top $k$ eigenvalues, $\\lambda_k /
        \\lambda_1$, $\\lambda_k / (\\lambda_k + \\epsilon)$, the share of
        $\\mathrm{tr}(C)$ in the top $k$, $\\epsilon$ and whether it is
        scoreable."""
        v = self.values
        has = len(v) > 0 and float(v[0]) > 0
        return {
            "values": v.tolist(),
            "ratio_k": float(v[-1] / v[0]) if has else None,
            "kept_k": float(v[-1] / (v[-1] + self.eps)) if has else None,
            "share": float(v.sum()) / self.total if self.total > 0 else None,
            "eps_value": self.eps,
            "scoreable": self.scoreable,
            "why": self.why,
        }


@dataclass(frozen=True)
class TripleTerms:
    r"""Per triple ``(T,)`` float64: ``num`` $\nu$, ``den`` $\delta$,
    ``a2`` $\lVert a \rVert^2$, ``b2`` $\lVert b \rVert^2$."""

    num: Tensor
    den: Tensor
    a2: Tensor
    b2: Tensor


def triple_terms(w: Tensor, centers: Tensor, lag: int) -> TripleTerms:
    """The terms of the triples $(u - n, u, u + n)$ of the ``centers``
    ``(T,)`` (each at least ``lag`` from both ends) of the codes ``w`` ``(N,
    k)`` (module docstring)."""
    x = w.to(_F64)
    w0, w1, w2 = x[centers - lag], x[centers], x[centers + lag]
    a, b = w1 - w0, w2 - w1
    a2, b2 = a.square().sum(1), b.square().sum(1)
    return TripleTerms((a - b).square().sum(1) / 4, (a2 + b2) / 2, a2, b2)


def spread(w: Tensor) -> float:
    """$V$: the mean of $\\lVert w - \\bar w \\rVert^2$ over the rows of
    ``w`` ``(N, k)``."""
    x = w.to(_F64)
    return float((x - x.mean(0)).square().sum(1).mean())


def g_mid_of(s1: Tensor, s2: Tensor) -> Tensor:
    r"""The stationary $G_{\mathrm{mid}} = 1 - s(2n) / (4 s(n))$ of $s(n)$
    ``s1`` and $s(2n)$ ``s2``."""
    return 1 - s2 / (4 * s1)


def stationary_terms(rho1: Tensor, rho2: Tensor) -> tuple[Tensor, Tensor]:
    r"""$E\nu = \tfrac32 + \tfrac12 \rho(2n) - 2 \rho(n)$ and $E\delta =
    2(1 - \rho(n))$ per unit-variance dimension of the autocorrelations
    ``rho1`` $\rho(n)$, ``rho2`` $\rho(2n)$."""
    return 1.5 + 0.5 * rho2 - 2 * rho1, 2 * (1 - rho1)


def random_walk_terms(lag: int, d: int) -> tuple[float, float]:
    """$E\\nu = d n / 2$ and $E\\delta = d n$ of a random walk of unit
    increments in ``d`` dimensions at the lag $n$."""
    return d * lag / 2, float(d * lag)


def permutation_terms(var: float, n: int) -> tuple[float, float]:
    r"""$E\nu = \tfrac32 V \frac{N}{N-1}$ and $E\delta = 2V \frac{N}{N-1}$
    of three distinct places of a uniform permutation of $N$ = ``n`` fixed
    values of variance ``var`` $V$ (over $N$; the cross term of two distinct
    values is $-V/(N - 1)$)."""
    f = n / (n - 1)
    return 1.5 * var * f, 2 * var * f


def spectral_rho(omegas: Tensor, weights: Tensor, lag: int) -> Tensor:
    r"""$\rho(n) = E_\mu[\cos \omega n]$ of the spectral measure of the
    angular ``omegas`` ``(J,)`` and their ``weights`` ``(J,)`` (summing to
    1)."""
    return (weights.to(_F64) * torch.cos(omegas.to(_F64) * lag)).sum()


def injected(g: float, s: float, nu: float) -> tuple[float, float]:
    r"""$G_\nu = (sG + \tfrac34 \nu)/(s + \nu)$ and $s_\nu = (s + \nu)/(1 +
    \nu)$: the expected values after the noise $\nu V$ (module
    docstring)."""
    return (s * g + 0.75 * nu) / (s + nu), (s + nu) / (1 + nu)


def injection_for(g: float, s: float, delta: float) -> float | None:
    r"""$\nu = \delta s / (\tfrac34 - G - \delta)$, the noise that raises
    $G$ by ``delta``; None where $\delta \ge \tfrac34 - G$ (no noise
    reaches it)."""
    room = 0.75 - g - delta
    return None if room <= 0 else delta * s / room


def rho(t: TripleTerms) -> Tensor:
    r"""$\rho_t = \langle a, b \rangle / \delta_t = 1 - 2 \nu_t / \delta_t$
    ``(T,)`` float64 (NaN where $\delta_t = 0$)."""
    return 1 - 2 * t.num / t.den


def wins(t: TripleTerms, q: float) -> Tensor:
    r"""Whether each triple wins at the bound ``q``: $q\,\delta_t - \nu_t >
    0$ ``(T,)`` bool (module docstring; $\delta_t = 0$ loses, a tie
    loses)."""
    return q * t.den - t.num > 0


def asym_wins(t: TripleTerms) -> Tensor:
    r"""Whether the extrapolation beats the copy of the future literally:
    $4 \nu_t < \lVert b \rVert^2$ ``(T,)`` bool."""
    return 4 * t.num < t.b2


def block_sums(values: Tensor, blocks: Tensor, n_blocks: int) -> Tensor:
    """The sums ``(n_blocks, ...)`` float64 of ``values`` ``(N, ...)`` by
    their ``blocks`` ``(N,)``."""
    v = values.double()
    out = torch.zeros(n_blocks, *v.shape[1:], dtype=torch.float64)
    return out.index_add_(0, blocks, v)


def win_sums(
    t: TripleTerms, blocks: Tensor, n_blocks: int, qs: tuple[float, ...]
) -> dict[str, Tensor]:
    r"""The blocks' sums ``(n_blocks,)`` of the triples ``t`` of the
    ``blocks`` ``(T,)``: ``num_b`` $\sum \nu$, ``den_b`` $\sum \delta$,
    ``n_b`` the count, ``still_b`` the triples of $\delta = 0$,
    ``win<100q>_b`` the winners at each bound of ``qs`` (``wins``) and
    ``winasym_b`` (``asym_wins``)."""
    out = {
        "num_b": block_sums(t.num, blocks, n_blocks),
        "den_b": block_sums(t.den, blocks, n_blocks),
        "n_b": block_sums(torch.ones(len(t.num)), blocks, n_blocks),
        "still_b": block_sums((t.den == 0).double(), blocks, n_blocks),
    }
    for q in qs:
        out[f"win{round(100 * q)}_b"] = block_sums(
            wins(t, q).double(), blocks, n_blocks
        )
    out["winasym_b"] = block_sums(asym_wins(t).double(), blocks, n_blocks)
    return out


def pair_sq(w: Tensor, starts: Tensor, lag: int) -> Tensor:
    r"""$\lVert w_{u+n} - w_u \rVert^2$ ``(P,)`` float64 of the ``starts``
    $u$ ``(P,)`` of ``w`` ``(N, k)``."""
    x = w.to(_F64)
    return (x[starts + lag] - x[starts]).square().sum(1)


def mid_covariances(w: Tensor, centers: Tensor, lag: int) -> tuple[Tensor, Tensor]:
    r"""$C_\nu$ and $C_\delta$ ``(k, k)`` float64 of the triples of the
    ``centers`` (module docstring): their traces are $\sum \nu$ and $\sum
    \delta$."""
    x = w.to(_F64)
    w0, w1, w2 = x[centers - lag], x[centers], x[centers + lag]
    e = w1 - (w0 + w2) / 2
    a, b = w1 - w0, w2 - w1
    return e.T @ e, (a.T @ a + b.T @ b) / 2


def slow_directions(c_nu: Tensor, c_delta: Tensor) -> tuple[Tensor, Tensor]:
    r"""The generalized eigenvalues $\lambda$ ``(k,)`` (ascending) and
    vectors $V$ ``(k, k)`` (columns, $V^\top C_\delta V = I$) of $C_\nu v =
    \lambda C_\delta v$, by the Cholesky factor $C_\delta = L L^\top$:
    the eigenpairs of $L^{-1} C_\nu L^{-\top}$, $v = L^{-\top} y$.

    Raises:
        torch.linalg.LinAlgError: $C_\delta$ is not positive definite (a
            direction that never moves).
    """
    low = torch.linalg.cholesky(c_delta.to(_F64))
    m = torch.linalg.solve_triangular(low, c_nu.to(_F64), upper=False)
    m = torch.linalg.solve_triangular(low, m.T, upper=False)  # L^-1 C L^-T
    lam, y = torch.linalg.eigh((m + m.T) / 2)
    v = torch.linalg.solve_triangular(low.T, y, upper=True)
    return lam, v
