r"""A centred ridge readout from codes to targets (V18c's E1, E1e, E2;
``exp/vision/plans/jepa.md``, 「未観測の部分: 読み出し」).

With $X$ ``(n, D)`` and $Y$ ``(n, K)`` the fit set less their means
$\bar x$, $\bar y$: $B = (X^\top X + \alpha I)^{-1} X^\top Y$ in float64,
$\alpha = c \cdot \mathrm{tr}(X^\top X) / D$, and $\hat y = (x - \bar x) B +
\bar y$. ``r2`` is the mean over the $K$ coordinates of $1 - \sum_i (\hat
y_{ik} - y_{ik})^2 / \sum_i (y_{ik} - \langle y_k \rangle)^2$ (the
evaluated set's mean). ``select_ridge`` takes the $c$ of the best ``r2`` on
a calibration set (the first on a tie), or of another ``score``.

V18c's main measure (``plans/jepa.md``, 「V18c: 主の指標の作り直し」):
``centered_r2`` pools the error over the coordinates after centring both
the prediction and the target on the evaluated set $E$, $\tilde p_i =
\hat y_i - \langle \hat y \rangle_E$, $\tilde y_i = y_i - \langle y
\rangle_E$:

$$m_E = 1 - \frac{\sum_{i \in E} \lVert \tilde p_i - \tilde y_i
\rVert^2}{\sum_{i \in E} \lVert \tilde y_i \rVert^2},$$

exactly 0 for a constant prediction, unchanged by a shift common to the
set. ``rank_bound`` is its ceiling over predictions whose centred rows are
of rank at most $r$: with $\mu_1 \ge \mu_2 \ge \dots$ the eigenvalues
of the covariance of $\tilde Y$, Eckart-Young gives $\lVert \tilde Y -
\tilde P \rVert_F^2 \ge \lvert E \rvert \sum_{j > r} \mu_j$, so $m_E
\le b_r(E) = \sum_{j \le r} \mu_j / \sum_j \mu_j$, exactly on each set.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from exp.lib.linear_dyn import ALPHA_SCALES

__all__ = [
    "Ridge",
    "centered_r2",
    "centered_terms",
    "fit_ridge",
    "predict",
    "r2",
    "rank_bound",
    "select_ridge",
]


@dataclass(frozen=True)
class Ridge:
    """A fit readout: the means ``x_mean`` ``(D,)``, ``y_mean`` ``(K,)``,
    the matrix ``b`` ``(D, K)`` (float64) and its ``alpha``."""

    x_mean: Tensor
    y_mean: Tensor
    b: Tensor
    alpha: float


def fit_ridge(x: Tensor, y: Tensor, scale: float) -> Ridge:
    r"""The readout of $\alpha$ = ``scale`` $\cdot \mathrm{tr}(X^\top X) /
    D$ from ``x`` ``(n, D)`` to ``y`` ``(n, K)`` (module docstring)."""
    xd, yd = x.double(), y.double()
    mx, my = xd.mean(0), yd.mean(0)
    xc, yc = xd - mx, yd - my
    gram = xc.T @ xc
    alpha = scale * float(gram.trace()) / x.shape[1]
    eye = torch.eye(x.shape[1], dtype=torch.float64, device=x.device)
    return Ridge(mx, my, torch.linalg.solve(gram + alpha * eye, xc.T @ yc), alpha)


def predict(fit: Ridge, x: Tensor) -> Tensor:
    """$\\hat y$ ``(N, K)`` float64 of the codes ``x`` ``(N, D)``."""
    return (x.double() - fit.x_mean) @ fit.b + fit.y_mean


def r2(pred: Tensor, y: Tensor) -> float:
    """The mean over the coordinates of $R^2$ of ``pred`` against ``y``
    ``(N, K)`` (module docstring)."""
    yd = y.double()
    err = (pred.double() - yd).square().sum(0)
    total = (yd - yd.mean(0)).square().sum(0)
    return float((1 - err / total).mean())


def centered_terms(pred: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    r"""The items' terms of ``centered_r2`` of ``pred`` against ``y`` ``(N,
    K)``: $e_i = \lVert \tilde p_i - \tilde y_i \rVert^2$ and $v_i =
    \lVert \tilde y_i \rVert^2$, both ``(N,)`` float64 (module docstring),
    so that $m = 1 - \sum_i e_i / \sum_i v_i$."""
    pd, yd = pred.double(), y.double()
    pc, yc = pd - pd.mean(0), yd - yd.mean(0)
    return (pc - yc).square().sum(1), yc.square().sum(1)


def centered_r2(pred: Tensor, y: Tensor) -> float:
    """$m$ of ``pred`` against ``y`` ``(N, K)`` (module docstring)."""
    e, v = centered_terms(pred, y)
    return float(1 - e.sum() / v.sum())


def rank_bound(y: Tensor, r: int) -> float:
    r"""$b_r$ of the targets ``y`` ``(N, K)`` (module docstring): the share
    of the $r$ largest eigenvalues of their covariance."""
    yd = y.double()
    s2 = torch.linalg.svdvals(yd - yd.mean(0)).square()
    return float(s2[:r].sum() / s2.sum())


def select_ridge(
    x: Tensor,
    y: Tensor,
    x_cal: Tensor,
    y_cal: Tensor,
    scales: Sequence[float] = ALPHA_SCALES,
    score: Callable[[Tensor, Tensor], float] = r2,
) -> Ridge:
    """The readout fit on ``x``, ``y`` of the scale of ``scales`` whose
    ``score`` (``r2``, or ``centered_r2``) on ``x_cal``, ``y_cal`` is the
    best (the first on a tie)."""
    fits = [fit_ridge(x, y, c) for c in scales]
    scores = [score(predict(f, x_cal), y_cal) for f in fits]
    return fits[max(range(len(fits)), key=lambda i: scores[i])]
