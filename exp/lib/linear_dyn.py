r"""Linear dynamics on codes, fit in closed form: $\hat z_{k+1} = A z_k + b$.

``fit`` solves the ridge regression
$\min_{A, b} \sum_i \lVert z^1_i - A z^0_i - b \rVert^2 + \alpha \lVert A
\rVert_F^2$ ($b$ not penalized): with the centered samples
$\tilde z^0 = z^0 - \bar z^0$, $\tilde z^1 = z^1 - \bar z^1$,
$C_{00} = \sum \tilde z^0 \tilde z^{0\top}$, $C_{10} = \sum \tilde z^1
\tilde z^{0\top}$, it is $A = C_{10} (C_{00} + \alpha I)^{-1}$ and
$b = \bar z^1 - A \bar z^0$, in float64.

``errors`` rolls $A$ out ``h`` steps from the first code of each window
(``rollout``) and gives the error relative to the variance of the target,
$\mathrm{dyn}_h = \overline{\lVert \hat z_h - z_h \rVert^2} /
\overline{\lVert z_h - \bar z_h \rVert^2}$ ($1 - R^2$, unchanged by a scale
of $z$), the same of the copy $\hat z_h = z_0$ (``copy_h``) and their ratio.
Given the training codes ``train``, it also gives the same errors in the
coordinates whitened by their covariance $C$ (``dynw_h``, ``copyw_h``):
$\overline{\lVert W(\hat z_h - z_h) \rVert^2} / \overline{\lVert W(z_h -
\bar z_h) \rVert^2}$, $W^\top W = (C + \epsilon I)^{-1}$, $\epsilon =
10^{-6} \mathrm{tr}(C) / D$. ``dyn_h`` weighs each direction by its variance,
so a code whose variance sits in slowly changing directions (the top
components of a PCA) scores well on it; ``dynw_h`` weighs every direction of
the training codes the same and is unchanged by any invertible linear map of
$z$ (up to $\epsilon$).
"""

from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["ALPHA_SCALES", "errors", "fit", "rollout", "select_alpha"]

# The ridge alpha = c tr(C00) / D, for c in 1e-8 .. 1e0.
ALPHA_SCALES = tuple(10.0**e for e in range(-8, 1))


def fit(z0: Tensor, z1: Tensor, *, alpha: float) -> tuple[Tensor, Tensor]:
    """$A$ ``(D, D)`` and $b$ ``(D,)`` (float64) of the pairs ``z0`` ``(n, D)``
    -> ``z1`` ``(n, D)`` (module docstring)."""
    z0, z1 = z0.double(), z1.double()
    m0, m1 = z0.mean(0), z1.mean(0)
    x, y = z0 - m0, z1 - m1
    c00 = x.T @ x
    c10 = y.T @ x
    eye = torch.eye(len(c00), dtype=c00.dtype, device=c00.device)
    # A (C00 + a I) = C10, and C00 + a I is symmetric: A^T = solve(C00 + a I, C10^T).
    a = torch.linalg.solve(c00 + alpha * eye, c10.T).T
    return a, m1 - a @ m0


def rollout(a: Tensor, b: Tensor, z0: Tensor, steps: int) -> Tensor:
    """$\\hat z_1 .. \\hat z_{steps}$ ``(n, steps, D)`` (float64) from ``z0``
    ``(n, D)``: $\\hat z_h = A \\hat z_{h-1} + b$, $\\hat z_0 = z_0$."""
    z = z0.double()
    out = []
    for _ in range(steps):
        z = z @ a.T + b
        out.append(z)
    return torch.stack(out, 1) if out else z.new_zeros(len(z), 0, z.shape[1])


def errors(
    a: Tensor,
    b: Tensor,
    zs: Tensor,
    horizons: Sequence[int],
    *,
    train: Tensor | None = None,
) -> dict[str, float]:
    """``dyn_h``, ``copy_h`` and ``dyn_h/copy_h`` for each ``h`` of
    ``horizons`` on the windows ``zs`` ``(n, 1 + H, D)`` (codes $z_0 .. z_H$
    of each window; module docstring), and with ``train`` ``(m, D)`` also
    ``dynw_h`` and ``copyw_h``.

    Raises:
        ValueError: a horizon is not in ``[1, H]``.
    """
    steps = zs.shape[1] - 1
    if not horizons or min(horizons) < 1 or max(horizons) > steps:
        raise ValueError(f"horizons must be in [1, {steps}], not {list(horizons)}")
    zs = zs.double()
    pred = rollout(a, b, zs[:, 0], max(horizons))
    w = None if train is None else _whitener(train)
    out: dict[str, float] = {}
    for h in horizons:
        target = zs[:, h]
        var = float(((target - target.mean(0)) ** 2).sum(1).mean())
        dyn = float(((pred[:, h - 1] - target) ** 2).sum(1).mean()) / var
        copy = float(((zs[:, 0] - target) ** 2).sum(1).mean()) / var
        out |= {f"dyn_{h}": dyn, f"copy_{h}": copy, f"dyn_{h}/copy_{h}": dyn / copy}
        if w is not None:
            var_w = _sq(w, target - target.mean(0))
            out[f"dynw_{h}"] = _sq(w, pred[:, h - 1] - target) / var_w
            out[f"copyw_{h}"] = _sq(w, zs[:, 0] - target) / var_w
    return out


def _whitener(train: Tensor) -> Tensor:
    """$L^{-1}$ ``(D, D)`` of the Cholesky factor $L L^\top = C + \\epsilon I$
    of the covariance of ``train`` (module docstring)."""
    x = train.double() - train.double().mean(0)
    c = x.T @ x / len(x)
    eps = 1e-6 * float(c.trace()) / len(c) + 1e-300
    eye = torch.eye(len(c), dtype=c.dtype, device=c.device)
    low = torch.linalg.cholesky(c + eps * eye)
    return torch.linalg.solve_triangular(low, eye, upper=False)


def _sq(w: Tensor, e: Tensor) -> float:
    """The mean of $\\lVert W e \rVert^2$ over the rows of ``e`` ``(n, D)``."""
    return float(((e.double() @ w.T) ** 2).sum(1).mean())


def select_alpha(
    z0: Tensor,
    z1: Tensor,
    calib: Tensor,
    *,
    scales: Sequence[float] = ALPHA_SCALES,
) -> float:
    """The $\\alpha = c\\,\\mathrm{tr}(C_{00}) / D$, $c$ of ``scales``, whose
    ``fit`` on ``z0`` -> ``z1`` has the least ``dyn_1`` on the windows
    ``calib`` ``(n, 1 + H, D)`` (the first of the least on a tie)."""
    x = z0.double() - z0.double().mean(0)
    unit = float((x**2).sum()) / z0.shape[1]  # tr(C00) / D
    best, best_err = 0.0, float("inf")
    for c in scales:
        alpha = c * unit
        a, b = fit(z0, z1, alpha=alpha)
        err = errors(a, b, calib[:, :2], [1])["dyn_1"]
        if err < best_err:
            best, best_err = alpha, err
    return best
