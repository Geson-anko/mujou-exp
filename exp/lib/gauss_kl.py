r"""Gaussian moment matching (M17, 2026-11-13): a regularizer that keeps the
first two moments of a representation those of $N(0, I)$, learned from one
sample per step of a temporally correlated stream with a buffer of past
samples.

**Population objective.** For the representation $z = f_\theta(x) \in
\mathbb{R}^D$ of the data's law, $\mu = E z$ and $\Sigma = \mathrm{Cov}\, z$,

$$
F(\theta) = \mathrm{KL}(N(\mu, \Sigma) \,\|\, N(0, I))
    = \tfrac12 \left[ \mathrm{tr}\, \Sigma + \lVert \mu \rVert^2 - D
    - \log \det \Sigma \right].
$$

$F$ is a lower bound of $\mathrm{KL}(q \,\|\, N(0, I))$ for the law $q$ of
$z$ (the gap is the negentropy of $q$): only the first two moments are
matched. A $z$ that is the same for every input (C), a function of the
position of the tokens only (P: every data point the same) or tokens
constant within a stack (Cw: rank $\Sigma \le d < D$) has $\det \Sigma = 0$
and $F = +\infty$, so at the optimum of $(1 - \lambda) L + \lambda F / D$
any $\lambda > 0$ rules them out.

**Its gradient.** $\Sigma = E[z z^\top] - \mu \mu^\top$, so $d\Sigma = E[dz
(z - \mu)^\top + (z - \mu) dz^\top]$ and $d\mu = E[dz]$, and

$$
dF = \tfrac12 \mathrm{tr}((I - \Sigma^{-1}) d\Sigma) + \mu^\top d\mu
   = E[g(z)^\top dz], \qquad
g(z) = (I - \Sigma^{-1})(z - \mu) + \mu = z - \Sigma^{-1} (z - \mu):
$$

$\nabla_\theta F = E[J_\theta(x)^\top g(z)]$, $J_\theta = \partial z /
\partial \theta$. The part $z$ is the gradient of $\tfrac12 (\mathrm{tr}\,
\Sigma + \lVert \mu \rVert^2) = \tfrac12 E \lVert z \rVert^2$, linear in the
samples: it needs no estimate. Only the log-determinant's $-\Sigma^{-1}(z -
\mu)$ needs the moments.

**The estimate of a step** (``buffer_moments``, ``gauss_kl_grad``). A step
sees the current samples $z_b$ and a buffer of $n$ past samples $p_j$
(detached). With $\hat\mu = \frac1n \sum_j p_j$, the scatter $S = \sum_j
(p_j - \hat\mu)(p_j - \hat\mu)^\top$ and

$$
\hat P = (n - D - 2)\, S^{-1}, \qquad \hat g_b = z_b - \hat P (z_b - \hat\mu),
$$

$E[J_\theta^\top \hat g_b] = \nabla_\theta F$ when the $p_j$ are iid
$N(\mu, \Sigma)$ of the current encoder (a stationary encoder) and
independent of $z_b$: then $\hat\mu$ and $S$ are independent, $E \hat\mu =
\mu$, $S \sim W(\Sigma, n - 1)$ and $E S^{-1} = \Sigma^{-1} / (n - D - 2)$
for $n > D + 2$, so $E \hat P = \Sigma^{-1}$; $\hat g_b$ is linear in $\hat
P$, in $\hat\mu$ and in their product, each independent of $z_b$, so its
expectation is $g(z_b)$'s. The other conventions are biased at a finite
buffer:

- the plug-in $\hat\Sigma = S / (n - 1)$ (or $S / n$) has $E \hat\Sigma^{-1}
  = \frac{n - 1}{n - D - 2} \Sigma^{-1}$: the expected gradient vanishes at
  $\Sigma = \frac{n - 1}{n - D - 2} I$, not at $I$ (1.34 $I$ at $D$ = 1024,
  $n$ = 4096; the factor is $1 + (D + 1) / (n - D - 2)$);
- the current sample among the $n + 1$ samples of the moments: the exact
  gradient of the plug-in $\hat F$ of the $n + 1$ samples (covariance with
  $1 / (n + 1)$), times $n + 1$, is $z_b - \hat\Sigma^{-1}(z_b - \hat\mu)$
  with $z_b$ in $\hat\mu$ and $\hat\Sigma$ (the reading of 2026-11-13). It
  has the bias of the plug-in, and its moments are no longer independent of
  $z_b$.

$\hat P$ exists only for $n \ge D + 3$ ($S$ is invertible from $n = D + 1$,
and $E S^{-1}$ is finite only for $n > D + 2$): below that the regularizer
is off (``GaussMoments.on`` 0). This is the rule the start follows (a
buffer filling from empty), defined by the estimator's existence; no prior
and no threshold of its own. (The variance of $\hat P$ is finite only for
$n > D + 4$.)

**What the stream breaks.** The derivation needs the $p_j$ iid and
independent of $z_b$. On one stream they are correlated with each other and
with $z_b$ (the recent rows are near it), and then the expected gradient
does depend on the correlation, through the log-determinant's part only:
$S / (n - 1)$ underestimates $\Sigma$ (the rows vary less about their own
mean), $S^{-1}$ is convex in $S$ so a larger spread of $S$ raises its mean
further, and $z_b - \hat\mu$ is smaller than $z_b - \mu$ along the slow
directions. The $\mathrm{tr}$ and mean parts ($z_b$) stay exact.
``exp.video.diag.gkl_ar1`` measures where this moves the fixed point.

**Several samples.** With $B$ current samples the step's gradient is the
mean over them of $J_b^\top \hat g_b$; the buffer's $n$ rows are of all the
streams, and the estimate of each is the one above.

**Scale.** The loss is $(1 - \lambda) L^{\mathrm{pred}} + \lambda F / D$:
$L^{\mathrm{pred}}$ is a mean over the coordinates of the tokens ($\frac1D
\lVert \cdot \rVert^2$), and $F$ a sum over the $D$ coordinates (for a
diagonal $\Sigma$, $F = \sum_i \tfrac12 (\sigma_i^2 - 1 - \log \sigma_i^2 +
\mu_i^2)$), so $F / D$ is its mean per coordinate and $\lambda$ weighs two
means per coordinate, whatever $D$. A sample's credit is $\lambda \hat g_b
/ D$.

**The value.** ``GaussMoments.kl`` is $\hat F = \mathrm{KL}(N(\hat\mu, S / (n
- 1)) \,\|\, N(0, I))$, the closed form of the buffer's sample moments: a
statistic to read, not the function whose gradient is used ($\hat g_b$).
"""

from typing import NamedTuple

import torch
from torch import Tensor

__all__ = ["GaussMoments", "buffer_moments", "gauss_kl_grad", "kl_gauss", "off"]


class GaussMoments(NamedTuple):
    r"""What a step needs of the buffer (module docstring), constants: the
    mean $\hat\mu$ ``(D,)``, the precision $\hat P = (n - D - 2) S^{-1}$
    ``(D, D)``, the value $\hat F$ (0-d) and ``on`` (0-d, 1 when $n \ge D +
    3$; 0 before, with the others 0)."""

    mean: Tensor
    precision: Tensor
    kl: Tensor
    on: Tensor


def kl_gauss(mean: Tensor, cov: Tensor) -> Tensor:
    r"""$\tfrac12 [\mathrm{tr}\, \Sigma + \lVert \mu \rVert^2 - D - \log \det
    \Sigma]$ of ``mean`` ``(D,)`` and ``cov`` ``(D, D)``: $+\infty$ for a
    singular $\Sigma$ (a determinant of 0)."""
    sign, logdet = torch.linalg.slogdet(cov)
    logdet = torch.where(sign > 0, logdet, -torch.inf)
    d = mean.shape[-1]
    return 0.5 * (torch.trace(cov) + mean @ mean - d - logdet)


def off(
    width: int, device: torch.device | str, dtype: torch.dtype = torch.float32
) -> GaussMoments:
    """The ``GaussMoments`` of a buffer of fewer than ``width + 3`` rows: all
    0 (the regularizer off)."""
    zero = torch.zeros((), device=device, dtype=dtype)
    return GaussMoments(
        torch.zeros(width, device=device, dtype=dtype),
        torch.zeros(width, width, device=device, dtype=dtype),
        zero,
        zero.clone(),
    )


def buffer_moments(rows: Tensor) -> GaussMoments:
    r"""The ``GaussMoments`` of the buffer ``rows`` ``(n, D)`` (module
    docstring), computed in float64 (the scatter and its Cholesky factor),
    returned in the dtype of ``rows``.

    Raises:
        ValueError: $n \ge D + 3$ and $S$ is not positive definite (the rows
            span fewer than $D$ dimensions: $\hat F = +\infty$, a collapse
            the regularizer cannot follow).
    """
    n, d = rows.shape
    if n < d + 3:
        return off(d, rows.device, rows.dtype)
    x = rows.double()
    mean = x.mean(0)
    xc = x - mean
    scatter = xc.T @ xc
    factor, info = torch.linalg.cholesky_ex(scatter)
    if int(info) != 0:
        raise ValueError(
            f"the scatter of the {n} rows is singular (of rank below {d}): F is +inf"
        )
    precision = (n - d - 2) * torch.cholesky_inverse(factor)
    logdet = 2 * factor.diagonal().log().sum() - d * torch.log(
        torch.tensor(n - 1.0, dtype=x.dtype, device=x.device)
    )
    kl = 0.5 * (torch.trace(scatter) / (n - 1) + mean @ mean - d - logdet)
    dtype = rows.dtype
    return GaussMoments(
        mean.to(dtype),
        precision.to(dtype),
        kl.to(dtype),
        torch.ones((), dtype=dtype, device=rows.device),
    )


def gauss_kl_grad(z: Tensor, m: GaussMoments) -> Tensor:
    r"""$\hat g_b = z_b - \hat P (z_b - \hat\mu)$ of each row of ``z`` ``(B,
    D)`` (module docstring), times ``m.on``: the estimate of $\partial F /
    \partial z_b$ (the population's $g$) from the buffer, a constant of
    ``z``'s graph (detached)."""
    z = z.detach()
    return m.on * (z - (z - m.mean) @ m.precision)
