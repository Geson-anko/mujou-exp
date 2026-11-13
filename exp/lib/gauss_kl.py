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

- the plug-in $\hat\Sigma = S / (n - 1)$ has $E \hat\Sigma^{-1} = \frac{n
  - 1}{n - D - 2} \Sigma^{-1}$: the expected gradient vanishes at $\Sigma =
  \frac{n - 1}{n - D - 2} I$, not at $I$ (1.34 $I$ at $D$ = 1024, $n$ =
  4096; the factor is $1 + (D + 1) / (n - D - 2)$); $S / n$ has the factor
  $\frac{n}{n - D - 2}$;
- the current sample among the $n + 1$ samples of the moments: the exact
  gradient of the plug-in $\hat F$ of the $n + 1$ samples (covariance with
  $1 / (n + 1)$), times $n + 1$, is $z_b - \hat\Sigma^{-1}(z_b - \hat\mu)$
  with $z_b$ in $\hat\mu$ and $\hat\Sigma$ (the reading of 2026-11-13). It
  has the bias of the plug-in, and its moments are no longer independent of
  $z_b$.

**When it is on** (``min_rows``). A buffer fills from empty (the first
$N k$ frames), and $\hat g_b$ is a step of a stochastic gradient descent:
an unbiased estimate of $\nabla F$ of finite variance (the conditions of a
stochastic approximation, Robbins-Monro). With $S \sim W(\Sigma, n - 1)$,
$S$ is invertible from $n = D + 1$, the mean of $S^{-1}$ (inverse Wishart of
$n - 1$ degrees of freedom) is finite for $n - 1 > D + 1$ and its variance
for $n - 1 > D + 3$. So $\hat P$ is unbiased from $n = D + 3$ and of finite
variance from $n = D + 5$: the term is on from $D + 5$ rows (``on`` 0
before). The other rules considered: from $D + 3$ (the mean exists, but
the first steps' variance is infinite); only once full (no condition on
the estimate gives $N$); a prior to shrink $S$ (a strength with no
derivation). At $n = D + 5$ the smallest eigenvalue of $S$ is near the edge
of Marchenko-Pastur, so the first steps are still large; the rule bounds
their variance, not their size.

**What the stream breaks.** The derivation needs the $p_j$ iid and
independent of $z_b$. On one stream they are correlated with each other and
with $z_b$ (the recent rows are near it), and then the expected gradient
does depend on the correlation, through the log-determinant's part only:
$S / (n - 1)$ underestimates $\Sigma$ (the rows vary less about their own
mean), $S^{-1}$ is convex in $S$ so a larger spread of $S$ raises its mean
further, and $z_b - \hat\mu$ is smaller than $z_b - \mu$ along the slow
directions. The $\mathrm{tr}$ and mean parts ($z_b$) stay exact.
``exp.video.diag.gkl_ar1`` measures where this moves the fixed point.

**What else the premises leave out.** (a) Rows of a non-Gaussian law: $S$
and $\hat\mu$ are no longer independent and $E S^{-1}$ depends on the
fourth moments, so $E \hat P \ne \Sigma^{-1}$ at order $D / n$ (no
estimate of $\Sigma^{-1}$ is unbiased for every law; $(n - D - 2)$ is exact
within the Gaussian family, the family of $F$).
``exp.video.diag.gkl_check nongauss`` measures $\mathrm{tr}\, E \hat P / D$
for heavy-tailed rows of covariance $I$. (b) A moving encoder: the rows are
the $z$ of past weights (a row's age is geometric of mean $N k$ frames),
and $\hat\mu$, $\hat P$ are those of a mixture of past laws, not of the
current one; the expected gradient is $\nabla F$ only when the encoder
moves little within $N k$ frames.

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

**The value** (``GaussMoments.kl``). The statistic logged is the estimate
of $F$ unbiased for iid Gaussian rows,

$$
\tilde F = \tfrac12 \left[ \tfrac1n \mathrm{tr}\, S + \lVert \hat\mu
\rVert^2 - D - \log \det S + D \log 2 + \textstyle\sum_{i=1}^{D}
\psi\!\left(\tfrac{n - i}{2}\right) \right]
$$

($\psi$ the digamma function): $E \tfrac1n \mathrm{tr}\, S = \frac{n -
1}{n} \mathrm{tr}\, \Sigma$ and $E \lVert \hat\mu \rVert^2 = \lVert \mu
\rVert^2 + \frac1n \mathrm{tr}\, \Sigma$ sum to $\mathrm{tr}\, \Sigma +
\lVert \mu \rVert^2$, and $E \log \det S = \log \det \Sigma + D \log 2 +
\sum_{i=1}^{D} \psi(\frac{n - i}{2})$ (Wishart of $n - 1$ degrees of
freedom). So $E \tilde F = F$ and it reads 0 on average at the optimum. The
closed form of the sample moments, $\mathrm{KL}(N(\hat\mu, S / (n - 1))
\,\|\, N(0, I))$, does not: its bias $\tfrac12 [\mathrm{tr}\,\Sigma / n + D
\log \frac{n - 1}{2} - \sum_i \psi(\frac{n - i}{2})]$ is 0.0687 per
coordinate at $\Sigma = I$, $D$ = 1024, $n$ = 4096
(``exp.video.diag.gkl_check floor``). $\tilde F$ is a statistic to read,
not the function whose gradient is used ($\hat g_b$).

**A singular scatter.** The rows spanning fewer than $D$ dimensions ($F =
+\infty$, a collapse the regularizer cannot follow) give $\tilde F = +\infty$
and a precision of NaN, without a host synchronization: the steps carry the
NaN on, and the run's row (its non-finite ``sigreg``) marks it diverged.
"""

import math
from typing import NamedTuple

import torch
from torch import Tensor

__all__ = [
    "GaussMoments",
    "buffer_moments",
    "gauss_kl_grad",
    "kl_gauss",
    "min_rows",
    "off",
    "plugin_bias",
]


class GaussMoments(NamedTuple):
    r"""What a step needs of the buffer (module docstring), constants: the
    mean $\hat\mu$ ``(D,)``, the precision $\hat P = (n - D - 2) S^{-1}$
    ``(D, D)``, the value $\tilde F$ (0-d) and ``on`` (0-d, 1 when $n \ge
    D + 5$, ``min_rows``; 0 before, with the others 0)."""

    mean: Tensor
    precision: Tensor
    kl: Tensor
    on: Tensor


def min_rows(width: int) -> int:
    r"""$D + 5$: the fewest rows from which $\hat P$ is unbiased and of
    finite variance (module docstring, "When it is on")."""
    return width + 5


def kl_gauss(mean: Tensor, cov: Tensor) -> Tensor:
    r"""$\tfrac12 [\mathrm{tr}\, \Sigma + \lVert \mu \rVert^2 - D - \log \det
    \Sigma]$ of ``mean`` ``(D,)`` and ``cov`` ``(D, D)``: $+\infty$ for a
    singular $\Sigma$ (a determinant of 0)."""
    sign, logdet = torch.linalg.slogdet(cov)
    logdet = torch.where(sign > 0, logdet, -torch.inf)
    d = mean.shape[-1]
    return 0.5 * (torch.trace(cov) + mean @ mean - d - logdet)


def _wishart_logdet_shift(n: int, d: int) -> float:
    r"""$D \log 2 + \sum_{i=1}^{D} \psi(\frac{n - i}{2})$: $E \log \det S -
    \log \det \Sigma$ of the scatter of $n$ iid Gaussian rows."""
    i = torch.arange(1, d + 1, dtype=torch.float64)
    return d * math.log(2) + float(torch.special.digamma((n - i) / 2).sum())


def plugin_bias(n: int, d: int, trace: float) -> float:
    r"""$E[\mathrm{KL}(N(\hat\mu, S / (n - 1)) \,\|\, N(0, I))] - F$ for $n$
    iid Gaussian rows of $\mathrm{tr}\, \Sigma$ = ``trace`` (module
    docstring, "The value")."""
    shift = _wishart_logdet_shift(n, d)
    return 0.5 * (trace / n + d * math.log(n - 1) - shift)


def off(
    width: int, device: torch.device | str, dtype: torch.dtype = torch.float32
) -> GaussMoments:
    """The ``GaussMoments`` of a buffer of fewer than ``min_rows(width)``
    rows: all 0 (the regularizer off)."""
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
    returned in the dtype of ``rows``. A singular scatter gives $\tilde F =
    +\infty$ and a precision of NaN (module docstring), with no host
    synchronization."""
    n, d = rows.shape
    if n < min_rows(d):
        return off(d, rows.device, rows.dtype)
    x = rows.double()
    mean = x.mean(0)
    xc = x - mean
    scatter = xc.T @ xc
    factor, info = torch.linalg.cholesky_ex(scatter)
    ok = info == 0
    # a factor that cholesky_inverse takes when there is none (then replaced)
    factor = torch.where(ok, factor, torch.eye(d, dtype=x.dtype, device=x.device))
    precision = (n - d - 2) * torch.cholesky_inverse(factor)
    logdet = 2 * factor.diagonal().log().sum()
    shift = _wishart_logdet_shift(n, d)
    kl = 0.5 * (torch.trace(scatter) / n + mean @ mean - d - logdet + shift)
    precision = torch.where(ok, precision, torch.nan)
    kl = torch.where(ok, kl, torch.inf)
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
