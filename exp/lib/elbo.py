r"""The parts of the evidence lower bound (ELBO) of a VAE with a diagonal
Gaussian posterior, a standard normal prior and a dequantized Gaussian pixel
likelihood of one scalar $\sigma$ (Kingma & Welling 2014), and the closed-form
log-likelihood of probabilistic PCA (Tipping & Bishop 1999), its linear
optimum (Lucas et al. 2019).

**Dequantization** ("A note on the evaluation of generative models", ICLR
2016). Integer data $x \in \{0, \dots, L - 1\}^P$ ($L$ = ``levels``) is
taken to $y = (x + u) / L$, $u \sim U[0, 1)^P$ (``dequantize``). For any density $p$ of $y$, Jensen's
inequality over the bin gives $\log P(x) \ge E_u[\log p(y)] - P \ln L$, so

$$\mathrm{bpd} = \frac{E_u[-\log p(y)]}{P \ln 2} + \log_2 L
\ \ge\ -\frac{\log_2 P(x)}{P} \ \ge\ 0$$

(``bits_per_dim``): a density of $y$ cannot buy likelihood by shrinking its
variance, since $y$ carries the bin's uniform noise of variance $1 / (12
L^2)$ in every value (``dequant_var``). The same holds for the ELBO in place
of $\log p(y)$.

**The ELBO of a data point** in nats, with $q = N(\mu, \mathrm{diag}\, s^2)$,
$p(z) = N(0, I_D)$, $p(y \mid z) = N(g(z), \sigma^2 I_P)$:

$$-\mathcal L = E_q\big[-\log p(y \mid z)\big] + \mathrm{KL}(q \,\Vert\, p),$$

$-\log p(y \mid z) = \frac{\lVert y - g(z) \rVert^2}{2 \sigma^2} + P \ln
\sigma + \frac P2 \ln 2\pi$ (``gauss_nll``) and $\mathrm{KL} = \tfrac12
\sum_j (\mu_j^2 + s_j^2 - 1 - \ln s_j^2)$ (``gauss_kl``, element by
element). For a fixed decoder, $-\mathcal L$ is smallest in $\sigma$ at
$\sigma^{*2} = E\lVert y - g(z) \rVert^2 / P$, and a $\sigma$ off it costs
$\frac P2 (r - 1 - \ln r)$ nats, $r = \sigma^{*2} / \sigma^2$
(``sigma_gap``).

**Probabilistic PCA of $k$ components** fitted by maximum likelihood to data
of covariance eigenvalues $\lambda_1 \ge \dots \ge \lambda_P$ (``ppca_fit``):
the model's covariance is $C = U_k \Lambda_k U_k^\top + \sigma^2 (I - U_k
U_k^\top)$ with $\sigma^2 = \frac{1}{P - k} \sum_{j > k} \lambda_j$, the
mean of the discarded eigenvalues. Directions with $\lambda_j \le \sigma^2$
carry no component (the weight $W = U_k (\Lambda_k - \sigma^2 I)^{1/2}$ is 0
there). Of a point at $c = y - m$ from the fitted mean, with coordinates
$a_j = u_j^\top c$ and residual $r = c - U_k U_k^\top c$,

$$-\log p(y) = \tfrac12 \Big[P \ln 2\pi + \sum_{j \le k} \ln \lambda_j +
(P - k) \ln \sigma^2 + \sum_{j \le k} \frac{a_j^2}{\lambda_j} +
\frac{\lVert r \rVert^2}{\sigma^2}\Big]$$

(``ppca_nll``, of the mean squares of $a_j$ and $\lVert r \rVert$ over a set
of points). $k = 0$ is the isotropic Gaussian of the mean and $\sigma^2 =
\mathrm{tr}\,\Sigma / P$: the best a VAE whose decoder ignores $z$ can do.
"""

import math

import torch
from torch import Tensor

__all__ = [
    "LEVELS",
    "LN2",
    "bits_per_dim",
    "dequant_var",
    "dequantize",
    "gauss_kl",
    "gauss_nll",
    "ppca_fit",
    "ppca_nll",
    "sigma_gap",
]

LEVELS = 256  # the values of an 8-bit pixel
LN2 = math.log(2)
_LN_2PI = math.log(2 * math.pi)


def dequantize(x: Tensor, u: Tensor, levels: int = LEVELS) -> Tensor:
    """$y = (x + u) / L$ (float32) of the integers ``x`` and the uniform
    noise ``u`` in $[0, 1)$ of the same shape."""
    return (x.float() + u) / levels


def dequant_var(levels: int = LEVELS) -> float:
    """The variance $1 / (12 L^2)$ of the noise $u / L$ in each value of
    ``dequantize``."""
    return 1 / (12 * levels**2)


def gauss_kl(mu: Tensor, logvar: Tensor) -> Tensor:
    r"""$\mathrm{KL}(N(\mu, s^2) \,\Vert\, N(0, 1))$ of each element of
    ``mu`` and ``logvar`` $= \ln s^2$ (same shape), in nats."""
    return 0.5 * (mu.square() + logvar.exp() - 1 - logvar)


def gauss_nll(y: Tensor, mean: Tensor, log_sigma: Tensor) -> Tensor:
    r"""$-\log N(y; \hat y, \sigma^2 I)$ ``(n,)`` in nats of each sample of
    ``y`` and ``mean`` $\hat y$ ``(n, ...)``, summed over all but the first
    dimension; ``log_sigma`` $\ln \sigma$ a 0-d tensor."""
    sq = (y - mean).square().flatten(1).sum(1)
    p = y[0].numel()
    return 0.5 * sq * torch.exp(-2 * log_sigma) + p * (log_sigma + 0.5 * _LN_2PI)


def bits_per_dim(
    nll: float | Tensor, dims: int, levels: int = LEVELS
) -> float | Tensor:
    r"""$\mathrm{nll} / (P \ln 2) + \log_2 L$: bits per value of the
    integers from the nats ``nll`` of a point's $P$ = ``dims`` dequantized
    values (module docstring)."""
    return nll / (dims * LN2) + math.log2(levels)


def sigma_gap(sigma_star_sq: float, sigma_sq: float, dims: int) -> float:
    r"""$\frac P2 (r - 1 - \ln r)$, $r = \sigma^{*2} / \sigma^2$: the nats
    of $-\mathcal L$ a point loses to $\sigma^2$ off its optimum
    $\sigma^{*2}$ (module docstring); 0 at $r = 1$."""
    r = sigma_star_sq / sigma_sq
    return 0.5 * dims * (r - 1 - math.log(r))


def ppca_fit(
    evals: Tensor, k: int, dims: int, *, floor: float = 0.0
) -> tuple[Tensor, float]:
    r"""The maximum-likelihood pPCA of ``k`` components of data of ``dims``
    values whose covariance has the eigenvalues ``evals`` (descending, the
    nonzero ones at least; the rest of the ``dims`` are 0) plus ``floor`` in
    every direction (the noise of ``dequantize``, ``dequant_var``, independent
    of the data): its top eigenvalues $\lambda_1 .. \lambda_k$ (float64)
    and $\sigma^2$, the mean of the other ``dims - k`` (module docstring).

    Raises:
        ValueError: ``k`` is not in ``[0, dims)`` or beyond ``evals``.
    """
    if not 0 <= k < dims or k > len(evals):
        raise ValueError(f"k must be in [0, {min(dims - 1, len(evals))}], not {k}")
    lam = evals.double()
    return lam[:k] + floor, float(lam[k:].sum()) / (dims - k) + floor


def ppca_nll(
    top: Tensor, sigma_sq: float, dims: int, coord_ms: Tensor, residual_ms: float
) -> float:
    r"""The mean $-\log p(y)$ in nats over a set of points of the pPCA of
    eigenvalues ``top`` $\lambda_1 .. \lambda_k$ and ``sigma_sq`` in
    ``dims`` values (module docstring), from the mean squares over the points
    of their coordinates ``coord_ms`` ``(k,)`` along the top eigenvectors and
    of their residuals' norms ``residual_ms``."""
    lam = top.double()
    k = len(lam)
    quad = float((coord_ms.double() / lam).sum()) + residual_ms / sigma_sq
    logdet = float(lam.log().sum()) + (dims - k) * math.log(sigma_sq)
    return 0.5 * (dims * _LN_2PI + logdet + quad)
