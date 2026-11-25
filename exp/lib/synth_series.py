r"""Synthetic time series of known second-order statistics, to check the
measures of time linearity and of the information's width on
(``exp.lib.midpoint``, ``exp.lib.twonn``).

Every series is ``(n, d)`` float64, time first (one row a step); each column
is stationary of unit variance where it is stationary at all, and ``gen`` (a
CPU ``torch.Generator``) draws everything, column by column in order:

- ``white``: independent $N(0, 1)$, $\rho(\ell) = 0$ for $\ell \ne 0$
- ``random_walk``: the sums of independent $N(0, 1)$ increments (not
  stationary: $E \lVert x_{t+\ell} - x_t \rVert^2 = \ell$ per column)
- ``ou``: the AR(1) $x_t = r x_{t-1} + \sqrt{1 - r^2}\,\varepsilon_t$ from
  $x_0 \sim N(0, 1)$ (its stationary law), $\rho(\ell) = r^{\lvert \ell
  \rvert}$ exactly; also the points of an AR(1) correlation of the TwoNN's
  resolution
- ``smooth``: white noise convolved with a kernel $k$ (``gauss_kernel``)
  over its whole support (``valid``: every output sees the full kernel, so
  the series is exactly stationary), $\rho(\ell) = \sum_u k_u k_{u+\ell} /
  \sum_u k_u^2$ (``kernel_acf``)
- ``spectral``: $\sqrt2 \cos(\omega_j t + \phi_j)$ of the angular
  frequencies $\omega_j$ (per step), $\phi_j$ uniform; over the phases,
  $E[x_{t+\ell} x_t] = \cos \omega_j \ell$
- ``chebyshev``: $\sqrt2 \cos(j \theta)$ of one quantity $\theta$ and the
  integers $j$ (a one-dimensional code spread over many columns: Fable's
  and Codex's counterexample); $\theta_t = \Theta + \omega t$ for a series,
  or independent uniform $\theta$ for the points of a TwoNN
"""

import math

import torch
from torch import Tensor

__all__ = [
    "chebyshev",
    "gauss_kernel",
    "kernel_acf",
    "ou",
    "random_walk",
    "smooth",
    "spectral",
    "white",
]

_F64 = torch.float64


def white(n: int, d: int, gen: torch.Generator) -> Tensor:
    """``(n, d)`` independent $N(0, 1)$."""
    return torch.randn(n, d, generator=gen, dtype=_F64)


def random_walk(n: int, d: int, gen: torch.Generator) -> Tensor:
    r"""``(n, d)``: $x_t = \sum_{i \le t} \varepsilon_i$, $\varepsilon_i
    \sim N(0, 1)$."""
    return white(n, d, gen).cumsum(0)


def ou(n: int, d: int, r: float, gen: torch.Generator) -> Tensor:
    r"""``(n, d)`` stationary AR(1) of coefficient ``r`` ($\lvert r \rvert <
    1$; module docstring): $\rho(\ell) = r^{\lvert \ell \rvert}$. A time
    constant $\tau$ (in steps) is $r = e^{-1/\tau}$."""
    e = white(n, d, gen)
    x = torch.empty_like(e)
    x[0] = e[0]
    c = math.sqrt(1 - r * r)
    for t in range(1, n):
        x[t] = r * x[t - 1] + c * e[t]
    return x


def gauss_kernel(sigma: float, reach: float = 5.0) -> Tensor:
    r"""$k_u \propto e^{-u^2 / \sigma^2}$ for the integers $\lvert u \rvert
    \le \lfloor \text{reach} \cdot \sigma \rfloor$ (``sigma`` in steps),
    scaled to $\sum_u k_u^2 = 1$ (so ``smooth`` has unit variance)."""
    half = math.floor(reach * sigma)
    u = torch.arange(-half, half + 1, dtype=_F64)
    k = torch.exp(-(u**2) / sigma**2)
    return k / torch.linalg.vector_norm(k)


def kernel_acf(kernel: Tensor, lags: Tensor) -> Tensor:
    r"""$\rho(\ell) = \sum_u k_u k_{u+\ell} / \sum_u k_u^2$ ``(L,)`` float64
    of the ``kernel`` ``(K,)`` at the integer ``lags`` ``(L,)`` (0 beyond
    its support): the autocorrelation of ``smooth``."""
    k = kernel.to(_F64)
    norm = float((k * k).sum())
    out = torch.zeros(len(lags), dtype=_F64)
    for i, lag in enumerate(lags.abs().tolist()):
        if lag < len(k):
            out[i] = (k[: len(k) - lag] * k[lag:]).sum() / norm
    return out


def smooth(n: int, d: int, kernel: Tensor, gen: torch.Generator) -> Tensor:
    r"""``(n, d)``: $x_t = \sum_u k_u \varepsilon_{t+u}$ of white noise
    $\varepsilon$ ``(n + K - 1, d)`` (the ``valid`` convolution, by the FFT),
    $\rho$ = ``kernel_acf``."""
    k = kernel.to(_F64)
    m = n + len(k) - 1
    e = white(m, d, gen)
    size = 1 << (m + len(k) - 1).bit_length()
    fe = torch.fft.rfft(e, size, dim=0)
    fk = torch.fft.rfft(k.flip(0), size)
    full = torch.fft.irfft(fe * fk[:, None], size, dim=0)  # e * reversed k
    return full[len(k) - 1 : len(k) - 1 + n]


def spectral(n: int, omegas: Tensor, gen: torch.Generator) -> Tensor:
    r"""``(n, J)``: $\sqrt2 \cos(\omega_j t + \phi_j)$, $t = 0 .. n - 1$, of
    the angular frequencies ``omegas`` ``(J,)`` (per step), $\phi_j$ uniform
    in $[0, 2\pi)$ (drawn by ``gen``)."""
    w = omegas.to(_F64)
    phi = 2 * math.pi * torch.rand(len(w), generator=gen, dtype=_F64)
    t = torch.arange(n, dtype=_F64)
    return math.sqrt(2) * torch.cos(t[:, None] * w + phi)


def chebyshev(theta: Tensor, js: Tensor) -> Tensor:
    r"""``(N, J)``: $\sqrt2 \cos(j \theta)$ of the quantities ``theta``
    ``(N,)`` and the integers ``js`` ``(J,)``. The phase $j\theta$ is taken
    modulo $2\pi$ in float64 from $\theta$ split into its multiples of
    $2\pi / 2^{20}$ and the rest, so that $j$ up to $10^9$ keeps it to about
    $10^{-6}$ rad."""
    th = theta.to(_F64)
    j = js.to(torch.int64)
    unit = 2 * math.pi / 2**20
    whole = torch.floor(th / unit)  # th = whole * unit + rest
    rest = th - whole * unit
    turns = (whole.to(torch.int64)[:, None] * j) % 2**20  # exact integers
    phase = turns.to(_F64) * unit + rest[:, None] * j.to(_F64)
    return math.sqrt(2) * torch.cos(phase)
