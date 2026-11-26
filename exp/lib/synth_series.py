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
- ``rotation``: $d/2$ planes turning at one angular speed $\omega$ (per
  step), each $\sqrt2 (\cos, \sin)(\omega t + \phi_j)$ of its own uniform
  phase: per triple $G_{\mathrm{mid}} = \sin^2(\omega n / 2)$ exactly (the
  chord's midpoint lies $r(1 - \cos\omega n)$ from the arc's)
- ``chained_rotation``: ``rotation`` of the period ``periods[b % J]`` in
  each block $b$ of rows, its phases drawn anew in each block
- ``ou_integral``: the running sum of an ``ou`` velocity, $E \lVert
  x_{t+n} - x_t \rVert^2 = D(n) = n \frac{1 + r}{1 - r} - \frac{2r(1 -
  r^n)}{(1 - r)^2}$ per column (``ou_integral_d``; not stationary, its
  increments are), whose pooled $G_{\mathrm{mid}} = 1 - D(2n) / (4 D(n))$:
  of a time constant $\tau$ in the limit of many steps, $G < 1/4$ for $n <
  1.1514\,\tau$ (the root of $u - 2 + 3e^{-u} - e^{-2u} = 0$) and $G < 1/2$
  at every lag

The direct prediction's checks (``exp.lib.dynpred``; ``plans/dyn.md``) add
processes of known autocovariance $\Gamma(\ell) = E[x_{t+\ell} x_t^\top]$
``(d, d)``:

- ``block_rotation``: in each block $b$ of rows, $d/2$ planes turning at
  $\omega_b = 2\pi$ / ``periods[b % J]`` (times ``radii[b % K]`` with
  ``radii``), of amplitude $\sqrt2\,\rho_b / \bar\rho$ ($\bar\rho^2$ the
  mean of the radii's squares; 1 without), the phases drawn anew in each
  block; mixed over its blocks of weights $\pi$, $\Gamma(\ell) = \sum_b
  \pi_b (\rho_b / \bar\rho)^2 R(\omega_b \ell)$ in every plane
  (``rotation_gamma``; $R(\theta)$ the plane's rotation by $\theta$)
- ``complex_ou``: in every plane the exact discretization of $\dot w =
  (-\lambda + \omega J) w$ + noise, $w_{t+1} = e^{-\lambda} R(\omega) w_t +
  \sqrt{1 - e^{-2\lambda}}\,\varepsilon_t$ from its stationary law:
  $\Gamma(\ell) = e^{-\lambda \ell} R(\omega \ell)$ (``complex_ou_gamma``)
- ``box_mean``: the mean of the last ``width`` rows, $y_t = \frac1W
  \sum_{i=0}^{W-1} x_{t-i}$; of an AR(1) of coefficient $r$, $\gamma_y(\ell)
  = \frac1{W^2} \sum_{i,j=0}^{W-1} r^{\lvert \ell + i - j \rvert}$ exactly
  (``box_ar1_acf``, a finite sum)
"""

import math

import torch
from torch import Tensor

__all__ = [
    "block_rotation",
    "box_ar1_acf",
    "box_mean",
    "chained_rotation",
    "chebyshev",
    "complex_ou",
    "complex_ou_gamma",
    "gauss_kernel",
    "kernel_acf",
    "ou",
    "ou_integral",
    "ou_integral_d",
    "random_walk",
    "rotation",
    "rotation_gamma",
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


def rotation(n: int, d: int, period: float, gen: torch.Generator) -> Tensor:
    r"""``(n, d)`` (``d`` even): $\sqrt2 (\cos, \sin)(\omega t + \phi_j)$
    in columns $2j$, $2j + 1$, $\omega = 2\pi$ / ``period`` (steps), the
    phases uniform (by ``gen``)."""
    if d % 2:
        raise ValueError(f"a rotation needs an even d, not {d}")
    phi = 2 * math.pi * torch.rand(d // 2, generator=gen, dtype=_F64)
    theta = torch.arange(n, dtype=_F64)[:, None] * (2 * math.pi / period) + phi
    out = torch.empty(n, d, dtype=_F64)
    out[:, 0::2], out[:, 1::2] = theta.cos(), theta.sin()
    return math.sqrt(2) * out


def chained_rotation(
    blocks: Tensor, d: int, periods: tuple[float, ...], gen: torch.Generator
) -> Tensor:
    r"""``(n, d)``: the rows of each block $b$ of ``blocks`` ``(n,)`` (each
    block's rows consecutive, numbered from 0 in order) a ``rotation`` of
    the period ``periods[b % len(periods)]``, its phases drawn by ``gen``
    block by block."""
    out = torch.empty(len(blocks), d, dtype=_F64)
    for b in range(int(blocks.max()) + 1):
        rows = (blocks == b).nonzero().flatten()
        rot = rotation(int(rows[-1]) + 1, d, periods[b % len(periods)], gen)
        out[rows] = rot[rows]
    return out


def ou_integral(n: int, d: int, r: float, gen: torch.Generator) -> Tensor:
    r"""``(n, d)``: $x_t = \sum_{i \le t} v_i$ of the stationary ``ou``
    velocity $v$ of coefficient ``r``."""
    return ou(n, d, r, gen).cumsum(0)


def ou_integral_d(lag: float, r: float) -> float:
    r"""$D(n) = n \frac{1 + r}{1 - r} - \frac{2r(1 - r^n)}{(1 - r)^2}$, the
    mean squared increment of ``ou_integral`` per column at the lag $n$
    (module docstring; $n$ may be real: the same formula, continued)."""
    return lag * (1 + r) / (1 - r) - 2 * r * (1 - r**lag) / (1 - r) ** 2


def block_rotation(
    blocks: Tensor,
    d: int,
    periods: tuple[float, ...],
    gen: torch.Generator,
    radii: tuple[float, ...] | None = None,
) -> Tensor:
    r"""``(n, d)`` (``d`` even): the rows of each block $b$ of ``blocks``
    ``(n,)`` (each block's rows consecutive, numbered from 0 in order) turn
    in the planes of the columns $2j$, $2j + 1$ at $\omega_b \rho_b$,
    $\omega_b = 2\pi$ / ``periods[b % J]``, $\rho_b$ = ``radii[b % K]``
    (1 without), of amplitude $\sqrt2 \rho_b / \bar\rho$ (module
    docstring): $\sqrt2 \tfrac{\rho_b}{\bar\rho} (\cos, \sin)(\omega_b
    \rho_b t + \phi_{bj})$ of the row $t$, the phases uniform, drawn by
    ``gen`` block by block (``chained_rotation``'s draws, in one pass)."""
    if d % 2:
        raise ValueError(f"a rotation needs an even d, not {d}")
    rs = (1.0,) if radii is None else radii
    norm = math.sqrt(sum(r * r for r in rs) / len(rs))
    out = torch.empty(len(blocks), d, dtype=_F64)
    edges = torch.cat([torch.tensor([0]), (blocks[1:] != blocks[:-1]).nonzero().flatten() + 1,
                       torch.tensor([len(blocks)])]).tolist()  # fmt: skip
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        b = int(blocks[lo])
        rho = rs[b % len(rs)]
        phi = 2 * math.pi * torch.rand(d // 2, generator=gen, dtype=_F64)
        t = torch.arange(lo, hi, dtype=_F64)[:, None]
        theta = t * (2 * math.pi / periods[b % len(periods)] * rho) + phi
        amp = math.sqrt(2) * rho / norm
        out[lo:hi, 0::2], out[lo:hi, 1::2] = amp * theta.cos(), amp * theta.sin()
    return out


def _plane(theta: float) -> Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=_F64)


def rotation_gamma(
    lag: int,
    d: int,
    omegas: Tensor,
    weights: Tensor,
    gains: Tensor | None = None,
) -> Tensor:
    r"""$\Gamma(\ell) = \sum_j \pi_j g_j R(\omega_j \ell)$ ``(d, d)`` in
    every plane of the columns $2j$, $2j + 1$ (block diagonal) of the angular
    ``omegas`` ``(J,)`` (per step), their ``weights`` $\pi$ ``(J,)`` and
    ``gains`` $g$ ``(J,)`` (the squared amplitudes over 2; 1 without):
    ``block_rotation``'s mixed over its blocks."""
    g = torch.ones(len(omegas), dtype=_F64) if gains is None else gains.to(_F64)
    plane = torch.zeros(2, 2, dtype=_F64)
    for om, w, gj in zip(omegas.tolist(), weights.tolist(), g.tolist(), strict=True):
        plane += w * gj * _plane(om * lag)
    return torch.block_diag(*[plane] * (d // 2))


def complex_ou(
    n: int, d: int, lam: float, period: float, gen: torch.Generator
) -> Tensor:
    r"""``(n, d)`` (``d`` even): in every plane $w_{t+1} = e^{-\lambda}
    R(\omega) w_t + \sqrt{1 - e^{-2\lambda}}\,\varepsilon_t$ (``lam``
    $\lambda$ per step, $\omega = 2\pi$ / ``period``) from $w_0 \sim N(0,
    I)$, each column of unit variance (module docstring)."""
    if d % 2:
        raise ValueError(f"a rotation needs an even d, not {d}")
    e = white(n, d, gen)
    z = torch.complex(e[:, 0::2], e[:, 1::2])  # (n, d / 2)
    step = complex(math.exp(-lam) * math.cos(2 * math.pi / period),
                   math.exp(-lam) * math.sin(2 * math.pi / period))  # fmt: skip
    c = math.sqrt(1 - math.exp(-2 * lam))
    x = torch.empty_like(z)
    x[0] = z[0]
    for t in range(1, n):
        x[t] = step * x[t - 1] + c * z[t]
    out = torch.empty(n, d, dtype=_F64)
    out[:, 0::2], out[:, 1::2] = x.real, x.imag
    return out


def complex_ou_gamma(lag: int, d: int, lam: float, period: float) -> Tensor:
    r"""$\Gamma(\ell) = e^{-\lambda \lvert \ell \rvert} R(\omega \ell)$
    ``(d, d)`` of ``complex_ou`` ($\Gamma(-\ell) = \Gamma(\ell)^\top$)."""
    plane = math.exp(-lam * abs(lag)) * _plane(2 * math.pi / period * lag)
    return torch.block_diag(*[plane] * (d // 2))


def box_mean(x: Tensor, width: int) -> Tensor:
    r"""``(n - W + 1, d)``: $y_t = \frac1W \sum_{i=0}^{W-1} x_{t+W-1-i}$, the
    mean of each ``width`` consecutive rows of ``x`` ``(n, d)`` (the row
    $t$ of the output ends at the row $t + W - 1$ of ``x``)."""
    cs = torch.cat([x.new_zeros(1, x.shape[1]), x.cumsum(0)])
    return (cs[width:] - cs[:-width]) / width


def box_ar1_acf(r: float, width: int, lags: Tensor) -> Tensor:
    r"""$\gamma_y(\ell) = \frac1{W^2} \sum_{i,j=0}^{W-1} r^{\lvert \ell +
    i - j \rvert}$ ``(L,)`` float64 at the integer ``lags`` of ``box_mean``
    of a unit-variance AR(1) of coefficient ``r`` (a finite sum: of the
    $W^2$ terms, $W - \lvert m \rvert$ have $i - j = m$)."""
    m = torch.arange(-(width - 1), width, dtype=_F64)
    count = width - m.abs()
    out = torch.empty(len(lags), dtype=_F64)
    for i, lag in enumerate(lags.tolist()):
        out[i] = (count * r ** (lag + m).abs()).sum() / width**2
    return out
