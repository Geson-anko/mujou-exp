r"""The linear horizon of a code from its curves over the lags
(``exp/vision/plans/jepa.md``, 「線形の幅（2026-11-26）」; README, 「線形の幅
（回す前に決めたこと）」).

A curve $y(\Delta)$ on an ascending grid of lags $x$ (seconds) and a
condition: $p_q(\Delta) \ge 1/2$ (``below=False``, the share of the triples
that win) or $G_{\mathrm{mid}}(\Delta) < q$ (``below=True``, pooled).

**Windows** (``windows``): the runs of consecutive grid points that meet the
condition. The **first** (of the shortest lags) is the window; the later
ones are **reappearances**, recorded and never used as the window (Taylor's
expansion vouches only for the run that starts on the short side; a rotation
of period $T$ wins again near $kT \pm T/6$). A run's ends are the crossings
of the threshold by the curve linear in $\log \Delta$ between the grid point
in and its neighbour out; a run that starts at the first grid point has $lo
\le x_0$ (``lo_cut``), one that ends at the last $hi \ge x_{-1}$
(``hi_cut``). No point meets it: no window (no number is given; nothing is
extrapolated).

**The widths**: $\Delta_{\mathrm{typ}}$ the first window's upper end of $p$
(``Windows.typical``), and $\bar\Delta = x_0 + \int_{x_0}^{x_{-1}} p\,
d\Delta$ (trapezoids linear in $\Delta$; ``mean_width``). Where each moment
$t$ wins below its own width $W_t$ and loses above it, $p(\Delta) = P(W_t >
\Delta)$: the crossing of 1/2 is the median of $W_t$ and $\bar\Delta =
E[\min(\max(W_t, x_0), x_{-1})]$ (on the grid: the median exactly where
the $W_t$ lie at the geometric midpoints of the grid's intervals, an odd
number of them, the median alone in its interval; the mean exactly where
they lie at the arithmetic midpoints). Where the wins are not monotone in
$\Delta$, $\int p\,d\Delta$ is the mean length of the lags each moment
wins and the crossing need not be a median: ``survival``, the share of the
moments whose first run of wins (from the first lag each wins at) reaches
$\Delta$, is monotone by construction; where each moment wins on one run
$[l_t, W_t]$ it is $P(W_t \ge \Delta)$ whatever the $l_t$, its crossing of
1/2 the median of $W_t$ (a run counted from the grid's first lag instead is
0 wherever the moments lose there: a floor of noise below the window).

**The bootstrap** (``draws_summary``, ``paired_ratio``): each resample's
curve ``(n, L)``; its first window's upper end (``first_upper``: NaN of no
window) and $\bar\Delta$. An interval is of the 2.5% and 97.5% order points
(``quantile``), a draw of no window counted as below every width: where more
than 2.5% of the draws have none, the lower end is None (「無い」; at
exactly 2.5% the next order point). The
paired ratio of two codes is $\log_2(W_a / W_b)$ of the same draws; of
$\Delta_{\mathrm{typ}}$ only where both have a window in at least 97.5% of
the draws (``WINDOWED``), else it is not given.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "WINDOWED",
    "DrawSummary",
    "Interval",
    "PairedRatio",
    "Span",
    "Windows",
    "draws_summary",
    "first_upper",
    "mean_width",
    "mean_widths",
    "meets",
    "n_lin",
    "paired_ratio",
    "quantile",
    "survival",
    "union",
    "windows",
]

WINDOWED = 0.975  # the share of draws with a window for a ratio of Delta_typ
_F64 = torch.float64


@dataclass(frozen=True)
class Span:
    """A run of the grid meeting the condition: its ends (seconds) and
    whether each is cut by the grid's end (``lo`` $\\le$, ``hi``
    $\\ge$)."""

    lo: float
    hi: float
    lo_cut: bool
    hi_cut: bool


type Interval = tuple[float | None, float | None]


@dataclass(frozen=True)
class DrawSummary:
    r"""``draws_summary``'s: the shares of draws of no window and of a window
    cut at the grid's end, the interval of the first window's upper end and
    of $\bar\Delta$ (None of a pooled $G$)."""

    none: float
    cut: float
    typ: Interval
    mean: Interval | None


@dataclass(frozen=True)
class PairedRatio:
    r"""``paired_ratio``'s: the intervals of $\log_2$ of the ratios of
    $\bar\Delta$ and of $\Delta_{\mathrm{typ}}$ (None: not given), the
    two codes' shares of draws with a window, whether a draw's first window
    is cut at the grid's end (「打ち切りを含む」) and the two codes' mean
    over the draws of $p$ at the grid's last lag (``tail``: the share of
    the moments $\bar\Delta$ clips there)."""

    mean: Interval
    typ: Interval | None
    windowed: tuple[float, float]
    cut: bool
    tail: tuple[float, float]


@dataclass(frozen=True)
class Windows:
    """The first window (None: no window) and the reappearances."""

    first: Span | None
    again: tuple[Span, ...]

    @property
    def typical(self) -> float | None:
        """The first window's upper end (None of no window)."""
        return None if self.first is None else self.first.hi


def meets(y: Tensor, threshold: float, *, below: bool) -> Tensor:
    """Whether each value of ``y`` meets the condition: $y <$ ``threshold``
    (``below``) or $y \\ge$ ``threshold``; NaN never meets it."""
    return y < threshold if below else y >= threshold


def _cross(x0: float, x1: float, y0: float, y1: float, c: float) -> float:
    """The crossing of ``c`` by the line through $(\\log x_0, y_0)$, $(\\log
    x_1, y_1)$ (in $\\Delta$)."""
    f = (c - y0) / (y1 - y0)
    return math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0)))


def windows(x: Tensor, y: Tensor, threshold: float, *, below: bool) -> Windows:
    """The windows (module docstring) of the curve ``y`` ``(L,)`` on the
    grid ``x`` ``(L,)`` (seconds, ascending)."""
    ok = meets(y, threshold, below=below).tolist()
    xs, ys = x.double().tolist(), y.double().tolist()
    spans: list[Span] = []
    i, n = 0, len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        lo = xs[0] if i == 0 else _cross(xs[i - 1], xs[i], ys[i - 1], ys[i], threshold)
        hi = (
            xs[-1]
            if j == n - 1
            else _cross(xs[j], xs[j + 1], ys[j], ys[j + 1], threshold)
        )
        spans.append(Span(lo, hi, i == 0, j == n - 1))
        i = j + 1
    return Windows(spans[0] if spans else None, tuple(spans[1:]))


def mean_width(x: Tensor, p: Tensor) -> float:
    r"""$\bar\Delta = x_0 + \int_{x_0}^{x_{-1}} p\,d\Delta$ (trapezoids) of
    ``p`` ``(L,)`` on ``x`` ``(L,)``."""
    return float(mean_widths(x, p[None])[0])


def mean_widths(x: Tensor, p: Tensor) -> Tensor:
    """``mean_width`` of each row of ``p`` ``(n, L)`` ``(n,)`` float64."""
    xd, pd = x.double(), p.double()
    dx = xd[1:] - xd[:-1]
    return xd[0] + ((pd[:, 1:] + pd[:, :-1]) / 2 * dx).sum(1)


def survival(won: Tensor) -> Tensor:
    """Whether each moment's first run of wins reaches each lag ``(N, L)``
    bool (module docstring): of ``won`` ``(N, L)`` bool along the lags,
    the lags up to the end of the run that starts at the first win (none of
    a moment that never wins)."""
    n, length = won.shape
    idx = torch.arange(length)
    has = won.any(1)
    start = won.int().argmax(1)
    out_after = ~won & (idx[None] > start[:, None])
    end = torch.where(
        out_after.any(1), out_after.int().argmax(1) - 1, torch.full((n,), length - 1)
    )
    return has[:, None] & (idx[None] <= end[:, None])


def first_upper(
    x: Tensor, y: Tensor, threshold: float, *, below: bool
) -> tuple[Tensor, Tensor]:
    """The first window's upper end ``(n,)`` float64 of each curve of ``y``
    ``(n, L)`` (NaN of no window; as ``windows``) and whether it is cut by
    the grid's end ``(n,)`` bool."""
    ok = meets(y, threshold, below=below)
    n, length = ok.shape
    idx = torch.arange(length)
    has = ok.any(1)
    start = ok.int().argmax(1)  # the first point in (0 where none)
    out_after = ~ok & (idx[None] > start[:, None])
    ends = out_after.any(1)
    stop = out_after.int().argmax(1)  # the first point out after it
    j = (stop - 1).clamp(min=0)
    xd, yd = x.double(), y.double()
    x0, x1 = xd[j], xd[stop]
    y0, y1 = yd.gather(1, j[:, None])[:, 0], yd.gather(1, stop[:, None])[:, 0]
    f = (threshold - y0) / (y1 - y0)
    crossed = torch.exp(torch.log(x0) + f * (torch.log(x1) - torch.log(x0)))
    hi = torch.where(ends, crossed, xd[-1].expand(n))
    hi = torch.where(has, hi, torch.full_like(hi, math.nan))
    return hi, has & ~ends


def quantile(values: Tensor, q: float) -> float | None:
    """The ``q`` quantile of ``values`` ``(n,)`` (linear between order
    points), NaN counted below every value: None where more than ``q`` of
    them are NaN; at most ``q`` and the lower order point NaN, the upper
    one."""
    v = values.double()
    if int(v.isnan().sum()) > q * len(v) + 1e-9:
        return None
    v = v.nan_to_num(nan=-math.inf).sort().values
    pos = q * (len(v) - 1)
    i, j = math.floor(pos), math.ceil(pos)
    a, b = float(v[i]), float(v[j])
    if a == -math.inf:
        return b
    return a + (pos - i) * (b - a)


def union(
    intervals: Sequence[tuple[float | None, float | None]],
) -> tuple[float | None, float | None]:
    """The smallest interval holding all of ``intervals`` (a None lower end
    is below all; a None upper end, of a quantity of no window, is
    dropped)."""
    los = [lo for lo, _ in intervals]
    his = [hi for _, hi in intervals if hi is not None]
    lo = None if any(v is None for v in los) else min(v for v in los if v is not None)
    return lo, max(his) if his else None


def draws_summary(
    x: Tensor, draws: Tensor, threshold: float, *, below: bool, level: float = 0.95
) -> DrawSummary:
    r"""Of the resampled curves ``draws`` ``(n, L)`` on ``x``: the share of
    draws of no window, of a window cut at the grid's end, the interval of
    the first window's upper end and, of a $p$ (not ``below``), of
    $\bar\Delta$ (module docstring)."""
    hi, cut = first_upper(x, draws, threshold, below=below)
    a = (1 - level) / 2
    mean = None
    if not below:
        m = mean_widths(x, draws)
        mean = (quantile(m, a), quantile(m, 1 - a))
    return DrawSummary(
        none=float(hi.isnan().double().mean()),
        cut=float(cut.double().mean()),
        typ=(quantile(hi, a), quantile(hi, 1 - a)),
        mean=mean,
    )


def paired_ratio(
    x: Tensor,
    draws_a: Tensor,
    draws_b: Tensor,
    threshold: float,
    *,
    level: float = 0.95,
) -> PairedRatio:
    r"""Of two codes' resampled $p$ ``(n, L)`` of the same draws: the
    interval of $\log_2(ar\Delta_a / ar\Delta_b)$, and of $\log_2$ of the
    ratio of the first windows' upper ends (None unless both have a window
    in at least ``WINDOWED`` of the draws; then of the draws where both have
    one), the shares of draws with a window, whether a draw's first window
    runs to the last point (「打ち切りを含む」) and the mean $p$ there."""
    a = (1 - level) / 2
    r = torch.log2(mean_widths(x, draws_a) / mean_widths(x, draws_b))
    ha, ca = first_upper(x, draws_a, threshold, below=False)
    hb, cb = first_upper(x, draws_b, threshold, below=False)
    wa = float((~ha.isnan()).double().mean())
    wb = float((~hb.isnan()).double().mean())
    typ = None
    if wa >= WINDOWED and wb >= WINDOWED:
        both = ~ha.isnan() & ~hb.isnan()
        t = torch.log2(ha[both] / hb[both])
        typ = (quantile(t, a), quantile(t, 1 - a))
    return PairedRatio(
        mean=(quantile(r, a), quantile(r, 1 - a)),
        typ=typ,
        windowed=(wa, wb),
        cut=bool((ca | cb).any()),
        tail=(float(draws_a[:, -1].mean()), float(draws_b[:, -1].mean())),
    )


def n_lin(widths: Tensor, ts: Sequence[float]) -> list[int]:
    r"""$N_{\mathrm{lin}}(T) = \#\{i : W_i \ge T\}$ of the directions' widths
    ``widths`` ``(k,)`` (0 of no window) at each $T$ of ``ts``."""
    return [int((widths >= t - 1e-12).sum()) for t in ts]
