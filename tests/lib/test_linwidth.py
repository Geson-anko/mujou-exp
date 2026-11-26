r"""``exp.lib.linwidth``: the windows, the widths and their bootstrap.

The oracles: hand-made curves (no window, a window cut at either end, a
first window and its reappearance, the crossing linear in $\log\Delta$ by
hand); the trapezoids of $\bar\Delta$ by hand; the main width
$\bar\Delta_1$ by hand (to the first window's upper end, $p = 1/2$ there;
the reappearance left out; $x_0$ of no window; $\bar\Delta$ of a window
to the grid's end), the vectorized one the loop's of ``windows``; of moments each with a width
$W_t$ ($p(\Delta) = P(W_t > \Delta)$), $\Delta_{\mathrm{typ}}$ is their
median exactly where they lie at the geometric midpoints of the grid's
intervals, an odd number, the median alone in its interval (the crossing
linear in $\log\Delta$ falls on it) and $\bar\Delta$
their mean clipped to the grid exactly where they lie at the arithmetic
midpoints (the trapezoid of a step at the middle is exact); the survival
curve is the first run of wins, $P(W_t \ge \Delta)$ of moments winning on
one run whatever its start; the vectorized first upper end is the loop's;
an order point of no window counts below all; and a ratio of
$\Delta_{\mathrm{typ}}$ is given only where both codes have a window in
97.5% of the draws.
"""

import math

import pytest
import torch

from exp.lib.linwidth import (
    draws_summary,
    first_mean,
    first_means,
    first_upper,
    mean_width,
    mean_widths,
    n_lin,
    paired_ratio,
    quantile,
    survival,
    union,
    windows,
)

F64 = torch.float64
X = torch.tensor([0.8, 1.6, 3.2, 6.4, 12.8], dtype=F64)


def curve(*ys: float) -> torch.Tensor:
    return torch.tensor(ys, dtype=F64)


class TestWindows:
    def test_no_point_meets_it(self):
        got = windows(X, curve(0.1, 0.2, 0.3, 0.4, 0.49), 0.5, below=False)
        assert got.first is None and got.again == () and got.typical is None

    def test_a_window_from_the_first_point_and_the_log_crossing(self):
        got = windows(X, curve(0.9, 0.7, 0.3, 0.1, 0.0), 0.5, below=False)
        assert got.first is not None and got.first.lo_cut and not got.first.hi_cut
        assert got.first.lo == 0.8
        # between 1.6 (0.7) and 3.2 (0.3): f = 0.5, the geometric midpoint
        assert got.typical == pytest.approx(math.sqrt(1.6 * 3.2), rel=1e-12)

    def test_a_window_to_the_last_point_is_cut(self):
        got = windows(X, curve(0.4, 0.45, 0.6, 0.8, 0.9), 0.5, below=False)
        assert got.first is not None and got.first.hi_cut and got.first.hi == 12.8
        f = (0.5 - 0.45) / (0.6 - 0.45)
        want = math.exp(math.log(1.6) + f * (math.log(3.2) - math.log(1.6)))
        assert got.first.lo == pytest.approx(want, rel=1e-12)

    def test_the_first_window_and_a_reappearance(self):
        # the second run is longer; the first is the window all the same
        got = windows(X, curve(0.6, 0.2, 0.7, 0.8, 0.9), 0.5, below=False)
        assert got.first is not None and got.first.lo_cut and got.first.hi < 1.6
        assert len(got.again) == 1 and got.again[0].hi_cut
        assert 1.6 < got.again[0].lo < 3.2

    def test_below_is_strict(self):
        # 0.25 at 0.8 is out (not < 0.25): the window starts at the crossing,
        # which is 0.8 itself, not cut by the grid
        got = windows(X, curve(0.25, 0.2, 0.3, 0.3, 0.3), 0.25, below=True)
        assert got.first is not None and not got.first.lo_cut
        assert got.first.lo == 0.8
        assert got.first.hi == pytest.approx(math.sqrt(1.6 * 3.2), rel=1e-12)


class TestMeanWidth:
    def test_the_trapezoids_by_hand(self):
        p = curve(1.0, 0.5, 0.0, 0.0, 0.0)
        assert mean_width(X, p) == pytest.approx(0.8 + 0.8 * 0.75 + 1.6 * 0.25)

    def test_of_none_and_of_all(self):
        assert mean_width(X, torch.zeros(5, dtype=F64)) == pytest.approx(0.8)
        assert mean_width(X, torch.ones(5, dtype=F64)) == pytest.approx(12.8)


class TestFirstMean:
    def test_to_the_first_windows_end_by_hand(self):
        p = curve(0.9, 0.7, 0.3, 0.1, 0.0)
        hi = math.sqrt(1.6 * 3.2)  # the crossing, linear in log
        want = 0.8 + 0.8 * (0.9 + 0.7) / 2 + (0.7 + 0.5) / 2 * (hi - 1.6)
        assert first_mean(X, p) == pytest.approx(want, rel=1e-12)

    def test_the_reappearance_is_left_out(self):
        p = curve(0.6, 0.2, 0.7, 0.8, 0.9)
        w = windows(X, p, 0.5, below=False)
        assert w.first is not None and len(w.again) == 1
        want = 0.8 + (0.6 + 0.5) / 2 * (w.first.hi - 0.8)
        assert first_mean(X, p) == pytest.approx(want, rel=1e-12)
        assert mean_width(X, p) > 4 * first_mean(X, p)

    def test_of_no_window_and_of_a_window_to_the_end(self):
        assert first_mean(X, curve(0.4, 0.49, 0.1, 0.3, 0.2)) == 0.8
        p = curve(0.4, 0.6, 0.9, 0.7, 0.5)  # from 0.8 to 12.8, no reappearance
        assert first_mean(X, p) == pytest.approx(mean_width(X, p), rel=1e-12)

    def test_the_vectorized_is_the_loops(self):
        g = torch.Generator().manual_seed(1)
        y = torch.rand(300, 5, generator=g, dtype=F64)
        got = first_means(X, y)
        xs = X.tolist()
        for i in range(300):
            w = windows(X, y[i], 0.5, below=False)
            if w.first is None:
                assert float(got[i]) == 0.8
                continue
            want, k = 0.8, 0
            while k + 1 < len(xs) and xs[k + 1] <= w.first.hi:
                want += (float(y[i, k]) + float(y[i, k + 1])) / 2 * (xs[k + 1] - xs[k])
                k += 1
            if not w.first.hi_cut:
                want += (float(y[i, k]) + 0.5) / 2 * (w.first.hi - xs[k])
            assert float(got[i]) == pytest.approx(want, rel=1e-12)


def p_of(widths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """$p(\\Delta) = P(W_t > \\Delta)$ on the grid ``x``."""
    return (widths[:, None] > x[None]).double().mean(0)


class TestMonotoneWidths:
    def test_typ_is_the_median_at_geometric_midpoints(self):
        mids = (X[1:] * X[:-1]).sqrt()  # 4 geometric midpoints
        # 5 moments, the median alone in its interval: p steps 3/5 to 2/5
        w = mids[torch.tensor([0, 0, 1, 2, 3])]
        got = windows(X, p_of(w, X), 0.5, below=False)
        assert got.typical == pytest.approx(float(mids[1]), rel=1e-12)

    def test_the_mean_is_the_clipped_mean_at_arithmetic_midpoints(self):
        mids = (X[1:] + X[:-1]) / 2
        w = torch.cat([mids[torch.tensor([0, 2, 2, 3])], torch.tensor([0.3, 40.0])])
        want = float(w.clamp(0.8, 12.8).mean())
        assert mean_width(X, p_of(w.double(), X)) == pytest.approx(want, rel=1e-12)

    def test_survival_is_the_first_run_of_wins(self):
        won = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0],
                            [0, 1, 0, 1]]).bool()  # fmt: skip
        want = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0],
                             [1, 1, 0, 0]]).bool()  # fmt: skip
        assert torch.equal(survival(won), want)

    def test_survival_of_runs_is_p_of_their_upper_ends(self):
        # moments winning on [l_t, W_t]: the share alive is P(W_t >= lag),
        # whatever the l_t (a floor of losses below the window included)
        lo = torch.tensor([0, 1, 2, 1, 3])
        hi = torch.tensor([2, 4, 2, 3, 4])
        idx = torch.arange(5)
        won = (idx[None] >= lo[:, None]) & (idx[None] <= hi[:, None])
        want = (hi[:, None] >= idx[None]).double().mean(0)
        assert torch.equal(survival(won).double().mean(0), want)


class TestDraws:
    def test_the_vectorized_upper_end_is_the_loops(self):
        g = torch.Generator().manual_seed(0)
        y = torch.rand(200, 5, generator=g, dtype=F64)
        hi, cut = first_upper(X, y, 0.5, below=False)
        for i in range(200):
            w = windows(X, y[i], 0.5, below=False)
            if w.first is None:
                assert math.isnan(float(hi[i]))
            else:
                assert float(hi[i]) == pytest.approx(w.first.hi, rel=1e-12)
                assert bool(cut[i]) == w.first.hi_cut

    def test_a_draw_of_no_window_is_below_all(self):
        v = torch.tensor([math.nan] * 3 + [1.0] * 97, dtype=F64)
        assert quantile(v, 0.025) is None
        assert quantile(v, 0.975) == 1.0
        v = torch.tensor([math.nan] * 2 + [1.0] * 98, dtype=F64)
        assert quantile(v, 0.025) == 1.0
        # exactly 2.5% of no window is not more than 2.5%: the next order point
        v = torch.tensor([math.nan] * 5 + [2.0] * 195, dtype=F64)
        assert quantile(v, 0.025) == 2.0
        v = torch.tensor([math.nan] * 6 + [2.0] * 194, dtype=F64)
        assert quantile(v, 0.025) is None

    def test_the_union_of_intervals(self):
        assert union([(1.0, 2.0), (0.5, 1.5)]) == (0.5, 2.0)
        assert union([(None, 2.0), (0.5, 3.0)]) == (None, 3.0)
        assert union([(None, None), (None, 1.0)]) == (None, 1.0)

    def test_the_summary_counts_the_draws_of_no_window(self):
        y = torch.stack([curve(0.9, 0.7, 0.3, 0.1, 0.0)] * 9 + [curve(0.1).expand(5)])
        got = draws_summary(X, y, 0.5, below=False)
        assert got.none == pytest.approx(0.1)
        assert got.typ[0] is None  # 10% > 2.5% of no window
        assert got.mean is not None
        lo, hi = got.mean
        assert lo is not None and hi is not None and lo < hi
        # the main width is 0.8 of a draw of no window: always given
        assert got.mean1 is not None
        w = first_mean(X, y[0])
        # one draw of 0.8 among ten: the 2.5% point 0.225 of the way to w
        assert got.mean1[0] == pytest.approx(0.8 + 0.225 * (w - 0.8))
        assert got.mean1[1] == pytest.approx(w)

    def test_a_ratio_of_typ_needs_windows_in_975_of_the_draws(self):
        a = torch.stack([curve(0.9, 0.7, 0.3, 0.1, 0.0)] * 100)
        b = torch.stack([curve(0.9, 0.9, 0.9, 0.3, 0.0)] * 100)
        got = paired_ratio(X, a, b, 0.5)
        assert got.typ is not None
        lo, hi = got.typ
        f = (0.5 - 0.9) / (0.3 - 0.9)
        want = math.log2(
            math.sqrt(1.6 * 3.2) / math.exp(math.log(3.2) + f * math.log(2))
        )
        assert lo == pytest.approx(want) and hi == pytest.approx(want)
        mean = float(torch.log2(mean_widths(X, a[:1]) / mean_widths(X, b[:1])))
        assert got.mean[0] == pytest.approx(mean)
        mean1 = math.log2(first_mean(X, a[0]) / first_mean(X, b[0]))
        assert got.mean1 == (pytest.approx(mean1), pytest.approx(mean1))
        none = b.clone()
        none[:3] = 0.0  # 3% of the draws of b have no window
        assert paired_ratio(X, a, none, 0.5).typ is None
        assert not got.cut and got.tail == (0.0, 0.0)
        # a p > 0 at the last point is a tail, not a cut; a window there is a cut
        c = torch.stack([curve(0.9, 0.7, 0.3, 0.1, 0.2)] * 100)
        got = paired_ratio(X, c, a, 0.5)
        assert not got.cut and got.tail[0] == pytest.approx(0.2)
        d = torch.stack([curve(0.9, 0.7, 0.3, 0.1, 0.6)] * 100)
        assert not paired_ratio(X, d, a, 0.5).cut
        e = torch.stack([curve(0.9, 0.9, 0.9, 0.9, 0.9)] * 100)
        assert paired_ratio(X, e, a, 0.5).cut

    def test_n_lin(self):
        assert n_lin(torch.tensor([0.0, 1.0, 3.3, 3.3, 30.0]), (0.8, 3.2, 25.6)) == [
            4,
            3,
            1,
        ]
