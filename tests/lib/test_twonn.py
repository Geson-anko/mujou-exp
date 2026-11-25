r"""``exp.lib.twonn``: the TwoNN estimate against hand counts and known
dimensions.

The oracles: unequally spaced points on a line whose $\mu_i = r_2/r_1$ are
counted by hand, $\hat d$ and $\hat d_{\mathrm{ML}}$ their formulas
(``plans/jepa.md``, V-d0f's hand count); a duplicate and a point within the
floor merge; the fit's $\lfloor 0.9 N \rfloor$ in exact arithmetic; a
similarity map changes nothing; excluded pairs are not neighbours; a
constant is not defined; and the recovery registered in V-d0f: an isotropic
Gaussian of 4 dimensions within $(3.5, 4.5)$ and the Chebyshev code of the
band $j = 1 .. 64$ at independent phases within $(0.5, 1.5)$ (487 points,
the mean of the seeds 601 to 608).
"""

import math

import pytest
import torch

from exp.lib.synth_series import chebyshev
from exp.lib.twonn import thinned, twonn

F64 = torch.float64
LINE = torch.tensor([0.0, 1, 3, 8, 20, 45, 100], dtype=F64)[:, None]
# each point's (r1, r2) by hand: 0 (1, 3), 1 (1, 2), 3 (2, 3), 8 (5, 7),
# 20 (12, 17), 45 (25, 37), 100 (55, 80)
MUS = [3.0, 2.0, 1.5, 7 / 5, 17 / 12, 37 / 25, 80 / 55]


def by_hand(mus: list[float], n_fit: int) -> tuple[float, float]:
    logs = sorted(math.log(m) for m in mus)
    n = len(mus)
    ys = [-math.log(1 - i / n) for i in range(1, n_fit + 1)]
    d = sum(x * y for x, y in zip(logs, ys, strict=False)) / sum(
        x * x for x in logs[:n_fit]
    )
    return d, (n - 1) / sum(logs)


class TestTwoNN:
    def test_hand_count_on_a_line(self):
        got = twonn(LINE)
        d, d_ml = by_hand(MUS, 6)  # floor(0.9 * 7) = 6
        assert got.n_fit == 6 and got.n_dup == 0
        assert got.d == pytest.approx(d, rel=1e-12)
        assert got.d_ml == pytest.approx(d_ml, rel=1e-12)
        assert got.nn[0].tolist() == [1, 2] and got.nn[6].tolist() == [5, 4]

    def test_a_duplicate_and_a_point_within_the_floor_merge(self):
        radius = 10.0
        x = torch.cat([LINE, LINE[3:4], LINE[5:6] + 1e-6 * radius / 2])
        got = twonn(x, radius=radius)
        assert got.n_dup == 2 and got.n_exact == 1 and got.n_kept == 7
        assert got.d == pytest.approx(by_hand(MUS, 6)[0], rel=1e-12)

    @pytest.mark.parametrize(("n", "fit"), [(10, 9), (30, 27), (487, 438), (488, 439)])
    def test_the_fit_keeps_floor_of_nine_tenths(self, n, fit):
        x = torch.randn(n, 3, generator=torch.Generator().manual_seed(n), dtype=F64)
        assert twonn(x).n_fit == fit

    def test_a_similarity_changes_nothing(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(200, 5, generator=g, dtype=F64)
        q, _ = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=F64))
        a, b = twonn(x), twonn(3.7 * x @ q.T + 11.0)
        assert b.d == pytest.approx(a.d, rel=1e-12)
        assert b.rho == pytest.approx(a.rho, rel=1e-12)

    def test_excluded_pairs_are_not_neighbours(self):
        exclude = torch.zeros(7, 7, dtype=torch.bool)
        exclude[0, 1] = True  # 0's neighbours become 3 and 8
        got = twonn(LINE, exclude=exclude)
        assert got.nn[0].tolist() == [2, 3]
        mus = [8 / 3, *MUS[1:]]
        assert got.d == pytest.approx(by_hand(mus, 6)[0], rel=1e-12)

    def test_a_constant_is_not_defined(self):
        got = twonn(torch.ones(50, 4))
        assert got.d is None and got.n_kept == 1 and got.n_dup == 49

    @pytest.mark.parametrize(
        ("make", "low", "high"), [("gauss4", 3.5, 4.5), ("cheb1", 0.5, 1.5)]
    )
    def test_recovery_of_v_d0f(self, make, low, high):
        ds = []
        for seed in range(601, 609):
            g = torch.Generator().manual_seed(seed)
            if make == "gauss4":
                x = torch.randn(487, 4, generator=g, dtype=F64)
            else:
                theta = 2 * math.pi * torch.rand(487, generator=g, dtype=F64)
                x = chebyshev(theta, torch.arange(1, 65))
            d = twonn(x).d
            assert d is not None
            ds.append(d)
        assert low < sum(ds) / len(ds) < high


class TestThinned:
    def test_offsets_partition_the_grid(self):
        sets = thinned(20, 8)
        assert len(sets) == 8 and sets[3].tolist() == [3, 11, 19]
        assert torch.equal(torch.cat(sets).sort().values, torch.arange(20))
