"""``exp.lib.readout``: the exact linear relation recovered, independent
targets read as nothing, the calibration's choice, and the constant
prediction's identification at chance; ``centered_r2``'s values of the
targets, a constant and permutations, and Eckart-Young's ``rank_bound``."""

import math

import pytest
import torch

from exp.lib.metrics import identification
from exp.lib.readout import (
    centered_r2,
    centered_terms,
    fit_ridge,
    predict,
    r2,
    rank_bound,
    select_ridge,
)


def data(n: int, d: int = 5, k: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    b = torch.randn(d, k, generator=g, dtype=torch.float64)
    return x, b, torch.randn(k, generator=g, dtype=torch.float64)


class TestRidge:
    def test_a_noiseless_linear_relation_is_recovered(self):
        x, b, c = data(50)
        fit = fit_ridge(x, x @ b + c, 1e-12)
        torch.testing.assert_close(fit.b, b, rtol=0, atol=1e-8)
        xt, _, _ = data(20, seed=1)
        assert r2(predict(fit, xt), xt @ b + c) == pytest.approx(1.0, abs=1e-10)

    def test_independent_targets_read_nothing_under_a_large_ridge(self):
        x, _, _ = data(200)
        g = torch.Generator().manual_seed(2)
        y = torch.randn(200, 4, generator=g)
        fit = fit_ridge(x, y, 1e6)
        held = torch.randn(200, 4, generator=g)
        assert r2(predict(fit, x), held) <= 0.01

    def test_the_calibrations_best_scale_is_chosen(self):
        x, b, c = data(30, d=20)
        g = torch.Generator().manual_seed(3)
        y = x @ b + c + 3 * torch.randn(30, 3, generator=g, dtype=torch.float64)
        xc, _, _ = data(30, d=20, seed=4)
        yc = xc @ b + c
        scales = (1e-8, 1e-2, 1.0)
        best = select_ridge(x, y, xc, yc, scales)
        scores = [r2(predict(fit_ridge(x, y, s), xc), yc) for s in scales]
        want = fit_ridge(x, y, scales[max(range(3), key=lambda i: scores[i])])
        assert best.alpha == want.alpha

    def test_a_constant_input_identifies_at_exactly_chance(self):
        x, b, c = data(16)
        y = x @ b + c
        fit = fit_ridge(x, y, 1e-3)
        mean_in = x.mean(0, keepdim=True).expand(16, -1)  # E1's mean stage
        # a centre other than the fit set's mean (fill_eval's: the held-out
        # grid's), so the constant prediction has a direction
        out = identification(predict(fit, mean_in), y, center=torch.zeros(y.shape[1]))
        assert out["top1"] == pytest.approx(1 / 16)


class TestCenteredR2:
    """``centered_r2``: 1 for the targets themselves, exactly 0 for a
    constant, blind to a common shift and to a common positive scale, and
    the theory's value for a permutation of the targets."""

    def test_the_targets_are_1_and_a_constant_exactly_0(self):
        g = torch.Generator().manual_seed(5)
        y = torch.randn(40, 6, generator=g, dtype=torch.float64)
        assert centered_r2(y, y) == 1.0
        const = torch.full((40, 6), 3.7, dtype=torch.float64)
        assert centered_r2(const, y) == 0.0
        e, v = centered_terms(const, y)
        assert torch.equal(e, v)

    def test_a_shift_and_a_common_scale_change_nothing(self):
        g = torch.Generator().manual_seed(6)
        y = torch.randn(30, 4, generator=g, dtype=torch.float64)
        p = y + 0.7 * torch.randn(30, 4, generator=g, dtype=torch.float64)
        m = centered_r2(p, y)
        shift = torch.randn(4, generator=g, dtype=torch.float64)
        assert centered_r2(p + shift, y) == pytest.approx(m, abs=1e-12)
        assert centered_r2(p, y + shift) == pytest.approx(m, abs=1e-12)
        assert centered_r2(2.5 * p, 2.5 * y) == pytest.approx(m, abs=1e-12)

    def test_permutations_average_minus_1_and_derangements_less(self):
        # m(pi) = -1 + 2 sum_i <y~_pi(i), y~_i> / |Y~|^2; a uniform pi has
        # E y~_pi(i) = 0, one without fixed points -y~_i / (n - 1)
        n, draws = 12, 10000
        g = torch.Generator().manual_seed(7)
        y = torch.randn(n, 3, generator=g, dtype=torch.float64)
        uniform, deranged = [], []
        while len(deranged) < draws:
            pi = torch.randperm(n, generator=g)
            m = centered_r2(y[pi], y)
            if len(uniform) < draws:
                uniform.append(m)
            if not bool((pi == torch.arange(n)).any()):
                deranged.append(m)
        for got, want in ((uniform, -1.0), (deranged, -1 - 2 / (n - 1))):
            x = torch.tensor(got, dtype=torch.float64)
            se = float(x.std()) / math.sqrt(draws)
            assert abs(float(x.mean()) - want) < 4 * se


class TestRankBound:
    def test_no_prediction_of_rank_r_passes_it_and_the_svd_reaches_it(self):
        g = torch.Generator().manual_seed(8)
        y = torch.randn(60, 10, generator=g, dtype=torch.float64)
        y = y @ torch.randn(10, 10, generator=g, dtype=torch.float64)
        r = 3
        bound = rank_bound(y, r)
        for _ in range(20):
            a = torch.randn(60, r, generator=g, dtype=torch.float64)
            b = torch.randn(r, 10, generator=g, dtype=torch.float64)
            assert centered_r2(a @ b, y) <= bound
        yc = y - y.mean(0)
        u, s, vh = torch.linalg.svd(yc, full_matrices=False)
        best = u[:, :r] @ torch.diag(s[:r]) @ vh[:r]
        assert centered_r2(best, y) == pytest.approx(bound, abs=1e-12)


class TestSelectRidgeScore:
    def test_another_score_chooses_its_own_best(self):
        x, b, c = data(30, d=20)
        g = torch.Generator().manual_seed(9)
        y = x @ b + c + 3 * torch.randn(30, 3, generator=g, dtype=torch.float64)
        xc, _, _ = data(30, d=20, seed=4)
        yc = xc @ b + 5.0  # a shift only centred_r2 forgives
        scales = (1e-8, 1e-2, 1.0)
        best = select_ridge(x, y, xc, yc, scales, score=centered_r2)
        scores = [centered_r2(predict(fit_ridge(x, y, s), xc), yc) for s in scales]
        want = fit_ridge(x, y, scales[max(range(3), key=lambda i: scores[i])])
        assert best.alpha == want.alpha
