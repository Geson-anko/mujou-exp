"""``exp.lib.readout``: the exact linear relation recovered, independent
targets read as nothing, the calibration's choice, and the constant
prediction's identification at chance."""

import pytest
import torch

from exp.lib.metrics import identification
from exp.lib.readout import fit_ridge, predict, r2, select_ridge


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
