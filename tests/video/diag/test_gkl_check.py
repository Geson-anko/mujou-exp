import json
import math

import pytest
import torch

from exp.lib.gauss_kl import plugin_bias
from exp.video.diag.gkl_check import LAWS, bench, floor, main, nongauss, sample


@pytest.mark.parametrize("law", LAWS)
def test_every_law_has_mean_0_and_covariance_i(law):
    """200000 rows of 3: the sample mean within 5 standard errors of 0 and
    the sample covariance within 0.03 of I (a few standard errors of a
    variance of a law of finite fourth moments)."""
    x = sample(law, 200_000, 3, torch.Generator().manual_seed(0))
    assert x.dtype == torch.float64 and x.shape == (200_000, 3)
    assert (x.mean(0).abs() <= 5 * x.std(0) / math.sqrt(len(x))).all()
    torch.testing.assert_close(torch.cov(x.T), torch.eye(3).double(), atol=0.03, rtol=0)


def test_the_laws_differ_in_their_fourth_moments():
    """Coordinate kurtosis: 3 for the Gaussian, 6 for the Laplace, and for
    the elliptical t of nu = 5 3 (nu - 2) / (nu - 4) = 9; the t's radial
    factor is shared, so its squared coordinates are correlated (0 for the
    other two, independent coordinates)."""
    g = torch.Generator().manual_seed(1)
    kurt = {}
    for law in LAWS:
        x = sample(law, 400_000, 2, g)
        kurt[law] = float((x**4).mean())
        r = torch.corrcoef((x**2).T)[0, 1]
        assert (float(r) > 0.1) == (law == "student")
    assert abs(kurt["gauss"] - 3) < 0.1 and abs(kurt["laplace"] - 6) < 0.3
    assert kurt["student"] > 6


def test_gaussian_rows_give_an_unbiased_precision():
    """The Wishart identity: tr E P^ / D = 1 for Gaussian rows, within 5
    standard errors; the plug-in's factor (n - 1) / (n - D - 2) is not."""
    out = nongauss("gauss", dim=4, rows=30, reps=2000, seed=2)
    assert abs(out["trace_ratio"] - 1) <= 5 * out["se"]
    assert out["plugin_ratio"] == 29 / 24
    assert out["plugin_ratio"] - 1 > 10 * out["se"]


def test_the_floor_is_the_plugin_bias_per_coordinate():
    out = floor(dim=8, rows=40)
    assert out["plugin_bias_per_coordinate"] == plugin_bias(40, 8, 8.0) / 8 > 0


def test_the_bench_times_every_part():
    out = bench(dim=8, rows=40, device="cpu", warmup=1, repeats=2)
    for k in ("scatter_ms", "cholesky_ms", "inverse_ms", "total_ms"):
        assert out[k] > 0


def test_the_cli_prints_the_floor(capsys):
    main(["floor", "--dim", "8", "--rows", "40"])
    assert json.loads(capsys.readouterr().out) == floor(dim=8, rows=40)
