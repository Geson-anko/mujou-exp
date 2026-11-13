"""The AR(1) diagnosis of the Gaussian moment matching and SIGReg (M17,
2026-11-13): the stream, the learning rates of one time constant, the
gradients against their oracles, and c* = 1 for an iid stream."""

import math

import pytest
import torch

from exp.lib.gauss_kl import GaussMoments, gauss_kl_grad
from exp.lib.sigreg import sigreg
from exp.video.diag.gkl_ar1 import (
    _grad,
    ar1_coefficients,
    eta_of,
    report,
    run,
    sigreg_curvature,
)
from exp.video.diag.ssig import expected_s
from exp.video.jepa.nodes import PastPoints


@pytest.mark.parametrize("tau", [0.0, 1.0, 10.0])
def test_the_stream_is_stationary_of_correlation_e_to_minus_lag_over_tau(tau):
    """a^2 + noise^2 = 1 keeps N(0, 1); 200000 steps from N(0, 1) have a
    variance of 1 and a lag-1 correlation of exp(-1 / tau) (0 for iid)
    within 0.02 (about 4 sd of these estimates at tau = 10)."""
    a, noise = ar1_coefficients(tau)
    assert a * a + noise * noise == pytest.approx(1)
    assert a == (math.exp(-1 / tau) if tau else 0.0)
    g = torch.Generator().manual_seed(0)
    e = torch.randn(200_000, generator=g, dtype=torch.float64)
    s = torch.empty_like(e)
    s[0] = e[0]
    for t in range(1, len(e)):
        s[t] = a * s[t - 1] + noise * e[t]
    assert float(s.var()) == pytest.approx(1, abs=0.02)
    r1 = float((s[1:] * s[:-1]).mean() / s.var())
    assert r1 == pytest.approx(a, abs=0.02)


def test_the_learning_rates_give_one_time_constant():
    """gauss: L = F / m = (w^2 - 1 - 2 log w) / 2, L''(1) = 2, eta = m / (2
    tau_learn); sigreg: L''(1) of E S(w^2) / n by finite differences of the
    closed form."""
    assert eta_of("gauss", dim=16, buffer=4096, tau_learn=1e5) == 16 / 2e5
    n, h = 4097, 1e-4

    def f(w: float) -> float:
        return float(expected_s(torch.tensor(w * w, dtype=torch.float64), n)) / n

    fd = (f(1 + h) - 2 * f(1) + f(1 - h)) / h**2
    assert sigreg_curvature(n) == pytest.approx(fd, rel=1e-4)
    eta = eta_of("sigreg", dim=16, buffer=4096, tau_learn=1e5)
    assert eta == pytest.approx(16 / (fd * 1e5), rel=1e-4)


def test_the_gauss_step_of_exact_moments_is_w_less_1_over_w():
    """z = w s with the exact moments (mu 0, Sigma w^2 I): g = (w - 1 / w)
    s, so E[g s^T] / m = (w - 1 / w) / m I, the linearization of the
    learning rate (rate 2 eta / m at w = 1)."""
    w, m = 1.3, 5
    s = torch.randn(
        7, m, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    one = torch.ones((), dtype=torch.float64)
    exact = GaussMoments(
        torch.zeros(m, dtype=torch.float64),
        torch.eye(m, dtype=torch.float64) / w**2,
        one,
        one,
    )
    torch.testing.assert_close(gauss_kl_grad(w * s, exact), (w - 1 / w) * s)


def test_the_sigreg_step_is_the_gradient_of_sigreg_of_z_and_the_buffer():
    """The step's g is d/dz of exp.lib.sigreg.sigreg of z with the buffer
    along the directions it draws (redrawn from the same generator)."""
    m = 4
    points = PastPoints(
        capacity=10,
        every=1,
        streams=1,
        width=m,
        dtype=torch.float64,
        directions=0,
        device="cpu",
    )
    g = torch.Generator().manual_seed(0)
    one = torch.ones(1, dtype=torch.bool)
    for _ in range(8):
        points.next_frame(None, g)
        points.keep(torch.randn(1, m, generator=g, dtype=torch.float64), one)
    z = torch.randn(m, dtype=torch.float64)
    got = _grad("sigreg", z, points, torch.Generator().manual_seed(3))
    a = torch.randn(m, m, generator=torch.Generator().manual_seed(3))
    dirs = a / a.norm(dim=1, keepdim=True)
    leaf = z.float().requires_grad_()
    (want,) = torch.autograd.grad(sigreg(leaf, points.filled.float(), dirs), [leaf])
    torch.testing.assert_close(got, want.double())


def test_an_iid_stream_has_c_star_1_and_stays_at_the_optimum():
    """iid rows: E P^ = Sigma^-1 and the rows are independent of s_t, so K =
    I and c* = 1 (the plug-in would give (n - 1) / (n - m - 2) = 1.086
    here); within 0.03, about 5 standard errors of 20000 steps of m = 4.
    Trained from W = I, the mean of Sigma - I stays within 4 of its se."""
    kw = {"dim": 4, "buffer": 64, "steps": 20_000, "tau_learn": 2000.0}
    frozen = run("gauss", 0.0, frozen=True, **kw)
    assert frozen["c_star"] == pytest.approx(1, abs=0.03)
    trained = run("gauss", 0.0, **kw)
    assert abs(trained["mean_dev"]) < 4 * trained["se"]
    again = run("gauss", 0.0, **kw)  # the same seed: the same numbers
    assert again | {"seconds": 0} == trained | {"seconds": 0}


def test_the_report_has_a_row_per_trained_run_and_c_star_over_the_seeds():
    """A row per trained run; the frozen runs of one m, N and tau make one
    row of the mean c* over their seeds and its standard error sd / sqrt(k)
    (c* 1.0 and 1.1: mean 1.05, se 0.05 / sqrt(2) * sqrt(2) = 0.05)."""
    base = {"dim": 4, "buffer": 8, "every": 1, "steps": 10, "seed": 0}
    rows = [
        base
        | {
            "law": "gauss",
            "tau": 1.0,
            "frozen": False,
            "mean_dev": 0.1,
            "se": 0.01,
            "sd_dev": 0.02,
            "eig_min": 0.9,
            "eig_max": 1.2,
            "mu_norm": 0.0,
        },
        base | {"law": "gauss", "tau": 2.0, "frozen": True, "c_star": 1.0},
        base | {"law": "gauss", "tau": 2.0, "frozen": True, "c_star": 1.1, "seed": 1},
    ]
    out = report(rows)
    assert "| gauss | 4 | 8 | 1 | +0.1000 |" in out
    assert "| 4 | 8 | 0.5 | 2 | 0.25 | 2 | 1.0500 | 0.0500 |" in out
