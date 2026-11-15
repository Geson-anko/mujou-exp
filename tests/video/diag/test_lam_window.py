"""The Gaussian model of the token SIGReg (diagnosis, M17): the closed form of
E S against the integral it comes from and against simulation, and the lambda
windows against the objective they balance."""

import math

import torch

from exp.lib.sigreg import T_POINTS, epps_pulley
from exp.video.diag import lam_window as lw
from exp.video.diag.sigreg_sims import gauss_stacks


def test_S0_is_f_of_1():
    want = math.sqrt(2 * math.pi) * (1 - 1 / math.sqrt(3))
    assert abs(lw.S0 - want) < 1e-12
    assert abs(lw.S0 - 1.0594) < 1e-4
    assert lw.f(0.0) == 0.0
    # one stack of T tokens collapsed: S(1) = T S0 (542 at T = 512)
    assert abs(lw.S_stack(1.0) - 512 * lw.S0) < 1e-9
    assert abs(lw.S_stack(0.0) - lw.S0) < 1e-12


def test_the_closed_form_is_the_integral_of_the_expected_error():
    """N times E|phi_N - phi|^2 against e^{-t^2/2} equals S0 + (1/N)
    sum_{j != l} f(r_jl), up to the error of the trapezoid of epps_pulley (17
    points on [-5, 5]: below 1e-5 for these Gaussians)."""
    t = T_POINTS.double()
    w = torch.exp(-(t**2) / 2)
    r = torch.tensor([[1.0, 0.3, -0.2], [0.3, 1.0, 0.7], [-0.2, 0.7, 1.0]]).double()
    n = len(r)
    off = [(j, k) for j in range(n) for k in range(n) if j != k]
    err = (
        1 / n
        - torch.exp(-(t**2))
        + sum(torch.exp(-(t**2) * (1 - r[j, k])) for j, k in off) / n**2
    )
    integral = float(n * torch.trapezoid(err * w, t))
    closed = lw.S0 + sum(lw.f(float(r[j, k])) for j, k in off) / n
    assert abs(integral - closed) < 1e-5


def test_the_stack_model_matches_simulation():
    """E S of n stacks of the Gaussian model is S0 + (m - 1) f(rho), whatever
    n: the mean of many draws (columns of epps_pulley) within 4 standard
    errors."""
    g = torch.Generator().manual_seed(0)
    m, rho, reps = 8, 0.3, 4000
    for n in (4, 16):
        s = gauss_stacks(n, m, reps, rho, g)  # d = reps: one column per draw
        stat = epps_pulley(s.reshape(n * m, reps).double())
        se = float(stat.std()) / math.sqrt(reps)
        assert abs(float(stat.mean()) - lw.S_stack(rho, m)) < 4 * se


def test_the_windows_balance_the_objective():
    """At lambda_lo, H and C have the same objective (1 - lambda) L + lambda S;
    at lambda_hi, H and I."""

    def J(lam: float, L: float, S: float) -> float:
        return (1 - lam) * L + lam * S

    rho = 0.12
    lo, hi = lw.stack_window(rho)
    S_H = lw.S_stack(rho)
    assert abs(J(lo, 1 - rho, S_H) - J(lo, 0.0, lw.S_stack(1.0))) < 1e-12
    assert abs(J(hi, 1 - rho, S_H) - J(hi, 1.0, lw.S0)) < 1e-12
    w = lw.window(S_H=135.7, S_C=2130.4, S_P=197.6, rho=0.104)
    assert abs(J(w["lo_C"], 0.896, 135.7) - J(w["lo_C"], 0.0, 2130.4)) < 1e-9
    assert abs(J(w["lo_P"], 0.896, 135.7) - J(w["lo_P"], 0.0, 197.6)) < 1e-9
    assert abs(J(w["hi_I"], 0.896, 135.7) - J(w["hi_I"], 1.0, lw.S0)) < 1e-9


def test_the_buffer_model_at_0_and_its_slope():
    """Uncorrelated tokens give S0; the slope at rho = 0 is the derivative of
    S_closed along r_k = rho kappa_k (a central difference)."""
    assert abs(lw.S_closed(0.0, lambda k: 0.0) - lw.S0) < 1e-9

    def kappa(k: int) -> float:
        return 0.9**k

    eps = 1e-4
    up = lw.S_closed(eps, lambda k: eps * kappa(k))
    down = lw.S_closed(-eps, lambda k: -eps * kappa(k))
    slope = lw.S_closed_slope(kappa)
    assert abs((up - down) / (2 * eps) - slope) < 1e-5 * slope


def test_S_P_closed_matches_the_binomial_counts():
    """n_p = B + sum_k Bin(B, q^(k-1)): E n_p (n_p - 1) by simulation of the
    counts gives the closed form's S_P."""
    g = torch.Generator().manual_seed(1)
    B, draws = 4, 100_000
    probs = torch.tensor([lw.q ** (k - 1) for k in range(1, 80)]).double()
    count = torch.full((draws, len(probs)), float(B), dtype=torch.float64)
    n = B + torch.binomial(count, probs.expand(draws, -1), generator=g).sum(1)
    e_nn = float((n * (n - 1)).mean())
    sim = lw.S0 + lw.f(1.0) * lw.T * e_nn / (B * lw.G)
    assert abs(sim - lw.S_P_closed(B)) < 0.01 * lw.S_P_closed(B)
