"""The SIGReg simulations (diagnosis, M17): the position-only tokens against
the closed form of E S for repeated points, and the Gaussian stacks' moments."""

import math

import torch

from exp.lib.sigreg import sigreg
from exp.video.diag import lam_window as lw
from exp.video.diag.sigreg_sims import gauss_stacks, position_only


def test_position_only_matches_the_repeated_points_closed_form():
    """P: t points each n_p times among N samples, n_p = B + Bin(cap, 1/t),
    pairs of one point have r = 1, others r = 0, so E S = S0 + f(1)/N sum_p
    E n_p (n_p - 1); the mean over draws of the points within 4 standard
    errors."""
    torch.manual_seed(0)
    t, d, B, cap, draws = 8, 2, 3, 40, 600
    vals = []
    for _ in range(draws):
        _, dirs, cur, buf = position_only(t, d, B, cap)
        vals.append(float(sigreg(cur.double(), buf.double(), dirs.double())))
    v = torch.tensor(vals)
    N = B * t + cap
    p = 1 / t
    e_nn = (B + cap * p) ** 2 + cap * p * (1 - p) - (B + cap * p)
    want = lw.S0 + lw.f(1.0) * t * e_nn / N
    se = float(v.std()) / math.sqrt(draws)
    assert abs(float(v.mean()) - want) < 4 * se


def test_gauss_stacks_have_unit_marginals_and_the_stack_correlation():
    g = torch.Generator().manual_seed(2)
    rho = 0.4
    s = gauss_stacks(20_000, 2, 3, rho, g).double()  # (n, 2 tokens, 3)
    assert torch.allclose(s.var((0, 1)), torch.ones(3).double(), atol=0.03)
    within = (s[:, 0] * s[:, 1]).mean(0)  # E s_1 s_2 = rho
    assert torch.allclose(within, torch.full((3,), rho).double(), atol=0.03)
    across = (s[:-1, 0] * s[1:, 0]).mean(0)  # other stacks: 0
    assert torch.allclose(across, torch.zeros(3).double(), atol=0.03)
