"""The estimators of the lambda calibration (diagnosis, M17) on tokens of known
correlation: the within-stack rho, the lag correlation of the stack sums, and
the whitening."""

import torch

from exp.video.diag.lam_calib import rho_of, stack_corr, whitening
from exp.video.diag.sigreg_sims import gauss_stacks
from exp.video.jepa.nodes import draw_directions


def test_rho_of_the_gaussian_stacks_and_of_the_position_alone():
    """The Gaussian model gives rho and no share of the position means; the
    tokens of the position alone (the same t points in every stack) give
    rho = -1/(t - 1) (the deviations of a stack sum to 0) and a share of 1."""
    g = torch.Generator().manual_seed(0)
    a = draw_directions(64, 4, g).double()
    s = gauss_stacks(2000, 16, 4, 0.3, g).double()
    r = rho_of(s, a)
    assert abs(r["rho"] - 0.3) < 0.03
    assert r["pos_share"] < 0.01
    t = 16
    pos = torch.randn(1, t, 4, generator=g).double().expand(50, -1, -1)
    r = rho_of(pos, a)
    assert abs(r["rho"] + 1 / (t - 1)) < 1e-9
    assert abs(r["pos_share"] - 1) < 1e-9


def test_stack_corr_of_a_common_component_carried_over_frames():
    """u_{f,t} = sqrt(rho) c_f + sqrt(1 - rho) e_{f,t}, c an AR(1) of
    coefficient phi over the frames of each stream: within-stack rho, lag k
    rho phi^k (the population moments mu = 0, var = 1)."""
    g = torch.Generator().manual_seed(1)
    F, B, T, K, rho, phi = 400, 64, 32, 4, 0.2, 0.8
    c = torch.empty(F, B, 1, dtype=torch.float64)
    c[0] = torch.randn(B, 1, generator=g)
    for f in range(1, F):
        eta = torch.randn(B, 1, generator=g)
        c[f] = phi * c[f - 1] + (1 - phi**2) ** 0.5 * eta
    e = torch.randn(F, B, T, generator=g).double()
    u = rho**0.5 * c + (1 - rho) ** 0.5 * e  # (F, B, T): one direction
    S1, S2 = u.sum(2, keepdim=True), u.pow(2).sum(2, keepdim=True)  # (F, B, 1)
    V = torch.ones(F, B, dtype=torch.bool)
    V[::7, 3] = False  # missing stacks are left out
    mu, var = torch.zeros(1).double(), torch.ones(1).double()
    within, lags = stack_corr(S1, S2, V, mu, var, T, K)
    assert abs(within - rho) < 0.02
    for k, lag in enumerate(lags, 1):
        assert abs(lag - rho * phi**k) < 0.03


def test_whitening_gives_the_identity_covariance():
    g = torch.Generator().manual_seed(2)
    x = (
        torch.randn(500, 3, generator=g).double()
        @ torch.tensor([[2.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.3, -0.2, 1.5]]).double()
        + 4.0
    )
    mu, W = whitening(x)
    y = (x - mu) @ W.T
    assert torch.allclose(torch.cov(y.T), torch.eye(3).double(), atol=1e-9)
    assert torch.allclose(y.mean(0), torch.zeros(3).double(), atol=1e-9)
