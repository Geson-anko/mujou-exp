r"""``exp.lib.midpoint``: the midpoint's error against its theory.

The oracles (``plans/jepa.md``, V-d0a to V-d0e in small): a line at constant
speed has $G_{\mathrm{mid}} = 0$; a rotation, whitened by four points of
orthogonal phases (an exactly isotropic covariance), has per triple $G =
(1 - \cos\omega n)/2$ and $s = 1 - \cos\omega n$; an invertible affine map
($k = D$, $\epsilon = 0$) and the reversal of time change no value; the
constant, the $D < k$ and the rank-deficient codes are not scoreable. The
closed forms: Fable's identity $4 G_{\mathrm{mid}} = 2 E_\mu[(1-c)^2] /
E_\mu[1-c] = 2s + 2 \mathrm{Var}_\mu[c] / s$ on 7 random spectra and 6
lags; OU's $G = (2 + s)/4$; the permutation's expectation against the exact
mean over all 720 permutations of 6 values; white noise, the random walk
and OU against Monte Carlo means (within 4.5 standard errors); and the
injection's $G_\nu$ against the noise's expected terms added to a small
code's sums. Per triple: $\nu_t / \delta_t = (1 - \rho_t)/2$ and a win at
$q$ is $\rho_t > 1 - 2q$ (a triple of no motion loses) on hand-made
triples; the literal extrapolation's win $4\nu_t < \lVert b \rVert^2$ by
hand; the blocks' sums add up; the pooled $G$ made of the squared
displacements $D$ is the triples' $G$ of the same set; and the generalized
eigenproblem gives each direction's pooled $G$ as its eigenvalue (the
Rayleigh quotient), the slow band's directions first.
"""

import itertools
import math

import pytest
import torch

from exp.lib.midpoint import (
    Whitener,
    asym_wins,
    g_mid_of,
    injected,
    injection_for,
    mid_covariances,
    pair_sq,
    permutation_terms,
    random_walk_terms,
    rho,
    slow_directions,
    spectral_rho,
    spread,
    stationary_terms,
    triple_terms,
    win_sums,
    wins,
)
from exp.lib.synth_series import ou, random_walk, spectral, white

F64 = torch.float64


def pooled(w: torch.Tensor, lag: int) -> tuple[float, float]:
    """$G_{\\mathrm{mid}}$ and $s$ of all the triples of ``w``."""
    centers = torch.arange(lag, len(w) - lag)
    t = triple_terms(w, centers, lag)
    return float(t.num.sum() / t.den.sum()), float(t.den.mean()) / (2 * spread(w))


class TestTriples:
    def test_a_line_has_no_midpoint_error(self):
        b = torch.randn(64, generator=torch.Generator().manual_seed(0), dtype=F64)
        z = torch.arange(500, dtype=F64)[:, None] * b
        for lag in (1, 8, 64):
            t = triple_terms(z, torch.arange(lag, 500 - lag), lag)
            assert float(t.num.max()) <= 1e-12 * float(t.den.min())

    @pytest.mark.parametrize("lag", [1, 8, 32, 100])
    def test_a_rotation_whitened_isotropically(self, lag):
        period = 200
        omega = 2 * math.pi / period
        phase = torch.arange(4, dtype=F64) * math.pi / 2  # C = I / 2 exactly
        ref = torch.stack([phase.cos(), phase.sin()], 1)
        t = torch.arange(3 * period, dtype=F64) * omega
        wh = Whitener.fit(ref, 2)
        w = wh.white(torch.stack([t.cos(), t.sin()], 1))
        terms = triple_terms(w, torch.arange(lag, len(w) - lag), lag)
        c = math.cos(omega * lag)
        torch.testing.assert_close(terms.num / terms.den,
                                   torch.full_like(terms.num, (1 - c) / 2),
                                   rtol=1e-10, atol=1e-12)  # fmt: skip
        _, s = pooled(w, lag)
        assert s == pytest.approx(1 - c, rel=1e-10)

    def test_an_invertible_affine_map_changes_nothing(self):
        g = torch.Generator().manual_seed(1)
        ref, ev = ou(800, 4, 0.9, g), ou(600, 4, 0.9, g)
        q1, _ = torch.linalg.qr(torch.randn(4, 4, generator=g, dtype=F64))
        q2, _ = torch.linalg.qr(torch.randn(4, 4, generator=g, dtype=F64))
        a = q1 @ torch.diag(torch.logspace(0, 3, 4, dtype=F64)) @ q2
        c = torch.randn(4, generator=g, dtype=F64) * 100
        plain = Whitener.fit(ref, 4, c_eps=0.0)
        moved = Whitener.fit(ref @ a.T + c, 4, c_eps=0.0)
        for lag in (1, 5, 20):
            got = pooled(moved.white(ev @ a.T + c), lag)
            want = pooled(plain.white(ev), lag)
            assert got == pytest.approx(want, rel=1e-8)

    def test_reversing_time_changes_nothing(self):
        w = ou(400, 6, 0.95, torch.Generator().manual_seed(2))
        for lag in (1, 7, 30):
            centers = torch.arange(lag, 400 - lag, 3)
            f = triple_terms(w, centers, lag)
            r = triple_terms(w.flip(0), 399 - centers, lag)
            assert float(r.num.sum() / r.den.sum()) == pytest.approx(
                float(f.num.sum() / f.den.sum()), rel=1e-12
            )
            torch.testing.assert_close(r.a2, f.b2, rtol=1e-12, atol=0)


class TestScoreable:
    def test_a_constant_is_not(self):
        assert not Whitener.fit(torch.ones(50, 8), 4).scoreable

    def test_fewer_dimensions_than_k_are_not(self):
        wh = Whitener.fit(
            torch.randn(50, 8, generator=torch.Generator().manual_seed(0)), 16
        )
        assert not wh.scoreable and "D = 8 < k = 16" in str(wh.why)

    def test_a_rank_deficient_code_is_not(self):
        g = torch.Generator().manual_seed(3)
        z = torch.randn(100, 3, generator=g, dtype=F64) @ torch.randn(
            3, 8, generator=g, dtype=F64
        )
        assert not Whitener.fit(z, 4).scoreable
        assert Whitener.fit(z, 3).scoreable

    def test_the_whitened_reference_has_variance_lambda_over_lambda_eps(self):
        g = torch.Generator().manual_seed(4)
        z = torch.randn(300, 5, generator=g, dtype=F64) * torch.tensor(
            [5.0, 3, 2, 1, 0.1]
        )
        wh = Whitener.fit(z, 5, c_eps=1e-3)
        var = wh.white(z).var(0, unbiased=False)
        torch.testing.assert_close(
            var, wh.values / (wh.values + wh.eps), rtol=1e-12, atol=0
        )


class TestClosedForms:
    LAGS = (1, 2, 4, 8, 16, 32)

    @pytest.mark.parametrize("seed", range(7))
    def test_fables_identity(self, seed):
        g = torch.Generator().manual_seed(seed)
        omegas = torch.rand(7, generator=g, dtype=F64) * math.pi
        weights = torch.rand(7, generator=g, dtype=F64)
        weights /= weights.sum()
        for lag in self.LAGS:
            c = torch.cos(omegas * lag)
            s1 = 1 - spectral_rho(omegas, weights, lag)
            s2 = 1 - spectral_rho(omegas, weights, 2 * lag)
            g_ext = 4 * float(g_mid_of(s1, s2))
            first = 2 * float(
                (weights * (1 - c) ** 2).sum() / (weights * (1 - c)).sum()
            )
            var = float((weights * c**2).sum() - (weights * c).sum() ** 2)
            assert g_ext == pytest.approx(first, rel=1e-12, abs=1e-12)
            assert g_ext == pytest.approx(
                2 * float(s1) + 2 * var / float(s1), rel=1e-12
            )
            num, den = stationary_terms(1 - s1, 1 - s2)
            assert float(num / den) == pytest.approx(g_ext / 4, rel=1e-12)

    @pytest.mark.parametrize("r", [0.1, 0.5, 0.97])
    def test_ou_is_two_plus_s_over_four(self, r):
        for n in self.LAGS:
            num, den = stationary_terms(
                torch.tensor(r**n, dtype=F64), torch.tensor(r ** (2 * n), dtype=F64)
            )
            s = 1 - r**n
            assert float(num / den) == pytest.approx((2 + s) / 4, rel=1e-12)

    def test_permutation_against_all_720(self):
        x = torch.tensor(
            [[0.0, 1], [2, -1], [5, 3], [1, 1], [-4, 2], [7, 0]], dtype=F64
        )
        lag, n = 1, 6
        centers = torch.arange(lag, n - lag)
        num = den = 0.0
        perms = list(itertools.permutations(range(n)))
        for p in perms:
            t = triple_terms(x[list(p)], centers, lag)
            num += float(t.num.mean())
            den += float(t.den.mean())
        want = permutation_terms(spread(x), n)
        assert num / len(perms) == pytest.approx(want[0], rel=1e-12)
        assert den / len(perms) == pytest.approx(want[1], rel=1e-12)

    @pytest.mark.parametrize("kind", ["white", "walk", "ou"])
    def test_monte_carlo_means(self, kind):
        # per-triple means of 64 realizations against the closed forms
        g = torch.Generator().manual_seed(5)
        lag, d, n = 4, 8, 400
        centers = torch.arange(lag, n - lag)
        nums, dens = [], []
        for _ in range(64):
            z = {"white": white, "walk": random_walk}.get(kind)
            x = ou(n, d, 0.8, g) if z is None else z(n, d, g)
            t = triple_terms(x, centers, lag)
            nums.append(float(t.num.mean()))
            dens.append(float(t.den.mean()))
        if kind == "walk":
            want = random_walk_terms(lag, d)
        else:
            r = 0.0 if kind == "white" else 0.8
            num, den = stationary_terms(
                torch.tensor(r**lag, dtype=F64), torch.tensor(r ** (2 * lag), dtype=F64)
            )
            want = (d * float(num), d * float(den))
        for got, w in zip((nums, dens), want, strict=True):
            v = torch.tensor(got, dtype=F64)
            assert abs(float((v.mean() - w) / (v.std() / 8))) < 4.5


class TestInjection:
    def test_g_nu_is_the_sums_plus_the_noises_expectation(self):
        w = ou(200, 3, 0.9, torch.Generator().manual_seed(6))
        lag = 5
        t = triple_terms(w, torch.arange(lag, 200 - lag), lag)
        v = spread(w)
        g, s = float(t.num.sum() / t.den.sum()), float(t.den.mean()) / (2 * v)
        nu = 0.3
        m = len(t.num)
        want_g = (float(t.num.sum()) + m * 1.5 * nu * v) / (
            float(t.den.sum()) + m * 2 * nu * v
        )
        want_s = (float(t.den.mean()) + 2 * nu * v) / (2 * v * (1 + nu))
        assert injected(g, s, nu) == pytest.approx((want_g, want_s), rel=1e-12)

    def test_the_injection_raises_g_by_delta(self):
        for g, s, delta in ((0.4, 0.2, 0.01), (0.6, 0.5, 0.08), (0.1, 0.05, 0.3)):
            nu = injection_for(g, s, delta)
            assert nu is not None and nu > 0
            assert injected(g, s, nu)[0] - g == pytest.approx(delta, rel=1e-12)

    def test_beyond_three_quarters_there_is_none(self):
        assert injection_for(0.7, 0.3, 0.06) is None


def hand_triples() -> torch.Tensor:
    """Four triples (rows 0..11) of 2-d points: straight on (rho 1), a right
    angle (rho 0), back and forth (rho -1), and still (delta 0)."""
    pts = [
        [0, 0], [1, 0], [2, 0],
        [0, 0], [1, 0], [1, 1],
        [0, 0], [1, 0], [0, 0],
        [3, 3], [3, 3], [3, 3],
    ]  # fmt: skip
    return torch.tensor(pts, dtype=F64)


class TestPerTriple:
    def test_rho_and_the_wins_by_hand(self):
        w = hand_triples()
        t = triple_terms(w, torch.tensor([1, 4, 7, 10]), 1)
        r = rho(t)
        assert r[:3].tolist() == [1.0, 0.0, -1.0] and math.isnan(float(r[3]))
        torch.testing.assert_close(t.num[:3] / t.den[:3], (1 - r[:3]) / 2)
        # at 1/4 only straight on wins (rho > 1/2); at 1/2 a right angle ties
        # (rho = 0 is not > 0) and loses; the still triple loses at both
        assert wins(t, 0.25).tolist() == [True, False, False, False]
        assert wins(t, 0.5).tolist() == [True, False, False, False]
        assert wins(t, 0.75).tolist() == [True, True, False, False]

    def test_the_literal_extrapolation_by_hand(self):
        # a = (1, 0); b = (1, 0): 4 nu = 0 < 1; b = (0, 1): |b - a|^2 = 2 > 1
        w = hand_triples()
        t = triple_terms(w, torch.tensor([1, 4, 7, 10]), 1)
        assert asym_wins(t).tolist() == [True, False, False, False]

    def test_the_blocks_sums_add_up(self):
        w = ou(300, 3, 0.9, torch.Generator().manual_seed(0))
        u = torch.arange(5, 295)
        t = triple_terms(w, u, 5)
        blocks = u // 100
        got = win_sums(t, blocks, 3, (0.25, 0.5))
        assert float(got["num_b"].sum()) == pytest.approx(float(t.num.sum()), rel=1e-12)
        assert float(got["n_b"].sum()) == len(u)
        assert float(got["win25_b"].sum()) == float(wins(t, 0.25).sum())
        assert float(got["win50_b"].sum()) == float(wins(t, 0.5).sum())
        assert float(got["still_b"].sum()) == 0.0

    @pytest.mark.parametrize("lag", [1, 4, 16])
    def test_the_pooled_g_of_d_is_the_triples(self, lag):
        w = ou(400, 5, 0.95, torch.Generator().manual_seed(1))
        u = torch.arange(lag, 400 - lag)
        t = triple_terms(w, u, lag)
        far = pair_sq(w, u - lag, 2 * lag).sum()  # |a + b|^2
        den = (pair_sq(w, u - lag, lag).sum() + pair_sq(w, u, lag).sum()) / 2
        assert float(den) == pytest.approx(float(t.den.sum()), rel=1e-12)
        g_of_d = 1 - far / (4 * den)
        assert float(g_of_d) == pytest.approx(
            float(t.num.sum() / t.den.sum()), rel=1e-12
        )


class TestDirections:
    def test_each_directions_g_is_its_eigenvalue_the_band_first(self):
        g = torch.Generator().manual_seed(3)
        n, slow = 4000, 3
        band = spectral(n, torch.linspace(0.02, 0.03, slow, dtype=F64), g)
        x = torch.cat([band, white(n, 4, g)], 1)  # slow then white columns
        q, _ = torch.linalg.qr(torch.randn(7, 7, generator=g, dtype=F64))
        w = x @ q.T  # mixed
        u = torch.arange(8, n - 8)
        c_nu, c_delta = mid_covariances(w, u, 8)
        t = triple_terms(w, u, 8)
        assert float(c_nu.trace()) == pytest.approx(float(t.num.sum()), rel=1e-12)
        assert float(c_delta.trace()) == pytest.approx(float(t.den.sum()), rel=1e-12)
        lam, v = slow_directions(c_nu, c_delta)
        assert bool((lam[1:] >= lam[:-1]).all())
        torch.testing.assert_close(v.T @ c_delta @ v, torch.eye(7, dtype=F64))
        for i in range(7):
            ti = triple_terms(w @ v[:, i : i + 1], u, 8)
            assert float(ti.num.sum() / ti.den.sum()) == pytest.approx(
                float(lam[i]), rel=1e-9
            )
        # the slow directions lie in the band's columns, not the white ones
        back = q.T @ v[:, :slow]  # in the unmixed columns
        back = back / torch.linalg.vector_norm(back, dim=0)
        assert float(back[slow:].square().sum(0).max()) < 1e-3
        assert float(lam[slow - 1]) < 0.05 < 0.6 < float(lam[slow])
