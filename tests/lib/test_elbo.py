import math

import pytest
import torch
from torch.distributions import MultivariateNormal, Normal, kl_divergence

from exp.lib.elbo import (
    bits_per_dim,
    dequant_var,
    dequantize,
    gauss_kl,
    gauss_nll,
    ppca_fit,
    ppca_nll,
    sigma_gap,
)

F64 = torch.float64


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestKL:
    def test_the_closed_form_is_torch_distributions(self):
        g = _gen()
        mu = torch.randn(5, 7, generator=g, dtype=F64)
        logvar = torch.randn(5, 7, generator=g, dtype=F64)
        want = kl_divergence(Normal(mu, (0.5 * logvar).exp()), Normal(0.0, 1.0))
        torch.testing.assert_close(gauss_kl(mu, logvar), want, rtol=1e-12, atol=0)

    def test_the_closed_form_is_the_monte_carlo_mean_of_log_q_over_p(self):
        """KL = E_q[log q(z) - log p(z)]: the mean over 2^20 draws of z = mu
        + s eps is within 5 standard errors of the closed form."""
        mu = torch.tensor([0.0, 1.5, -0.7], dtype=F64)
        logvar = torch.tensor([0.0, -2.0, 1.0], dtype=F64)
        s = (0.5 * logvar).exp()
        eps = torch.randn(2**20, 3, generator=_gen(1), dtype=F64)
        z = mu + s * eps
        term = Normal(mu, s).log_prob(z) - Normal(0.0, 1.0).log_prob(z)
        mean, se = term.mean(0), term.std(0) / math.sqrt(len(z))
        got = gauss_kl(mu, logvar)
        assert ((mean - got).abs() < 5 * se + 1e-12).all(), (mean, got, se)
        assert float(got[0]) == 0.0  # q = p


class TestGaussNLL:
    def test_is_the_sum_of_the_normal_log_probs(self):
        g = _gen()
        y = torch.rand(4, 2, 3, generator=g, dtype=F64)
        mean = torch.rand(4, 2, 3, generator=g, dtype=F64)
        log_sigma = torch.tensor(-1.3, dtype=F64)
        want = -Normal(mean, log_sigma.exp()).log_prob(y).flatten(1).sum(1)
        torch.testing.assert_close(
            gauss_nll(y, mean, log_sigma), want, rtol=1e-12, atol=0
        )

    def test_sigma_star_is_the_mean_squared_residual(self):
        """d/d log sigma of the sum over the samples vanishes at sigma*^2 =
        sum ||y - y_hat||^2 / (n P), and any other sigma costs exactly
        sigma_gap a sample."""
        g = _gen(2)
        y = torch.rand(6, 10, generator=g, dtype=F64)
        mean = y + 0.2 * torch.randn(6, 10, generator=g, dtype=F64)
        star_sq = float((y - mean).square().mean())
        ls = torch.tensor(0.5 * math.log(star_sq), dtype=F64, requires_grad=True)
        (grad,) = torch.autograd.grad(gauss_nll(y, mean, ls).sum(), ls)
        assert abs(float(grad)) < 1e-9
        best = float(gauss_nll(y, mean, ls.detach()).mean())
        for sigma_sq in (0.3 * star_sq, 2.0 * star_sq, 10 * star_sq):
            other = torch.tensor(0.5 * math.log(sigma_sq), dtype=F64)
            got = float(gauss_nll(y, mean, other).mean()) - best
            assert got == pytest.approx(sigma_gap(star_sq, sigma_sq, 10), rel=1e-10)
            assert got > 0
        assert sigma_gap(star_sq, star_sq, 10) == 0.0


class TestBitsPerDim:
    def test_the_uniform_density_of_y_is_8_bits(self):
        """p(y) = 1 on [0, 1)^P (-log p = 0) is the uniform law of the 256
        integers: exactly 8 bits a value."""
        assert bits_per_dim(0.0, 100) == 8.0

    def test_the_best_gaussian_of_uniform_integers(self):
        """Dequantized uniform integers are uniform on [0, 1): of variance
        1/12, whose best Gaussian costs 8 + 1/2 log2(2 pi e / 12) bits, above
        the 8 of the true law (the Gaussian is not the uniform density); the
        mean nll of 2^16 points is within 5 standard errors of it."""
        g = _gen(3)
        n, p = 2**14, 4
        x = torch.randint(0, 256, (n, p), generator=g)
        y = dequantize(x, torch.rand(n, p, generator=g)).double()
        ls = torch.tensor(0.5 * math.log(1 / 12), dtype=F64)
        nll = gauss_nll(y, torch.full_like(y, 0.5), ls)
        want = 8 + 0.5 * math.log2(2 * math.pi * math.e / 12)
        got = bits_per_dim(float(nll.mean()), p)
        se = float(nll.std()) / math.sqrt(n) / (p * math.log(2))
        assert got == pytest.approx(want, abs=5 * se)
        assert want > 8

    def test_dequantize_and_its_variance(self):
        x = torch.tensor([0, 255], dtype=torch.uint8)
        u = torch.tensor([0.5, 0.25])
        torch.testing.assert_close(dequantize(x, u), torch.tensor([0.5, 255.25]) / 256)
        u = torch.rand(2**20, generator=_gen(4), dtype=F64)
        assert float((u / 256).var()) == pytest.approx(dequant_var(), rel=1e-2)


def _spectrum_data(n: int, scales: list[float], seed: int) -> torch.Tensor:
    g = _gen(seed)
    p = len(scales)
    q, _ = torch.linalg.qr(torch.randn(p, p, generator=g, dtype=F64))
    sd = torch.tensor(scales, dtype=F64).sqrt()
    return (torch.randn(n, p, generator=g, dtype=F64) * sd) @ q.T + 0.3


def _moments(y: torch.Tensor, m: torch.Tensor, u: torch.Tensor):
    c = y - m
    a = c @ u
    r = c - a @ u.T
    return a.square().mean(0), float(r.square().sum(1).mean())


class TestPPCA:
    @pytest.mark.parametrize("k", [0, 1, 3])
    def test_the_closed_form_is_the_explicit_covariance(self, k):
        """The mean -log p of held-out points of the ML pPCA of training
        points (P = 6) is MultivariateNormal's of C = U diag(lambda) U^T +
        sigma^2 (I - U U^T), sigma^2 the mean of the discarded eigenvalues
        (k = 0: the isotropic Gaussian of variance tr S / P)."""
        p = 6
        train = _spectrum_data(500, [9, 4, 1, 0.3, 0.2, 0.1], 0)
        test = _spectrum_data(40, [5, 4, 2, 0.5, 0.2, 0.1], 5)
        m = train.mean(0)
        cov = (train - m).T @ (train - m) / len(train)
        evals, vecs = torch.linalg.eigh(cov)
        evals, vecs = evals.flip(0), vecs.flip(1)
        top, sigma_sq = ppca_fit(evals, k, p)
        assert sigma_sq == pytest.approx(float(evals[k:].mean()), rel=1e-12)
        u = vecs[:, :k]
        c = u @ torch.diag(top) @ u.T + sigma_sq * (torch.eye(p, dtype=F64) - u @ u.T)
        want = -float(MultivariateNormal(m, c).log_prob(test).mean())
        coord_ms, residual_ms = _moments(test, m, u)
        got = ppca_nll(top, sigma_sq, p, coord_ms, residual_ms)
        assert got == pytest.approx(want, rel=1e-10)

    def test_more_components_fit_the_training_points_better(self):
        """In sample (a_j^2 averaging lambda_j, ||r||^2 the discarded sum),
        the ML likelihood rises with k."""
        p = 6
        evals = torch.tensor([9, 4, 1, 0.3, 0.2, 0.1], dtype=F64)
        nll = []
        for k in range(p):
            top, s2 = ppca_fit(evals, k, p)
            nll.append(ppca_nll(top, s2, p, evals[:k], float(evals[k:].sum())))
        assert all(a > b for a, b in zip(nll, nll[1:]))

    def test_the_floor_is_the_padded_spectrum_plus_isotropic_noise(self):
        """The eigenvalues of S + v I are those of S plus v, the dims
        beyond the listed ones v: the fit with ``floor`` v of the nonzero
        eigenvalues is the fit of that whole spectrum."""
        evals, p, v = torch.tensor([9, 4, 1], dtype=F64), 6, 0.05
        whole = torch.cat([evals, torch.zeros(p - 3, dtype=F64)]) + v
        for k in range(4):
            top, s2 = ppca_fit(evals, k, p, floor=v)
            want_top, want_s2 = ppca_fit(whole, k, p)
            torch.testing.assert_close(top, want_top, rtol=1e-15, atol=0)
            assert s2 == pytest.approx(want_s2, rel=1e-15)

    def test_k_must_leave_a_direction(self):
        with pytest.raises(ValueError, match="k must be"):
            ppca_fit(torch.ones(3), 3, 3)


class _LinearVAE(torch.nn.Module):
    """The linear VAE of Lucas et al. (2019): mu = A (y - m0) + a, a log
    variance b of each latent independent of y, g(z) = W z + m, and the
    scalar log sigma (fixed when ``sigma_sq`` is given)."""

    def __init__(self, p: int, d: int, m0: torch.Tensor, sigma_sq: float | None):
        super().__init__()
        g = _gen(7)
        self.m0 = m0
        self.enc = torch.nn.Parameter(0.1 * torch.randn(d, p, generator=g, dtype=F64))
        self.a = torch.nn.Parameter(torch.zeros(d, dtype=F64))
        self.b = torch.nn.Parameter(torch.zeros(d, dtype=F64))
        self.dec = torch.nn.Parameter(0.1 * torch.randn(p, d, generator=g, dtype=F64))
        self.m = torch.nn.Parameter(m0.clone())
        ls = 0.0 if sigma_sq is None else 0.5 * math.log(sigma_sq)
        self.log_sigma = torch.nn.Parameter(
            torch.tensor(ls, dtype=F64), requires_grad=sigma_sq is None
        )

    def posterior(self, y):
        mu = (y - self.m0) @ self.enc.T + self.a
        return mu, self.b.expand_as(mu)

    def neg_elbo(self, y, eps):
        """-L of each point with z = mu + s eps (the estimate of a step)."""
        mu, logvar = self.posterior(y)
        z = mu + (0.5 * logvar).exp() * eps
        nll = gauss_nll(y, z @ self.dec.T + self.m, self.log_sigma)
        return nll + gauss_kl(mu, logvar).sum(1)

    def exact_neg_elbo(self, y):
        """-L with E_q taken exactly: the decoder is linear, so E||y - W z -
        m||^2 = ||y - W mu - m||^2 + sum_j s_j^2 ||W_j||^2."""
        mu, logvar = self.posterior(y)
        trace = (logvar.exp() * self.dec.square().sum(0)).sum(1)
        nll = gauss_nll(y, mu @ self.dec.T + self.m, self.log_sigma)
        nll = nll + 0.5 * trace * torch.exp(-2 * self.log_sigma)
        return nll + gauss_kl(mu, logvar).sum(1)


def _fit(y: torch.Tensor, d: int, sigma_sq: float | None, steps: int = 4000):
    vae = _LinearVAE(y.shape[1], d, y.mean(0), sigma_sq)
    params = [q for q in vae.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=0.02)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [steps // 2, 3 * steps // 4])
    g = _gen(8)
    for _ in range(steps):
        eps = torch.randn(len(y), d, generator=g, dtype=F64)
        opt.zero_grad()
        vae.neg_elbo(y, eps).mean().backward()
        opt.step()
        sched.step()
    return vae


@pytest.mark.slow
class TestLinearVAE:
    def test_the_elbo_reaches_the_ppca_likelihood(self):
        """Lucas et al. (2019): the global optimum of a linear VAE's ELBO is
        the ML pPCA of d components, and the ELBO is tight there. Trained by
        the reparameterized one-sample estimate on points of a 2-component
        pPCA (P = 6), its exact -L comes within 1e-4 nats a value of the
        closed-form ML -log p in sample, and the learned sigma^2 is pPCA's
        (the mean of the discarded eigenvalues)."""
        y = _spectrum_data(2000, [9, 4, 0.3, 0.3, 0.3, 0.3], 1)
        p, d = 6, 2
        vae = _fit(y, d, None)
        c = y - y.mean(0)
        evals = torch.linalg.eigvalsh(c.T @ c / len(y)).flip(0)
        top, s2 = ppca_fit(evals, d, p)
        want = ppca_nll(top, s2, p, top, float(evals[d:].sum()))
        with torch.no_grad():
            got = float(vae.exact_neg_elbo(y).mean())
            sigma_sq = float((2 * vae.log_sigma).exp())
        assert got >= want - 1e-9  # a bound
        assert (got - want) / p < 1e-4, (got, want)
        assert sigma_sq == pytest.approx(s2, rel=2e-3)

    def test_directions_below_sigma_sq_go_unused(self):
        """At a fixed sigma^2 the ML weights are U (Lambda - sigma^2)^{1/2}
        on the eigenvalues above sigma^2 and 0 below (Tipping & Bishop
        1999): with eigenvalues 9, 4, 0.5, ... and sigma^2 = 1, the third of
        three latents is unused, its posterior the prior (KL 0) while the
        other two carry information; the exact -L reaches the closed form of
        C with lambda_1, lambda_2 and sigma^2 elsewhere."""
        y = _spectrum_data(2000, [9, 4, 0.5, 0.1, 0.1, 0.1], 2)
        p, sigma_sq = 6, 1.0
        vae = _fit(y, 3, sigma_sq)
        with torch.no_grad():
            mu, logvar = vae.posterior(y)
            kl = gauss_kl(mu, logvar).mean(0).sort().values
            got = float(vae.exact_neg_elbo(y).mean())
        assert float(kl[0]) < 1e-4 and float(kl[1]) > 0.3, kl
        c = y - y.mean(0)
        evals, vecs = torch.linalg.eigh(c.T @ c / len(y))
        evals, vecs = evals.flip(0), vecs.flip(1)
        above = int((evals > sigma_sq).sum())
        assert above == 2
        u = vecs[:, :above]
        coord_ms, residual_ms = _moments(y, y.mean(0), u)
        want = ppca_nll(evals[:above], sigma_sq, p, coord_ms, residual_ms)
        assert got >= want - 1e-9
        assert (got - want) / p < 1e-4, (got, want)
