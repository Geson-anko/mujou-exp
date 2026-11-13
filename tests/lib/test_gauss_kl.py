import math

import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence

from exp.lib.gauss_kl import (
    GaussMoments,
    buffer_moments,
    gauss_kl_grad,
    kl_gauss,
)

D = 4


def _law(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A mean ``(D,)`` and a covariance ``(D, D)`` of a generic Gaussian."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(D, D, generator=g, dtype=torch.float64)
    return torch.randn(D, generator=g, dtype=torch.float64), a @ a.T + 0.5 * torch.eye(
        D, dtype=torch.float64
    )


def _rows(n: int, mean, cov, g: torch.Generator) -> torch.Tensor:
    e = torch.randn(n, len(mean), generator=g, dtype=torch.float64)
    return mean + e @ torch.linalg.cholesky(cov).T


class TestValue:
    def test_kl_gauss_is_the_kl_of_the_two_normals(self):
        mean, cov = _law()
        ref = kl_divergence(
            MultivariateNormal(mean, cov),
            MultivariateNormal(torch.zeros(D).double(), torch.eye(D).double()),
        )
        torch.testing.assert_close(kl_gauss(mean, cov), ref)

    def test_the_buffer_value_is_the_kl_of_its_sample_moments(self):
        """kl is KL(N(mean, S / (n - 1)) || N(0, I)) of the rows' sample mean
        and unbiased covariance."""
        mean, cov = _law()
        rows = _rows(40, mean, cov, torch.Generator().manual_seed(1))
        m = buffer_moments(rows)
        ref = kl_divergence(
            MultivariateNormal(rows.mean(0), torch.cov(rows.T)),
            MultivariateNormal(torch.zeros(D).double(), torch.eye(D).double()),
        )
        torch.testing.assert_close(m.kl, ref)
        torch.testing.assert_close(m.mean, rows.mean(0))
        assert float(m.on) == 1

    def test_the_precision_is_n_less_d_less_2_times_the_inverse_scatter(self):
        rows = _rows(40, *_law(), torch.Generator().manual_seed(1))
        xc = rows - rows.mean(0)
        expect = (40 - D - 2) * torch.linalg.inv(xc.T @ xc)
        torch.testing.assert_close(buffer_moments(rows).precision, expect)


class TestGradient:
    def test_g_is_the_gradient_of_f_of_a_population(self):
        """For a population of m points of equal weight (its mean and its
        covariance of 1 / m), dF / dz_i = g(z_i) / m with g(z) = z -
        Sigma^-1 (z - mu): autograd of the closed form against
        gauss_kl_grad with the population's mean and precision. (The same
        identity, on the current sample and a buffer as the population,
        is the plug-in reading: (n + 1) dF^/dz_t = z_t - Sigma^-1 (z_t -
        mu^) with z_t in the moments.)"""
        mean, cov = _law()
        z = _rows(12, mean, cov, torch.Generator().manual_seed(2)).requires_grad_()
        mu = z.mean(0)
        sigma = (z - mu).T @ (z - mu) / len(z)
        (grad,) = torch.autograd.grad(kl_gauss(mu, sigma), [z])
        one = torch.ones((), dtype=torch.float64)
        m = GaussMoments(mu.detach(), torch.linalg.inv(sigma.detach()), one, one)
        torch.testing.assert_close(grad, gauss_kl_grad(z, m) / len(z))

    def test_the_gradient_is_0_at_the_optimum_and_off_below_d_plus_3(self):
        """At mu = 0, Sigma = I (exact moments) g(z) = z - z = 0; a buffer
        of D + 2 rows gives no precision (off: 0), one of D + 3 does."""
        z = torch.randn(5, D, dtype=torch.float64)
        eye = torch.eye(D, dtype=torch.float64)
        one = torch.ones((), dtype=torch.float64)
        exact = GaussMoments(torch.zeros(D, dtype=torch.float64), eye, one * 0, one)
        assert gauss_kl_grad(z, exact).abs().max() == 0
        g = torch.Generator().manual_seed(3)
        few = buffer_moments(torch.randn(D + 2, D, generator=g, dtype=torch.float64))
        assert float(few.on) == 0 and gauss_kl_grad(z, few).abs().max() == 0
        enough = buffer_moments(torch.randn(D + 3, D, generator=g, dtype=torch.float64))
        assert float(enough.on) == 1 and float(enough.precision.abs().max()) > 0

    def test_the_expected_gradient_is_the_populations_for_iid_rows(self):
        """Rows iid N(mu, Sigma) independent of the current z: E P^ =
        Sigma^-1 (P^ = (n - D - 2) S^-1, S Wishart), so E[g^ z^T] = E[g(z)
        z^T] = Sigma + mu mu^T - I, which at mu = 0, Sigma = I (the expected
        gradient of a linear encoder at its optimum) is 0. The plug-in
        inverse (S / (n - 1))^-1
        has the mean (n - 1) / (n - D - 2) Sigma^-1 instead (1.16 here),
        far outside the tolerance. 4000 buffers of n = 30, the tolerance 5
        standard errors of the Monte Carlo."""
        n, reps = 30, 4000
        mean, cov = _law()
        g = torch.Generator().manual_seed(4)
        ps = torch.stack(
            [buffer_moments(_rows(n, mean, cov, g)).precision for _ in range(reps)]
        )
        inv = torch.linalg.inv(cov)
        se = ps.std(0) / math.sqrt(reps)
        assert ((ps.mean(0) - inv).abs() <= 5 * se).all()
        plug = ps * (n - 1) / (n - D - 2)  # (S / (n - 1))^-1
        bias = (plug.mean(0) - inv).diagonal() / inv.diagonal()
        assert (bias > 0.1).all()
        # mu = 0, Sigma = I: the expected gradient of a linear encoder is 0
        zero, eye = torch.zeros(D, dtype=torch.float64), torch.eye(D).double()
        outer = []
        for _ in range(reps):
            m = buffer_moments(_rows(n, zero, eye, g))
            z = torch.randn(1, D, generator=g, dtype=torch.float64)
            outer.append(gauss_kl_grad(z, m).T @ z)
        o = torch.stack(outer)
        assert (o.mean(0).abs() <= 5 * o.std(0) / math.sqrt(reps)).all()


class TestCollapses:
    """The trivial solutions have det Sigma = 0: F = +inf, and a buffer of
    them is refused (its scatter has no Cholesky factor)."""

    @staticmethod
    def _collapses() -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(5)
        t, d, n = 4, 2, 20  # D = T d = 8
        same = torch.randn(t * d, generator=g, dtype=torch.float64)
        c = torch.randn(n, d, generator=g, dtype=torch.float64)
        return {
            # C: every data point the same
            "C": same.expand(n, -1).clone(),
            # P: the tokens a function of their position only: every data
            # point the same flattened sequence
            "P": torch.arange(t * d, dtype=torch.float64).expand(n, -1).clone(),
            # Cw: tokens constant within a stack, c_b repeated T times
            "Cw": c.repeat(1, t),
        }

    @pytest.mark.parametrize("kind", ["C", "P", "Cw"])
    def test_f_is_inf_and_the_buffer_is_refused(self, kind):
        rows = self._collapses()[kind]
        cov = torch.cov(rows.T)
        assert torch.linalg.matrix_rank(cov) < rows.shape[1]
        assert kl_gauss(rows.mean(0), cov) == math.inf
        with pytest.raises(ValueError, match="singular"):
            buffer_moments(rows)
