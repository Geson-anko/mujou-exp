import math

import pytest
import torch

from exp.lib.metrics import bootstrap_se, effective_rank, probe_r2


class TestEffectiveRank:
    def test_isotropic_gaussian_is_about_d(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(20_000, 64, generator=g)
        assert abs(effective_rank(z) - 64) < 1.5

    def test_data_of_rank_r_is_r(self):
        g = torch.Generator().manual_seed(1)
        q, _ = torch.linalg.qr(torch.randn(64, 5, generator=g))
        z = torch.randn(20_000, 5, generator=g) @ q.T
        assert abs(effective_rank(z) - 5) < 0.1


class TestProbe:
    def _split(self, x, y):
        n = len(x) // 2
        return x[:n], y[:n], x[n:], y[n:]

    def test_a_linear_function_of_the_state_is_read_out(self):
        g = torch.Generator().manual_seed(2)
        y = torch.randn(4000, 6, generator=g)
        x = y @ torch.randn(6, 32, generator=g) + 0.01 * torch.randn(
            4000, 32, generator=g
        )
        assert probe_r2(*self._split(x, y)).min() > 0.99

    def test_unrelated_noise_reads_nothing(self):
        g = torch.Generator().manual_seed(3)
        x, y = torch.randn(4000, 32, generator=g), torch.randn(4000, 6, generator=g)
        assert probe_r2(*self._split(x, y)).abs().max() < 0.05


class TestBootstrapSE:
    @pytest.mark.parametrize(("n", "sigma"), [(200, 1.0), (1024, 3.0)])
    def test_it_is_the_se_of_the_mean_of_normal_values(self, n, sigma):
        """For n values of N(0, sigma^2) the standard error of their mean is
        sigma / sqrt(n); 4000 resamples estimate it within 10% (the sampling
        error is about 1/sqrt(2n) + 1/sqrt(2 * 4000), below 6%)."""
        g = torch.Generator().manual_seed(4)
        x = sigma * torch.randn(n, generator=g, dtype=torch.float64)
        got = bootstrap_se(x, gen=g, resamples=4000)
        assert abs(got / (sigma / math.sqrt(n)) - 1) < 0.1

    def test_the_same_generator_gives_the_same_value(self):
        x = torch.randn(100, generator=torch.Generator().manual_seed(5))
        a = bootstrap_se(x, gen=torch.Generator().manual_seed(6))
        b = bootstrap_se(x, gen=torch.Generator().manual_seed(6))
        assert a == b

    def test_constant_values_have_no_error(self):
        g = torch.Generator().manual_seed(7)
        assert bootstrap_se(torch.full((50,), 2.5), gen=g) == 0.0
