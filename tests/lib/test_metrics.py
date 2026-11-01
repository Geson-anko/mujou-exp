import torch

from exp.lib.metrics import effective_rank, probe_r2


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
