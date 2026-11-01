import torch

from experiments.continuous_stream.evaluate import (
    effective_rank,
    probe_r2,
    rollout_errors,
)


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


class TestRollout:
    def test_an_identity_model_errs_by_the_change_over_n_frames(self):
        """With s = z and z_hat = s, the model predicts the last observed frame
        for every frame after it."""
        g = torch.Generator().manual_seed(4)
        z = torch.randn(3, 40, 5, generator=g).cumsum(1)

        def step(z_in: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return z_in

        errors = rollout_errors(z, step, lambda s: s, d_s=5, observed=16)
        for n, err in errors.items():
            want = float(((z[:, 15] - z[:, 15 + n]) ** 2).mean())
            assert abs(err - want) < 1e-6 * max(1.0, want)
