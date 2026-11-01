import torch

from exp.video.evaluate import mean_errors, persistence_errors, rollout_errors


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


class TestPersistence:
    def test_a_ramp_changes_by_n_times_its_slope(self):
        """z_k = c k: saying z_15 again for z_{15+n} errs by (c n)^2."""
        c = torch.tensor([0.5, -2.0])
        z = (torch.arange(40.0)[:, None] * c).expand(3, 40, 2)
        for n, err in persistence_errors(z, observed=16).items():
            assert abs(err - float((c**2).mean()) * n**2) < 1e-4

    def test_it_is_the_rollout_of_the_identity_model(self):
        g = torch.Generator().manual_seed(5)
        z = torch.randn(2, 40, 3, generator=g)
        want = rollout_errors(z, lambda z_in, s: z_in, lambda s: s, d_s=3)
        assert persistence_errors(z) == want


class TestMean:
    def test_it_errs_by_the_distance_to_the_mean(self):
        """Two clips, constant in time at +c and -c: the mean is 0, so the
        error is c^2 at every horizon."""
        c = torch.tensor([1.0, -3.0])
        z = torch.stack([c.expand(40, 2), -c.expand(40, 2)])
        for err in mean_errors(z).values():
            assert abs(err - float((c**2).mean())) < 1e-6


class TestResidualRollout:
    def test_a_residual_model_that_adds_nothing_is_persistence(self):
        g = torch.Generator().manual_seed(6)
        z = torch.randn(2, 40, 3, generator=g)
        got = rollout_errors(
            z, lambda z_in, s: s, lambda s: torch.zeros_like(s), d_s=3, residual=True
        )
        assert got == persistence_errors(z)
