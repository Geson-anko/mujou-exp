import torch

from exp.video.evaluate import rollout_errors


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
