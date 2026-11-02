import torch

from exp.video.evaluate import (
    halves,
    mean_errors,
    persistence_errors,
    rollout_errors,
    rollout_shape,
    shrink_errors,
    shrink_predictions,
)


def _held(v: torch.Tensor, after: torch.Tensor, observed: int = 16) -> torch.Tensor:
    """Clips ``(n, 40, D)`` at ``v`` up to the last observed frame and at
    ``after`` from then on."""
    early = v[:, None].expand(-1, observed, -1)
    late = after[:, None].expand(-1, 40 - observed, -1)
    return torch.cat([early, late], 1)


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


class TestShrink:
    V = torch.tensor([[1.0, -2.0], [-1.0, 2.0], [3.0, 0.5], [-3.0, -0.5]])
    HALF = torch.tensor([True, True, False, False])

    def test_a_clip_that_halves_its_offset_is_met_exactly(self):
        """z goes from v to v / 2 around the mean 0: lambda = 1/2, no error."""
        z = _held(self.V, self.V / 2)
        for err in shrink_errors(z, self.HALF).values():
            assert abs(err) < 1e-10

    def test_each_half_is_predicted_with_the_lambda_of_the_other(self):
        """The first half goes to v / 2 (lambda 1/2), the second stays at v
        (lambda 1): each half is predicted with the other's lambda, so both
        miss by v / 2."""
        after = torch.cat([self.V[:2] / 2, self.V[2:]])
        z = _held(self.V, after)
        preds = shrink_predictions(z, self.HALF)
        for h in (1, 10):
            torch.testing.assert_close(preds[h][:2], self.V[:2])
            torch.testing.assert_close(preds[h][2:], self.V[2:] / 2)
        want = float(((self.V / 2) ** 2).mean())
        for err in shrink_errors(z, self.HALF).values():
            assert abs(err - want) < 1e-6

    def test_with_nothing_to_carry_it_is_the_mean(self):
        """The late z does not depend on the last one in either half:
        lambda = 0, and the error is that of the mean."""
        v = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        after = torch.tensor([[0.0, 1.0], [0.0, -1.0], [2.0, 0.0], [-2.0, 0.0]])
        z = _held(v, after)
        assert shrink_errors(z, self.HALF) == mean_errors(z)

    def test_the_halves_split_each_phase(self):
        assert halves(2, 4).tolist() == [True, True, False, False] * 2


class TestShape:
    V = torch.tensor([[1.0, -2.0], [-1.0, 2.0], [3.0, 0.5]])

    def test_persistence_on_clips_that_hold_still_is_exact(self):
        z = _held(self.V, self.V)
        shape = rollout_shape(z, lambda z_in, s: z_in, lambda s: s, d_s=2)
        for h in (1, 10):
            assert abs(shape[f"amp{h}"] - 1) < 1e-6
            assert abs(shape[f"corr{h}"] - 1) < 1e-6

    def test_doubling_each_step_overshoots_by_4_to_the_n(self):
        z = _held(self.V, self.V)
        shape = rollout_shape(z, lambda z_in, s: z_in, lambda s: 2 * s, d_s=2)
        for h in (1, 5):
            assert abs(shape[f"amp{h}"] / 4**h - 1) < 1e-6
            assert abs(shape[f"corr{h}"] - 1) < 1e-6

    def test_a_constant_prediction_has_no_amplitude(self):
        z = _held(self.V, self.V)
        zero = lambda s: torch.zeros_like(s)  # noqa: E731
        shape = rollout_shape(z, lambda z_in, s: z_in, zero, d_s=2)
        assert shape["amp1"] == 0
        assert shape["corr1"] == 0
