import math
from typing import cast

import pytest
import torch

from exp.video.compose import compose
from exp.video.measure.encoder_eval import stream_windows
from exp.video.programs.vision import DATA_ENV, PackedMeasures
from exp.video.vae.measures import (
    KL_QUANTILES,
    elbo_errors,
    held_out_stacks,
    stack_errors,
    training_moments,
)
from exp.video.vae.modules import AEModules, build_vae
from tests.video.packed import smooth_tours
from tests.video.vae.test_program import tiny_ae, tiny_vae


@pytest.fixture
def setup(tmp_path):
    root = smooth_tours(tmp_path / "d", {"A": 120, "B": 80, "H": 200})
    return root, compose(tiny_ae(root.name))


class _Constant:
    """A stand-in of the autoencoder whose reconstruction is ``out``."""

    def __init__(self, out: torch.Tensor) -> None:
        self.out = out

    def __call__(self, stacks: torch.Tensor) -> torch.Tensor:
        return self.out.expand(len(stacks), *self.out.shape)


class TestTrainingMoments:
    def test_the_mean_and_variance_of_every_stack_of_the_windows(self, setup):
        """Against the stacks of stream_windows made one by one: stack k of a
        window is its frames k .. k + N - 1, y = x / 256."""
        root, conf = setup
        got = training_moments(conf, root, windows=16, steps=3)
        w = stream_windows(conf, root, windows=16, steps=3)
        n = conf.vision.frames
        y = torch.cat([w[:, k : k + n] for k in range(4)]).double() / 256
        assert got.count == len(y) == 4 * len(w)
        torch.testing.assert_close(got.mean, y.mean(0), rtol=0, atol=1e-12)
        want = float(y.var(0, unbiased=False).mean())
        assert got.var == pytest.approx(want, rel=1e-10)


class TestStackErrors:
    def test_one_for_the_mean_stack_and_zero_for_the_stacks(self, setup):
        root, conf = setup
        x = held_out_stacks(conf, root)["val"]
        mean = x.double().mean(0) / 256 + 0.01
        mse, loss = stack_errors(cast(AEModules, _Constant(mean.float())), x, mean)
        assert mse == pytest.approx(1.0, rel=1e-6)
        y = x.double() / 256
        assert loss == pytest.approx(
            float(((y - mean) ** 2).sum()) / 2 / len(x), rel=1e-6
        )

        class _Same:
            def __call__(self, stacks: torch.Tensor) -> torch.Tensor:
                return stacks.float() / 256

        mse, loss = stack_errors(cast(AEModules, _Same()), x, mean, batch=3)
        assert mse == 0.0 and loss == 0.0


class TestHeldOutStacks:
    def test_the_first_stacks_of_the_windows_of_val_and_test(self, setup, monkeypatch):
        root, conf = setup
        monkeypatch.setenv(DATA_ENV, str(root.parent))
        m = PackedMeasures(conf)
        got = held_out_stacks(conf, root)
        n = conf.vision.frames
        for s in ("val", "test"):
            assert torch.equal(got[s], m.windows[s][:, :n])


class TestELBOErrors:
    @pytest.fixture
    def vae(self, tmp_path):
        root = smooth_tours(tmp_path / "d", {"A": 120, "B": 80, "H": 200})
        conf = compose(tiny_vae(root.name))
        mods = build_vae(conf, seed=conf.seed)
        d = mods.code_shape[1]
        with torch.no_grad():  # a posterior variance off the prior's
            mods.head.bias[d:] = -0.5
        x = held_out_stacks(conf, root)["val"]
        return mods, x, x.double().mean(0) / 256

    def test_the_bits_the_rate_and_the_mean_stacks_error(self, vae):
        """bpd = -L / (P ln 2) + 8 and the rate kl / (P ln 2); stack_mse is
        that of g(mu), the autoencoder's yardstick; the quantiles of the
        latents' mean KL bracket their mean; the same seed draws the same
        noise (with the same batch)."""
        mods, x, mean = vae
        got = elbo_errors(mods, x, mean, seed=7, batch=3)
        p = mean.numel()
        assert got["bpd"] == pytest.approx(got["loss"] / (p * math.log(2)) + 8)
        assert got["rate_bpd"] == pytest.approx(got["kl"] / (p * math.log(2)))
        mse, _ = stack_errors(mods, x, mean)
        assert got["stack_mse"] == pytest.approx(mse, rel=1e-6)
        q = [got[f"kl_q{c}"] for c in KL_QUANTILES]
        assert q == sorted(q)
        dims = math.prod(mods.code_shape)
        assert q[0] <= got["kl"] / dims <= q[-1]
        assert elbo_errors(mods, x, mean, seed=7, batch=3) == got
        assert elbo_errors(mods, x, mean, seed=8, batch=3)["bpd"] != got["bpd"]

    def test_the_sigma_gap_is_what_sigma_star_would_save(self, vae):
        """Of the same samples, -L at log sigma = ln sigma* is -L less
        sigma_gap (the loss is quadratic-logarithmic in sigma), and sigma*
        is then sigma."""
        mods, x, mean = vae
        got = elbo_errors(mods, x, mean, seed=7)
        assert got["sigma_gap"] > 0
        with torch.no_grad():
            mods.likelihood.log_sigma.fill_(math.log(got["sigma_star"]))
        best = elbo_errors(mods, x, mean, seed=7)
        assert best["loss"] == pytest.approx(got["loss"] - got["sigma_gap"], rel=1e-6)
        assert best["sigma_gap"] == pytest.approx(0.0, abs=1e-6 * got["sigma_gap"])
        assert best["sigma_star"] == pytest.approx(got["sigma_star"], rel=1e-6)
