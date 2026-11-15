from typing import cast

import pytest
import torch

from exp.video.compose import compose
from exp.video.measure.encoder_eval import stream_windows
from exp.video.programs.vision import DATA_ENV, PackedMeasures
from exp.video.vae.measures import held_out_stacks, stack_errors, training_moments
from exp.video.vae.modules import AEModules
from tests.video.packed import smooth_tours
from tests.video.vae.test_program import tiny_ae


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
