import math
from typing import Any, cast

import pytest
import torch

from exp.video.jepa.masks import make_views
from exp.video.jepa.vit import (
    Block,
    Embed,
    Stage,
    build_vision,
    extra_tokens,
    patchify,
    sincos_2d,
)
from exp.video.schema import BallsConf, DataConf, OptimConf, RunConf, VisionConf

SMALL: dict[str, Any] = {
    "patch": 4,
    "dim": 16,
    "depth": 2,
    "heads": 2,
    "nodes": 2,
    "pred_depth": 1,
}


def _conf(**vision) -> RunConf:
    return RunConf(
        name="test",
        program="vision",
        data=DataConf(streams=2, balls=BallsConf(obs="frames", size=8)),
        vision=VisionConf(**(SMALL | vision)),
        optim_vision=OptimConf(),
    )


def test_patchify_makes_a_tube_of_each_patch_in_row_major_order():
    """Pixel (f, c, r, q) holds its own coordinates: token t = (r // p) gw +
    q // p is the (N, C, p, p) block of its patch through all frames."""
    n, f, c, h, w, p = 1, 2, 3, 4, 6, 2
    f_, c_, r_, q_ = torch.meshgrid(
        *(torch.arange(k) for k in (f, c, h, w)), indexing="ij"
    )
    x = (1000 * f_ + 100 * c_ + 10 * r_ + q_).float()[None]
    tokens = patchify(x, p)
    assert tokens.shape == (n, (h // p) * (w // p), f * c * p * p)
    for t in range(tokens.shape[1]):
        gr, gq = divmod(t, w // p)
        block = x[0, :, :, gr * p : (gr + 1) * p, gq * p : (gq + 1) * p]
        assert torch.equal(tokens[0, t], block.flatten())


def test_sincos_rows_are_distinct_and_bounded():
    pos = sincos_2d((4, 4), 16)
    assert pos.shape == (16, 16) and pos.abs().max() <= 1
    assert len({tuple(r.tolist()) for r in pos}) == 16
    # row 0, column 0 is sin 0 = 0 and cos 0 = 1 in each half
    assert torch.equal(pos[0], torch.tensor(([0.0] * 4 + [1.0] * 4) * 2))


@pytest.mark.parametrize(("pool", "extra"), [("mean", 0), ("cls", 1), ("registers", 3)])
def test_the_extra_tokens_of_each_pool(pool, extra):
    conf = _conf(z_pool=pool, registers=3)
    assert conf.vision is not None
    assert extra_tokens(conf.vision) == extra
    mods = build_vision(conf, seed=0)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(5, 1, 2, 8, 8, generator=g)  # a channel and the mask channel
    assert mods.encoder(x).shape == (5, extra + 4, 16)
    assert mods.encode(torch.rand(5, 1, 1, 8, 8, generator=g)).shape == (5, 16)


class TestEmbed:
    def test_pixels_at_the_mean_and_masked_patches_add_nothing(self):
        """The pixels enter as (x - mean) / std, and as 0 where masked: a
        chunk all at the mean, and any chunk masked everywhere, give tokens
        of the positions alone (the bias is 0 at the initialization)."""
        conf = _conf(pixel_mean=0.3, pixel_std=0.2)
        embed = build_vision(conf, seed=0).encoder.stages[0].embed
        assert isinstance(embed, Embed)
        g = torch.Generator().manual_seed(0)
        mean = torch.full((2, 1, 1, 8, 8), 0.3)
        all_masked = torch.ones(2, 1, 2, 2, dtype=torch.bool)
        h_mean, _ = make_views(mean, all_masked[:, :0], 4)  # the target view
        h_masked, _ = make_views(torch.rand(2, 1, 1, 8, 8, generator=g), all_masked, 4)
        pos = cast(torch.Tensor, embed.pos).expand(2, -1, -1)
        torch.testing.assert_close(embed(h_mean), pos)
        masked = embed(h_masked).unflatten(0, (2, 2))[:, 1]  # the context view
        mask_part = embed.proj.weight[:, 16:].sum(1)  # the mask channel, all 1
        torch.testing.assert_close(masked, pos + mask_part)

    def test_one_standardized_pixel_moves_its_token_by_its_weight(self):
        """Linear in the standardized pixel: pixel (r, q) = m + s moves token
        (r // p) gw + q // p by the weight's column of that pixel."""
        conf = _conf(pixel_mean=0.3, pixel_std=0.2)
        embed = build_vision(conf, seed=0).encoder.stages[0].embed
        assert isinstance(embed, Embed)
        x = torch.full((1, 1, 1, 8, 8), 0.3)
        base = embed(make_views(x, torch.zeros(1, 0, 2, 2, dtype=torch.bool), 4)[0])
        x[0, 0, 0, 5, 2] = 0.5  # one std above the mean, in patch (1, 0)
        moved = embed(make_views(x, torch.zeros(1, 0, 2, 2, dtype=torch.bool), 4)[0])
        column = embed.proj.weight[:, (5 % 4) * 4 + 2]  # pixel (1, 2) of the patch
        torch.testing.assert_close(moved[0, 2] - base[0, 2], column)
        others = [0, 1, 3]
        torch.testing.assert_close(moved[0, others], base[0, others])


class TestBuild:
    def test_the_patch_embedding_is_lecun_and_the_rest_002(self):
        """Normal weights truncated at 2 std have std sigma sqrt(1 - 2 * 2
        phi(2) / (2 Phi(2) - 1)) = 0.880 sigma: sigma = 1 / sqrt(fan-in) for
        the patch embedding, 0.02 for the other linear maps."""
        conf = _conf(dim=64, heads=4, depth=2)
        mods = build_vision(conf, seed=0)
        stage = cast(Stage, mods.encoder.stages[0])
        phi2 = math.exp(-2) / math.sqrt(2 * math.pi)
        mass = math.erf(2 / math.sqrt(2))
        factor = math.sqrt(1 - 4 * phi2 / mass)
        embed = stage.embed
        assert embed is not None
        w = embed.proj.weight.detach()  # (64, 1 x 2 x 16)
        sigma = 1 / math.sqrt(w.shape[1])
        assert w.abs().max() <= 2 * sigma
        # The standard error of a std of n normal draws is about std / sqrt(2n).
        tol = 4 * factor * sigma / math.sqrt(2 * w.numel())
        assert abs(float(w.std()) - factor * sigma) < tol
        qkv = cast(Block, stage.blocks[0]).qkv.weight.detach()
        tol = 4 * factor * 0.02 / math.sqrt(2 * qkv.numel())
        assert abs(float(qkv.std()) - factor * 0.02) < tol

    def test_the_seed_fixes_the_weights_and_the_global_rng_is_left(self):
        torch.manual_seed(123)
        before = torch.rand(3)
        torch.manual_seed(123)
        a = build_vision(_conf(), seed=7)
        after = torch.rand(3)
        b = build_vision(_conf(), seed=7)
        c = build_vision(_conf(), seed=8)
        assert torch.equal(before, after)
        pa = [*a.encoder.parameters(), *a.predictor.parameters()]
        pb = [*b.encoder.parameters(), *b.predictor.parameters()]
        pc_ = [*c.encoder.parameters(), *c.predictor.parameters()]
        assert all(torch.equal(x, y) for x, y in zip(pa, pb, strict=True))
        assert not all(torch.equal(x, y) for x, y in zip(pa, pc_, strict=True))

    def test_the_stages_split_the_depth(self):
        mods = build_vision(_conf(depth=6, nodes=3), seed=0)
        stages = mods.encoder.stages
        assert [len(cast(Stage, s).blocks) for s in stages] == [2, 2, 2]
        assert stages[0].embed is not None and stages[1].embed is None
        assert stages[-1].norm is not None and stages[0].norm is None

    def test_a_patch_that_does_not_divide_the_side_is_an_error(self):
        with pytest.raises(ValueError, match="does not divide"):
            build_vision(_conf(patch=3), seed=0)


def _cross_jacobian(f, x: torch.Tensor, i: int) -> torch.Tensor:
    """d f(x)[i] / dx: the gradient of the sum of sample i's output with
    respect to every sample's input, ``(n, *x)``."""
    x = x.clone().requires_grad_()
    (g,) = torch.autograd.grad(f(x)[i].sum(), [x])
    return g


class TestSamplesAreIndependent:
    """A node computes each sample on its own (pc.Layer needs it): the output
    of sample i does not depend on the input of any other sample."""

    def test_each_stage_and_the_predictor(self):
        mods = build_vision(_conf(z_pool="cls"), seed=0)
        g = torch.Generator().manual_seed(0)
        x = torch.rand(4, 1, 2, 8, 8, generator=g)
        tokens = torch.randn(4, 5, 16, generator=g)
        for f, inp in [
            (mods.encoder.stages[0], x),
            (mods.encoder.stages[1], tokens),
            (mods.predictor, tokens),
        ]:
            g = _cross_jacobian(f, inp, 1)
            assert (g[[0, 2, 3]] == 0).all() and g[1].abs().sum() > 0

    def test_a_block(self):
        block = Block(8, 2)
        x = torch.randn(3, 5, 8, generator=torch.Generator().manual_seed(0))
        g = _cross_jacobian(block, x, 0)
        assert (g[1:] == 0).all()
