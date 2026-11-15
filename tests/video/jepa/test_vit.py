import math
from typing import Any, cast

import pytest
import torch

from exp.video.jepa.masks import (
    expected_keep,
    make_views,
    multiblock_masks,
    pack_views,
)
from exp.video.jepa.vit import (
    Block,
    BroadcastDecoder,
    Embed,
    PackedPredictor,
    PixelDecoder,
    Stage,
    build_vision,
    extra_tokens,
    patchify,
    patchify_tubelets,
    sincos_2d,
    sincos_3d,
)
from exp.video.schema import BallsConf, DataConf, OptimConf, RunConf, VisionConf
from tests.video.packed import masks as packed_masks, packed_conf, stacks

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


class TestBroadcastDecoder:
    def test_it_makes_the_frames_and_the_default_width_follows_the_decoder(self):
        mlp = VisionConf(recon=1.0)
        bc = VisionConf(recon=1.0, recon_decoder="broadcast")
        assert (mlp.decoder_hidden, bc.decoder_hidden) == (256, 64)
        assert VisionConf(recon_decoder="broadcast", recon_hidden=8).decoder_hidden == 8
        with pytest.raises(ValueError, match="recon_decoder"):
            VisionConf(recon_decoder="conv")
        dec = BroadcastDecoder(6, 8, (2, 4, 4))
        assert dec(torch.randn(3, 6)).shape == (3, 2, 4, 4)
        assert dec.coords.min() > -1 and dec.coords.max() < 1
        assert "coords" not in dec.state_dict()

    def test_a_pixel_depends_on_z_and_its_coordinates_only(self):
        torch.manual_seed(0)
        dec = BroadcastDecoder(6, 8, (1, 4, 5))
        z = torch.randn(3, 6)
        perm = torch.randperm(20)
        full = dec.at(z, dec.coords)
        assert torch.equal(dec.at(z, dec.coords[perm]), full[:, perm])
        one = dec.at(z, dec.coords[7:8])  # alone, as with the others
        torch.testing.assert_close(one, full[:, 7:8])
        # and the decoder is the per-pixel MLP of [z, x, y]
        x = torch.cat([z[1], dec.coords[7]])
        torch.testing.assert_close(dec.net(x), full[1, 7])

    def test_the_output_is_laid_out_row_major_as_the_loss_expects(self):
        # out[n, c, i, j] is the net at [z, x_j, y_i] (H != W): the layout of
        # ``split_y``'s ``unflatten(1, (C, H, W))``
        torch.manual_seed(0)
        c, h, w = 2, 3, 5
        dec = BroadcastDecoder(6, 8, (c, h, w))
        z = torch.randn(4, 6)
        out = dec(z)
        for i in range(h):
            for j in range(w):
                xy = torch.tensor([(j + 0.5) / w * 2 - 1, (i + 0.5) / h * 2 - 1])
                for n in range(len(z)):
                    expected = dec.net(torch.cat([z[n], xy]))
                    torch.testing.assert_close(out[n, :, i, j], expected)

    def test_the_default_decoder_and_the_other_weights_are_unchanged(self):
        base = _conf(recon=1.0)
        bc = _conf(recon=1.0, recon_decoder="broadcast")
        off = _conf()
        a, b, c = (build_vision(x, seed=2) for x in (base, bc, off))
        assert isinstance(a.decoder, PixelDecoder)
        assert isinstance(b.decoder, BroadcastDecoder)
        # "mlp" is what it was: width 256, drawn after the encoder and predictor
        assert [tuple(p.shape) for p in a.decoder.parameters()] == [
            (256, 16),
            (256,),
            (64, 256),
            (64,),
        ]
        for m in (a, b):
            for x, y in zip(
                [*m.encoder.parameters(), *m.predictor.parameters()],
                [*c.encoder.parameters(), *c.predictor.parameters()],
                strict=True,
            ):
                assert torch.equal(x, y)


# --- the packed layout (M17) ---------------------------------------------------


class TestTubelets:
    def test_a_tubelet_of_all_the_frames_is_a_tube(self):
        x = torch.rand(2, 4, 3, 8, 8)
        assert torch.equal(patchify_tubelets(x, 4, 4), patchify(x, 4))

    def test_tubelets_of_one_frame_are_the_frames_patches_slot_by_slot(self):
        x = torch.rand(2, 3, 3, 8, 8)
        tokens = patchify_tubelets(x, 4, 1)  # (2, 3 x 4, 3 x 16)
        for s in range(3):
            got = tokens[:, 4 * s : 4 * (s + 1)]
            assert torch.equal(got, patchify(x[:, s : s + 1], 4))

    def test_a_tubelet_is_its_frames_in_order(self):
        """Token (slot s, patch q) of tau = 2 is frames 2s, 2s + 1 of patch q."""
        x = torch.rand(1, 4, 2, 8, 8)
        tokens = patchify_tubelets(x, 4, 2)
        both = patchify(x[:, 2:4], 4)  # slot 1 as a tube of 2 frames
        assert torch.equal(tokens[:, 4:], both)


class TestSincos3d:
    def test_the_slot_row_and_column_parts(self):
        """dim / 4 dimensions of the slot, 3 dim / 8 of the row, 3 dim / 8 of
        the column, each sin, cos of pos 10000^(-i / half)."""
        grid, dim = (2, 3, 4), 32
        pos = sincos_3d(grid, dim).double()
        assert pos.shape == (24, 32)

        def code(p: int, d: int) -> torch.Tensor:
            half = d // 2
            a = p / 10000 ** (torch.arange(half, dtype=torch.float64) / half)
            return torch.cat([a.sin(), a.cos()])

        for t in range(2):
            for r in range(3):
                for c in range(4):
                    row = pos[t * 12 + r * 4 + c]
                    expect = torch.cat([code(t, 8), code(r, 12), code(c, 12)])
                    torch.testing.assert_close(row, expect, rtol=0, atol=1e-6)
        assert len({tuple(r.tolist()) for r in pos}) == 24


class TestSegments:
    def test_a_segment_is_computed_on_its_own(self):
        """Block(segments=(a, b)): the first a tokens are those of the block on
        the first a alone, the others those on the others alone."""
        torch.manual_seed(0)
        block = Block(16, 2)
        x = torch.randn(3, 7, 16)
        out = block(x, (4, 3))
        torch.testing.assert_close(out[:, :4], block(x[:, :4]), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out[:, 4:], block(x[:, 4:]), rtol=1e-6, atol=1e-6)
        assert not torch.allclose(block(x)[:, :4], out[:, :4])


class TestPackedModules:
    def test_z_of_the_packed_sequence_is_that_of_the_target_view_alone(self):
        """The context views in the same sequence do not change the target's
        tokens: the mean of its segment is encode's z (rtol 1e-6)."""
        conf = packed_conf(z_dim=8, sigreg_on="both")
        mods = build_vision(conf, seed=0)
        x = stacks(3)
        g = torch.Generator().manual_seed(1)
        t, *keeps = mods.segments
        views = [
            multiblock_masks(3, mods.grid, m, keep=k // 2, generator=g)
            for m, k in zip(packed_masks(), keeps, strict=True)
        ]
        h, _ = pack_views(x, views, keeps=[k // 2 for k in keeps], patch=4, tubelet=2)
        tokens = mods.encoder(h, mods.segments)
        torch.testing.assert_close(
            mods.z_of(tokens[:, :t]), mods.encode(x), rtol=1e-6, atol=1e-7
        )
        assert mods.encode(x).shape == (3, 8)

    def test_the_segments_projection_and_names(self):
        conf = packed_conf(z_dim=8, sigreg_on="z")
        mods = build_vision(conf, seed=0)
        keeps = [expected_keep((4, 4), m) for m in packed_masks()]
        assert mods.segments == (32, 2 * keeps[0], 2 * keeps[1])
        assert mods.proj is not None and set(mods.named()) == {
            "encoder",
            "predictor",
            "proj",
        }
        assert build_vision(packed_conf(), seed=0).proj is None

    def test_the_seed_fixes_the_weights(self):
        a = build_vision(packed_conf(z_dim=8), seed=3)
        b = build_vision(packed_conf(z_dim=8), seed=3)
        for (_, m), (_, n) in zip(a.named().items(), b.named().items()):
            for p, q in zip(m.parameters(), n.parameters()):
                assert torch.equal(p, q)


def test_a_predictor_of_another_width_maps_to_it_and_back():
    conf = packed_conf(pred_dim=32, pred_heads=4)
    mods = build_vision(conf, seed=0)
    pred = mods.predictor
    assert isinstance(pred, PackedPredictor) and pred.embed is not None
    assert pred.mask_token.shape == (32,) and pred.out.out_features == 16
    ctx = torch.randn(2, 5, 16)
    ids = torch.arange(5).expand(2, 5)
    assert pred(ctx, ids, torch.arange(5, 9).expand(2, 4)).shape == (2, 4, 16)
    plain = build_vision(packed_conf(), seed=0).predictor
    assert isinstance(plain, PackedPredictor) and plain.embed is None


def test_the_predictors_map_of_the_context_is_lecun_and_sees_the_context():
    """The map of the context's tokens (width d = 8) to the predictor's width
    (32): weights of std 1 / sqrt(d) truncated at 2 std (std 0.880 / sqrt(d)),
    zero bias, so a context token of unit scale per dimension adds a content
    of about unit scale per dimension (0.880, the truncation's factor) to its
    position, whose sines and cosines have the mean square 1/2 (with std 0.02
    the content would be 0.05)."""
    mods = build_vision(packed_conf(token_dim=8, pred_dim=32, pred_heads=4), seed=0)
    pred = mods.predictor
    assert isinstance(pred, PackedPredictor) and pred.embed is not None
    w = pred.embed.weight.detach()  # (32, 8)
    sigma = 1 / math.sqrt(8)
    assert (pred.embed.bias == 0).all() and float(w.abs().max()) <= 2 * sigma
    phi2 = math.exp(-2) / math.sqrt(2 * math.pi)
    factor = math.sqrt(1 - 4 * phi2 / math.erf(2 / math.sqrt(2)))
    tol = 4 * factor * sigma / math.sqrt(2 * w.numel())
    assert abs(float(w.std()) - factor * sigma) < tol
    g = torch.Generator().manual_seed(0)
    ctx = torch.randn(4096, 8, generator=g)  # unit scale per dimension
    with torch.no_grad():
        content = float(pred.embed(ctx).pow(2).mean().sqrt())
    position = float(pred.pos.pow(2).mean().sqrt())
    assert position == pytest.approx(math.sqrt(0.5), rel=0.05)
    # E (w_i . x)^2 = |w_i|^2 for unit x: the mean square is 8 mean(w^2).
    want = math.sqrt(8 * float(w.pow(2).mean()))
    assert content == pytest.approx(want, rel=0.05)
    assert 0.5 * position < content < 2 * position
