import dataclasses
import math

import pytest
import torch
from torch import nn

from exp.vision.models import (
    Block,
    PatchEmbed,
    ResNetPart,
    UpStage,
    ViTPart,
    build_vae,
    early_fusion,
    mup_lr,
    param_table,
)
from exp.vision.schema import DataConf, ModelConf, RunConf, ToursConf

FRAMES, SIZE, D = 2, 32, 8


def tiny(arch: str, size: str = "S", streams: int = 2) -> RunConf:
    """32 px stacks of 2 frames, a 4 x 4 grid of patches of 8 for the ViT."""
    return RunConf(
        name="tiny",
        data=DataConf(streams=streams, tours=ToursConf(size=SIZE, frames=FRAMES)),
        model=ModelConf(
            arch=arch,
            size=size,
            latent=D,
            width=16,
            heads=2,
            patch=8,
            resnet_channels=[4, 4, 8, 8],
            decoder_channels=[8, 8, 4, 4, 4],
            sigma0_sq=0.05,
        ),
    )


def stacks(b: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, FRAMES, 3, SIZE, SIZE), generator=g).to(
        torch.uint8
    )


class TestEarlyFusion:
    def test_frame_f_channel_c_is_channel_3f_plus_c_over_255(self):
        x = stacks(2)
        img = early_fusion(x)
        assert img.shape == (2, 3 * FRAMES, SIZE, SIZE) and img.dtype == torch.float32
        for f in range(FRAMES):
            for c in range(3):
                assert torch.equal(img[:, 3 * f + c], x[:, f, c].float() / 255)


@pytest.mark.parametrize("arch", ["vit", "resnet"])
class TestShapes:
    def test_four_encoder_nodes_two_decoder_nodes_and_the_shapes(self, arch):
        """Each size has 4 encoder parts of k blocks (S, M, L = 1, 2, 4), the
        encoder ends in (B, 2D), and the decoder maps z (B, D) to the stacks'
        shape."""
        for size, k in [("S", 1), ("M", 2), ("L", 4)]:
            mods = build_vae(tiny(arch, size), seed=0)
            parts = mods.encoder.parts
            assert len(parts) == 4 and len(mods.decoder.parts) == 2
            kind = ViTPart if arch == "vit" else ResNetPart
            assert all(isinstance(p, kind) for p in parts)
            assert all(len(getattr(p, "blocks")) == k for p in parts)
            mu, logvar = mods.posterior(stacks(3))
            assert mu.shape == logvar.shape == (3, D)
            y = mods.decode(mu)
            assert y.shape == (3, FRAMES, 3, SIZE, SIZE)

    def test_the_grid_of_each_node(self, arch):
        """The ResNet halves the side at every node (to size / 16 in node 4),
        the ViT keeps its (size / patch)^2 tokens."""
        mods = build_vae(tiny(arch), seed=0)
        h = early_fusion(stacks(1))
        sides = []
        for part in list(mods.encoder.parts)[:-1]:
            h = part(h)
            sides.append(h.shape[-1] if arch == "resnet" else h.shape[1])
        if arch == "resnet":
            assert sides == [16, 8, 4]  # then 2 = 32 / 16 in node 4
        else:
            assert sides == [16, 16, 16]  # (32 / 8)^2 tokens throughout

    def test_the_log_variance_starts_at_zero_and_mu_does_not(self, arch):
        """The head's rows of ln s^2 and their bias are 0, so s^2 = 1 for any
        stack; the rows of mu are drawn."""
        mods = build_vae(tiny(arch), seed=0)
        head = mods.encoder.head
        assert not head.weight[D:].any() and not head.bias[D:].any()
        assert head.weight[:D].std() > 0
        _, logvar = mods.posterior(stacks(2))
        assert not logvar.any()

    def test_stacks_are_computed_each_on_its_own(self, arch):
        """No statistic crosses the batch (GroupNorm, attention within a
        sample): a stack's code is the same alone or beside another."""
        mods = build_vae(tiny(arch), seed=0)
        x = stacks(2)
        both = mods.mu(x)
        torch.testing.assert_close(mods.mu(x[:1]), both[:1])
        torch.testing.assert_close(mods.mu(x[1:]), both[1:])


class TestBuild:
    def test_the_decoder_does_not_depend_on_the_encoder(self):
        """The decoder is drawn from seed + DECODER_SEED alone: the same for
        the ViT and the ResNet of every size."""
        ref = build_vae(tiny("vit", "S"), seed=3).decoder.state_dict()
        for arch, size in [("vit", "L"), ("resnet", "S"), ("resnet", "M")]:
            other = build_vae(tiny(arch, size), seed=3).decoder.state_dict()
            assert all(torch.equal(ref[k], other[k]) for k in ref)

    def test_the_seed_fixes_the_weights_and_the_global_rng_is_left(self):
        before = torch.get_rng_state()
        a = build_vae(tiny("vit"), seed=1).parameters()
        assert torch.equal(torch.get_rng_state(), before)
        b = build_vae(tiny("vit"), seed=1).parameters()
        c = build_vae(tiny("vit"), seed=2).parameters()
        assert all(torch.equal(p, q) for p, q in zip(a, b, strict=True))
        assert not all(torch.equal(p, q) for p, q in zip(a, c, strict=True))

    def test_log_sigma_starts_at_half_the_log_of_sigma0_sq(self):
        mods = build_vae(tiny("vit"), seed=0)
        assert float(mods.log_sigma) == pytest.approx(0.5 * math.log(0.05))


class TestDecode:
    def test_the_mean_of_the_target_of_pixel_c_decodes_to_c_over_255(self):
        """A readout that outputs (c + 1/2) / 256, the mean of the target
        (c + u) / 256 of the pixel value c, decodes to c / 255."""
        mods = build_vae(tiny("vit"), seed=0)
        readout = mods.decoder.readout
        with torch.no_grad():
            readout.weight.zero_()
            assert readout.bias is not None
            readout.bias.fill_((200 + 0.5) / 256)
        x = mods.decode(torch.randn(2, D))
        assert x.shape == (2, FRAMES, 3, SIZE, SIZE)
        torch.testing.assert_close(x, torch.full_like(x, 200 / 255))


class TestParamTable:
    def test_resnet_m_matches_vit_m_within_ten_percent(self):
        """At the defaults (8 frames of 64 px, D = 256): the two M encoders
        within 10 %, S < M < L for each arch, one decoder for all."""
        rows = param_table()
        table = "\n".join(str(r) for r in rows)
        enc = {(r["arch"], r["size"]): int(r["encoder"]) for r in rows}
        ratio = enc["resnet", "M"] / enc["vit", "M"]
        assert 0.9 <= ratio <= 1.1, table
        for arch in ("vit", "resnet"):
            assert enc[arch, "S"] < enc[arch, "M"] < enc[arch, "L"], table
        assert len({r["decoder"] for r in rows}) == 1, table

    def test_the_counts_are_those_of_the_built_modules(self):
        conf = tiny("resnet", "M")
        rows = param_table(conf)
        row = next(r for r in rows if r["arch"] == "resnet" and r["size"] == "M")
        mods = build_vae(conf, seed=0)
        assert row["encoder"] == sum(p.numel() for p in mods.encoder.parameters())
        decoder = sum(p.numel() for p in mods.decoder.parameters()) + 1
        assert row["decoder"] == decoder


def _first[M: nn.Module](module: nn.Module, kind: type[M]) -> M:
    """The first module of ``kind`` in ``module``."""
    return next(m for m in module.modules() if isinstance(m, kind))


def wide(width: int, param: str = "mup", *, scale_decoder: bool = True) -> RunConf:
    """``tiny("vit")`` of ``width`` (heads ``width / 64``) and the decoder of
    v0/base scaled with it (``scale_decoder``), in ``param``."""
    conf = tiny("vit")
    conf.model = dataclasses.replace(
        conf.model,
        width=width,
        heads=None,
        decoder_channels=[256, 128, 64, 32, 32],
        scale_decoder=scale_decoder,
        param=param,
    )
    return conf


class TestWidth:
    @pytest.mark.parametrize("width", [128, 256, 512, 1024])
    def test_heads_of_64_and_the_decoder_scaled_with_the_width(self, width):
        """heads = width / 64; the decoder's channels are v0/base's times
        width / 256, every GroupNorm of groups dividing its channels, and the
        VAE decodes to the stacks' shape."""
        mods = build_vae(wide(width, "sp"), seed=0)
        block = _first(mods.encoder, Block)
        assert block.qkv.in_features == width
        assert getattr(block, "_heads") == width // 64
        k = width / 256
        convs = [
            m.conv.out_channels
            for m in mods.decoder.modules()
            if isinstance(m, UpStage)
        ]
        assert convs == [round(c * k) for c in [128, 64, 32, 32]]
        norms = [m for m in mods.decoder.modules() if isinstance(m, nn.GroupNorm)]
        assert norms and all(n.num_channels % n.num_groups == 0 for n in norms)
        assert mods.decode(mods.mu(stacks(2))).shape == (2, FRAMES, 3, SIZE, SIZE)


class TestMuP:
    def test_at_the_base_width_mup_is_sp_bit_for_bit(self):
        a = build_vae(wide(256, "sp"), seed=0).parameters()
        b = build_vae(wide(256, "mup"), seed=0).parameters()
        assert all(torch.equal(p, q) for p, q in zip(a, b, strict=True))

    def test_the_initialization_is_sps_rescaled_by_role(self):
        r"""Width 512 ($k = 2$): the ViT's hidden matrices (std 0.02 in SP)
        and the output matrices (head, readout; std $\propto$ fan-in$^{-1/2}$
        in SP) are SP's of the same seed times $k^{-1/2}$, so the hidden
        variance is $\propto 1/\text{fan-in}$ and the output's std is that of
        width 256 over $k$; the rest is SP's."""
        sp, mup = (
            build_vae(wide(512, "sp"), seed=0),
            build_vae(wide(512, "mup"), seed=0),
        )
        f = 2**-0.5
        vit_hidden = {
            id(lin.weight)
            for m in mup.encoder.modules()
            if isinstance(m, Block)
            for lin in (m.qkv, m.proj, m.fc1, m.fc2)
        }
        outputs = {id(mup.encoder.head.weight), id(mup.decoder.readout.weight)}
        for p, q in zip(sp.parameters(), mup.parameters(), strict=True):
            want = p * f if id(q) in vit_hidden | outputs else p
            assert torch.equal(q, want)

    @pytest.mark.parametrize("scale_decoder", [True, False])
    def test_the_learning_rate_of_each_role(self, scale_decoder):
        """Width 512: 1 / k = 0.5 for the hidden matrices (the ViT's, the
        decoder's convs) and the outputs (head, readout), 1 for the inputs
        (patch embedding, the decoder's Linear of z) and the vectors (biases,
        norms, ln sigma); the decoder's are all 1 when it is not scaled."""
        mods = build_vae(wide(512, scale_decoder=scale_decoder), seed=0)
        lr = mup_lr(mods, wide(512, scale_decoder=scale_decoder).model)
        assert set(map(id, lr)) == set(map(id, [*mods.parameters(), mods.log_sigma]))
        enc, dec = mods.encoder, mods.decoder
        block, patch = _first(enc, Block), _first(enc, PatchEmbed)
        z_map, stage = (
            _first(dec, nn.Linear),
            [m for m in dec.modules() if isinstance(m, UpStage)][2],
        )
        dk = 0.5 if scale_decoder else 1.0
        want = {
            patch.proj.weight: 1.0,
            patch.proj.bias: 1.0,
            block.qkv.weight: 0.5,
            block.qkv.bias: 1.0,
            block.fc2.weight: 0.5,
            block.norm1.weight: 1.0,
            enc.head.weight: 0.5,
            enc.head.bias: 1.0,
            z_map.weight: 1.0,
            z_map.bias: 1.0,
            stage.conv.weight: dk,
            stage.norm.weight: 1.0,
            dec.readout.weight: dk,
            dec.readout.bias: 1.0,
            mods.log_sigma: 1.0,
        }
        for p, rate in want.items():
            assert lr[p] == rate
        assert sorted(set(lr.values())) == sorted({0.5, 1.0, dk})

    def test_at_the_base_width_every_rate_is_one(self):
        conf = wide(256)
        assert set(mup_lr(build_vae(conf, seed=0), conf.model).values()) == {1.0}
