import math

import pytest
import torch

from exp.vision.models import (
    ResNetPart,
    ViTPart,
    build_vae,
    early_fusion,
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
