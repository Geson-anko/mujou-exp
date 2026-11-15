import re

import pytest
import torch

from exp.lib.optim import fan_ins
from exp.video.compose import compose
from exp.video.jepa.vit import Block, build_vision, patchify_tubelets
from exp.video.vae.modules import TokenDecoder, build_ae, unpatchify_tubelets


def _m18():
    return compose(["+experiment=m18/ae_t128", "device=cpu"])


class TestUnpatchifyTubelets:
    @pytest.mark.parametrize(("frames", "tubelet", "patch"), [(4, 2, 4), (6, 3, 2)])
    def test_inverts_patchify_tubelets(self, frames, tubelet, patch):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, frames, 3, 8, 12, generator=g)
        grid3 = (frames // tubelet, 8 // patch, 12 // patch)
        tokens = patchify_tubelets(x, patch, tubelet)
        assert torch.equal(unpatchify_tubelets(tokens, grid3, patch, 3), x)


class TestTokenDecoder:
    def test_a_token_draws_its_own_tubelet_without_blocks(self):
        """With no blocks the decoder is per token: changing one token changes
        only its tubelet's pixels (its slot's frames, its patch)."""
        dec = TokenDecoder(
            token_dim=3,
            dim=16,
            heads=2,
            depth=0,
            grid3=(2, 2, 2),
            patch=4,
            tubelet=2,
            channels=3,
            mean=0.5,
            std=0.25,
        )
        g = torch.Generator().manual_seed(1)
        t = torch.randn(1, 8, 3, generator=g)
        moved = t.clone()
        moved[0, 5] += 1.0  # slot 1, row 0, column 1
        with torch.no_grad():
            diff = (dec(moved) - dec(t)).abs()[0]  # (4, 3, 8, 8)
        changed = diff.amax(1) > 0  # (4, 8, 8)
        want = torch.zeros(4, 8, 8, dtype=torch.bool)
        want[2:4, 0:4, 4:8] = True
        assert torch.equal(changed, want)


# The fan-in of each parameter of the decoder of m18/ae_t128 by name (the first
# pattern that matches; every other is a vector): the encoder's are those of
# M17 (tests/video/jepa/test_nets.py).
DECODER_FAN_INS = (
    (r"embed\.weight", 8),  # a token, d = 8 to the width 256
    (r"stage\.blocks\.\d+\.(qkv|proj|fc1)\.weight", 256),
    (r"stage\.blocks\.\d+\.fc2\.weight", 1024),
    (r"stage\.out\.weight", 256),  # 256 to a tubelet's 2 x 3 x 32 x 32 pixels
)


class TestBuildAE:
    def test_the_encoder_is_m17s_of_the_seed(self):
        conf = _m18()
        ae = build_ae(conf, seed=100)
        ref = build_vision(conf, seed=100)
        for (a, p), (b, q) in zip(
            ae.vision.encoder.named_parameters(),
            ref.encoder.named_parameters(),
            strict=True,
        ):
            assert a == b and torch.equal(p, q)

    def test_the_decoder_is_lighter_than_the_encoder(self):
        """2 blocks of the encoder's 8: about 3.2 M of its 7.9 M parameters."""
        ae = build_ae(_m18(), seed=100)
        enc = sum(p.numel() for p in ae.vision.encoder.parameters())
        dec = sum(p.numel() for p in ae.decoder.parameters())
        assert len(ae.decoder.stage.blocks) == 2
        blocks = [m for m in ae.vision.encoder.modules() if isinstance(m, Block)]
        assert len(blocks) == 8
        assert dec < 0.5 * enc

    def test_the_fan_ins_of_the_decoder(self):
        ae = build_ae(_m18(), seed=100)
        n = fan_ins(ae.decoder)
        got, want = {}, {}
        for name, p in ae.decoder.named_parameters():
            got[name] = n[p]
            rule = (f for pattern, f in DECODER_FAN_INS if re.fullmatch(pattern, name))
            want[name] = next(rule, None)
            if want[name] is None:  # a bias or a gain
                assert p.dim() == 1, name
                want[name] = 1
        assert got == want
        assert ae.decoder.stage.out is not None
        assert ae.decoder.stage.out.out_features == 6144

    def test_the_initial_output_is_standardized_pixels_around_the_mean(self):
        """LeCun's last map of the unit LayerNorm: each output about N(0, 1)
        before the data's std and mean, so g(f(x)) starts with the mean pixel
        (0.46) and a spread of about the pixel std (0.28)."""
        conf = _m18()
        ae = build_ae(conf, seed=100)
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 256, (2, 16, 3, 128, 128), generator=g, dtype=torch.uint8)
        with torch.no_grad():
            y = ae(x)
        assert y.shape == x.shape
        assert abs(float(y.mean()) - 0.46) < 0.02
        assert 0.8 * 0.28 < float(y.std()) < 1.2 * 0.28
