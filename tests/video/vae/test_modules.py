import math
import re
from dataclasses import replace

import pytest
import torch

from exp.lib.optim import fan_ins
from exp.video.compose import compose
from exp.video.jepa.vit import Block, build_vision, patchify_tubelets
from exp.video.schema import AEConf
from exp.video.vae.modules import (
    TokenDecoder,
    build_ae,
    build_vae,
    unpatchify_tubelets,
)
from tests.video.packed import packed_conf, stacks


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

    @pytest.mark.parametrize("depth", [4, 8])
    def test_a_deeper_decoder_leaves_the_encoder_as_it_is(self, depth):
        """The decoder axis (results/m18.md): the encoder's initial weights do
        not depend on the decoder's depth (the decoder is drawn after it from
        its own seed), so the untrained ViT's yardstick of U is one for every
        depth; the decoder has ``depth`` blocks, and 8 is the encoder's."""
        conf = _m18()
        assert conf.ae is not None
        deep = replace(conf, ae=replace(conf.ae, decoder_depth=depth))
        a, b = build_ae(conf, seed=100), build_ae(deep, seed=100)
        for (n, p), (m, q) in zip(
            a.vision.encoder.named_parameters(),
            b.vision.encoder.named_parameters(),
            strict=True,
        ):
            assert n == m and torch.equal(p, q)
        assert len(b.decoder.stage.blocks) == depth
        blocks = [m for m in b.vision.encoder.modules() if isinstance(m, Block)]
        assert depth <= len(blocks)

    @pytest.mark.parametrize("depth", [2, 4, 8])
    def test_the_fan_ins_of_the_decoder(self, depth):
        """The same per-layer rate at every depth: each block's maps have the
        fan-ins of the table, so a deeper decoder adds blocks of the same
        rate and the meaning of eta does not change with the depth."""
        conf = _m18()
        assert conf.ae is not None
        ae = build_ae(replace(conf, ae=replace(conf.ae, decoder_depth=depth)), seed=100)
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

    def test_the_initial_output_is_the_mean_pixel_and_a_spread_of_the_last_map(
        self,
    ):
        """At the start the unit LayerNorm gives tokens h of mean 0 and ||h||^2
        = dim s^2 / (s^2 + eps) <= dim (s^2 a token's variance before it),
        and the last map W (rows = 6144 of dim, zero bias) makes z = (y - mu)
        / sigma = W h. So the mean square of z lies between s_min(W)^2 and
        s_max(W)^2 times dim / rows (W's extreme singular values; the lower
        one less eps / s^2, below 1e-3 for s^2 above 1e-2), and |mean z| =
        |mean row of W . mean h| <= ||mean row of W|| sqrt(dim). With
        LeCun's truncated normal W these are about 0.77 (1 +- 0.2)^2 and
        0.18: g(f(x)) starts near the mean pixel with about the pixel std."""
        conf = _m18()
        v = conf.vision
        assert v is not None
        ae = build_ae(conf, seed=100)
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 256, (2, 16, 3, 128, 128), generator=g, dtype=torch.uint8)
        with torch.no_grad():
            y = ae(x).double()
        assert y.shape == x.shape
        last = ae.decoder.stage.out
        assert last is not None and not last.bias.any()
        w = last.weight.detach().double()
        rows, dim = w.shape
        s = torch.linalg.svdvals(w)
        z = (y - v.pixel_mean) / v.pixel_std
        ms = float((z**2).mean())
        lo, hi = float(s[-1]) ** 2 * dim / rows, float(s[0]) ** 2 * dim / rows
        assert lo * (1 - 1e-3) <= ms <= hi * (1 + 1e-5)
        bound = float(w.mean(0).norm()) * dim**0.5
        assert abs(float(z.mean())) <= bound * (1 + 1e-5)


def _vae(depth: int = 4, sigma0_sq: float = 0.07):
    conf = compose(
        [
            "+experiment=m18/vae_t128",
            "device=cpu",
            f"ae.decoder_depth={depth}",
            f"ae.sigma0_sq={sigma0_sq}",
        ]
    )
    ae = compose(
        [
            "+experiment=m18/ae_t128",
            "device=cpu",
            f"ae.decoder_depth={depth}",
            "optim_vision.lr=1e-3",
        ]
    )
    return conf, ae


class TestBuildVAE:
    def test_it_starts_as_the_autoencoder_with_the_priors_variance(self):
        """Every weight of the autoencoder of the same seed and depth is the
        VAE's, the encoder's last map widened from d to 2 d: its first d rows
        (mu) are the autoencoder's, the last d (log s^2) 0, so the untrained
        posterior is N(code, I); log sigma starts at 1/2 ln sigma_0^2. No RNG
        is drawn beyond the autoencoder's."""
        conf, base = _vae(sigma0_sq=0.0625)
        before = torch.random.get_rng_state()
        vae = build_vae(conf, seed=100)
        assert torch.equal(torch.random.get_rng_state(), before)
        ae = build_ae(base, seed=100)
        got = dict(vae.vision.encoder.named_parameters())
        for name, p in ae.vision.encoder.named_parameters():
            if name.endswith("out.weight") or name.endswith("out.bias"):
                d = len(p)
                assert torch.equal(got[name][:d], p), name
                assert not got[name][d:].any(), name
                assert len(got[name]) == 2 * d
            else:
                assert torch.equal(got[name], p), name
        for (a, p), (b, q) in zip(
            vae.decoder.named_parameters(), ae.decoder.named_parameters(), strict=True
        ):
            assert a == b and torch.equal(p, q)
        assert float(vae.log_sigma) == pytest.approx(math.log(0.25))
        assert vae.code_shape == (128, 8)

    def test_the_posterior_of_a_stack(self):
        """On a small ViT: mu is the autoencoder's code, log s^2 is 0, and
        the VAE's g(mu) is the autoencoder's g(f(x))."""
        conf = replace(
            packed_conf("bp", token_dim=4),
            program="vae",
            ae=AEConf(decoder_depth=1, sigma0_sq=0.1),
        )
        vae = build_vae(conf, seed=3)
        ae = build_ae(replace(conf, program="ae"), seed=3)
        x = stacks(2)
        with torch.no_grad():
            mu, logvar = vae.posterior(x)
            assert torch.equal(mu, ae.code(x)) and torch.equal(vae.code(x), mu)
            assert mu.shape == (2, *vae.code_shape) and not logvar.any()
            assert torch.equal(vae(x), ae(x))

    def test_log_sigma_steps_by_eta_and_the_head_by_eta_over_its_width(self):
        """``scale=fan_in``: log sigma, held by LogSigma itself, has fan-in 1
        (a vector: the step eta); the widened head keeps the fan-in 256 of
        the encoder's last map."""
        conf, _ = _vae()
        vae = build_vae(conf, seed=100)
        n = fan_ins(*vae.named().values())
        assert n[vae.log_sigma] == 1
        assert n[vae.head.weight] == 256 and n[vae.head.bias] == 1
        assert set(n) == set(vae.parameters())

    def test_it_needs_sigma0(self):
        conf, _ = _vae()
        assert conf.ae is not None
        with pytest.raises(ValueError, match="sigma0_sq"):
            build_vae(replace(conf, ae=replace(conf.ae, sigma0_sq=None)), seed=0)
