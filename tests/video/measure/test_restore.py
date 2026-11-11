import torch

from exp.video.jepa.vit import build_vision, patchify_tubelets
from exp.video.measure.restore import (
    fit_token_decoder,
    last_slot,
    patches_of,
    picture_of,
    restore,
    token_decoder,
)
from tests.video.packed import packed_conf, stacks


def _mask(n: int) -> torch.Tensor:
    """Spatial masks (n, 4, 4) of 10 masked patches, 6 visible."""
    g = torch.Generator().manual_seed(0)
    out = torch.ones(n, 16, dtype=torch.bool)
    for i in range(n):
        out[i, torch.randperm(16, generator=g)[:6]] = False
    return out.view(n, 4, 4)


class TestRestore:
    def test_it_never_reads_the_masked_pixels(self):
        """Changing the pixels of the masked tubelets (in every frame) leaves
        the restoration as it was: it sees the context view only."""
        mods = build_vision(packed_conf(), seed=0)
        torch.manual_seed(0)
        dec = token_decoder(16, 4, 3)
        x, m = stacks(3), _mask(3)
        a, pix = restore(mods, x, m, dec, keep=6)
        y = x.clone()
        px = m.repeat_interleave(4, 1).repeat_interleave(4, 2)  # (3, 16, 16)
        y[px[:, None, None].expand_as(y)] = 7
        b, _ = restore(mods, y, m, dec, keep=6)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.equal(pix, px)
        z, _ = restore(mods, stacks(3, seed=1), m, dec, keep=6)
        assert not torch.equal(a, z)

    def test_the_visible_patches_are_the_context_tokens_drawn(self):
        """A visible patch of the last frame is dec of the context's own token
        of the last slot (ids in order: slot 0's 6, then slot 1's 6)."""
        mods = build_vision(packed_conf(), seed=0)
        torch.manual_seed(0)
        dec = token_decoder(16, 4, 3)
        x, m = stacks(2), _mask(2)
        img, _ = restore(mods, x, m, dec, keep=6)
        masked = m.flatten(1).repeat(1, 2)  # (2, 32): both slots
        for i in range(2):
            ids = (~masked[i]).nonzero().flatten()
            seq = patchify_tubelets(x[i : i + 1].float() / 255, 4, 2)[:, ids]
            h = torch.cat([seq, ids[None, :, None].float()], 2)
            with torch.no_grad():
                ctx = mods.encoder(h, (len(ids),))[0]  # (12, D)
                drawn = dec(ctx[6:])  # the last slot's visible tokens
            got = patches_of(img[i : i + 1], 4)[ids[6:] - 16]
            torch.testing.assert_close(got, drawn)


class TestPatches:
    def test_picture_inverts_patches(self):
        x = torch.rand(2, 3, 16, 16)
        assert torch.equal(picture_of(patches_of(x, 4), 2, (4, 4)), x)

    def test_the_token_decoder_learns_a_linear_map(self):
        """A map the decoder can represent: the error falls well below that of
        the mean patch."""
        g = torch.Generator().manual_seed(0)
        tokens = torch.randn(512, 8, generator=g)
        w = torch.randn(8, 12, generator=g) / 8
        patches = (0.5 + tokens @ w).clamp(0, 1).view(-1, 3, 2, 2)
        dec = fit_token_decoder(tokens, patches, steps=300, batch=128, lr=3e-3, gen=g)
        with torch.no_grad():
            err = float(((dec(tokens) - patches) ** 2).mean())
        base = float(((patches.mean(0) - patches) ** 2).mean())
        assert err < 0.1 * base

    def test_last_slot(self):
        mods = build_vision(packed_conf(), seed=0)
        t = torch.arange(32.0)[None, :, None]
        assert last_slot(mods, t).flatten().tolist() == list(range(16, 32))
