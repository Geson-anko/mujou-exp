"""``exp.lib.regions``: the masks' cells and pixels by hand, the regions'
images, and the harmonic fill against its definition (constants kept, each
filled pixel the mean of its neighbours, the maximum principle), and the
transplant's pixels. That the
masks are of the JEPA's mask family is in ``tests/vision/test_jepa.py``."""

import pytest
import torch

from exp.lib.regions import (
    EVAL_MASKS,
    GREY,
    SWAP_PAIRS,
    harmonic_fill,
    pixel_mask,
    region_image,
    transplant,
)


def frames(b: int = 2, size: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(256, (b, 3, size, size), generator=g, dtype=torch.uint8)


class TestEvalMasks:
    def test_the_counts_and_the_quarters_tile_the_frame(self):
        counts = {k: int(m.sum()) for k, m in EVAL_MASKS.items()}
        assert counts == {
            **dict.fromkeys(("Q_tl", "Q_tr", "Q_bl", "Q_br"), 4),
            **dict.fromkeys(("H_l", "H_r", "H_t", "H_b", "S_1", "S_2"), 8),
            "R": 12,
        }
        q = torch.stack([EVAL_MASKS[k] for k in ("Q_tl", "Q_tr", "Q_bl", "Q_br")])
        assert torch.equal(q.sum(0), torch.ones(16, dtype=torch.long))
        for s, (a, b) in SWAP_PAIRS.items():
            assert torch.equal(EVAL_MASKS[s], EVAL_MASKS[a] | EVAL_MASKS[b])

    def test_the_pixels_of_q_tl_and_of_rs_visible_centre(self):
        q = pixel_mask(EVAL_MASKS["Q_tl"], 64)
        assert q[:32, :32].all() and int(q.sum()) == 32 * 32
        visible = ~pixel_mask(EVAL_MASKS["R"], 64)
        assert visible[16:48, 16:48].all() and int(visible.sum()) == 32 * 32


class TestRegionImage:
    def test_a_quarter_and_a_half_are_their_boxes(self):
        f = frames(size=64)
        q = region_image(f, pixel_mask(EVAL_MASKS["Q_br"], 64))
        assert torch.equal(q, f[:, :, 32:, 32:])
        h = region_image(f, pixel_mask(EVAL_MASKS["H_t"], 64))
        assert torch.equal(h, f[:, :, :32])

    def test_the_rim_is_the_frame_with_its_centre_grey(self):
        f = frames(size=64)
        r = region_image(f, pixel_mask(EVAL_MASKS["R"], 64))
        assert r.shape == f.shape
        assert bool((r[:, :, 16:48, 16:48] == GREY).all())
        inner = torch.zeros(64, 64, dtype=torch.bool)
        inner[16:48, 16:48] = True
        assert torch.equal(r[:, :, ~inner], f[:, :, ~inner])


class TestHarmonicFill:
    @pytest.mark.parametrize("name", ["Q_tl", "R", "S_1"])
    def test_its_definition_on_random_frames(self, name):
        size = 16
        hidden = pixel_mask(EVAL_MASKS[name], size)
        f = frames(size=size)
        out = harmonic_fill(f, hidden)
        x = out.flatten(0, 1)  # (B C, H, W)
        pad = torch.nn.functional.pad(x, (1, 1, 1, 1))
        ones = torch.nn.functional.pad(
            torch.ones(size, size, dtype=x.dtype), (1, 1, 1, 1)
        )
        nb_sum = (
            pad[:, :-2, 1:-1] + pad[:, 2:, 1:-1] + pad[:, 1:-1, :-2] + pad[:, 1:-1, 2:]
        )
        deg = ones[:-2, 1:-1] + ones[2:, 1:-1] + ones[1:-1, :-2] + ones[1:-1, 2:]
        mean = nb_sum / deg
        torch.testing.assert_close(x[:, hidden], mean[:, hidden], rtol=0, atol=1e-6)
        # the visible pixels kept; the maximum principle
        torch.testing.assert_close(out[:, :, ~hidden], f[:, :, ~hidden].double())
        vis = f.double()[:, :, ~hidden]
        lo, hi = vis.amin(2, keepdim=True), vis.amax(2, keepdim=True)
        filled = out[:, :, hidden]
        assert bool((filled >= lo - 1e-9).all() and (filled <= hi + 1e-9).all())

    def test_a_constant_frame_is_kept(self):
        f = torch.full((1, 3, 16, 16), 77, dtype=torch.uint8)
        out = harmonic_fill(f, pixel_mask(EVAL_MASKS["R"], 16))
        torch.testing.assert_close(out, f.double(), rtol=0, atol=1e-9)


class TestTransplant:
    def stacks(self, b: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(256, (b, 2, 3, 16, 16), generator=g, dtype=torch.uint8)

    def test_the_region_is_the_sources_and_the_rest_the_destinations(self):
        dst, src = self.stacks(3, 0), self.stacks(3, 1)
        hidden = pixel_mask(EVAL_MASKS["Q_br"], 16)
        out = transplant(dst, src, hidden)
        assert out.dtype == torch.uint8 and out.shape == dst.shape
        assert torch.equal(out[..., hidden], src[..., hidden])
        assert torch.equal(out[..., ~hidden], dst[..., ~hidden])
        assert torch.equal(transplant(dst, dst, hidden), dst)

    def test_one_destination_takes_each_source(self):
        dst, src = self.stacks(1, 2), self.stacks(4, 3)
        hidden = pixel_mask(EVAL_MASKS["Q_tl"], 16)
        out = transplant(dst, src, hidden)
        for i in range(4):
            assert torch.equal(out[i], transplant(dst, src[i : i + 1], hidden)[0])
