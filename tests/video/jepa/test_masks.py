import math

import pytest
import torch

from exp.video.jepa.masks import (
    ASPECT,
    block_masks,
    block_sides,
    expected_keep,
    make_views,
    multiblock_masks,
    pack_views,
    paint,
)
from exp.video.jepa.vit import patchify_tubelets
from exp.video.schema.vision import MaskConf

GRID = (8, 8)


def _draw(n: int, seed: int = 0, **kw) -> torch.Tensor:
    kw = {"ratio": 0.5, "blocks": 4} | kw
    return block_masks(n, GRID, generator=torch.Generator().manual_seed(seed), **kw)


def _side_range(ratio: float, blocks: int) -> tuple[range, range]:
    """The heights and widths a block can take: h = round(sqrt(a s)) grows and
    w = round(sqrt(a / s)) falls with the aspect s in ASPECT."""
    lo, hi = torch.tensor(ASPECT)
    h_lo, w_hi = block_sides(lo, GRID, ratio=ratio, blocks=blocks)
    h_hi, w_lo = block_sides(hi, GRID, ratio=ratio, blocks=blocks)
    return range(int(h_lo), int(h_hi) + 1), range(int(w_lo), int(w_hi) + 1)


class TestBlockMasks:
    def test_one_block_is_a_filled_rectangle_of_the_drawn_sides(self):
        """With one block, the mask is its bounding box, whose sides are those
        that block_sides gives for some aspect in ASPECT."""
        heights, widths = _side_range(0.3, 1)
        for m in _draw(200, ratio=0.3, blocks=1):
            rows = m.any(1).nonzero().flatten()
            cols = m.any(0).nonzero().flatten()
            h, w = len(rows), len(cols)
            assert m.sum() == h * w  # filled
            assert rows[-1] - rows[0] + 1 == h and cols[-1] - cols[0] + 1 == w
            assert h in heights and w in widths

    def test_the_union_of_blocks_covers_at_most_their_areas(self):
        """smallest block <= |union| <= blocks x the largest block, and the
        union is never the whole grid at ratio 0.5 (the blocks are about
        ratio T / blocks each, so they cover about ratio T at most)."""
        heights, widths = _side_range(0.5, 4)
        largest, smallest = max(heights) * max(widths), min(heights) * min(widths)
        assert 4 * largest < math.prod(GRID)
        counts = _draw(2000, ratio=0.5, blocks=4).flatten(1).sum(1)
        assert (counts <= 4 * largest).all() and (counts >= smallest).all()

    @pytest.mark.parametrize(("ratio", "blocks"), [(0.3, 1), (0.5, 4), (0.6, 2)])
    def test_the_mean_coverage_is_that_of_independent_uniform_blocks(
        self, ratio, blocks
    ):
        """A block of aspect s covers cell (r, c) with probability
        P_h(r) P_w(c), P_h(r) = #{top <= r < top + h} / (gh - h + 1) (the top
        is uniform on 0..gh - h), and the blocks are independent, so
        E|mask| = sum_cells 1 - (1 - E_s[P_h(r) P_w(c)])^blocks, with s
        log-uniform on ASPECT (by quadrature over it). The mean of 20000
        masks is within 4 standard errors of it."""
        gh, gw = GRID
        lo, hi = math.log(ASPECT[0]), math.log(ASPECT[1])
        q = 20000
        aspect = torch.exp(lo + (hi - lo) * (torch.arange(q) + 0.5) / q)
        h, w = block_sides(aspect, GRID, ratio=ratio, blocks=blocks)  # (q,)
        r, c = torch.arange(gh), torch.arange(gw)

        def cover(side: torch.Tensor, cells: torch.Tensor, n: int) -> torch.Tensor:
            # (q, n): the tops t in [max(0, x - side + 1), min(x, n - side)]
            first = (cells[None] - side[:, None] + 1).clamp(min=0)
            last = torch.minimum(cells[None], (n - side)[:, None])
            return (last - first + 1).clamp(min=0) / (n - side + 1)[:, None]

        ph = cover(h, r, gh).double()  # (q, gh)
        pw = cover(w, c, gw).double()  # (q, gw)
        p_cell = torch.einsum("qr,qc->rc", ph, pw) / q
        expect = float((1 - (1 - p_cell) ** blocks).sum())
        counts = _draw(20000, ratio=ratio, blocks=blocks).flatten(1).sum(1).double()
        se = float(counts.std()) / math.sqrt(len(counts))
        assert abs(float(counts.mean()) - expect) < 4 * se

    def test_the_same_generator_draws_the_same_masks(self):
        assert torch.equal(_draw(16, seed=3), _draw(16, seed=3))
        assert not torch.equal(_draw(16, seed=3), _draw(16, seed=4))

    def test_paint_is_the_union_of_the_rectangles(self):
        top, left = torch.tensor([[0, 5]]), torch.tensor([[1, 6]])
        h, w = torch.tensor([[2, 3]]), torch.tensor([[3, 2]])
        m = paint(top, left, h, w, GRID)[0]
        expect = torch.zeros(GRID, dtype=torch.bool)
        expect[0:2, 1:4] = True
        expect[5:8, 6:8] = True
        assert torch.equal(m, expect)


class TestMakeViews:
    @pytest.mark.parametrize("frames", [1, 3])
    def test_masked_patches_are_zero_in_every_frame_and_marked(self, frames):
        """A tube: the patches of the mask are 0 in all N frames of a context
        view, the others keep the chunk; the extra channel is the mask in
        every frame; the target view is the chunk with a channel of 0."""
        g = torch.Generator().manual_seed(0)
        b, c, patch = 2, 3, 2
        chunks = torch.rand(b, frames, c, 8, 8, generator=g) + 0.1  # no zeros
        masks = block_masks(b * 2, (4, 4), ratio=0.4, blocks=2, generator=g)
        masks = masks.unflatten(0, (b, 2))
        h, y = make_views(chunks, masks, patch)
        assert h.shape == (b * 3, frames, c + 1, 8, 8) and y.shape == (b * 3, 16)
        h = h.unflatten(0, (b, 3))
        y = y.unflatten(0, (b, 3))
        assert torch.equal(h[:, 0, :, :c], chunks)
        assert (h[:, 0, :, c] == 0).all() and (y[:, 0] == 0).all()
        for v in (1, 2):
            m = masks[:, v - 1]  # (b, 4, 4)
            px = m.repeat_interleave(patch, 1).repeat_interleave(patch, 2)
            px = px[:, None, None].expand(b, frames, c, 8, 8)
            view = h[:, v]
            assert (view[:, :, :c][px] == 0).all()
            assert torch.equal(view[:, :, :c][~px], chunks[~px])
            assert torch.equal(view[:, :, c], px[:, :, 0].float())
            assert torch.equal(y[:, v], m.flatten(1).float())


class TestMultiblockMasks:
    @pytest.mark.parametrize("keep", [1, 10, 40, 63])
    def test_exactly_keep_patches_are_visible(self, keep):
        g = torch.Generator().manual_seed(0)
        for mask in (MaskConf(blocks=8, scale=0.15), MaskConf(blocks=2, scale=0.7)):
            m = multiblock_masks(64, GRID, mask, keep=keep, generator=g)
            assert m.shape == (64, *GRID) and m.dtype == torch.bool
            assert ((~m).flatten(1).sum(1) == keep).all()

    def test_the_same_generator_draws_the_same_masks(self):
        mask = MaskConf(blocks=8, scale=0.15)
        a, b = (
            multiblock_masks(
                5, GRID, mask, keep=20, generator=torch.Generator().manual_seed(1)
            )
            for _ in range(2)
        )
        assert torch.equal(a, b)

    def test_a_block_of_the_kept_count_is_left_as_drawn(self):
        """One square block of 5 x 5 on 10 x 10 (scale 0.25, aspect 1) leaves
        75 visible: with keep 75 the mask is exactly one filled 5 x 5 square."""
        mask = MaskConf(blocks=1, scale=0.25, aspect=[1.0, 1.0])
        g = torch.Generator().manual_seed(2)
        m = multiblock_masks(16, (10, 10), mask, keep=75, generator=g)
        for one in m:
            rows, cols = one.nonzero(as_tuple=True)
            assert one.sum() == 25
            assert rows.max() - rows.min() == 4 and cols.max() - cols.min() == 4


class TestExpectedKeep:
    def test_one_square_block_by_hand(self):
        mask = MaskConf(blocks=1, scale=0.25, aspect=[1.0, 1.0])
        assert expected_keep((10, 10), mask) == 75

    @pytest.mark.parametrize(
        ("mask", "expect"),
        [(MaskConf(blocks=4, scale=1.0), 1), (MaskConf(blocks=1, scale=0.001), 63)],
    )
    def test_it_is_never_none_or_all(self, mask, expect):
        assert expected_keep(GRID, mask) == expect


class TestPackViews:
    def test_the_target_then_the_visible_tokens_of_each_view_in_id_order(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 256, (2, 4, 3, 8, 8), generator=g).to(torch.uint8)
        views = [
            torch.tensor([[[True, False], [False, True]]] * 2),
            torch.tensor(
                [[[False, True], [True, True]], [[True, True], [True, False]]]
            ),
        ]
        h, y = pack_views(x, views, patch=4, tubelet=2)
        tokens = patchify_tubelets(x.float() / 255, 4, 2)  # (2, 8, 96)
        t = 8
        assert h.shape == (2, t + 4 + 2, 96 + 1) and y.shape == (2, 2 * t)
        assert torch.equal(h[:, :t, :-1], tokens)
        assert h[:, :t, -1].tolist() == [list(range(t))] * 2
        # view 0 shows spatial patches 1 and 2 in both slots: ids 1, 2, 5, 6
        assert h[:, t : t + 4, -1].tolist() == [[1, 2, 5, 6]] * 2
        assert torch.equal(h[:, t : t + 4, :-1], tokens[:, [1, 2, 5, 6]])
        # view 1 shows patch 0 of stream 0 and patch 3 of stream 1
        assert h[:, t + 4 :, -1].tolist() == [[0, 4], [3, 7]]
        assert torch.equal(h[1, t + 4 :, :-1], tokens[1, [3, 7]])
        tube = [v.flatten(1).repeat(1, 2).float() for v in views]
        assert torch.equal(y, torch.cat(tube, 1))
