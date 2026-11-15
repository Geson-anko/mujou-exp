import math

import pytest
import torch

from exp.video.jepa.masks import (
    ASPECT,
    CANDIDATES,
    KEEP_QUANTILE,
    _union,
    block_masks,
    block_sides,
    expected_keep,
    make_views,
    multiblock_masks,
    pack_views,
    paint,
    spatial_keep,
    token_keep,
    view_tokens,
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

    @pytest.mark.parametrize(
        "mask", [MaskConf(blocks=8, scale=0.15), MaskConf(blocks=2, scale=0.7)]
    )
    def test_the_visible_patches_are_outside_the_first_union_that_leaves_keep(
        self, mask
    ):
        """V-JEPA's masks on 8 x 8 at their expected_keep: replaying the
        generator, mask i is drawn from the first of its CANDIDATES unions
        that leaves at least keep patches visible, and none of its visible
        patches is in that union (the blocks are never shown)."""
        n, keep = 256, expected_keep(GRID, mask)
        m = multiblock_masks(
            n, GRID, mask, keep=keep, generator=torch.Generator().manual_seed(3)
        )
        unions = _union(
            n * CANDIDATES, GRID, mask, torch.Generator().manual_seed(3)
        ).view(n, CANDIDATES, -1)
        for i in range(n):
            first = next(c for c in range(CANDIDATES) if (~unions[i, c]).sum() >= keep)
            assert not (~m[i].flatten() & unions[i, first]).any()

    def test_with_no_union_of_enough_visible_patches_masked_ones_are_shown(self):
        """Two blocks of 0.7 on 8 x 8 never leave 63 visible: the mask keeps
        all the visible patches of the union of the most and shows masked
        ones up to 63."""
        mask = MaskConf(blocks=2, scale=0.7)
        n = 32
        m = multiblock_masks(
            n, GRID, mask, keep=63, generator=torch.Generator().manual_seed(4)
        )
        unions = _union(
            n * CANDIDATES, GRID, mask, torch.Generator().manual_seed(4)
        ).view(n, CANDIDATES, -1)
        for i in range(n):
            visible = (~unions[i]).sum(1)
            best = unions[i, int(visible.argmax())]
            assert (~m[i].flatten())[~best].all() and int((~m[i]).sum()) == 63

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

    @pytest.mark.parametrize(
        "mask", [MaskConf(blocks=8, scale=0.15), MaskConf(blocks=2, scale=0.7)]
    )
    def test_it_is_the_lower_quartile_of_the_visible_counts(self, mask):
        """The lower KEEP_QUANTILE quantile k of the same 4096 draws: at most
        a quarter leave fewer than k visible, at least a quarter at most k."""
        keep = expected_keep(GRID, mask)
        g = torch.Generator().manual_seed(0)
        visible = (~_union(4096, GRID, mask, g)).flatten(1).sum(1)
        assert float((visible < keep).double().mean()) <= KEEP_QUANTILE
        assert float((visible <= keep).double().mean()) >= KEEP_QUANTILE


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
        h, y = pack_views(x, views, keeps=[2, 1], patch=4, tubelet=2)
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

    def test_a_future_view_shows_the_tube_outside_its_blocks_in_the_past_only(self):
        """Four slots of 2 x 2 patches; the view masks patch 3 and the last
        slot whole: it shows patches 0, 1, 2 of slots 0, 1, 2 (ids 0-2, 4-6,
        8-10), and y marks every other token."""
        g = torch.Generator().manual_seed(1)
        x = torch.randint(0, 256, (1, 8, 3, 8, 8), generator=g).to(torch.uint8)
        spatial = torch.tensor([[[False, False], [False, True]]])
        h, y = pack_views(x, [spatial], keeps=[3], patch=4, tubelet=2, futures=[1])
        tokens = patchify_tubelets(x.float() / 255, 4, 2)
        t = 16
        shown = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        assert h.shape == (1, t + 9, 96 + 1)
        assert h[0, t:, -1].tolist() == shown
        assert torch.equal(h[0, t:, :-1], tokens[0, shown])
        want = torch.ones(t)
        want[shown] = 0.0
        assert torch.equal(y[0], want)


class TestSpaceTimeViews:
    @pytest.mark.parametrize("future", [1, 3])
    def test_no_visible_token_is_in_a_block_or_in_the_future(self, future):
        """Drawn as in training: every visible token of a space-time view is
        outside its blocks' tube and before the future slots, and there are
        token_keep of them."""
        slots, per = 4, GRID[0] * GRID[1]
        mask = MaskConf(blocks=4, scale=0.15, future=future)
        k = spatial_keep(GRID, mask)
        g = torch.Generator().manual_seed(2)
        spatial = multiblock_masks(64, GRID, mask, keep=k, generator=g)
        x = torch.zeros(64, 2 * slots, 1, 32, 32)
        h, y = pack_views(x, [spatial], keeps=[k], patch=4, tubelet=2, futures=[future])
        ids = h[:, slots * per :, -1].long()  # (64, K)
        assert ids.shape[1] == token_keep(GRID, slots, mask) == k * (slots - future)
        assert int(ids.max()) < (slots - future) * per
        hidden = spatial.flatten(1)  # (64, per)
        assert not bool(hidden.gather(1, ids % per).any())
        assert torch.equal(y.gather(1, ids), torch.zeros_like(ids, dtype=y.dtype))
        assert int(y.sum()) == 64 * (slots * per - ids.shape[1])
        assert torch.equal(y, view_tokens(spatial, slots, future))

    def test_the_past_alone_draws_nothing_and_keeps_every_patch(self):
        mask = MaskConf(blocks=0, scale=0.1, future=2)
        g = torch.Generator().manual_seed(3)
        state = g.get_state()
        m = multiblock_masks(5, GRID, mask, keep=spatial_keep(GRID, mask), generator=g)
        assert not bool(m.any()) and torch.equal(g.get_state(), state)
        assert token_keep(GRID, 4, mask) == 2 * GRID[0] * GRID[1]


class TestTheFourByFourGrid:
    """V-JEPA's two context views (8 blocks of 0.15, 2 of 0.7) on the 4 x 4
    grid of 128 x 128 frames in patches of 32 (M17's arms of T = 128 and 64),
    against the 8 x 8 grid of patches of 16."""

    SHORT, LONG = MaskConf(blocks=8, scale=0.15), MaskConf(blocks=2, scale=0.7)

    def test_the_sides_of_the_blocks(self):
        """Short: area 0.15 x 16 = 2.4, sides round(sqrt(2.4 s)) and
        round(sqrt(2.4 / s)) for s in [0.75, 1.5]: 1 or 2. Long: area 11.2,
        sides 3 or 4, so one block covers at least 9 of the 16 patches and a
        union leaves at most 7 visible."""
        grid = (4, 4)
        for mask, sides in ((self.SHORT, {1, 2}), (self.LONG, {3, 4})):
            s = torch.exp(
                torch.linspace(math.log(mask.aspect[0]), math.log(mask.aspect[1]), 101)
            )
            h, w = block_sides(
                s, grid, ratio=mask.scale * mask.blocks, blocks=mask.blocks
            )
            assert set(h.tolist()) == sides and set(w.tolist()) == sides
        g = torch.Generator().manual_seed(0)
        visible = (~_union(4096, grid, self.LONG, g)).flatten(1).sum(1)
        assert int(visible.max()) <= 7

    def test_the_masks_are_valid_and_their_masked_fractions(self):
        """The visible patches of a slot (expected_keep): 4 and 1 of 16 on 4 x
        4 (0.75 and 0.9375 masked), 18 and 8 of 64 on 8 x 8 (0.72 and 0.875);
        between 1 and T_s - 1 (a valid view), and every drawn mask keeps
        exactly that many. A union leaves fewer with probability at most
        KEEP_QUANTILE (the lower quartile), so masked patches are shown at most
        with KEEP_QUANTILE ** CANDIDATES."""
        for grid, keeps in (((4, 4), (4, 1)), ((8, 8), (18, 8))):
            t = grid[0] * grid[1]
            for mask, keep in zip((self.SHORT, self.LONG), keeps, strict=True):
                assert expected_keep(grid, mask) == spatial_keep(grid, mask) == keep
                assert 1 <= keep <= t - 1
                g = torch.Generator().manual_seed(1)
                drawn = multiblock_masks(256, grid, mask, keep=keep, generator=g)
                assert ((~drawn).flatten(1).sum(1) == keep).all()
        assert KEEP_QUANTILE**CANDIDATES < 2e-5
