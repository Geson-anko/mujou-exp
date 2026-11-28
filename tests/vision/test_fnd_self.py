r"""``exp.vision.fnd_self``: the predictor's own completion of a half.

The oracles: the nearest seen position of a 4-times x 4 x 4 grid by hand
(the future's hidden times 2 and 3 copy time 1, the right half's columns 2
and 3 copy column 1, a seen position itself); the sums of ``cut_terms`` by
hand (a perfect prediction scores 0, tokens constant along time have a copy
of 0 on the temporal cut, tokens at their mean have a mean's sum of 0); the
predictor's $\hat t$ of a half does not move, to the bit, when the pixels
outside the seen half change, and equals ``fnd_jepa.complete`` of the
hidden mask of the other half.
"""

import pytest
import torch

from exp.vision.fnd_guard import half_keep
from exp.vision.fnd_jepa import (
    TOKEN_DIM,
    build_fnd,
    complete,
    context_index,
    encode,
)
from exp.vision.fnd_self import CUTS, cut_terms, nearest_seen, predict_half
from exp.vision.models import early_fusion
from tests.vision.test_fnd_guard import _outside
from tests.vision.test_fnd_jepa import GRID, TIMES, conf_of, stacks

G = (TIMES, GRID, GRID)


def _flat(t: int, r: int, c: int) -> int:
    return (t * GRID + r) * GRID + c


class TestNearestSeen:
    def test_the_future_copies_the_last_time_seen(self):
        n = nearest_seen("early", G)
        assert int(n[_flat(2, 1, 3)]) == _flat(1, 1, 3)
        assert int(n[_flat(3, 0, 0)]) == _flat(1, 0, 0)
        assert int(n[_flat(0, 2, 2)]) == _flat(0, 2, 2)

    def test_the_past_copies_the_first_time_seen(self):
        n = nearest_seen("late", G)
        assert int(n[_flat(0, 3, 1)]) == _flat(2, 3, 1)
        assert int(n[_flat(1, 3, 1)]) == _flat(2, 3, 1)

    def test_the_present_copies_the_column_by_the_cut(self):
        n = nearest_seen("left", G)
        assert int(n[_flat(1, 2, 3)]) == _flat(1, 2, 1)
        assert int(n[_flat(1, 2, 2)]) == _flat(1, 2, 1)
        n = nearest_seen("bottom", G)
        assert int(n[_flat(3, 0, 0)]) == _flat(3, 2, 0)

    @pytest.mark.parametrize("axis,src,dst", CUTS)
    def test_every_cut_maps_into_its_seen_half(self, axis, src, dst):
        seen = half_keep(src, G).flatten()
        assert bool(seen[nearest_seen(src, G)].all())
        assert not bool((seen & half_keep(dst, G).flatten()).any())

    def test_an_unknown_half(self):
        with pytest.raises(ValueError):
            nearest_seen("middle", G)


class TestCutTerms:
    def test_by_hand(self):
        g = torch.Generator().manual_seed(0)
        t = torch.randn(3, TIMES * GRID * GRID, TOKEN_DIM, generator=g)
        dst = half_keep("late", G).flatten()
        near = nearest_seen("early", G)
        mean = t.mean(0)
        t_hat = t + 0.5
        got = cut_terms(t_hat, t, dst, near, mean)
        want_num = 0.25 * TOKEN_DIM * int(dst.sum())
        assert torch.allclose(
            got[:, 0], torch.full((3,), want_num, dtype=torch.float64)
        )
        copy = (t[:, near] - t).double().square().sum(-1)[:, dst].sum(1)
        clim = (mean - t).double().square().sum(-1)[:, dst].sum(1)
        assert torch.allclose(got[:, 1], copy)
        assert torch.allclose(got[:, 2], clim)

    def test_degenerate_cases_score_zero(self):
        g = torch.Generator().manual_seed(1)
        per_pos = torch.randn(3, 1, GRID * GRID, TOKEN_DIM, generator=g)
        t = per_pos.expand(3, TIMES, -1, -1).flatten(1, 2)  # constant along time
        dst = half_keep("late", G).flatten()
        got = cut_terms(t, t, dst, nearest_seen("early", G), t[0])
        assert float(got[:, 0].abs().max()) == 0.0
        assert float(got[:, 1].abs().max()) == 0.0
        assert float(got[0, 2]) == 0.0


class TestPredictHalf:
    @pytest.fixture
    def mods(self):
        conf = conf_of(seed=4)
        mods = build_fnd(conf, seed=4)
        with torch.no_grad():
            for p in mods.parameters():
                p.add_(0.01 * torch.randn_like(p))
        return mods, conf.model.head_count

    @pytest.mark.parametrize("axis,src,dst", CUTS)
    @pytest.mark.parametrize("scaled", [False, True])
    def test_sees_its_half_only(self, mods, axis, src, dst, scaled):
        m, heads = mods
        seen = half_keep(src, G).flatten()
        x, y = stacks(3), stacks(3, seed=9)
        out = _outside(src)
        got = predict_half(m, x, heads, TIMES, seen, scaled)
        assert torch.equal(predict_half(m, torch.where(out, y, x), heads, TIMES, seen,
                                        scaled), got)  # fmt: skip
        assert not torch.allclose(
            predict_half(m, torch.where(out, x, y), heads, TIMES, seen, scaled), got
        )

    def test_is_the_completion_of_the_other_half(self, mods):
        m, heads = mods
        x = stacks(2)
        hidden = half_keep("late", G).flatten().expand(2, -1)
        images = early_fusion(x)
        keep, valid, slot = context_index(hidden)
        with torch.no_grad():
            want = complete(
                m, encode(m, images, heads, keep, valid), slot, hidden, TIMES
            )
        got = predict_half(m, x, heads, TIMES, half_keep("early", G).flatten(), False)
        assert torch.equal(got, want)
