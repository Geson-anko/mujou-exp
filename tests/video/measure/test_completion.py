"""The completion error of the packed layout (M17): the scale-free $1 - R^2$
(perfect 0, the targets' mean 1, the same under any scale of the tokens), the
future mask, and the path of the training loss."""

import math

import pytest
import torch

from exp.video.jepa.loss import jepa_loss_packed
from exp.video.jepa.masks import expected_keep, multiblock_masks, pack_views
from exp.video.jepa.vit import build_vision
from exp.video.measure.completion import (
    Unexplained,
    complete,
    completion,
    future_slots,
    future_tokens,
    mean_token,
    tube_tokens,
)
from exp.video.schema.vision import MaskConf
from tests.video.packed import packed_conf, stacks


def _targets(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(5, 7, 3, generator=g) * torch.tensor([1.0, 3.0, 0.5]) + 2


def _value(target: torch.Tensor, pred: torch.Tensor) -> float:
    u = Unexplained(["p"])
    u.add(target, {"p": pred})
    return u.values()["p"]


class TestUnexplained:
    def test_a_perfect_prediction_is_zero(self):
        s = _targets()
        assert _value(s, s) == 0.0

    def test_the_mean_of_the_targets_is_one(self):
        s = _targets()
        mean = s.flatten(0, 1).mean(0)
        assert _value(s, mean) == pytest.approx(1.0, rel=1e-12)

    def test_the_value_is_the_same_under_any_scale_of_the_tokens(self):
        s, p = _targets(0).double(), _targets(1).double()
        base = _value(s, p)
        for c in (1e-3, 0.5, 40.0):
            assert _value(c * s, c * p) == pytest.approx(base, rel=1e-9)

    def test_a_hand_computed_value(self):
        # targets 0, 2 (mean 1, sum of squares about it 2), predictions 1, 1:
        # the squared errors 1 + 1 over 2
        s = torch.tensor([[[0.0], [2.0]]])
        p = torch.tensor([[[1.0], [3.0]]])
        assert _value(s, p) == pytest.approx((1 + 1) / 2)

    def test_batches_sum_to_the_whole(self):
        s, p = _targets(0), _targets(1)
        u = Unexplained(["p"])
        u.add(s[:2], {"p": p[:2]})
        u.add(s[2:], {"p": p[2:]})
        assert u.values()["p"] == pytest.approx(_value(s, p), rel=1e-12)

    def test_constant_targets_are_not_finite(self):
        s = torch.ones(2, 3, 4)
        assert math.isnan(_value(s, s))


class TestMasks:
    @pytest.mark.parametrize(("slots", "hidden"), [(1, 0), (2, 1), (4, 1), (8, 2)])
    def test_the_future_is_a_quarter_of_the_slots(self, slots, hidden):
        assert future_slots(slots) == hidden

    def test_the_future_hides_every_token_of_the_last_slots(self):
        m = future_tokens(3, (4, 2, 2), 1)  # 4 slots of 2 x 2 tokens
        assert m.shape == (3, 16)
        assert m[:, :12].sum() == 0 and bool((m[:, 12:] == 1).all())

    def test_a_tube_is_the_spatial_mask_in_every_slot(self):
        spatial = torch.tensor([[[True, False], [False, True]]])
        assert tube_tokens(spatial, 3).tolist() == [[1.0, 0.0, 0.0, 1.0] * 3]


class TestComplete:
    def test_the_squared_error_is_the_training_losss_prediction_term(self):
        """For one context view, the mean squared error of ``complete``'s
        prediction is ``pred`` of ``jepa_loss_packed`` (l2): the evaluation
        sees the context and the targets as training does."""
        conf = packed_conf()
        v = conf.vision
        assert v is not None
        mods = build_vision(conf, seed=0)
        x = stacks(3, frames=v.frames)
        grid = (4, 4)
        mask = MaskConf(blocks=2, scale=0.2)
        keep = expected_keep(grid, mask)
        g = torch.Generator().manual_seed(1)
        m = multiblock_masks(3, grid, mask, keep=keep, generator=g)
        slots = v.frames // v.tubelet
        with torch.no_grad():
            h, y = pack_views(x, [m], keeps=[keep], patch=v.patch, tubelet=v.tubelet)
            terms = jepa_loss_packed(
                mods.encoder(h, (mods.tokens_per_view, keep * slots)),
                y,
                torch.ones(3, dtype=torch.bool),
                torch.eye(v.dim)[:4],
                None,
                mods.predictor,
                keeps=[keep * slots],
                proj=None,
                lam=0.0,
                sg=False,
                sigreg_on="z",
                pred_loss="l2",
            )
            pred, goal, _ = complete(
                mods, x, mods.tokens(x), tube_tokens(m, slots), keep * slots
            )
        assert float(((pred - goal) ** 2).mean()) == pytest.approx(
            float(terms.pred), rel=1e-5
        )

    def test_the_trivial_predictors(self):
        """``_vis`` is the mean of the visible target tokens of each stack and
        ``_mean`` the fixed token, through the same normalization."""
        conf = packed_conf()
        v = conf.vision
        assert v is not None
        mods = build_vision(conf, seed=0)
        x = stacks(4, frames=v.frames)
        masked = future_tokens(4, mods.grid3, 1)
        keep = int((masked[0] == 0).sum())
        fixed = mean_token(mods, x[2:], batch=2, device=torch.device("cpu"))
        got = completion(
            mods,
            x,
            {"future": (masked, keep)},
            mean_token=fixed,
            batch=3,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            target = mods.tokens(x)
            pred, goal, vis = complete(mods, x, target, masked, keep)
        assert got["comp_future"] == pytest.approx(_value(goal, pred), rel=1e-6)
        assert got["comp_future_vis"] == pytest.approx(
            _value(goal, vis[:, None]), rel=1e-6
        )
        assert got["comp_future_mean"] == pytest.approx(_value(goal, fixed), rel=1e-6)
        assert torch.allclose(fixed, mods.tokens(x[2:]).mean((0, 1)), atol=1e-6)
