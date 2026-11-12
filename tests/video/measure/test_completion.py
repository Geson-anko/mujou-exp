"""The completion error of the packed layout (M17): the scale-free $1 - R^2$
(perfect 0, the targets' mean 1, the same under any scale of the tokens), the
future mask, and the path of the training loss."""

import math

import pytest
import torch

from exp.video.jepa.loss import jepa_loss_packed, view_ids
from exp.video.jepa.masks import expected_keep, multiblock_masks, pack_views
from exp.video.jepa.vit import build_vision
from exp.video.measure import completion as completion_module
from exp.video.measure.completion import (
    Unexplained,
    complete,
    completion,
    future_slots,
    future_tokens,
    position_mean,
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


def _within(target: torch.Tensor, pred: torch.Tensor) -> float:
    u = Unexplained(["p"])
    u.add(target, {"p": pred})
    return u.within()["p"]


class TestWithin:
    """$E^n$: the error over the spread of the masked tokens within each
    stack."""

    def test_each_stacks_own_mean_is_one_and_a_perfect_prediction_zero(self):
        s = _targets()
        assert _within(s, s.mean(1, keepdim=True)) == pytest.approx(1.0, rel=1e-12)
        assert _within(s, s) == 0.0

    def test_a_hand_computed_value(self):
        # stack 0: targets 0, 2 (its mean 1, spread 2); stack 1: 5, 5 (spread
        # 0); predictions off by 1 everywhere: 4 over 2
        s = torch.tensor([[[0.0], [2.0]], [[5.0], [5.0]]])
        assert _within(s, s + 1) == pytest.approx(4 / 2)

    def test_the_value_is_the_same_under_any_scale_and_batches_sum(self):
        s, p = _targets(0).double(), _targets(1).double()
        base = _within(s, p)
        for c in (1e-3, 40.0):
            assert _within(c * s, c * p) == pytest.approx(base, rel=1e-9)
        u = Unexplained(["p"])
        u.add(s[:2], {"p": p[:2]})
        u.add(s[2:], {"p": p[2:]})
        assert u.within()["p"] == pytest.approx(base, rel=1e-12)

    def test_tokens_collapsed_within_the_stacks_are_caught(self):
        """The audit's pilot: tokens nearly constant within a stack, spread
        between the stacks. The mean of the stack's tokens predicts them with
        a small E (the spread between the stacks is in its denominator), but
        not with a small E^n."""
        g = torch.Generator().manual_seed(0)
        clip = torch.randn(64, 1, 8, generator=g)
        s = clip + 1e-2 * torch.randn(64, 12, 8, generator=g)
        guess = clip + 1e-2 * torch.randn(64, 1, 8, generator=g)
        assert _value(s, guess) < 1e-3
        assert _within(s, guess) > 1.5


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
        """``_vis`` is the mean of the visible target tokens of each stack,
        ``_mean`` the fixed token and ``_pos`` the reference's mean token at
        each masked position, through the same normalization."""
        conf = packed_conf()
        v = conf.vision
        assert v is not None
        mods = build_vision(conf, seed=0)
        x = stacks(4, frames=v.frames)
        masked = future_tokens(4, mods.grid3, 1)
        keep = int((masked[0] == 0).sum())
        ref = position_mean(mods, x[2:], batch=1, device=torch.device("cpu"))
        got = completion(
            mods,
            x,
            {"future": (masked, keep)},
            reference=ref,
            batch=3,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            target = mods.tokens(x)
            pred, goal, vis = complete(mods, x, target, masked, keep)
        _, ids = view_ids(masked, keep)
        fixed = ref.mean(0)
        assert torch.allclose(ref, mods.tokens(x[2:]).mean(0), atol=1e-6)
        assert torch.allclose(fixed, mods.tokens(x[2:]).mean((0, 1)), atol=1e-6)
        assert got["comp_future"] == pytest.approx(_value(goal, pred), rel=1e-6)
        assert got["comp_future_vis"] == pytest.approx(
            _value(goal, vis[:, None]), rel=1e-6
        )
        assert got["comp_future_mean"] == pytest.approx(_value(goal, fixed), rel=1e-6)
        assert got["comp_future_pos"] == pytest.approx(_value(goal, ref[ids]), rel=1e-6)
        assert got["ncomp_future"] == pytest.approx(_within(goal, pred), rel=1e-6)
        assert got["ncomp_future_vis"] == pytest.approx(
            _within(goal, vis[:, None]), rel=1e-6
        )
        assert got["ncomp_future_pos"] == pytest.approx(
            _within(goal, ref[ids]), rel=1e-6
        )
        assert set(got) == {
            f"{k}_future{s}"
            for k in ("comp", "ncomp")
            for s in ("", "_vis", "_mean", "_pos")
        }


def _of_position(g: torch.Tensor, noise: float):
    """``targets`` of tokens $g(t)$ ``(T, D)`` plus i.i.d. noise of variance
    ``noise``, the noise of a stack drawn from a seed of its pixels (the same
    tokens for the same stack in any batch)."""

    def targets(mods, x, *, normalize):
        seeds = x.long().flatten(1).sum(1).tolist()
        e = [
            torch.randn(g.shape, generator=torch.Generator().manual_seed(k))
            for k in seeds
        ]
        return g + noise**0.5 * torch.stack(e)

    return targets


class TestPosition:
    r"""``_pos`` against tokens of the position alone, $s(x, t) = g(t) + e$,
    $e$ i.i.d. of variance $v$ in each of the $D$ dimensions: per token its
    error is $D v$ and the spread of the masked tokens over the set $D v +
    \mathrm{Var}_t g$ ($\mathrm{Var}_t g$ over the masked positions, summed
    over the dimensions), so $E = D v / (D v + \mathrm{Var}_t g)$, 0 at $v =
    0$; and on the reference itself it is at most ``_mean``."""

    def _run(self, monkeypatch, noise: float, n: int, same: bool):
        conf = packed_conf()
        v = conf.vision
        assert v is not None
        mods = build_vision(conf, seed=0)
        t, d = mods.tokens_per_view, mods.predictor.out.out_features
        g = torch.randn(t, d, generator=torch.Generator().manual_seed(7))
        monkeypatch.setattr(completion_module, "targets", _of_position(g, noise))
        x = stacks(n, frames=v.frames, seed=1)
        ref_x = x if same else stacks(n, frames=v.frames, seed=2)
        masked = future_tokens(n, mods.grid3, 1)
        keep = int((masked[0] == 0).sum())
        cpu = torch.device("cpu")
        ref = position_mean(mods, ref_x, batch=16, device=cpu)
        got = completion(
            mods, x, {"future": (masked, keep)}, reference=ref, batch=16, device=cpu
        )
        hidden = g[keep:].double()  # the future: the last ids
        spread = float(hidden.var(0, unbiased=False).sum())
        return got, spread, d

    def test_tokens_of_the_position_alone_are_zero(self, monkeypatch):
        got, _, _ = self._run(monkeypatch, 0.0, 8, same=False)
        assert got["comp_future_pos"] == pytest.approx(0.0, abs=1e-12)
        assert got["comp_future_mean"] > 0.5

    def test_with_noise_it_is_v_over_v_plus_the_spread_of_g(self, monkeypatch):
        noise = 0.5
        got, spread, d = self._run(monkeypatch, noise, 2048, same=False)
        expect = d * noise / (d * noise + spread)  # the noise of d dimensions
        assert got["comp_future_pos"] == pytest.approx(expect, rel=0.03)

    def test_on_the_reference_it_is_at_most_mean(self, monkeypatch):
        got, _, _ = self._run(monkeypatch, 2.0, 16, same=True)
        assert got["comp_future_pos"] <= got["comp_future_mean"]
