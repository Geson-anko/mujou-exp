r"""``exp.lib.gru_dyn``: the sets, the window the predictor reads, the
denominators and the window's oracle.

The oracles: the anchors of a margin by counting; the reversed set is an
involution whose $t + \Delta$ is the original's $t - \Delta$; the
prediction at an anchor is a function of its window alone (codes outside
it changed, the prediction to the bit; inside, changed); the denominators
of ``gru_sums`` by hand; ``best_window`` of white noise is $k$, of a
scalar AR(1) $1 - r^{2\Delta}$, of a window of 2 at lag 1 the two-point
oracle ``dynpred.best_linear``, and of a long window the Kalman filter's
error (the window's optimum converges to the whole past's, geometrically);
the prediction before training is the two-point linear by hand; the
training of one seed is the same twice.
"""

import math
from dataclasses import replace

import pytest
import torch

from exp.lib.dynpred import best_linear, fit_linear
from exp.lib.gru_dyn import (
    GRUConf,
    GRUPredictor,
    best_window,
    fit_gru,
    gru_sums,
    predict_at,
    seq_set,
    step_blocks,
    tiled_loss,
)
from exp.lib.synth_series import complex_ou_gamma, kalman_errors, lgssm_gamma

_F64 = torch.float64
SMALL = GRUConf(
    width=8, window=6, burn=2, lags=(1, 2), batch=4, patience=2, max_epochs=3
)


def _w(n: int, k: int = 2, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, k, generator=torch.Generator().manual_seed(seed))


class TestSets:
    def test_anchors_keep_the_margin_in_their_sequence(self):
        s = seq_set(_w(30), [12, 18], margin=3, block=5)
        want = list(range(3, 9)) + list(range(15, 27))
        assert s.anchors.tolist() == want

    def test_keep_drops_anchors(self):
        keep = torch.ones(30, dtype=torch.bool)
        keep[4] = keep[20] = False
        s = seq_set(_w(30), [12, 18], margin=3, block=5, keep=keep)
        assert 4 not in s.anchors.tolist() and 20 not in s.anchors.tolist()
        assert len(s.anchors) == 16

    def test_blocks_are_per_sequence_and_renumbered(self):
        """Blocks of 5 steps from each sequence's start, the remainder joined
        to the last, numbered over the blocks that hold anchors."""
        assert step_blocks([12, 7], 5, 0).tolist() == [0] * 5 + [1] * 7 + [2] * 7
        assert step_blocks([12], 5, 2).tolist() == [0] * 7 + [1] * 5
        s = seq_set(_w(19), [12, 7], margin=1, block=5, shifted=True)
        assert s.n_blocks == 3 and s.blocks.max() == 2
        assert s.blocks_s is not None and s.n_blocks_s == 3

    def test_lengths_must_sum_to_the_rows(self):
        with pytest.raises(ValueError, match="sum to"):
            seq_set(_w(10), [4, 5], margin=1, block=5)

    def test_reversed_is_an_involution_and_turns_the_lags(self):
        s = seq_set(_w(30), [12, 18], margin=3, block=5)
        r = s.reversed()
        twice = r.reversed()
        torch.testing.assert_close(twice.w, s.w)
        assert twice.anchors.tolist() == s.anchors.tolist()
        for lag in (1, 3):
            torch.testing.assert_close(r.w[r.anchors + lag], s.w[s.anchors - lag])
            torch.testing.assert_close(r.w[r.anchors], s.w[s.anchors])
        assert r.blocks.tolist() == s.blocks.tolist()


def _random(conf: GRUConf) -> GRUPredictor:
    model = GRUPredictor.build(2, conf, seed=1, dev=torch.device("cpu"))
    g = torch.Generator().manual_seed(8)
    with torch.no_grad():
        model.head.weight.copy_(torch.randn(model.head.weight.shape, generator=g))
    model.set_linear(torch.randn(len(conf.lags), 2, 4, generator=g),
                     torch.randn(len(conf.lags), 2, generator=g))  # fmt: skip
    return model


class TestWindow:
    def test_a_prediction_sees_its_window_alone(self):
        """Changing a code before the window or after the anchor leaves the
        prediction at the anchor to the bit; changing one in it moves it
        (the heads and the linear set to random values)."""
        s = seq_set(_w(40), [40], margin=SMALL.margin, block=10)
        model = _random(SMALL)
        t = int(s.anchors[5])
        base = predict_at(model, s, SMALL, torch.device("cpu"))[5]
        for at, moves in ((t - SMALL.window, False), (t + 1, False),
                          (t - SMALL.window + 1, True), (t, True)):  # fmt: skip
            w = s.w.clone()
            w[at] += 1.0
            got = predict_at(model, seq_set(w, [40], SMALL.margin, 10), SMALL,
                             torch.device("cpu"))[5]  # fmt: skip
            assert bool((got != base).any()) == moves, at

    def test_the_linear_by_hand(self):
        """$\\hat w = M_\\Delta [w_t; w_{t-\\Delta}] + c_\\Delta$ plus the
        heads' residual, which is 0 before any training."""
        s = seq_set(_w(40), [40], margin=SMALL.margin, block=10)
        model = GRUPredictor.build(2, SMALL, seed=1, dev=torch.device("cpu"))
        g = torch.Generator().manual_seed(9)
        m, c = torch.randn(2, 2, 4, generator=g), torch.randn(2, 2, generator=g)
        model.set_linear(m, c)
        got = predict_at(model, s, SMALL, torch.device("cpu"))
        for i, lag in enumerate(SMALL.lags):
            x = torch.cat([s.w[s.anchors], s.w[s.anchors - lag]], 1)
            torch.testing.assert_close(got[:, i].float(), x @ m[i].T + c[i])

    def test_the_denominators_by_hand(self):
        s = seq_set(_w(40, seed=2), [40], margin=SMALL.margin, block=10)
        model = _random(SMALL)
        mean = torch.tensor([0.5, -0.5], dtype=_F64)
        sums = gru_sums(model, s, mean, SMALL, torch.device("cpu"))
        w = s.w.double()
        for i, lag in enumerate(SMALL.lags):
            fut = w[s.anchors + lag]
            torch.testing.assert_close(
                sums["copy_b"][i].sum(), (w[s.anchors] - fut).square().sum()
            )
            torch.testing.assert_close(
                sums["clim_b"][i].sum(), (mean - fut).square().sum()
            )
        assert float(sums["n_b"].sum()) == len(s.anchors)


class TestBestWindow:
    def test_white_noise_is_its_variance(self):
        def gamma(lag: int) -> torch.Tensor:
            return torch.eye(3, dtype=_F64) * (lag == 0)

        torch.testing.assert_close(
            best_window(gamma, 5, (1, 4)), torch.full((2,), 3.0, dtype=_F64)
        )

    @pytest.mark.parametrize("lag", [1, 2, 5])
    def test_a_scalar_ar1(self, lag):
        r = 0.7

        def gamma(ell: int) -> torch.Tensor:
            return torch.tensor([[r ** abs(ell)]], dtype=_F64)

        got = float(best_window(gamma, 4, (lag,))[0])
        assert got == pytest.approx(1 - r ** (2 * lag), rel=1e-10)

    def test_a_window_of_two_at_lag_one_is_the_two_point_oracle(self):
        def gamma(ell: int) -> torch.Tensor:
            g = complex_ou_gamma(abs(ell), 4, 0.1, 12.0)
            return g if ell >= 0 else g.T

        got = float(best_window(gamma, 2, (1,))[0])
        assert got == pytest.approx(
            best_linear(gamma(0), gamma(1), gamma(2)), rel=1e-10
        )

    def test_a_long_window_is_the_kalman_filter(self):
        """A rotation seen through one coordinate with noise: the window's
        optimum falls to the Kalman filter's error as the window grows
        ($0.8^{80} \\approx 2 \\times 10^{-8}$)."""
        c, s = math.cos(1.0), math.sin(1.0)
        f = 0.8 * torch.tensor([[c, -s], [s, c]], dtype=_F64)
        q = torch.eye(2, dtype=_F64)
        h = torch.tensor([[1.0, 0.0]], dtype=_F64)
        r = torch.tensor([[0.5]], dtype=_F64)
        kal = kalman_errors(f, q, h, r, (1, 3))
        short = best_window(lambda ell: lgssm_gamma(ell, f, q, h, r), 2, (1, 3))
        long = best_window(lambda ell: lgssm_gamma(ell, f, q, h, r), 80, (1, 3))
        assert (short > kal + 1e-3).all()
        torch.testing.assert_close(long, kal, rtol=1e-6, atol=0)


class TestFit:
    def test_the_linear_is_learned_from_the_two_point_fit(self):
        """The linear starts at dyn/copy's two-point fit and is learned with
        the rest (the family holds the lower bound and any weight of
        $w_{t-\\Delta}$)."""
        s = seq_set(_w(120, seed=3), [60, 60], margin=SMALL.margin, block=20)
        v = seq_set(_w(40, seed=4), [40], margin=SMALL.margin, block=20)
        untrained, _ = fit_gru(
            s, v, replace(SMALL, max_epochs=0), 5, torch.device("cpu")
        )
        got = predict_at(untrained, v, SMALL, torch.device("cpu"))
        for i, lag in enumerate(SMALL.lags):
            want, _ = fit_linear(s.triples(), v.triples(), lag).predict(v.triples())
            torch.testing.assert_close(got[:, i], want, rtol=1e-5, atol=1e-5)
        learned = {n for n, p in untrained.named_parameters() if p.requires_grad}
        assert {"base_m", "base_c"} <= learned

    def test_one_seed_learns_the_same_twice(self):
        s = seq_set(_w(120, seed=3), [60, 60], margin=SMALL.margin, block=20)
        v = seq_set(_w(40, seed=4), [40], margin=SMALL.margin, block=20)
        m1, f1 = fit_gru(s, v, SMALL, 5, torch.device("cpu"))
        m2, f2 = fit_gru(s, v, SMALL, 5, torch.device("cpu"))
        assert f1 == f2
        for p, q in zip(m1.parameters(), m2.parameters(), strict=True):
            assert torch.equal(p, q)

    def test_the_best_epoch_is_kept(self):
        """The returned weights are those of the least validation loss."""
        s = seq_set(_w(120, seed=3), [60, 60], margin=SMALL.margin, block=20)
        v = seq_set(_w(40, seed=4), [40], margin=SMALL.margin, block=20)
        model, fit = fit_gru(s, v, SMALL, 5, torch.device("cpu"))
        assert fit.best == min(range(len(fit.val)), key=lambda i: (fit.val[i], i))
        assert tiled_loss(model, v, SMALL, torch.device("cpu")) == pytest.approx(
            fit.val[fit.best], rel=1e-6
        )
