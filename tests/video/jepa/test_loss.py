import pytest
import torch

from exp.lib.sigreg import sigreg
from exp.video.jepa.loss import jepa_loss, pool, prediction_errors, weighted_sigreg

B, V, E, T, D = 3, 3, 1, 4, 5


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(B, V, E + T, D, generator=g, dtype=torch.float64)
    masks = torch.rand(B, V - 1, T, generator=g) < 0.5
    masks[:, :, 0] = True  # at least one masked token per view
    directions = torch.randn(7, D, generator=g, dtype=torch.float64)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return tokens, masks, directions


def test_pool():
    x = torch.arange(2 * 5 * 3.0).view(2, 5, 3)
    assert torch.equal(pool(x, z_pool="cls", extra=1), x[:, 0])
    assert torch.equal(pool(x, z_pool="registers", extra=2), x[:, :2].mean(1))
    assert torch.equal(pool(x, z_pool="mean", extra=2), x[:, 2:].mean(1))


def test_the_prediction_error_by_hand():
    """With the identity predictor, view v's error is the mean over its
    masked patch tokens of the mean squared difference to the target's."""
    tokens, masks, _ = _inputs()
    got = prediction_errors(tokens, masks, _identity, extra=E, sg=False)
    for b in range(B):
        per_view = []
        for v in range(1, V):
            sq = ((tokens[b, v, E:] - tokens[b, 0, E:]) ** 2).mean(-1)  # (T,)
            per_view.append(sq[masks[b, v - 1]].mean())
        torch.testing.assert_close(got[b], torch.stack(per_view).mean())


class TestJEPALoss:
    @pytest.mark.parametrize("lam", [0.0, 0.3, 1.0])
    def test_all_valid_is_the_lejepa_mix(self, lam):
        """(1 - lam) mean_b L_pred + lam SIGReg of the B target z."""
        tokens, masks, dirs = _inputs()
        valid = torch.ones(B, dtype=torch.bool)
        t = jepa_loss(
            tokens,
            masks,
            valid,
            dirs,
            _identity,
            extra=E,
            z_pool="mean",
            lam=lam,
            sg=False,
        )
        pred = prediction_errors(tokens, masks, _identity, extra=E, sg=False)
        z = tokens[:, 0, E:].mean(1)
        s = sigreg(z, z.new_zeros(0, D), dirs)
        torch.testing.assert_close(t.loss, (1 - lam) * pred.mean() + lam * s)
        torch.testing.assert_close(t.pred, pred.mean())
        torch.testing.assert_close(t.sigreg, s)
        assert t.valid == B and torch.equal(t.z, z)

    def test_a_stream_without_a_chunk_gets_no_gradient(self):
        """The loss is the sum over the valid streams over B: an invalid
        stream's tokens get exactly zero gradient, and the loss is that of the
        valid ones with SIGReg of their z only, times |M| / B."""
        tokens, masks, dirs = _inputs()
        valid = torch.tensor([True, False, True])
        x = tokens.clone().requires_grad_()
        t = jepa_loss(
            x, masks, valid, dirs, _identity, extra=E, z_pool="mean", lam=0.4, sg=False
        )
        (g,) = torch.autograd.grad(t.loss, [x])
        assert (g[1] == 0).all() and (g[0] != 0).any()
        pred = prediction_errors(tokens, masks, _identity, extra=E, sg=False)
        z = tokens[valid, 0, E:].mean(1)
        s = sigreg(z, z.new_zeros(0, D), dirs)
        expect = (0.6 * pred[valid].sum() + 0.4 * 2 * s) / B
        torch.testing.assert_close(t.loss, expect)

    def test_no_valid_stream_is_zero(self):
        tokens, masks, dirs = _inputs()
        t = jepa_loss(
            tokens,
            masks,
            torch.zeros(B, dtype=torch.bool),
            dirs,
            _identity,
            extra=E,
            z_pool="mean",
            lam=0.4,
            sg=False,
        )
        assert t.loss == 0 and t.valid == 0

    def test_sg_stops_the_prediction_at_the_target_view(self):
        """With sg and no SIGReg, the target view (0) gets zero gradient and
        the context views theirs unchanged; without sg the target gets some."""
        tokens, masks, dirs = _inputs()
        valid = torch.ones(B, dtype=torch.bool)
        grads = {}
        for sg in (False, True):
            x = tokens.clone().requires_grad_()
            t = jepa_loss(
                x,
                masks,
                valid,
                dirs,
                _identity,
                extra=E,
                z_pool="mean",
                lam=0.0,
                sg=sg,
            )
            (grads[sg],) = torch.autograd.grad(t.loss, [x])
        assert (grads[True][:, 0] == 0).all()
        assert (grads[False][:, 0] != 0).any()
        torch.testing.assert_close(grads[True][:, 1:], grads[False][:, 1:])

    def test_sg_leaves_the_sigreg_credit_of_the_target_view(self):
        tokens, masks, dirs = _inputs()
        valid = torch.ones(B, dtype=torch.bool)
        x = tokens.clone().requires_grad_()
        t = jepa_loss(
            x, masks, valid, dirs, _identity, extra=E, z_pool="mean", lam=1.0, sg=True
        )
        (g,) = torch.autograd.grad(t.loss, [x])
        assert (g[:, 0, E:] != 0).any() and (g[:, 1:] == 0).all()


class TestWeightedSIGReg:
    def test_it_is_sigreg_of_the_samples_of_weight_one(self):
        """0/1 weights select the samples: the statistic of the selected ones
        (exp.lib.sigreg), and a gradient of exactly 0 for the others."""
        tokens, _, dirs = _inputs()
        x = tokens.flatten(0, 2)[:9].clone().requires_grad_()  # (9, D)
        w = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=torch.bool)
        got = weighted_sigreg(x, w, dirs)
        torch.testing.assert_close(got, sigreg(x[w], x.new_zeros(0, D), dirs))
        (g,) = torch.autograd.grad(got, [x])
        assert (g[~w] == 0).all() and (g[w] != 0).any()

    def test_no_weight_is_zero(self):
        tokens, _, dirs = _inputs()
        x = tokens.flatten(0, 2)[:4]
        assert weighted_sigreg(x, torch.zeros(4, dtype=torch.bool), dirs) == 0


class TestSIGRegOn:
    @pytest.mark.parametrize("on", ["tokens", "both"])
    def test_the_tokens_statistic_over_t(self, on):
        """ "tokens": S_t = SIGReg of the valid streams' |M| T target patch
        tokens together, over T; "both": (S_z + S_t) / 2."""
        tokens, masks, dirs = _inputs()
        valid = torch.tensor([True, False, True])
        t = jepa_loss(
            tokens,
            masks,
            valid,
            dirs,
            _identity,
            extra=E,
            z_pool="mean",
            lam=0.4,
            sg=False,
            sigreg_on=on,
        )
        patches = tokens[valid, 0, E:].flatten(0, 1)  # (2 T, D)
        s_t = sigreg(patches, patches.new_zeros(0, D), dirs) / T
        z = tokens[valid, 0, E:].mean(1)
        s_z = sigreg(z, z.new_zeros(0, D), dirs)
        expect = s_t if on == "tokens" else (s_z + s_t) / 2
        torch.testing.assert_close(t.sigreg, expect)
        pred = prediction_errors(tokens, masks, _identity, extra=E, sg=False)
        loss = (0.6 * pred[valid].sum() + 0.4 * 2 * expect) / B
        torch.testing.assert_close(t.loss, loss)

    def test_an_unknown_choice_is_an_error(self):
        tokens, masks, dirs = _inputs()
        with pytest.raises(ValueError, match="sigreg_on"):
            jepa_loss(
                tokens,
                masks,
                torch.ones(B, dtype=torch.bool),
                dirs,
                _identity,
                extra=E,
                z_pool="mean",
                lam=0.4,
                sg=False,
                sigreg_on="zz",
            )
