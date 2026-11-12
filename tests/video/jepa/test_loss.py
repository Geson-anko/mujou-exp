from typing import override

import pytest
import torch

from exp.lib.sigreg import T_POINTS, sigreg
from exp.video.jepa.loss import (
    Reconstruction,
    jepa_loss,
    jepa_loss_packed,
    pool,
    prediction_errors,
    split_y,
    token_sigreg_of,
    view_embeddings,
    weighted_sigreg,
    z_errors,
)
from exp.video.jepa.vit import ZPredictor

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


class TestReconstruction:
    P = 6  # pixels of a frame: (1, 2, 3)

    def _setup(self):
        tokens, masks, dirs = _inputs()
        g = torch.Generator().manual_seed(3)
        w = torch.randn(self.P, D, generator=g, dtype=torch.float64)
        frames = torch.rand(B, 1, 2, 3, generator=g, dtype=torch.float64)

        def decoder(z: torch.Tensor) -> torch.Tensor:
            return (z @ w.T).view(-1, 1, 2, 3)

        return tokens, masks, dirs, w, frames, decoder

    def _loss(self, tokens, masks, dirs, valid, rec, mu=0.7, lam=0.2):
        return jepa_loss(
            tokens,
            masks,
            valid,
            dirs,
            _identity,
            extra=E,
            z_pool="mean",
            lam=lam,
            sg=True,
            recon=None if rec is None else rec._replace(mu=mu),
        )

    def test_the_term_is_the_mean_squared_error_to_the_standardized_frame(self):
        """L = L_without + mu * mean over the valid streams of the mean over
        the pixels of (g(z) - (x - m) / s)^2, g linear here."""
        tokens, masks, dirs, w, frames, dec = self._setup()
        valid = torch.tensor([True, False, True])
        rec = Reconstruction(0.7, dec, 0.4, 2.0, frames)
        t = self._loss(tokens, masks, dirs, valid, rec)
        base = self._loss(tokens, masks, dirs, valid, None)
        z = tokens[:, 0, E:].mean(1)
        err = ((z @ w.T - ((frames - 0.4) / 2.0).flatten(1)) ** 2).mean(1)
        mean = err[valid].mean()
        torch.testing.assert_close(t.recon, mean)
        torch.testing.assert_close(t.loss, base.loss + 0.7 * mean * 2 / B)
        assert base.recon == 0

    def test_the_gradient_to_z_matches_a_central_difference(self):
        """With sg the prediction gives the target view no credit; the
        reconstruction does, through z, equal to the central difference of the
        loss (float64) and to 2 / P W^T (W z - t) / |M| spread over the T
        patch tokens of the mean pool."""
        tokens, masks, dirs, w, frames, dec = self._setup()
        valid = torch.ones(B, dtype=torch.bool)
        rec = Reconstruction(1.0, dec, 0.0, 1.0, frames)

        def f(x: torch.Tensor) -> torch.Tensor:
            return self._loss(x, masks, dirs, valid, rec, mu=1.0, lam=0.0).loss

        def r(x: torch.Tensor) -> torch.Tensor:
            return self._loss(x, masks, dirs, valid, rec, mu=1.0, lam=0.0).recon

        x = tokens.clone().requires_grad_()
        (g,) = torch.autograd.grad(f(x), [x])
        assert (g[:, 0, E:] != 0).any()
        z = tokens[:, 0, E:].mean(1)
        dz = 2 / self.P * (z @ w.T - frames.flatten(1)) @ w / B
        torch.testing.assert_close(g[:, 0, E:], dz[:, None].expand(-1, T, -1) / T)
        eps = 1e-6
        for idx in [(0, 0, E, 1), (1, 0, E + 2, 3), (2, 0, E + 3, 0)]:
            up, down = tokens.clone(), tokens.clone()
            up[idx] += eps
            down[idx] -= eps
            # terms.recon, not the loss: sg hides the prediction from the
            # gradient but not from a finite difference
            fd = (r(up) - r(down)) / (2 * eps)
            torch.testing.assert_close(g[idx], fd, rtol=1e-6, atol=1e-9)

    def test_split_y_reads_the_masks_and_the_target_frame(self):
        y = torch.arange(2 * 3 * (4 + 6.0)).view(6, 10)
        masks, frames = split_y(y, 2, 3, (1, 2, 3))
        assert frames is not None
        assert torch.equal(masks, y.view(2, 3, 10)[:, 1:, :4])
        assert torch.equal(frames.flatten(1), y.view(2, 3, 10)[:, 0, 4:])
        masks, frames = split_y(y, 2, 3, None)
        assert frames is None and masks.shape == (2, 2, 10)


class _Shift(torch.nn.Module):
    """A predictor of the packed layout one can compute by hand: each masked
    token is predicted as the mean of the context plus a learned offset and
    its id / 10."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.linspace(-1, 1, d, dtype=torch.float64))

    @override
    def forward(self, ctx, ctx_ids, tgt_ids):
        m = tgt_ids.shape[1]
        base = ctx.mean(1, keepdim=True).expand(-1, m, -1)
        return base + self.offset + tgt_ids[..., None].to(ctx) / 10


def _packed_inputs(seed: int = 0):
    """B = 3 streams, T = 6 tokens of D = 5, two context views keeping 2 and 4."""
    g = torch.Generator().manual_seed(seed)
    t, keeps = 6, [2, 4]
    y = torch.zeros(3, 2 * t)
    for b in range(3):
        for v, k in enumerate(keeps):
            masked = torch.randperm(t, generator=g)[: t - k]
            y[b, v * t + masked] = 1.0
    tokens = torch.randn(3, t + sum(keeps), 5, generator=g, dtype=torch.float64)
    dirs = torch.randn(7, 5, generator=g, dtype=torch.float64)
    return tokens, y, keeps, dirs / dirs.norm(dim=1, keepdim=True)


class TestJEPALossPacked:
    @pytest.mark.parametrize("pred_loss", ["l2", "l1"])
    def test_by_hand(self, pred_loss):
        """L = (1 / B) sum_valid [(1 - lam) L_pred_b + lam S], L_pred_b the mean
        over the views of the mean over the masked tokens of l / D."""
        tokens, y, keeps, dirs = _packed_inputs()
        valid = torch.tensor([True, False, True])
        pred = _Shift(5)
        lam, t = 0.3, 6
        got = jepa_loss_packed(
            tokens,
            y,
            valid,
            dirs,
            None,
            pred,
            keeps=keeps,
            proj=None,
            lam=lam,
            sg=False,
            sigreg_on="z",
            pred_loss=pred_loss,
        )
        target = tokens[:, :t]
        per_b = []
        for b in range(3):
            views, at = [], t
            for v, k in enumerate(keeps):
                ctx = tokens[b, at : at + k]
                at += k
                masked = (y[b, v * t : (v + 1) * t] == 1).nonzero().flatten()
                guess = ctx.mean(0) + pred.offset + masked[:, None] / 10
                diff = guess - target[b, masked]
                err = diff.abs() if pred_loss == "l1" else diff**2
                views.append(err.mean())
            per_b.append(sum(views) / 2)
        z = target.mean(1)
        s = weighted_sigreg(z, valid.double(), dirs)
        expect = ((1 - lam) * (per_b[0] + per_b[2]) + lam * 2 * s) / 3
        torch.testing.assert_close(got.loss, expect)
        torch.testing.assert_close(got.pred, (per_b[0] + per_b[2]) / 2)
        torch.testing.assert_close(got.z, z)
        assert got.recon == 0

    def test_the_projection_and_sigreg_on_both(self):
        """z = proj(mean); "both" averages S_z (on z, its directions) and S_t / T
        (on the target tokens, the other directions)."""
        tokens, y, keeps, dirs = _packed_inputs(1)
        g = torch.Generator().manual_seed(9)
        proj = torch.nn.Linear(5, 3).double()
        dz = torch.randn(7, 3, generator=g, dtype=torch.float64)
        valid = torch.ones(3, dtype=torch.bool)
        got = jepa_loss_packed(
            tokens,
            y,
            valid,
            dz,
            dirs,
            _Shift(5),
            keeps=keeps,
            proj=proj,
            lam=1.0,
            sg=False,
            sigreg_on="both",
            pred_loss="l2",
        )
        target = tokens[:, :6]
        z = proj(target.mean(1))
        s_z = weighted_sigreg(z, valid.double(), dz)
        s_t = weighted_sigreg(target.reshape(-1, 5), torch.ones(18).double(), dirs)
        torch.testing.assert_close(got.sigreg, (s_z + s_t / 6) / 2)
        torch.testing.assert_close(got.z, z)

    def test_sg_stops_the_prediction_at_the_target(self):
        tokens, y, keeps, dirs = _packed_inputs(2)
        tokens.requires_grad_()
        valid = torch.ones(3, dtype=torch.bool)
        for sg in (False, True):
            t = jepa_loss_packed(
                tokens,
                y,
                valid,
                dirs,
                None,
                _Shift(5),
                keeps=keeps,
                proj=None,
                lam=0.0,
                sg=sg,
                sigreg_on="z",
                pred_loss="l2",
            )
            (g,) = torch.autograd.grad(t.loss, [tokens])
            target_zero = bool((g[:, :6] == 0).all())
            assert target_zero == sg
            assert (g[:, 6:] != 0).any()


class TestTokenSigreg:
    """The SIGReg of the target tokens of the packed layout."""

    def _dirs(self, d: int = 5) -> torch.Tensor:
        g = torch.Generator().manual_seed(3)
        a = torch.randn(9, d, generator=g, dtype=torch.float64)
        return a / a.norm(dim=1, keepdim=True)

    def test_scaled_and_raw_are_of_all_the_tokens(self):
        g = torch.Generator().manual_seed(0)
        target = torch.randn(3, 6, 5, generator=g, dtype=torch.float64)
        w, dirs = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64), self._dirs()
        raw = weighted_sigreg(target.reshape(-1, 5), w.repeat_interleave(6), dirs)
        torch.testing.assert_close(token_sigreg_of(target, w, dirs, "raw"), raw)
        torch.testing.assert_close(token_sigreg_of(target, w, dirs, "scaled"), raw / 6)

    def test_clip_of_tokens_constant_within_each_stack_is_that_of_zeros(self):
        """Tokens that collapse to a constant per stack (z of any spread) are
        zeros less the stack's mean: the empirical characteristic function is
        1, so the statistic is n_w T int (1 - e^{-t^2/2})^2 e^{-t^2/2} dt,
        growing with the tokens; that of "scaled" stays below S_z."""
        g = torch.Generator().manual_seed(1)
        clip = torch.randn(4, 1, 5, generator=g, dtype=torch.float64)
        target = clip.expand(4, 8, 5)
        w, dirs = torch.ones(4, dtype=torch.float64), self._dirs()
        t = T_POINTS.double()
        phi = torch.exp(-(t**2) / 2)
        by_hand = 4 * 8 * torch.trapezoid((1 - phi) ** 2 * phi, t)
        torch.testing.assert_close(token_sigreg_of(target, w, dirs, "clip"), by_hand)
        s_z = weighted_sigreg(clip[:, 0], w, dirs)
        torch.testing.assert_close(token_sigreg_of(target, w, dirs, "scaled"), s_z)

    def test_clip_does_not_see_the_stacks_means(self):
        g = torch.Generator().manual_seed(2)
        target = torch.randn(3, 6, 5, generator=g, dtype=torch.float64)
        shift = 7 * torch.randn(3, 1, 5, generator=g, dtype=torch.float64)
        w, dirs = torch.ones(3, dtype=torch.float64), self._dirs()
        torch.testing.assert_close(
            token_sigreg_of(target + shift, w, dirs, "clip"),
            token_sigreg_of(target, w, dirs, "clip"),
        )

    @pytest.mark.parametrize("kind", ["raw", "clip"])
    def test_the_loss_averages_it_with_s_z_on_both(self, kind):
        tokens, y, keeps, dirs = _packed_inputs(4)
        valid = torch.ones(3, dtype=torch.bool)
        got = jepa_loss_packed(
            tokens,
            y,
            valid,
            dirs,
            None,
            _Shift(5),
            keeps=keeps,
            proj=None,
            lam=1.0,
            sg=False,
            sigreg_on="both",
            pred_loss="l2",
            token_sigreg=kind,
        )
        target = tokens[:, :6]
        w = valid.double()
        s_z = weighted_sigreg(target.mean(1), w, dirs)
        s_t = token_sigreg_of(target, w, dirs, kind)
        torch.testing.assert_close(got.sigreg, (s_z + s_t) / 2)


class TestSpaceTimeCompletion:
    """The completion of the whole stack's embedding u_g from each context
    view's u_v (LeJEPA's form; module docstring)."""

    def test_u_g_is_the_pooled_target_view_and_u_v_the_pooled_context(self):
        tokens, _, keeps, _ = _packed_inputs(5)
        proj = torch.nn.Linear(5, 4).double()
        head = torch.nn.Linear(4, 3).double()
        u = view_embeddings(tokens, keeps, 6, proj, head)
        assert u.shape == (3, 3, 3)
        torch.testing.assert_close(u[:, 0], head(proj(tokens[:, :6].mean(1))))
        torch.testing.assert_close(u[:, 1], head(proj(tokens[:, 6:8].mean(1))))
        torch.testing.assert_close(u[:, 2], head(proj(tokens[:, 8:].mean(1))))

    def test_by_hand(self):
        """L = (1 / B) sum_valid [(1 - lam)(beta L_pred + alpha L_z) + lam S],
        L_z the mean over the context views of |u_g - q(u_v, v)|^2 / d_u, S
        the mean over the views of SIGReg of u_v."""
        tokens, y, keeps, dirs = _packed_inputs(6)
        valid = torch.tensor([True, True, False])
        g = torch.Generator().manual_seed(4)
        du = torch.randn(7, 5, generator=g, dtype=torch.float64)
        lam, alpha, beta = 0.2, 0.7, 0.4

        def zpred(u: torch.Tensor, kind: int) -> torch.Tensor:
            return u * (kind + 2)

        got = jepa_loss_packed(
            tokens,
            y,
            valid,
            dirs,
            None,
            _Shift(5),
            keeps=keeps,
            proj=None,
            lam=lam,
            sg=False,
            sigreg_on="views",
            pred_loss="l2",
            token_weight=beta,
            z_weight=alpha,
            zpred=zpred,
            directions_u=du,
        )
        base = jepa_loss_packed(
            tokens,
            y,
            valid,
            dirs,
            None,
            _Shift(5),
            keeps=keeps,
            proj=None,
            lam=0.0,
            sg=False,
            sigreg_on="z",
            pred_loss="l2",
        )
        u = view_embeddings(tokens, keeps, 6, None, None)
        lz = torch.stack(
            [((u[:, 0] - zpred(u[:, v], v - 1)) ** 2).mean(1) for v in (1, 2)], 1
        ).mean(1)
        w = valid.double()
        s = torch.stack([weighted_sigreg(u[:, v], w, du) for v in range(3)]).mean()
        pred_sum = base.pred * 2
        expect = (
            (1 - lam) * (beta * pred_sum + alpha * lz[:2].sum()) + lam * 2 * s
        ) / 3
        torch.testing.assert_close(got.loss, expect)
        torch.testing.assert_close(got.sigreg, s)
        assert got.zpred is not None
        torch.testing.assert_close(got.zpred, lz[:2].mean())
        torch.testing.assert_close(
            z_errors(u, zpred),
            torch.stack(
                [((u[:, 0] - zpred(u[:, v], v - 1)) ** 2).mean(1) for v in (1, 2)], 1
            ),
        )

    def test_a_view_that_pools_as_the_whole_completes_it_exactly(self):
        """Tokens constant within each stack: every view pools to u_g, so the
        completion term is 0 (and the loss is lam S alone without tokens)."""
        g = torch.Generator().manual_seed(7)
        clip = torch.randn(3, 1, 5, generator=g, dtype=torch.float64)
        tokens = clip.expand(3, 12, 5).contiguous()
        _, y, keeps, dirs = _packed_inputs(0)

        def never(*_: torch.Tensor) -> torch.Tensor:
            raise AssertionError("token_weight 0 calls no predictor")

        got = jepa_loss_packed(
            tokens,
            y,
            torch.ones(3, dtype=torch.bool),
            dirs,
            None,
            never,
            keeps=keeps,
            proj=None,
            lam=0.0,
            sg=False,
            sigreg_on="z",
            pred_loss="l2",
            token_weight=0.0,
            z_weight=1.0,
        )
        assert float(got.loss) < 1e-30 and float(got.pred) == 0.0

    def test_sg_stops_the_completion_at_u_g(self):
        tokens, y, keeps, dirs = _packed_inputs(8)
        tokens.requires_grad_()
        for sg in (False, True):
            t = jepa_loss_packed(
                tokens,
                y,
                torch.ones(3, dtype=torch.bool),
                dirs,
                None,
                _Shift(5),
                keeps=keeps,
                proj=None,
                lam=0.0,
                sg=sg,
                sigreg_on="z",
                pred_loss="l2",
                token_weight=0.0,
                z_weight=1.0,
            )
            (g,) = torch.autograd.grad(t.loss, [tokens])
            assert bool((g[:, :6] == 0).all()) == sg
            assert (g[:, 6:] != 0).any()

    def test_the_z_predictor_starts_as_the_identity(self):
        q = ZPredictor(4, kinds=2, hidden=8)
        u = torch.randn(5, 4)
        torch.testing.assert_close(q(u, 0), u)
        torch.testing.assert_close(q(u, 1), u)


class TestTargets:
    def test_target_norm_predicts_the_tokens_through_a_layernorm(self):
        """The goal of each token is its target token through a LayerNorm
        without affine (zero mean, unit variance over its D); the context and
        SIGReg see the tokens as they are."""
        tokens, y, keeps, dirs = _packed_inputs(9)
        valid = torch.ones(3, dtype=torch.bool)
        kw = {
            "keeps": keeps,
            "proj": None,
            "lam": 0.5,
            "sg": False,
            "sigreg_on": "z",
            "pred_loss": "l2",
        }
        got = jepa_loss_packed(
            tokens, y, valid, dirs, None, _Shift(5), target_norm=True, **kw
        )
        target = tokens[:, :6]
        mu = target.mean(-1, keepdim=True)
        var = target.var(-1, unbiased=False, keepdim=True)
        by_hand = (target - mu) / torch.sqrt(var + 1e-5)
        want = jepa_loss_packed(
            tokens, y, valid, dirs, None, _Shift(5), goal=by_hand, **kw
        )
        torch.testing.assert_close(got.loss, want.loss)
        torch.testing.assert_close(got.sigreg, want.sigreg)

    def test_a_goal_is_a_constant_target(self):
        """With a goal (the EMA's tokens) the prediction aims at it, and the
        target tokens get no credit from the prediction."""
        tokens, y, keeps, dirs = _packed_inputs(10)
        tokens.requires_grad_()
        g = torch.Generator().manual_seed(11)
        goal = torch.randn(3, 6, 5, generator=g, dtype=torch.float64)
        kw = {
            "keeps": keeps,
            "proj": None,
            "lam": 0.0,
            "sg": False,
            "sigreg_on": "z",
            "pred_loss": "l2",
        }
        t = jepa_loss_packed(
            tokens,
            y,
            torch.ones(3, dtype=torch.bool),
            dirs,
            None,
            _Shift(5),
            goal=goal,
            **kw,
        )
        (grad,) = torch.autograd.grad(t.loss, [tokens])
        assert bool((grad[:, :6] == 0).all()) and (grad[:, 6:] != 0).any()
        swapped = tokens.detach().clone()
        swapped[:, :6] = goal
        want = jepa_loss_packed(
            swapped, y, torch.ones(3, dtype=torch.bool), dirs, None, _Shift(5), **kw
        )
        torch.testing.assert_close(t.pred, want.pred)
