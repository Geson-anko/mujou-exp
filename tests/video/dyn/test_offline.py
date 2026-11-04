import math
from typing import cast

import pytest
import torch
from mujou.pc.optim import AdamW
from torch import nn

from exp.lib.mingru import MinGRUTrace, ResidualMinGRU
from exp.video.data.clips import fit_standardizer
from exp.video.data.observe import observe
from exp.video.data.physics import BallStreams
from exp.video.dyn.modules import DynamicsModules, Top, build_dynamics
from exp.video.dyn.offline import (
    OfflineFit,
    fit_offline,
    pixel_encoder,
    stream_windows,
    window_loss,
)
from exp.video.programs.dynamics import PROGRAM
from exp.video.schema.data import BallsConf, DataConf
from exp.video.schema.dyn import DynamicsConf
from exp.video.schema.learn import OfflineConf
from exp.video.schema.measure import MeasureConf
from exp.video.schema.run import RunConf

FRAME = 1 / 30


def _conf(task: str = "offline", **dynamics) -> RunConf:
    return RunConf(
        name="test",
        task=task,
        data=DataConf(streams=3, balls=BallsConf(radii=[0.06, 0.10], obs="pos")),
        dynamics=DynamicsConf(d_s=8, **dynamics),
        offline=OfflineConf(frames=72, window=36, steps=4, batch=5, every=2),
        measure=MeasureConf(clips=4, windows=2),
    )


def _mods(residual: bool, nodes: int = 2) -> DynamicsModules:
    conf = DynamicsConf(d_s=5, nodes=nodes, residual=residual)
    mods = build_dynamics(conf, 4, 4, seed=3)
    mods.step.double()
    mods.readout.double()
    if residual:  # a readout that is not 0, so the gradients reach the step
        with torch.no_grad():
            g = torch.Generator().manual_seed(4)
            mods.readout.weight.copy_(
                0.3 * torch.randn(4, 5, generator=g, dtype=torch.float64)
            )
    return mods


def _params(mods: DynamicsModules) -> list[torch.Tensor]:
    return [*mods.step.parameters(), *mods.readout.parameters()]


def _cut_loss(mods: DynamicsModules, x, consts, *, residual: bool) -> torch.Tensor:
    """The loss with the input and the state of each frame held at
    ``consts``: frame j costs (f(c_{j-1}, s_{j-1}) - x_j)^2."""
    top = Top(mods.step.d_s)
    losses = []
    for j, (c, s) in enumerate(consts, start=1):
        s_j = mods.step(c, s)
        y = mods.readout(top(s_j))
        x_hat = c + y if residual else y
        losses.append(((x_hat - x[:, j]) ** 2).mean(-1))
    return torch.stack(losses, 1).mean()


def _consts(mods: DynamicsModules, x, observed: int, residual: bool):
    """The input and the state each frame starts from, at the weights now."""
    top = Top(mods.step.d_s)
    s = x.new_zeros(x.shape[0], mods.step.d_state)
    c = x[:, 0]
    out = []
    with torch.no_grad():
        for j in range(1, x.shape[1]):
            out.append((c, s))
            s = mods.step(c, s)
            y = mods.readout(top(s))
            x_hat = c + y if residual else y
            c = x[:, j] if j < observed else x_hat
    return out


def _finite_difference(loss_of, params, entries: int = 4, eps: float = 1e-6):
    """Central differences of ``loss_of()`` at a few entries of each
    parameter: (indices, values) per parameter."""
    g = torch.Generator().manual_seed(0)
    out = []
    for p in params:
        idx = torch.randint(p.numel(), (entries,), generator=g)
        vals = []
        flat = p.data.view(-1)
        for i in idx.tolist():
            old = float(flat[i])
            flat[i] = old + eps
            up = float(loss_of())
            flat[i] = old - eps
            down = float(loss_of())
            flat[i] = old
            vals.append((up - down) / (2 * eps))
        out.append((idx, torch.tensor(vals, dtype=torch.float64)))
    return out


class TestStreamWindows:
    def test_a_window_is_consecutive_frames_of_one_stream(self):
        conf = _conf()
        balls = conf.data.balls
        assert balls is not None
        std = fit_standardizer(balls, frame=FRAME, clips=16)
        w = stream_windows(conf, 20, 8, std=std)  # 2 windows per stream, 4 frames left
        assert w.shape == (3 * 2, 8, 4)
        streams = BallStreams.of(balls, 3, frame=FRAME, seed=conf.seed)
        frames = torch.stack([observe(streams.next(), "pos", 2) for _ in range(20)], 1)
        frames = std.apply(frames)
        for b in range(3):
            for j in range(2):
                torch.testing.assert_close(
                    w[b * 2 + j], frames[b, 8 * j : 8 * (j + 1)], rtol=0, atol=0
                )


class TestPixelWindows:
    def test_images_rendered_per_batch_are_the_images_of_the_windows(self):
        """``states=True`` windows with ``pixel_encoder`` are the standardized
        image windows, frame for frame; the images are never all in memory."""
        conf = _conf()
        balls = conf.data.balls
        assert balls is not None
        balls.obs, balls.size = "pixels", 8
        std = fit_standardizer(balls, frame=FRAME, clips=8)
        whole = stream_windows(conf, 20, 8, std=std)
        states = stream_windows(conf, 20, 8, std=std, states=True)
        assert states.shape == (6, 8, 8)  # 4 values of each of 2 balls
        assert whole.shape == (6, 8, 64)
        torch.testing.assert_close(
            pixel_encoder(conf, std)(states), whole, rtol=0, atol=0
        )

    def test_a_fit_on_encoded_states_is_the_fit_on_their_images(self):
        conf = _conf()
        balls = conf.data.balls
        assert balls is not None
        balls.obs, balls.size = "pixels", 8
        std = fit_standardizer(balls, frame=FRAME, clips=8)
        whole = stream_windows(conf, 20, 8, std=std)
        states = stream_windows(conf, 20, 8, std=std, states=True)

        def losses(windows, encode):
            mods = build_dynamics(DynamicsConf(d_s=6), 64, 64, seed=1)
            fit = OfflineFit(
                mods,
                windows,
                batch=3,
                lr=1e-2,
                local=False,
                observed=3,
                residual=False,
                gen=torch.Generator().manual_seed(0),
                encode=encode,
            )
            return [float(fit.step()) for _ in range(3)]

        assert losses(states, pixel_encoder(conf, std)) == losses(whole, None)


class TestWindowLoss:
    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_local_is_the_gradient_of_the_loss_cut_at_one_frame(self, residual):
        """With ``local``, the gradient is that of the loss whose input and
        state of each frame are constants (held at their values now), by
        central differences in float64."""
        mods = _mods(residual)
        x = torch.randn(
            3, 7, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
        )
        params = _params(mods)
        loss = window_loss(mods, x, observed=4, local=True, residual=residual)
        grads = torch.autograd.grad(loss, params)
        consts = _consts(mods, x, 4, residual)
        with torch.no_grad():
            fd = _finite_difference(
                lambda: _cut_loss(mods, x, consts, residual=residual), params
            )
        for g, (idx, want) in zip(grads, fd, strict=True):
            torch.testing.assert_close(g.reshape(-1)[idx], want, rtol=1e-5, atol=1e-9)

    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_through_time_is_the_gradient_of_the_whole_loss(self, residual):
        mods = _mods(residual)
        x = torch.randn(
            3, 7, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1)
        )
        params = _params(mods)
        loss = window_loss(mods, x, observed=4, local=False, residual=residual)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            fd = _finite_difference(
                lambda: window_loss(
                    mods, x, observed=4, local=False, residual=residual
                ),
                params,
            )
        for g, (idx, want) in zip(grads, fd, strict=True):
            torch.testing.assert_close(g.reshape(-1)[idx], want, rtol=1e-5, atol=1e-9)
        local = window_loss(mods, x, observed=4, local=True, residual=residual)
        assert float(local.detach()) == pytest.approx(
            float(loss.detach()), rel=1e-12
        )  # the same value


def _old_loss(mods, x, *, observed, residual, detach=False):
    """The loss as it was before ``feedback``, ``credit`` and ``mask``: the
    mean of the squared errors, written out (``detach`` cuts the feedback)."""
    top = Top(mods.step.d_s)
    s = x.new_zeros(x.shape[0], mods.step.d_state)
    c = x[:, 0]
    losses = []
    for j in range(1, x.shape[1]):
        s = mods.step(c, s)
        y = mods.readout(top(s))
        x_hat = c + y if residual else y
        losses.append(((x_hat - x[:, j]) ** 2).mean(-1))
        fed = x_hat.detach() if detach else x_hat
        c = x[:, j] if j < observed else fed
    return torch.stack(losses, 1).mean()


def _data(n: int, w: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(
        n, w, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)
    )


def _grads(loss, mods):
    return torch.autograd.grad(
        loss, _params(mods), materialize_grads=True, retain_graph=True
    )


class TestFeedbackAndMask:
    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_without_a_mask_the_loss_is_bit_for_bit_the_old_one(self, residual):
        mods, x = _mods(residual), _data(3, 8)
        new = window_loss(mods, x, observed=3, local=False, residual=residual)
        old = _old_loss(mods, x, observed=3, residual=residual)
        assert torch.equal(new, old)
        for a, b in zip(_grads(new, mods), _grads(old, mods), strict=True):
            assert torch.equal(a, b)

    def test_a_mask_drops_windows_under_the_fixed_normalization(self):
        """Sum over the kept windows over n P: the loss of the kept windows
        alone, times k / n; an all-True mask is the mean."""
        mods, x = _mods(True), _data(4, 7)
        kw = {"observed": 3, "local": False, "residual": True}
        mask = torch.tensor([True, False, True, False])
        got = window_loss(mods, x, mask=mask, **kw)
        kept = window_loss(mods, x[mask], **kw)
        torch.testing.assert_close(got, kept * 2 / 4, rtol=1e-12, atol=0)
        ones = torch.ones(4, dtype=torch.bool)
        torch.testing.assert_close(
            window_loss(mods, x, mask=ones, **kw),
            window_loss(mods, x, **kw),
            rtol=1e-12,
            atol=0,
        )
        zero = window_loss(mods, x, mask=torch.zeros(4), **kw)
        assert float(zero.detach()) == 0.0
        # a dropped window adds nothing to the gradient
        for g, h in zip(
            _grads(got, mods),
            [g * 2 / 4 for g in _grads(kept, mods)],
            strict=True,
        ):
            torch.testing.assert_close(g, h, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_teacher_is_every_frame_observed(self, residual):
        mods, x = _mods(residual), _data(3, 8)
        kw = {"local": False, "residual": residual}
        teacher = window_loss(mods, x, observed=3, feedback="teacher", **kw)
        whole = window_loss(mods, x, observed=8, **kw)
        assert torch.equal(teacher, whole)
        for a, b in zip(_grads(teacher, mods), _grads(whole, mods), strict=True):
            assert torch.equal(a, b)

    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_detach_is_the_loss_that_feeds_back_the_prediction_detached(self, residual):
        mods, x = _mods(residual), _data(3, 8)
        got = window_loss(
            mods, x, observed=3, local=False, residual=residual, feedback="detach"
        )
        want = _old_loss(mods, x, observed=3, residual=residual, detach=True)
        assert torch.equal(got, want)
        for a, b in zip(_grads(got, mods), _grads(want, mods), strict=True):
            assert torch.equal(a, b)
        full = window_loss(mods, x, observed=3, local=False, residual=residual)
        assert not all(
            torch.equal(a, b)
            for a, b in zip(_grads(got, mods), _grads(full, mods), strict=True)
        )  # the feedback path does carry credit

    def test_a_choice_that_does_not_exist_or_two_cuts_are_errors(self):
        mods, x = _mods(False), _data(2, 5)
        kw = {"observed": 2, "residual": False}
        with pytest.raises(ValueError, match="feedback"):
            window_loss(mods, x, local=False, feedback="stop", **kw)
        with pytest.raises(ValueError, match="credit"):
            window_loss(mods, x, local=False, credit="rtrl", **kw)
        with pytest.raises(ValueError, match="own"):
            window_loss(mods, x, local=True, credit="own", **kw)


def _held_loss(mods, x, cs, *, residual: bool):
    """The loss whose input of frame j is the constant ``cs[j-1]`` and whose
    state runs through time: what ``feedback="detach"`` differentiates."""
    top = Top(mods.step.d_s)
    s = x.new_zeros(x.shape[0], mods.step.d_state)
    losses = []
    for j, c in enumerate(cs, start=1):
        s = mods.step(c, s)
        y = mods.readout(top(s))
        x_hat = c + y if residual else y
        losses.append(((x_hat - x[:, j]) ** 2).mean(-1))
    return torch.stack(losses, 1).mean()


def _inputs(mods, x, observed: int, residual: bool) -> list[torch.Tensor]:
    """The input of each frame of the rollout, at the weights now."""
    top = Top(mods.step.d_s)
    s = x.new_zeros(x.shape[0], mods.step.d_state)
    c, out = x[:, 0], []
    with torch.no_grad():
        for j in range(1, x.shape[1]):
            out.append(c)
            s = mods.step(c, s)
            y = mods.readout(top(s))
            x_hat = c + y if residual else y
            c = x[:, j] if j < observed else x_hat
    return out


class TestFiniteDifferences:
    @staticmethod
    def _tiny() -> tuple[DynamicsModules, torch.Tensor]:
        conf = DynamicsConf(d_s=4, nodes=2, residual=True)
        mods = build_dynamics(conf, 3, 3, seed=5)
        mods.step.double()
        mods.readout.double()
        with torch.no_grad():
            g = torch.Generator().manual_seed(6)
            mods.readout.weight.copy_(
                0.4 * torch.randn(3, 4, generator=g, dtype=torch.float64)
            )
        x = torch.randn(
            2, 6, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(7)
        )
        return mods, x

    def test_grad_is_the_gradient_of_the_loss_through_the_feedback(self):
        mods, x = self._tiny()
        params = _params(mods)
        loss = window_loss(mods, x, observed=3, local=False, residual=True)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            fd = _finite_difference(
                lambda: window_loss(mods, x, observed=3, local=False, residual=True),
                params,
                entries=6,
            )
        for g, (idx, want) in zip(grads, fd, strict=True):
            torch.testing.assert_close(g.reshape(-1)[idx], want, rtol=1e-6, atol=1e-9)

    def test_detach_is_the_gradient_with_the_feedback_held(self):
        mods, x = self._tiny()
        params = _params(mods)
        loss = window_loss(
            mods, x, observed=3, local=False, residual=True, feedback="detach"
        )
        grads = torch.autograd.grad(loss, params)
        cs = _inputs(mods, x, 3, True)  # constants, at the weights now
        with torch.no_grad():
            fd = _finite_difference(
                lambda: _held_loss(mods, x, cs, residual=True), params, entries=6
            )
        for g, (idx, want) in zip(grads, fd, strict=True):
            torch.testing.assert_close(g.reshape(-1)[idx], want, rtol=1e-6, atol=1e-9)


class TestOwnCredit:
    @pytest.mark.parametrize("nodes", [1, 3])
    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_it_is_the_sum_of_the_per_frame_trace_gradients(self, nodes, residual):
        """``credit="own"`` is the gradient the eligibility traces give,
        frame after frame from $s = 0$ with the weights held: the loss of each
        frame (its share of the window's) differentiated in the frame for the
        layers' outputs and the readout, the layers' by their traces, as
        ``DynamicsBP._grads`` does with ``trace``."""
        mods = _mods(residual, nodes)
        n, w, observed = 3, 8, 3
        x = _data(n, w)
        p = w - 1
        loss = window_loss(
            mods,
            x,
            observed=observed,
            local=False,
            residual=residual,
            feedback="detach",
            credit="own",
        )
        got = _grads(loss, mods)

        st, top = mods.step, Top(mods.step.d_s)
        cells = [st.first] + [cast(ResidualMinGRU, layer).cell for layer in st.rest]
        traces = [MinGRUTrace(cell, (n,)) for cell in cells]
        readout = list(mods.readout.parameters())
        g_readout = [torch.zeros_like(q) for q in readout]
        g_cells = [[torch.zeros_like(q) for q in cell.parameters()] for cell in cells]
        s = x.new_zeros(n, st.d_state)
        c = x[:, 0]
        for j in range(1, w):
            parts = s.split(st.d_s, -1)
            with torch.no_grad():  # the traces move to this frame's state
                traces[0].step_(c, parts[0])
                y = st.first(c, parts[0])
                for tr, layer, h in zip(traces[1:], st.rest, parts[1:], strict=True):
                    tr.step_(nn.functional.gelu(y), h)
                    y = layer(y, h)
            y = st.first(c, parts[0])
            ys, out = [y], [y]
            for layer, h in zip(st.rest, parts[1:], strict=True):
                y_next = layer(y, h)
                out.append(y_next - y)
                y = y_next
                ys.append(y)
            x_hat = mods.readout(top(torch.cat(out, -1)))
            x_hat = c + x_hat if residual else x_hat
            frame = ((x_hat - x[:, j]) ** 2).mean(-1).sum() / (n * p)
            g = torch.autograd.grad(frame, [*readout, *ys], materialize_grads=True)
            for acc, d in zip(g_readout, g[: len(readout)], strict=True):
                acc += d
            for acc, trace, d in zip(g_cells, traces, g[len(readout) :], strict=True):
                for a, b in zip(acc, trace.grads(d), strict=True):
                    a += b
            with torch.no_grad():
                s = st(c, s)
                x_hat = x_hat.detach()
            c = x[:, j] if j < observed else x_hat
        want = [q for acc in g_cells for q in acc] + g_readout
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-13)

    def test_it_has_the_value_of_the_loss_and_less_credit(self):
        mods, x = _mods(True, 3), _data(3, 8)
        kw = {"observed": 3, "local": False, "residual": True, "feedback": "detach"}
        full = window_loss(mods, x, **kw)
        own = window_loss(mods, x, credit="own", **kw)
        torch.testing.assert_close(own, full, rtol=1e-12, atol=0)
        diff = [
            float((a - b).abs().max())
            for a, b in zip(_grads(own, mods), _grads(full, mods), strict=True)
        ]
        assert max(diff) > 1e-6  # not the same gradient: the cross-layer time path


def _adam_by(lr: float, elapsed: float) -> AdamW:
    """``adam(lr)`` (per step) as the optimizer of time, stepped by ``elapsed``."""
    return AdamW(
        tau=elapsed / lr,
        tau_m=-elapsed / math.log(0.9),
        tau_v=-elapsed / math.log(0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


class TestMaskShapeAndGradNorm:
    def test_a_mask_of_another_shape_is_an_error(self):
        mods, x = _mods(False), _data(4, 5)
        kw = {"observed": 2, "local": False, "residual": False}
        with pytest.raises(ValueError, match="mask"):
            window_loss(mods, x, mask=torch.ones(4, 1), **kw)
        with pytest.raises(ValueError, match="mask"):
            window_loss(mods, x, mask=torch.ones(3), **kw)

    def test_the_fit_keeps_the_norm_of_the_gradient_of_its_step(self):
        mods, x = _mods(True), _data(3, 6)
        fit = OfflineFit(
            mods,
            x,
            batch=3,
            lr=1e-2,
            local=False,
            observed=3,
            residual=True,
            gen=torch.Generator().manual_seed(0),
        )
        assert float(fit.grad_norm) == 0.0
        # the batch is drawn by the generator: draw it again to get the loss
        idx = torch.randint(3, (3,), generator=torch.Generator().manual_seed(0))
        loss = window_loss(mods, x[idx], observed=3, local=False, residual=True)
        grads = torch.autograd.grad(loss, _params(mods))
        want = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
        fit.step()
        torch.testing.assert_close(fit.grad_norm.double(), want, rtol=1e-9, atol=0)


class TestOfflineConfChecks:
    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"order": "shuffle"}, "offline.order"),
            ({"feedback": "stop"}, "offline.feedback"),
            ({"credit": "rtrl"}, "offline.credit"),
            ({"stride": -1}, "stride"),
            ({"credit": "own", "feedback": "detach", "local": True}, "twice"),
            ({"credit": "own"}, "feedback='detach'"),
            ({"order": "stream", "frames": 30, "window": 36}, "windows of 36"),
            (
                {
                    "order": "stream",
                    "frames": 72,
                    "window": 36,
                    "stride": 18,
                    "steps": 4,
                },
                "offline.steps=4",
            ),
        ],
    )
    def test_a_setting_that_cannot_run_is_refused(self, kw, match):
        with pytest.raises(ValueError, match=match):
            OfflineConf(**kw)

    def test_the_steps_the_streams_hold_are_accepted(self):
        OfflineConf(order="stream", frames=72, window=36, stride=18, steps=3)
        OfflineConf(order="stream", frames=72, window=36, steps=2)  # stride 0: W
        OfflineConf(order="random", frames=72, window=36, steps=999)
        OfflineConf(credit="own", feedback="detach")


class TestStreamOrder:
    W, S, FRAMES = 6, 3, 15  # (15 - 6) // 3 + 1 = 4 steps

    def _fit(self, mods, data, **kw):
        return OfflineFit(
            mods,
            data,
            batch=2,
            lr=1e-2,
            local=False,
            observed=3,
            residual=True,
            gen=torch.Generator().manual_seed(0),
            order="stream",
            window=self.W,
            stride=self.S,
            **kw,
        )

    def test_step_t_takes_the_windows_t_s_to_t_s_plus_w_of_every_stream(self):
        data = _data(3, self.FRAMES, seed=4)
        seen = []

        def record(x):
            seen.append(x.clone())
            return x

        fit = self._fit(_mods(True), data, encode=record)
        for _ in range(4):
            fit.step()
        assert len(seen) == 4
        for t, x in enumerate(seen):
            assert x.shape == (3, self.W, 4)
            assert torch.equal(x, data[:, t * self.S : t * self.S + self.W])
        with pytest.raises(ValueError, match="4 steps"):
            fit.step()

    def test_the_windows_that_do_not_fit_are_errors(self):
        data = _data(3, 5)
        with pytest.raises(ValueError, match="6 frames in 5"):
            self._fit(_mods(True), data)
        with pytest.raises(ValueError, match="stride"):
            OfflineFit(
                _mods(True),
                _data(3, 9),
                batch=1,
                lr=1e-2,
                local=False,
                observed=3,
                residual=True,
                gen=torch.Generator(),
                order="stream",
            )

    def test_an_optimizer_of_time_stepped_by_elapsed_is_adam_per_step(self):
        """``adam(lr)`` per step is the ``AdamW`` of time constants ``elapsed /
        lr`` ... stepped by ``elapsed``: the same trajectory."""
        data = _data(3, self.FRAMES, seed=4)
        a, b = _mods(True), _mods(True)
        by_step = self._fit(a, data)
        by_time = self._fit(b, data, optimizer=_adam_by(1e-2, 0.6), elapsed=0.6)
        for _ in range(4):
            la, lb = by_step.step(), by_time.step()
            torch.testing.assert_close(la, lb, rtol=1e-6, atol=0)
        for p, q in zip(_params(a), _params(b), strict=True):
            torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-12)

    def test_a_fit_resumed_from_its_state_takes_the_next_window(self):
        data = _data(3, self.FRAMES, seed=4)
        a = _mods(True)
        whole = self._fit(a, data)
        losses = [float(whole.step()) for _ in range(4)]
        b = _mods(True)
        first = self._fit(b, data)
        for _ in range(2):
            first.step()
        c = _mods(True)
        with torch.no_grad():
            for p, q in zip(_params(c), _params(b), strict=True):
                p.copy_(q)
        second = self._fit(c, data)
        second.load_state_dict(first.state_dict())
        assert [float(second.step()) for _ in range(2)] == losses[2:]
        for p, q in zip(_params(a), _params(c), strict=True):
            assert torch.equal(p, q)


class TestOfflineFit:
    def test_a_fit_resumed_from_its_state_goes_on_as_one_never_stopped(self):
        x = torch.randn(10, 6, 4, generator=torch.Generator().manual_seed(2))

        def fit(mods):
            return OfflineFit(
                mods,
                x,
                batch=3,
                lr=1e-2,
                local=False,
                observed=3,
                residual=False,
                gen=torch.Generator().manual_seed(0),
            )

        a = build_dynamics(DynamicsConf(d_s=6), 4, 4, seed=1)
        whole = fit(a)
        losses = [float(whole.step()) for _ in range(4)]
        b = build_dynamics(DynamicsConf(d_s=6), 4, 4, seed=1)
        first = fit(b)
        for _ in range(2):
            first.step()
        state = first.state_dict()
        weights = [p.detach().clone() for p in _params(b)]
        c = build_dynamics(DynamicsConf(d_s=6), 4, 4, seed=9)
        with torch.no_grad():
            for p, w in zip(_params(c), weights, strict=True):
                p.copy_(w)
        second = fit(c)
        second.load_state_dict(state)
        assert [float(second.step()) for _ in range(2)] == losses[2:]
        for p, q in zip(_params(a), _params(c), strict=True):
            assert torch.equal(p, q)

    def test_fit_offline_calls_back_every_steps(self):
        x = torch.randn(10, 6, 4, generator=torch.Generator().manual_seed(2))
        mods = build_dynamics(DynamicsConf(d_s=6), 4, 4, seed=1)
        seen = []
        fit_offline(
            mods,
            x,
            steps=5,
            batch=3,
            lr=1e-2,
            local=True,
            observed=3,
            gen=torch.Generator().manual_seed(0),
            every=2,
            on_eval=lambda step, m: seen.append((step, m is mods)),
        )
        assert seen == [(2, True), (4, True)]


class TestOfflineEvaluation:
    def test_the_same_weights_evaluate_as_online(self, tmp_path):
        """The offline task evaluates with the clips, the standardization and
        the rollout of the online one: the same weights, the same row."""
        online = PROGRAM.build(_conf("online", nodes=2), tmp_path, None)
        offline = PROGRAM.build(_conf("offline", nodes=2), tmp_path, None)
        try:
            src, dst = online.modules(), offline.modules()
            with torch.no_grad():
                for name, m in src.items():
                    for p in m.parameters():
                        p.normal_(0, 0.3, generator=torch.Generator().manual_seed(7))
                    dst[name].load_state_dict(m.state_dict())
            a, b = online.evaluate("test"), offline.evaluate("test")
        finally:
            online.close()
            offline.close()
        rolls = [k for k in a if k.startswith("roll")]
        assert len(rolls) > 10
        torch.testing.assert_close(
            torch.tensor([a[k] for k in rolls]),
            torch.tensor([b[k] for k in rolls]),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
