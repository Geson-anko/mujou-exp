import math
from typing import override

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import nn

import mujou
from exp.video.dyn.bp import DynamicsBP
from exp.video.dyn.modules import build_dynamics
from exp.video.dyn.nets import StateWorldBP, StateWorldPC, window_stats
from exp.video.dyn.nodes import Source, is_open
from exp.video.dyn.offline import OfflineFit, window_loss
from exp.video.schema.data import DataConf
from exp.video.schema.dyn import DynamicsConf, WindowConf
from exp.video.schema.learn import LearnerConf, OptimConf
from exp.video.schema.run import RunConf, TimeConf
from mujou import TensorSpec

D = 4
STEPS = 17


def _net(
    streams: int, *, learn_interval: float | None = math.inf, **dynamics
) -> StateWorldBP:
    dyn = DynamicsConf(**({"d_s": 6, "open_loop": 0.0} | dynamics))
    conf = RunConf(
        name="test",
        learner=LearnerConf(kind="bp"),
        data=DataConf(streams=streams),
        dynamics=dyn,
        time=TimeConf(learn_interval=learn_interval),
        optim=OptimConf(warmup=0.0),
    )
    mods = build_dynamics(dyn, D, D, seed=0)
    return StateWorldBP(mods, conf=conf, optimizer=pc.optim.SGD(tau=0.5))


def _params(net: StateWorldBP) -> list[torch.Tensor]:
    return [*net.world.step.parameters(), *net.world.readout.parameters()]


def _sample(x: torch.Tensor, reset) -> pc.msg.Sample:
    b = x.shape[0]
    reset = torch.as_tensor(reset).expand(b).clone()
    return pc.msg.Sample(x=x, y=torch.zeros(b, 1), mask=~reset, reset=reset)


def _show(run, net, sample) -> None:
    run.write({net.prev: sample})
    for _ in range(STEPS):
        run.step(net.tau.period)


class TestDynamicsBP:
    def test_it_takes_its_input_without_sending_credit(self):
        net = _net(2)
        assert isinstance(net.world, DynamicsBP)
        assert net.world.prev.send_spec is None

    @pytest.mark.parametrize(
        "dynamics",
        [{}, {"nodes": 2, "residual": True, "trace": True}],
        ids=["", "nodes-residual-trace"],
    )
    def test_identical_streams_learn_as_one_stream(self, dynamics):
        """B streams of the same inputs and resets, with learning on: the loss
        is the mean over the samples, so the parameters move as for the one
        stream, and every sample predicts what the one stream does."""
        one = _net(1, learn_interval=25e-3, **dynamics)
        many = _net(3, learn_interval=25e-3, **dynamics)
        before = [p.detach().clone() for p in _params(one)]
        g = torch.Generator().manual_seed(1)
        xs = torch.randn(6, 1, D, generator=g)
        with mujou.testing.run(one) as r1, mujou.testing.run(many) as r3:
            for k, x in enumerate(xs):
                reset = k in (0, 4)  # a second video from frame 4
                _show(r1, one, _sample(x, reset))
                _show(r3, many, _sample(x.expand(3, D), reset))
            p1, p3 = r1.read(one.next), r3.read(many.next)
        for b in range(3):
            torch.testing.assert_close(p3["y_hat"][b], p1["y_hat"][0])
        for a, b in zip(_params(one), _params(many), strict=True):
            torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)
        assert all(not torch.equal(a, b) for a, b in zip(before, _params(one)))

    @pytest.mark.parametrize("streams", [1, 2])
    def test_an_open_prediction_starts_from_the_one_before(self, streams):
        """In a cycle of 4 with half open, the 3rd and 4th predictions of a
        video remember the prediction of the frame before as x_{k-1}, the
        others the observed x_{k-1}."""
        net = _net(streams, open_loop=0.5, open_cycle=4)
        nn.init.normal_(net.world.readout.weight, std=0.3)
        g = torch.Generator().manual_seed(2)
        xs = torch.randn(9, streams, D, generator=g)
        last = None
        with mujou.testing.run(net) as run:
            for k in range(9):
                _show(run, net, _sample(xs[k], k == 0))
                if k >= 2:
                    want = last if is_open(k, 0.5, 4) else xs[k - 1]
                    torch.testing.assert_close(net.world.z_prev, want)
                last = run.read(net.next)["y_hat"]


# --- the learning on windows ----------------------------------------------------

W, O, S = 6, 3, 4
B64 = 3
PERIOD = 2e-3
FRAME = STEPS * PERIOD  # the frames are shown STEPS steps of PERIOD apart


class _World64(mujou.Network):
    """``StateWorldBP``'s ``Source -> DynamicsBP`` in float64."""

    def __init__(
        self, mods, dyn: DynamicsConf, optimizer, learn_interval: float = 25e-3
    ) -> None:
        super().__init__()
        tau = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
        assert tau.period == PERIOD
        self.tau = tau
        self.source = Source(
            x=TensorSpec((B64, D), dtype=torch.float64), period=PERIOD, batch=B64
        )
        self.world = DynamicsBP(
            mods,
            act_spec=self.source.act_spec,
            conf=dyn,
            tau=tau,
            optimizer=optimizer,
            learn_interval=learn_interval,
        )
        self.source.out.connect(self.world.prev)
        self.prev, self.next = self.source.prev, self.world.next


def _mods64(dyn: DynamicsConf):
    mods = build_dynamics(dyn, D, D, seed=0)
    mods.step.double()
    mods.readout.double()
    with torch.no_grad():  # a readout that is not 0, so that every weight moves
        g = torch.Generator().manual_seed(5)
        mods.readout.weight.copy_(
            0.3 * torch.randn(mods.readout.weight.shape, generator=g)
        )
    return mods


def _window_dyn(**window) -> DynamicsConf:
    win = WindowConf(
        **({"window": W, "observed": O, "stride": S, "relax": 0.01} | window)
    )
    return DynamicsConf(d_s=5, nodes=2, residual=True, open_loop=0.0, window=win)


def _stream(frames: int, seed: int = 3) -> torch.Tensor:
    """(frames, B, D) float64."""
    return torch.randn(
        frames,
        B64,
        D,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.float64,
    )


def _show_stream(
    net: _World64,
    xs: torch.Tensor,
    after: int = 0,
    resets: torch.Tensor | None = None,
) -> None:
    """Show the frames ``xs`` (the first resets, and the streams of
    ``resets`` ``(frames, B)`` where it holds), then ``after`` more steps
    without a new frame."""
    with mujou.testing.run(net) as run:
        for k, x in enumerate(xs):
            reset = torch.full((B64,), k == 0)
            if resets is not None:
                reset = reset | resets[k]
            run.write(
                {
                    net.prev: pc.msg.Sample(
                        x=x, y=torch.zeros(B64, 1), mask=~reset, reset=reset
                    )
                }
            )
            for _ in range(STEPS):
                run.step(PERIOD)
        for _ in range(after):
            run.step(PERIOD)


class _Recording(pc.optim.Optimizer):
    """``inner``, recording the gradients and the elapsed of every step."""

    def __init__(self, inner: pc.optim.Optimizer) -> None:
        self.inner = inner
        self.steps: list[tuple[list[torch.Tensor], float]] = []

    @override
    def bind(self, params):
        bound, steps = self.inner.bind(params), self.steps

        class _Bound(pc.optim.BoundOptimizer):
            @override
            def step_(self, grads, elapsed):
                steps.append(([g.clone() for g in grads], elapsed))
                bound.step_(grads, elapsed)

            @override
            def state_dict(self):
                return bound.state_dict()

            @override
            def load_state_dict(self, state):
                bound.load_state_dict(state)

        return _Bound()


def _params_of(mods) -> list[torch.Tensor]:
    return [*mods.step.parameters(), *mods.readout.parameters()]


def _fit_like(mods, xs: torch.Tensor, optimizer, dyn: DynamicsConf) -> OfflineFit:
    win = dyn.window
    assert win is not None
    return OfflineFit(
        mods,
        xs.transpose(0, 1),  # (B, frames, D)
        batch=B64,
        lr=1.0,
        local=False,
        observed=win.observed,
        residual=dyn.residual,
        gen=torch.Generator(),
        order="stream",
        window=win.window,
        stride=win.stride,
        optimizer=optimizer,
        elapsed=win.stride * FRAME,
        feedback=win.feedback,
        credit=win.credit,
    )


class TestWindowLearning:
    @pytest.mark.parametrize(
        "window",
        [{}, {"feedback": "grad"}, {"feedback": "teacher"}, {"credit": "own"}],
        ids=["detach", "grad", "teacher", "own"],
    )
    def test_the_first_step_is_the_gradient_of_the_window_loss(self, window):
        """With SGD, the gradient given to the optimizer at the first window is
        autograd's of ``window_loss`` on the first W frames (from the initial
        weights, normalized by B P), and the elapsed is S frames."""
        dyn = _window_dyn(**window)
        rec = _Recording(pc.optim.SGD(tau=0.5))
        net = _World64(_mods64(dyn), dyn, rec)
        ref = _mods64(dyn)
        xs = _stream(W)
        _show_stream(net, xs)
        assert len(rec.steps) == 1
        grads, elapsed = rec.steps[0]
        win = dyn.window
        assert win is not None
        loss = window_loss(
            ref,
            xs.transpose(0, 1),
            observed=O,
            local=False,
            residual=True,
            feedback=win.feedback,
            credit=win.credit,
        )
        want = torch.autograd.grad(loss, _params_of(ref))
        for g, r in zip(grads, want, strict=True):
            torch.testing.assert_close(g, r, rtol=0, atol=1e-10)
        assert elapsed == pytest.approx(S * FRAME, rel=1e-12)
        assert net.world.win_steps == 1
        torch.testing.assert_close(net.world.win_loss, loss.detach())

    @pytest.mark.parametrize("feedback", ["detach", "grad"])
    @pytest.mark.parametrize("relax", [0.0, 0.01, 0.05])
    @pytest.mark.parametrize("kind", ["sgd", "adamw"])
    @pytest.mark.parametrize("frames", [W, 17, 23])
    def test_a_run_learns_the_offline_fit_in_stream_order(
        self, kind, frames, relax, feedback
    ):
        """N frames learned on windows are ``(N - W) // S + 1`` steps of
        ``OfflineFit(order="stream")`` on the same streams, with the optimizer
        stepped by S frames: the same weights (once the last window is
        learned, ``relax`` after it)."""
        dyn = _window_dyn(relax=relax, feedback=feedback)

        def opt():
            if kind == "sgd":
                return pc.optim.SGD(tau=0.5)
            return pc.optim.AdamW(tau=2.0, tau_m=0.3, tau_v=5.0)

        net = _World64(_mods64(dyn), dyn, opt())
        xs = _stream(frames)
        _show_stream(net, xs, after=round(relax / PERIOD) + 1)
        steps = (frames - W) // S + 1
        assert net.world.win_steps == steps and net.world.win_skipped == 0
        ref = _mods64(dyn)
        fit = _fit_like(ref, xs, opt(), dyn)
        for _ in range(steps):
            fit.step()
        for p, q in zip(_params_of(net.world.mods), _params_of(ref), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=1e-8)
        assert not torch.equal(net.world.readout.weight, _mods64(dyn).readout.weight)

    def test_a_window_with_a_stream_that_resets_inside_leaves_it_out(self):
        """Stream 1 starts a new video at push S + 1, inside the first window
        (pushes 1..W): that window learns from streams 0 and 2 only, still
        normalized by B P (``window_loss`` with the mask); the second window
        (pushes S + 1..W + S) starts at the reset, so stream 1 is in it."""
        dyn = _window_dyn()
        rec = _Recording(pc.optim.SGD(tau=0.5))
        net = _World64(_mods64(dyn), dyn, rec)
        ref = _mods64(dyn)
        xs = _stream(W + S)
        resets = torch.zeros(W + S, B64, dtype=torch.bool)
        resets[S, 1] = True  # push S + 1
        _show_stream(net, xs, after=round(0.01 / PERIOD) + 1, resets=resets)
        assert len(rec.steps) == 2
        assert net.world.win_masked == 1 and net.world.win_skipped == 0
        win = dyn.window
        assert win is not None
        loss = window_loss(
            ref,
            xs[:W].transpose(0, 1),
            observed=O,
            local=False,
            residual=True,
            feedback=win.feedback,
            credit=win.credit,
            mask=torch.tensor([True, False, True]),
        )
        want = torch.autograd.grad(loss, _params_of(ref))
        for g, r in zip(rec.steps[0][0], want, strict=True):
            torch.testing.assert_close(g, r, rtol=0, atol=1e-10)

    def test_it_learns_in_the_first_update_at_or_after_relax(self):
        """The weights stay in every update before t_w + ``relax`` and move in
        the first one at or after it; the frame shown in between (``relax`` >
        a frame) is predicted at the weights before the window."""
        relax = 0.05
        dyn = _window_dyn(relax=relax)
        plain = DynamicsConf(d_s=5, nodes=2, residual=True, open_loop=0.0)
        net = _World64(_mods64(dyn), dyn, pc.optim.SGD(tau=0.5))
        frozen = _World64(_mods64(plain), plain, pc.optim.SGD(tau=0.5), math.inf)
        xs = _stream(W + 2)
        relax_ns = round(relax * 1e9)
        at_ns, learned_at, before_at = None, None, None
        preds: list[list[torch.Tensor]] = [[], []]
        for i, n in enumerate((net, frozen)):
            with mujou.testing.run(n) as run:
                for k, x in enumerate(xs):
                    reset = torch.full((B64,), k == 0)
                    run.write(
                        {
                            n.prev: pc.msg.Sample(
                                x=x, y=torch.zeros(B64, 1), mask=~reset, reset=reset
                            )
                        }
                    )
                    for _ in range(STEPS):
                        run.step(PERIOD)
                        if n is not net:
                            continue
                        world = n.world
                        if at_ns is None and world.pending is not None:
                            at_ns = world.pending["at_ns"]
                        now = run.timer.now().ns
                        if world.win_steps == 0:
                            before_at = now
                        elif learned_at is None:
                            learned_at = now
                    preds[i].append(run.read(n.next)["y_hat"].clone())
        assert at_ns is not None and learned_at is not None
        assert before_at is not None
        assert before_at < at_ns <= learned_at
        assert learned_at - before_at == round(PERIOD * 1e9)  # the next update
        # t_w: the time of frame W, the one the window is due at
        assert at_ns - relax_ns == pytest.approx((W - 1) * FRAME * 1e9, abs=1e3)
        # frames 1..W+1 are shown before the window is learned (W FRAME + a
        # frame < t_w + relax), frame W + 2 after it
        assert FRAME < relax < 2 * FRAME
        for k in range(W + 1):
            assert torch.equal(preds[0][k], preds[1][k])
        assert not torch.equal(preds[0][W + 1], preds[1][W + 1])

    def test_a_pending_window_whose_time_has_come_is_learned_before_the_next(
        self,
    ):
        """With ``relax`` half a period short of S frames, the time of each
        window comes in the update that pushes the next: it learns it there
        (before holding the next), so none is skipped."""
        dyn = _window_dyn(relax=S * FRAME - PERIOD / 2)
        net = _World64(_mods64(dyn), dyn, pc.optim.SGD(tau=0.5))
        frames = W + 3 * S
        _show_stream(net, _stream(frames), after=round(S * FRAME / PERIOD) + 1)
        assert net.world.win_skipped == 0
        assert net.world.win_steps == 4

    def test_a_window_due_while_one_is_pending_replaces_it(self):
        """With ``relax`` longer than S frames, every window is replaced by the
        next before its time; only the last is learned."""
        dyn = _window_dyn(relax=S * FRAME + 0.01)
        net = _World64(_mods64(dyn), dyn, pc.optim.SGD(tau=0.5))
        frames = W + 3 * S
        _show_stream(net, _stream(frames), after=round(0.2 / PERIOD))
        assert net.world.win_skipped == 3
        assert net.world.win_steps == 1

    def test_without_a_window_nothing_is_learned_until_w_frames(self):
        dyn = _window_dyn()
        net = _World64(_mods64(dyn), dyn, pc.optim.SGD(tau=0.5))
        before = [p.detach().clone() for p in _params_of(net.world.mods)]
        _show_stream(net, _stream(W - 1), after=50)
        assert net.world.win_steps == 0
        assert all(
            torch.equal(a, b) for a, b in zip(before, _params_of(net.world.mods))
        )
        assert math.isnan(float(net.world.win_loss))

    def test_before_its_first_window_it_predicts_as_one_that_never_learns(self):
        """No frame learns: before the first window the prediction of each
        frame is, to the bit, that of the per-frame model that does not
        learn."""
        dyn = _window_dyn()
        plain = DynamicsConf(d_s=5, nodes=2, residual=True, open_loop=0.0)
        a = _World64(_mods64(dyn), dyn, pc.optim.SGD(tau=0.5))
        b = _World64(_mods64(plain), plain, pc.optim.SGD(tau=0.5), math.inf)
        xs = _stream(W - 1)
        preds = []
        for net in (a, b):
            got = []
            with mujou.testing.run(net) as run:
                for k, x in enumerate(xs):
                    reset = torch.full((B64,), k == 0)
                    run.write(
                        {
                            net.prev: pc.msg.Sample(
                                x=x, y=torch.zeros(B64, 1), mask=~reset, reset=reset
                            )
                        }
                    )
                    for _ in range(STEPS):
                        run.step(PERIOD)
                    got.append(run.read(net.next)["y_hat"].clone())
            preds.append(got)
        for p, q in zip(*preds, strict=True):
            assert torch.equal(p, q)

    def test_its_state_resumes_bit_for_bit_with_a_window_pending(self):
        """Stopped with a window pending and restored from the run's state, the
        node goes on as one never stopped: the ring, the pending window, the
        counters and the optimizer."""
        dyn = _window_dyn(relax=0.05)
        xs = _stream(W + 2 * S)

        def show(run, net, k):
            reset = torch.full((B64,), k == 0)
            run.write(
                {
                    net.prev: pc.msg.Sample(
                        x=xs[k], y=torch.zeros(B64, 1), mask=~reset, reset=reset
                    )
                }
            )
            for _ in range(STEPS):
                run.step(PERIOD)

        def opt():
            return pc.optim.AdamW(tau=2.0, tau_m=0.3, tau_v=5.0)

        whole = _World64(_mods64(dyn), dyn, opt())
        with mujou.testing.run(whole) as run:
            for k in range(len(xs)):
                show(run, whole, k)
            for _ in range(40):
                run.step(PERIOD)
        first = _World64(_mods64(dyn), dyn, opt())
        cut = W + S  # a window due at push W + S, learned 0.05 s later
        with mujou.testing.run(first) as run:
            for k in range(cut):
                show(run, first, k)
            assert first.world.pending is not None
            state = run.state_dict()
        second = _World64(_mods64(dyn), dyn, opt())
        with mujou.testing.run(second, state=state) as run:
            assert second.world.pending is not None
            for k in range(cut, len(xs)):
                show(run, second, k)
            for _ in range(40):
                run.step(PERIOD)
        assert second.world.win_steps == whole.world.win_steps == 3
        for p, q in zip(
            _params_of(whole.world.mods), _params_of(second.world.mods), strict=True
        ):
            assert torch.equal(p, q)
        assert torch.equal(whole.world.win_loss, second.world.win_loss)
        assert window_stats(whole.world) == window_stats(second.world)

    def test_window_stats_needs_a_window(self):
        net = _net(1)
        with pytest.raises(ValueError, match="windows"):
            window_stats(net.world)


class TestWindowConf:
    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"window": 1, "observed": 0}, "window must be >= 2"),
            ({"observed": 36}, "observed"),
            ({"observed": 0}, "observed"),
            ({"stride": 0}, "stride"),
            ({"relax": -1.0}, "relax"),
            ({"feedback": "x"}, "feedback"),
            ({"credit": "x"}, "credit"),
            ({"credit": "own", "feedback": "grad"}, "detaches"),
        ],
    )
    def test_a_window_that_cannot_run_is_refused(self, kw, match):
        with pytest.raises(ValueError, match=match):
            WindowConf(**kw)

    def test_a_relax_of_a_stride_is_refused(self):
        """``relax`` >= S frames would learn each window late, at the push of
        the next."""
        dyn = _window_dyn(relax=S / 30)
        with pytest.raises(ValueError, match="relax"):
            RunConf(name="test", dynamics=dyn, time=TimeConf(frame=1 / 30))
        RunConf(name="test", dynamics=_window_dyn(relax=S / 30 - 1e-3))

    def test_a_window_with_traces_is_refused(self):
        with pytest.raises(ValueError, match="trace"):
            DynamicsConf(trace=True, window=WindowConf())

    def test_the_pc_learner_refuses_the_own_credit(self):
        """``credit="own"`` is an ablation of BP; PC-ALM learns windows with
        ``credit="bptt"`` only."""
        dyn = _window_dyn(credit="own")
        conf = RunConf(
            name="test",
            learner=LearnerConf(kind="pc"),
            data=DataConf(streams=2),
            dynamics=dyn,
        )
        mods = build_dynamics(dyn, D, D, seed=0)
        with pytest.raises(ValueError, match="ablation of BP only"):
            StateWorldPC(mods, conf=conf, optimizer=pc.optim.SGD(tau=0.5))
        StateWorldPC(
            mods,
            conf=RunConf(
                name="test",
                learner=LearnerConf(kind="pc"),
                data=DataConf(streams=2),
                dynamics=_window_dyn(),
            ),
            optimizer=pc.optim.SGD(tau=0.5),
        )
