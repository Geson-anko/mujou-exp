import math
from dataclasses import replace
from typing import override

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from mujou.pc.optim import BoundOptimizer, Optimizer

from exp.lib.optim import AdamDirection, Grafted, StepAdamW, Warmup
from exp.video.dyn.modules import Top, build_dynamics
from exp.video.dyn.nets import (
    StateWorldBP,
    StateWorldPC,
    build_world,
    live_states,
    make_optimizer,
    modules,
)
from exp.video.dyn.nodes import compare_energy
from exp.video.schema.data import DataConf
from exp.video.schema.dyn import DynamicsConf
from exp.video.schema.learn import LearnerConf, OptimConf
from exp.video.schema.run import RunConf, TimeConf

D = 4  # the observation
STEPS = 17  # of the period in a frame of 1/30 s


def _conf(
    learner: str = "pc",
    *,
    streams: int = 1,
    learn_interval: float | None = math.inf,
    **dynamics,
) -> RunConf:
    dyn = {"d_s": 8, "open_loop": 0.0} | dynamics
    return RunConf(
        name="test",
        learner=LearnerConf(kind=learner),
        data=DataConf(streams=streams),
        dynamics=DynamicsConf(**dyn),
        time=TimeConf(learn_interval=learn_interval),
        optim=OptimConf(warmup=0.0),
    )


def _sample(x: torch.Tensor, reset) -> pc.msg.Sample:
    b = x.shape[0]
    reset = torch.as_tensor(reset).expand(b).clone()
    return pc.msg.Sample(x=x, y=torch.zeros(b, 1), mask=~reset, reset=reset)


def _show(run, net, sample, steps: int = STEPS) -> None:
    run.write({net.prev: sample})
    for _ in range(steps):
        run.step(net.tau.period)


def _hold(run, net, steps: int) -> None:
    for _ in range(steps):
        run.step(net.tau.period)


def _params(net) -> list[torch.Tensor]:
    mods = modules(net)
    return [*mods.step.parameters(), *mods.readout.parameters()]


class _Recording(Optimizer):
    """Records the gradients of each step of each bound, and moves nothing."""

    def __init__(self) -> None:
        self.bound: list[tuple[list[torch.Tensor], list[list[torch.Tensor]]]] = []

    @override
    def bind(self, params) -> BoundOptimizer:
        steps: list[list[torch.Tensor]] = []
        self.bound.append((list(params), steps))

        class Bound(BoundOptimizer):
            @override
            def step_(self, grads, elapsed):
                steps.append([g.clone() for g in grads])

            @override
            def state_dict(self):
                return {}

            @override
            def load_state_dict(self, state, /):
                pass

        return Bound()

    def last(self, params) -> list[torch.Tensor]:
        """The last gradients of the bound whose first parameter is
        ``params[0]``."""
        (steps,) = (s for p, s in self.bound if p[0] is params[0])
        return steps[-1]


class TestMakeOptimizer:
    def test_a_step_of_one_frame_moves_sgd_by_lr(self):
        frame, lr = 1 / 30, 0.03
        p = torch.zeros(3)
        g = torch.tensor([1.0, -2.0, 0.5])
        opt = make_optimizer(OptimConf(kind="sgd", lr=lr, warmup=0.0), frame)
        opt.bind([p]).step_([g], frame)
        torch.testing.assert_close(p, -lr * g, rtol=1e-6, atol=0.0)

    def test_the_kinds_take_the_time_constant_of_the_rate(self):
        frame = 1 / 30
        c = OptimConf(lr=0.01, warmup=0.0, tau_m=0.2, tau_v=20.0)
        tau = frame / 0.01
        assert make_optimizer(c, frame) == pc.optim.SGD(tau=tau)
        adamw = replace(c, kind="adamw", weight_decay=0.1)
        assert make_optimizer(adamw, frame) == pc.optim.AdamW(
            tau=tau, tau_m=0.2, tau_v=20.0, weight_decay=0.1
        )
        graft = replace(c, kind="graft_adam")
        assert make_optimizer(graft, frame) == Grafted(
            inner=AdamDirection(tau=tau, tau_m=0.2, tau_v=20.0)
        )
        warm = replace(c, warmup=300.0)
        assert make_optimizer(warm, frame) == Warmup(
            inner=pc.optim.SGD(tau=tau), seconds=300.0
        )

    def test_adamw_step_is_step_adamw_of_the_schedules(self):
        c = OptimConf(
            kind="adamw_step",
            lr=7.5e-5,
            warmup=0.0,
            betas=[0.9, 0.99],
            weight_decay=0.04,
            weight_decay_end=0.4,
            warmup_steps=10,
            decay="cosine",
            decay_steps=100,
            min_lr=1e-3,
            clip=1.0,
        )
        assert make_optimizer(c, 1 / 30) == StepAdamW(
            lr=7.5e-5,
            betas=(0.9, 0.99),
            weight_decay=0.04,
            weight_decay_end=0.4,
            warmup_steps=10,
            decay="cosine",
            decay_steps=100,
            min_lr=1e-3,
            clip=1.0,
        )

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"kind": "adamw_step"}, "warmup_steps"),  # warmup 300 s
            ({"kind": "adamw", "warmup_steps": 5}, "adamw_step"),
            ({"kind": "sgd", "clip": 1.0}, "adamw_step"),
            ({"kind": "adamw", "betas": [0.8, 0.9]}, "adamw_step"),
        ],
    )
    def test_the_step_settings_belong_to_adamw_step(self, kw, match):
        with pytest.raises(ValueError, match=match):
            OptimConf(**kw)

    @pytest.mark.parametrize("lr", [0.0, -0.01])
    def test_a_rate_not_above_zero_is_an_error(self, lr):
        with pytest.raises(ValueError, match="lr must be > 0"):
            make_optimizer(OptimConf(lr=lr), 1 / 30)

    @pytest.mark.parametrize("kind", ["sgd", "graft_adam"])
    def test_weight_decay_without_adamw_is_an_error(self, kind):
        """A sweep of ``weight_decay`` would otherwise change nothing."""
        with pytest.raises(ValueError, match="has no weight decay"):
            make_optimizer(OptimConf(kind=kind, weight_decay=0.1), 1 / 30)


class TestBuildWorld:
    def test_the_learner_picks_the_network(self):
        assert isinstance(build_world(_conf("pc"), D), StateWorldPC)
        assert isinstance(build_world(_conf("bp"), D), StateWorldBP)

    def test_none_learns_no_world(self):
        with pytest.raises(ValueError, match="learns no world model"):
            build_world(_conf("none"), D)

    @pytest.mark.parametrize(
        "dynamics", [{}, {"nodes": 3, "residual": True}], ids=["", "big-residual"]
    )
    def test_pc_and_bp_start_from_the_same_weights(self, dynamics):
        pc_net = build_world(_conf("pc", **dynamics), D)
        bp_net = build_world(_conf("bp", **dynamics), D)
        mine, theirs = _params(pc_net), _params(bp_net)
        assert len(mine) == len(theirs) > 0
        for a, b in zip(mine, theirs, strict=True):
            assert a is not b and torch.equal(a, b)


def _randomize_readout(*nets) -> None:
    """The same random W_o in each of ``nets`` (residual starts it at 0)."""
    for net in nets:
        g = torch.Generator().manual_seed(9)
        w = modules(net).readout.weight
        with torch.no_grad():
            w.copy_(0.3 * torch.randn(w.shape, generator=g))


class TestEquilibrium:
    @pytest.mark.parametrize("streams", [1, 3])
    @pytest.mark.parametrize(
        "dynamics",
        [
            {"nodes": 1},
            {"nodes": 3, "residual": True, "open_loop": 0.5, "open_cycle": 4},
        ],
        ids=["small", "big-residual-open"],
    )
    def test_lam_of_each_node_is_the_credit_of_bp(self, dynamics, streams):
        """Hold a frame with learning off: each node's lam relaxes to the
        credit -B dL/dy_i of the loss L of ``DynamicsBP`` (the mean over the
        samples), for the output y_i of each layer, from the same modules and
        the memory of the same frames; r relaxes to 0."""
        nets = [
            build_world(_conf(k, streams=streams, **dynamics), D) for k in ("pc", "bp")
        ]
        _randomize_readout(*nets)
        pc_net, bp_net = nets
        assert isinstance(pc_net, StateWorldPC) and isinstance(bp_net, StateWorldBP)
        g = torch.Generator().manual_seed(1)
        xs = torch.randn(8, streams, D, generator=g)
        with mujou.testing.run(pc_net) as r1, mujou.testing.run(bp_net) as r2:
            for k, x in enumerate(xs):
                _show(r1, pc_net, _sample(x, k == 0))
                _show(r2, bp_net, _sample(x, k == 0))
            _hold(r1, pc_net, 3000)  # 6 s
        bp = bp_net.world
        ys: list[torch.Tensor] = []
        loss = bp.loss(xs[-1], torch.ones(streams, dtype=torch.bool), ys)
        credits = torch.autograd.grad(streams * loss, ys)
        block = pc_net.block
        nodes = [getattr(block, name) for name in block.names]
        assert len(nodes) == len(credits) == dynamics["nodes"]
        for node, credit in zip(nodes, credits, strict=True):
            torch.testing.assert_close(node.lam, -credit, rtol=1e-3, atol=1e-6)
            assert node.r.abs().max() < 1e-6
        assert credits[0].abs().max() > 1e-4  # the credit is not trivially zero

    @pytest.mark.parametrize("trace", [False, True], ids=["", "trace"])
    @pytest.mark.parametrize(
        "dynamics",
        [
            {"nodes": 1},
            {"nodes": 3, "residual": True, "open_loop": 0.5, "open_cycle": 4},
        ],
        ids=["small", "big-residual-open"],
    )
    def test_each_node_learns_by_the_gradient_of_bp(self, dynamics, trace):
        """Hold a frame with the weights still: the gradient each node of
        PC-ALM gives its optimizer (``g_theta / B``) is that of ``DynamicsBP``
        for the same parameters, the gradient of the mean loss L, with and
        without the eligibility traces (B = 3)."""
        recs = [_Recording(), _Recording()]
        nets = []
        for kind, rec in zip((StateWorldPC, StateWorldBP), recs, strict=True):
            conf = _conf(streams=3, learn_interval=25e-3, trace=trace, **dynamics)
            assert conf.dynamics is not None
            mods = build_dynamics(conf.dynamics, D, D, seed=0)
            nets.append(kind(mods, conf=conf, optimizer=rec))
        _randomize_readout(*nets)
        g = torch.Generator().manual_seed(8)
        xs = torch.randn(6, 3, D, generator=g)
        for net in nets:
            with mujou.testing.run(net) as run:
                for k, x in enumerate(xs):
                    _show(run, net, _sample(x, k == 0), STEPS if k < 5 else 2000)
        pc_net, bp_net = nets
        assert isinstance(pc_net, StateWorldPC)
        block = pc_net.block
        got = [
            g
            for node in [*(getattr(block, n) for n in block.names), block.compare]
            for g in recs[0].last(list(node.module.parameters()))
        ]
        want = recs[1].last(_params(bp_net))
        assert len(got) == len(want) == len(_params(pc_net))
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-3, atol=1e-6)
        assert max(float(w.abs().max()) for w in want) > 1e-4


class TestTrace:
    """``DynamicsConf.trace``: the dynamics learn through time by eligibility
    traces, with the inputs it remembered held."""

    def _run(self, learner: str, nodes: int = 1):
        rec = _Recording()
        conf = _conf(learner, learn_interval=25e-3, trace=True, nodes=nodes)
        assert conf.dynamics is not None
        mods = build_dynamics(conf.dynamics, D, D, seed=0)
        kind = StateWorldPC if learner == "pc" else StateWorldBP
        net = kind(mods, conf=conf, optimizer=rec)
        g = torch.Generator().manual_seed(2)
        xs = torch.randn(5, 1, D, generator=g)
        with mujou.testing.run(net) as run:
            for k, x in enumerate(xs):
                _show(run, net, _sample(x, k == 0), STEPS if k < 4 else 1500)
        return net, mods, rec, xs

    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_the_dynamics_learn_by_bptt_through_its_state(self, learner):
        """After K frames and a frame held, the gradient the minGRU learns by is
        that of autograd of the loss through s_K, ..., s_1 from s = 0 with
        the inputs held (the weights do not move), for PC-ALM (at
        equilibrium) as for BP."""
        net, mods, rec, xs = self._run(learner)
        cell = mods.step.first
        s = torch.zeros(1, cell.gate.out_features)
        for x in xs[:-1]:
            s = cell(x, s)
        # PC-ALM at equilibrium learns by the gradient of the loss, as BP.
        loss = compare_energy(lambda v: mods.readout(Top(8)(v)), s, xs[-1]).sum()
        params = list(cell.parameters())
        want = torch.autograd.grad(loss, params)
        got = rec.last(params)[: len(params)]
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-3, atol=1e-6)
        assert max(float(w.abs().max()) for w in want) > 1e-5

    def test_a_layer_learns_through_its_own_state(self):
        """With two nodes, the second learns by BPTT through its own state
        h_K, ..., h_1 with its inputs y1 (the first node's) held, of the
        loss at equilibrium."""
        net, mods, rec, xs = self._run("pc", nodes=2)
        assert isinstance(net, StateWorldPC)
        cell, layer = mods.step.first, mods.step.rest[0]
        with torch.no_grad():
            s = torch.zeros(1, 8)
            y1s = []
            for x in xs[:-1]:
                s = cell(x, s)
                y1s.append(s)
        h, y = torch.zeros_like(s), s
        for y1 in y1s:
            y = layer(y1, h)
            h = y - y1
        loss = compare_energy(mods.readout, y, xs[-1]).sum()
        params = list(layer.parameters())
        want = torch.autograd.grad(loss, params)
        for a, b in zip(rec.last(params), want, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-3, atol=1e-6)
        assert max(float(w.abs().max()) for w in want) > 1e-5


class TestTraceAtTheFirstFrame:
    def test_the_trace_is_the_gradient_of_one_frame(self):
        """At the first prediction of a video (K = 1, from s = 0) the traces
        are the Jacobian of one frame: each PC node learns by the same
        gradients with the trace as without it, in every learning step."""
        recs = []
        for trace in (False, True):
            rec = _Recording()
            conf = _conf(learn_interval=4e-3, trace=trace, nodes=2)
            assert conf.dynamics is not None
            net = StateWorldPC(
                build_dynamics(conf.dynamics, D, D, seed=0), conf=conf, optimizer=rec
            )
            _randomize_readout(net)
            g = torch.Generator().manual_seed(5)
            xs = torch.randn(2, 1, D, generator=g)
            with mujou.testing.run(net) as run:
                for k, x in enumerate(xs):
                    _show(run, net, _sample(x, k == 0))
            recs.append(rec)
        plain, traced = recs
        for (_, a), (_, b) in zip(plain.bound, traced.bound, strict=True):
            assert len(a) == len(b) > 0
            for step_a, step_b in zip(a, b, strict=True):
                for x, y in zip(step_a, step_b, strict=True):
                    torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)
        nonzero = [g for _, s in plain.bound for step in s for g in step]
        assert max(float(g.abs().max()) for g in nonzero) > 1e-5


class TestStates:
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_the_predictions_are_those_of_the_stack(self, learner, residual):
        """Over two videos with learning off, the prequential prediction of
        each frame is that of ``DynamicsStack`` and the readout from the same
        inputs, from s = 0 at each video; ``live_states`` is the stack's
        state after the last frame."""
        net = build_world(_conf(learner, streams=2, nodes=3, residual=residual), D)
        _randomize_readout(net)
        mods = modules(net)
        g = torch.Generator().manual_seed(3)
        xs = torch.randn(8, 2, D, generator=g)
        resets = torch.zeros(8, 2, dtype=torch.bool)
        resets[0] = True
        resets[5, 1] = True  # stream 1 alone starts a second video
        preds = []
        with mujou.testing.run(net) as run:
            for k in range(8):
                _show(run, net, _sample(xs[k], resets[k]))
                preds.append(run.read(net.next))
            live = live_states(net)
        state = torch.zeros(2, mods.step.d_state)
        with torch.no_grad():
            for k in range(8):
                state = torch.where(
                    resets[k][:, None], 0.0, mods.step(xs[k - 1], state)
                )
                want = mods.readout(Top(8)(state)) + (xs[k - 1] if residual else 0)
                p = preds[k]
                assert p["mask"].tolist() == (~resets[k]).tolist()
                keep = ~resets[k]
                torch.testing.assert_close(
                    p["y_hat"][keep], want[keep], rtol=1e-5, atol=1e-6
                )
        torch.testing.assert_close(live, state, rtol=1e-5, atol=1e-6)


class TestLearnedStates:
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_the_state_carried_is_of_the_weights_learned(self, learner):
        """With learning on, the state each learner carries to the next frame
        (``live_states``) is the minGRU of its remembered x_{k-1}, s_{k-1}
        at the weights after the frame's learning, for PC and BP alike."""
        conf = _conf(learner, streams=2, learn_interval=25e-3)
        net = build_world(replace(conf, optim=OptimConf(lr=0.3, warmup=0.0)), D)
        _randomize_readout(net)
        before = [p.detach().clone() for p in _params(net)]
        g = torch.Generator().manual_seed(6)
        with mujou.testing.run(net) as run:
            for k in range(4):
                _show(run, net, _sample(torch.randn(2, D, generator=g), k == 0))
            live = live_states(net)
        memory = net.world if isinstance(net, StateWorldBP) else net.block.dynamics
        with torch.no_grad():
            want = modules(net).step(memory.z_prev, memory.s_prev)
        torch.testing.assert_close(live, want, rtol=1e-6, atol=1e-7)
        assert not torch.equal(before[0], _params(net)[0])  # the weights learned


class TestResidual:
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_the_first_predictions_copy_the_observation_before(self, learner):
        """W_o starts at 0: the residual prediction of each frame is the
        observation before it (the baseline ``copy``)."""
        net = build_world(_conf(learner, streams=2, nodes=2, residual=True), D)
        g = torch.Generator().manual_seed(4)
        xs = torch.randn(6, 2, D, generator=g)
        with mujou.testing.run(net) as run:
            for k in range(6):
                _show(run, net, _sample(xs[k], k == 0))
                p = run.read(net.next)
                if k:
                    torch.testing.assert_close(p["y_hat"], xs[k - 1], rtol=0, atol=1e-7)
