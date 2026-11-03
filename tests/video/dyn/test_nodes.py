import math
from typing import Any, cast, override

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import nn

import mujou
from exp.lib.mingru import MinGRU
from exp.video.dyn.modules import build_dynamics
from exp.video.dyn.nodes import (
    Compare,
    Dynamics,
    DynamicsBlock,
    Source,
    compare_energy,
    is_open,
)
from exp.video.schema.dyn import DynamicsConf
from mujou import Port, TensorSpec

TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
STEPS = 17  # of TAU.period in a frame of 1/30 s
D, S = 4, 6


def _fd(f, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The central differences of the scalar ``f`` at ``x``."""
    out = torch.zeros_like(x)
    for i in range(x.numel()):
        e = torch.zeros_like(x).flatten()
        e[i] = eps
        e = e.view_as(x)
        out.view(-1)[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return out


def _sample(x: torch.Tensor, reset, mask=None) -> pc.msg.Sample:
    """A sample of the batch of ``x`` ``(B, d)``; ``mask`` is ``~reset`` unless
    given."""
    b = x.shape[0]
    reset = torch.as_tensor(reset).expand(b).clone()
    mask = ~reset if mask is None else torch.as_tensor(mask).expand(b).clone()
    return pc.msg.Sample(x=x, y=torch.zeros(b, 1), mask=mask, reset=reset)


def _show(run, port, sample, steps: int = STEPS) -> None:
    run.write({port: sample})
    for _ in range(steps):
        run.step(TAU.period)


def _source(b: int, d: int = D) -> Source:
    return Source(x=TensorSpec((b, d)), period=TAU.period, batch=b)


class TestNoCredit:
    """The input and the target are constants by the types of the ports."""

    def test_a_port_that_sends_credit_cannot_take_the_source(self):
        source = _source(2)
        back = Port(send=source.act_spec.field("h", TensorSpec), recv=source.act_spec)
        with pytest.raises(TypeError, match="do not match"):
            source.out.connect(cast(Any, back))  # the mismatch under test
        layer = pc.Layer(
            nn.Linear(D, D),
            act_spec=source.act_spec,
            tau=TAU,
            optimizer=pc.optim.SGD(tau=1.0),
            learn_interval=None,
        )
        with pytest.raises(TypeError, match="do not match"):
            source.out.connect(cast(Any, layer.prev))

    def test_the_input_and_the_target_of_the_block_only_receive(self):
        source = _source(2)
        conf = DynamicsConf(d_s=S, nodes=2, residual=True, open_loop=0.5)
        block = DynamicsBlock(
            build_dynamics(conf, D, D, seed=0),
            act_spec=source.act_spec,
            conf=conf,
            tau=TAU,
            optimizer=pc.optim.SGD(tau=1.0),
            learn_interval=None,
        )
        block.connect(source.out)
        assert block.dynamics.z.send_spec is None
        assert block.compare.target.send_spec is None
        assert block.compare.base.send_spec is None
        assert block.dynamics.z_hat.send_spec is None
        assert source.out.recv_spec is None


class _CompareNet(mujou.Network):
    def __init__(
        self, b: int, residual: bool = False, learn_interval: float = math.inf
    ) -> None:
        super().__init__()
        self.s_in = pc.Input(
            x=TensorSpec((b, S)), y=TensorSpec((b, 1)), period=TAU.period, batch=b
        )
        self.x_in = _source(b)
        torch.manual_seed(0)
        self.compare = Compare(
            nn.Linear(S, D),
            s_spec=self.s_in.act_spec,
            target_spec=self.x_in.act_spec,
            tau=TAU,
            optimizer=pc.optim.SGD(tau=1.0),
            learn_interval=learn_interval,
            residual=residual,
        )
        self.s_in.next.connect(self.compare.s)
        self.x_in.out.connect(self.compare.target)
        if residual:
            self.b_in = _source(b)
            self.b_in.out.connect(self.compare.base)


class TestCompare:
    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_the_gradient_of_the_energy_is_its_central_difference(self, residual):
        """In float64: autograd of ``compare_energy`` with respect to s is the
        central difference of it."""
        g = torch.Generator().manual_seed(3)
        readout = nn.Linear(S, D).double()
        s = torch.randn(S, generator=g, dtype=torch.float64)
        x = torch.randn(D, generator=g, dtype=torch.float64)
        base = torch.randn(D, generator=g, dtype=torch.float64) if residual else None
        leaf = s.clone().requires_grad_()
        (got,) = torch.autograd.grad(compare_energy(readout, leaf, x, base), leaf)
        want = _fd(lambda v: compare_energy(readout, v, x, base).detach(), s)
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_each_sample_gets_the_gradient_of_its_own_energy(self, residual):
        """Per sample, the credit sent on ``s`` is the central difference of
        its own energy (float64; the node computes in float32), zeros for the
        sample without a prediction; the error is the mean over the samples
        that have one."""
        b = 3
        net = _CompareNet(b, residual)
        g = torch.Generator().manual_seed(3)
        s, x = torch.randn(b, S, generator=g), torch.randn(b, D, generator=g)
        base = torch.randn(b, D, generator=g)
        mask = torch.tensor([True, False, True])
        writes = {
            net.s_in.prev: _sample(s, False, mask),
            net.x_in.prev: _sample(x, False),
        }
        if residual:
            writes[net.b_in.prev] = _sample(base, False)
        with mujou.testing.run(net) as run:
            run.write(writes)
            run.step(TAU.period)
            run.step(TAU.period)  # the base comes after the node in the tree
            g_s = run.read(net.compare.s)
            p = run.read(net.compare.next)
            error = run.read(net.compare.error)
        readout = net.compare.module.double()
        for i in range(b):
            want = torch.zeros(S, dtype=torch.float64)
            if mask[i]:
                xi = x[i].double()
                bi = base[i].double() if residual else None
                want = _fd(
                    lambda v: compare_energy(readout, v, xi, bi).detach(), s[i].double()
                )
            torch.testing.assert_close(g_s[i].double(), want, rtol=1e-4, atol=1e-6)
        assert p["mask"].tolist() == mask.tolist()
        torch.testing.assert_close(p["y"], x)
        err = ((p["y_hat"] - p["y"]) ** 2).mean(-1)[mask].mean()
        torch.testing.assert_close(error, err)

    def test_inputs_of_two_frames_make_no_prediction(self):
        """``s`` of one write and the target of another: no credit, and no
        prediction stamped with the target's time."""
        net = _CompareNet(2)
        with mujou.testing.run(net) as run:
            run.write({net.s_in.prev: _sample(torch.ones(2, S), False)})
            run.step(TAU.period)
            written = run.write({net.x_in.prev: _sample(torch.ones(2, D), False)})
            run.step(TAU.period)
            assert not run.read(net.compare.s).any()
            p = run.read(net.compare.next)
            assert p["time"] != written and not p["mask"].any()

    def test_the_error_is_sent_once_a_frame_before_learning(self):
        """With the readout learning in the frame, ``error`` stays the error
        of the first prediction (the prequential one, of the weights the frame
        began with), while ``next`` follows the weights."""
        net = _CompareNet(2, learn_interval=4 * TAU.period)
        g = torch.Generator().manual_seed(6)
        s, x = torch.randn(2, S, generator=g), torch.randn(2, D, generator=g)
        with torch.no_grad():
            want = ((net.compare.module(s) - x) ** 2).mean()
        with mujou.testing.run(net) as run:
            run.write(
                {net.s_in.prev: _sample(s, False), net.x_in.prev: _sample(x, False)}
            )
            run.step(TAU.period)
            first = run.read(net.compare.next)["y_hat"]
            for _ in range(STEPS):
                run.step(TAU.period)
            last = run.read(net.compare.next)["y_hat"]
            error = run.read(net.compare.error)
        assert not torch.allclose(first, last)  # the readout learned
        torch.testing.assert_close(error, want)


class _DynamicsNet(mujou.Network):
    def __init__(self, b: int, *, open_loop: float = 0.0) -> None:
        super().__init__()
        self.source = _source(b)
        torch.manual_seed(0)
        self.dynamics = Dynamics(
            MinGRU(D, S),
            act_spec=self.source.act_spec,
            tau=TAU,
            optimizer=pc.optim.SGD(tau=1.0),
            learn_interval=math.inf,
            open_loop=open_loop,
            open_cycle=4,
        )
        self.source.out.connect(self.dynamics.z)


class TestDynamics:
    def test_the_state_carried_is_that_of_the_sequential_mingru(self):
        """Frame by frame, ``mu`` is the minGRU of the input before from the
        state before, from s = 0 at each video; stream 1 alone resets at
        frame 4, and the others go on."""
        b = 3
        net = _DynamicsNet(b)
        g = torch.Generator().manual_seed(4)
        xs = torch.randn(9, b, D, generator=g)
        resets = torch.zeros(9, b, dtype=torch.bool)
        resets[0] = True
        resets[4, 1] = True
        got, masks = [], []
        with mujou.testing.run(net) as run:
            for k in range(9):
                _show(run, net.source.prev, _sample(xs[k], resets[k]))
                act = run.read(net.dynamics.next)
                got.append(act["h"])
                masks.append(act["mask"])
        cell = net.dynamics.module
        s = torch.zeros(b, S)
        with torch.no_grad():
            for k in range(9):
                s = torch.where(resets[k][:, None], 0.0, cell(xs[k - 1], s))
                torch.testing.assert_close(got[k], s, rtol=1e-6, atol=1e-6)
                assert masks[k].tolist() == (~resets[k]).tolist()

    def test_an_open_prediction_starts_from_its_own_prediction(self):
        """In a cycle of 4 with half open, the 3rd and 4th predictions of a
        video remember the prediction of the frame before as x_{k-1} (here
        written by hand on ``z_hat``), the others the observed x_{k-1}."""
        net = _DynamicsNet(1, open_loop=0.5)
        g = torch.Generator().manual_seed(5)
        xs = torch.randn(9, 1, D, generator=g)
        hats = torch.randn(9, 1, D, generator=g)
        with mujou.testing.run(net) as run:
            for k in range(9):
                written = run.write({net.source.prev: _sample(xs[k], k == 0)})
                hat = pc.msg.Prediction(
                    y_hat=hats[k], y=xs[k], mask=torch.tensor([True]), time=written
                )
                run.write({net.dynamics.z_hat: hat})
                for _ in range(STEPS):
                    run.step(TAU.period)
                if k >= 1:  # z_prev was set at the start of frame k (count k)
                    want = hats[k - 1] if is_open(k, 0.5, 4) else xs[k - 1]
                    torch.testing.assert_close(net.dynamics.z_prev, want)


class TestLocality:
    def test_no_parameter_gets_a_grad_and_only_the_optimizers_move_them(self):
        """After a few frames of learning, no parameter holds ``.grad``; SGD
        moved every node's weights, and an optimizer that does not step moves
        none of them."""
        conf = DynamicsConf(d_s=S, nodes=2, trace=True, residual=True, open_loop=0.5)
        moved = {}
        for name, optimizer in (("sgd", pc.optim.SGD(tau=1.0)), ("none", _Still())):
            mods = build_dynamics(conf, D, D, seed=0)
            nn.init.normal_(mods.readout.weight, std=0.3)
            before = [p.detach().clone() for p in mods.step.parameters()]
            before += [p.detach().clone() for p in mods.readout.parameters()]
            net: Any = mujou.Network()
            net.source = _source(2)
            net.block = DynamicsBlock(
                mods,
                act_spec=net.source.act_spec,
                conf=conf,
                tau=TAU,
                optimizer=optimizer,
                learn_interval=25e-3,
            )
            net.block.connect(net.source.out)
            g = torch.Generator().manual_seed(7)
            with mujou.testing.run(net) as run:
                for k in range(8):
                    x = torch.randn(2, D, generator=g)
                    _show(run, net.source.prev, _sample(x, k == 0))
            after = [*mods.step.parameters(), *mods.readout.parameters()]
            assert all(p.grad is None for p in after)
            moved[name] = [not torch.equal(a, b) for a, b in zip(before, after)]
        assert all(moved["sgd"])
        assert not any(moved["none"])


class _Still(pc.optim.Optimizer):
    """Takes the gradients and moves nothing."""

    @override
    def bind(self, params):
        class Bound(pc.optim.BoundOptimizer):
            @override
            def step_(self, grads, elapsed):
                pass

            @override
            def state_dict(self):
                return {}

            @override
            def load_state_dict(self, state, /):
                pass

        return Bound()
