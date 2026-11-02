"""PC-ALM against backpropagation, the oracles of stage 0."""

import math
from collections.abc import Iterable, Sequence
from typing import Any, override

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from mujou.pc.optim import BoundOptimizer, Optimizer
from torch import nn

from exp.lib.host import Host
from exp.lib.linear_stream import LinearStream
from exp.lib.mingru import MinGRU
from exp.lib.sigreg import sigreg
from exp.video.bp import WorldBP, bp_chain
from exp.video.model import DynamicsStack, Top, decoder
from exp.video.nodes import compare_energy, recon_energy
from mujou import TensorSpec

TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
FRAME = 1 / 30


class _Spy(Optimizer):
    """SGD that records each step's gradients and elapsed."""

    def __init__(self, tau: float) -> None:
        self.steps: list[tuple[list[torch.Tensor], float]] = []
        self._sgd = pc.optim.SGD(tau=tau)

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        inner = self._sgd.bind(params)
        steps = self.steps

        class Bound(BoundOptimizer):
            @override
            def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
                steps.append(([g.clone() for g in grads], elapsed))
                inner.step_(grads, elapsed)

            @override
            def state_dict(self):
                return {}

            @override
            def load_state_dict(self, state, /):
                pass

        return Bound()


def _stream_through(
    net: pc.Chain, frames: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    stream = LinearStream(d=8, tau_x=1.0, frame=FRAME, noise=0.1, seed=0)
    seen = []
    yes = torch.tensor(True)
    with mujou.testing.run(net) as run:
        host = Host(run, frame=FRAME, period=TAU.period)
        for k in range(frames):
            x, y = stream.next()
            seen.append((x, y))
            sample = pc.msg.Sample(x=x, y=y, mask=yes, reset=torch.tensor(k == 0))
            host.show({net.prev: sample}, net.next)
    return seen


class TestOneLayer:
    def test_pc_alm_steps_exactly_as_bp(self):
        """An Input -> Output chain has no r to relax: each frame's step is the
        gradient of the loss at the weights it had, as plain BP computes it."""
        torch.manual_seed(0)
        linear = nn.Linear(8, 1)
        start = {k: v.clone() for k, v in linear.state_dict().items()}
        spy = _Spy(tau=10.0)
        net = pc.Chain(
            linear,
            x=TensorSpec((8,)),
            y=TensorSpec((1,)),
            loss=pc.loss.MSELoss(),
            tau=TAU,
            optimizer=spy,
            learn_interval=25e-3,
        )
        frames = 60
        seen = _stream_through(net, frames)
        assert len(spy.steps) == frames  # once per frame
        ref = nn.Linear(8, 1)
        ref.load_state_dict(start)
        for (x, y), (grads, elapsed) in zip(seen, spy.steps, strict=True):
            loss = pc.loss.MSELoss()(ref(x), y)
            want = torch.autograd.grad(loss, list(ref.parameters()))
            for g, w in zip(grads, want, strict=True):
                torch.testing.assert_close(g, w)
            with torch.no_grad():
                for p, w in zip(ref.parameters(), want, strict=True):
                    p.sub_(w, alpha=elapsed / 10.0)
        for got, want in zip(linear.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(got, want)
        # Each step weighs the time since the one before: about one frame.
        assert all(abs(e - FRAME) < 2.5e-3 for _, e in spy.steps[1:])


class TestTwoLayers:
    @pytest.mark.parametrize("bp", [False, True], ids=["pc", "bp"])
    def test_lam_is_the_credit_of_bp_at_the_fixed_point(self, bp):
        torch.manual_seed(0)
        first = nn.Sequential(nn.Linear(4, 3), nn.Tanh())
        last = nn.Linear(3, 2)
        x = torch.tensor([0.5, -1.0, 2.0, 0.1])
        y = torch.tensor([1.0, -1.0])
        kwargs: dict[str, Any] = dict(
            x=TensorSpec((4,)),
            y=TensorSpec((2,)),
            loss=pc.loss.MSELoss(),
            tau=TAU,
            optimizer=pc.optim.SGD(tau=1.0),
            learn_interval=math.inf,
        )
        if bp:  # one node: nothing to relax, the weights do not move
            net = bp_chain(first, last, **kwargs)
        else:
            net = pc.Chain(first, last, **kwargs)
        yes = torch.tensor(True)
        sample = pc.msg.Sample(x=x, y=y, mask=yes, reset=yes)
        with mujou.testing.run(net, inputs={net.prev: sample}) as run:
            for _ in range(2000):
                run.step(TAU.period)
            y_hat = run.read(net.next)["y_hat"]
        h = first(x).detach().requires_grad_()
        loss = pc.loss.MSELoss()(last(h), y)
        (credit,) = torch.autograd.grad(-loss, h)
        torch.testing.assert_close(y_hat, last(first(x)).detach())
        if not bp:
            torch.testing.assert_close(net.layers[0].lam, credit, atol=1e-6, rtol=1e-5)
            assert net.layers[0].r.abs().max() < 1e-6


class _Frozen(_Spy):
    """Records the gradients of each step and moves nothing."""

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        steps = self.steps

        class Bound(BoundOptimizer):
            @override
            def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
                steps.append(([g.clone() for g in grads], elapsed))

            @override
            def state_dict(self):
                return {}

            @override
            def load_state_dict(self, state, /):
                pass

        return Bound()


class TestWorldBPBatch:
    def test_it_learns_from_the_mean_of_the_per_sample_losses(self):
        """B = 2 streams, the second resetting at the last frame (no target):
        the gradient it learns by is that of autograd on the mean over the
        samples of the prediction error (0 for the sample without a target)
        plus SIGReg of the two current z (the sample without a target sends
        none), at the weights and memory it holds."""
        b, d_x, d_z, d_s, kappa = 2, 5, 4, 6, 0.05
        torch.manual_seed(0)
        encoder = nn.Linear(d_x, d_z)
        step = DynamicsStack(MinGRU(d_z, d_s), [])
        readout = nn.Linear(d_s, d_z)
        inp = pc.Input(
            x=TensorSpec((b, d_x)), y=TensorSpec((b, 1)), period=TAU.period, batch=b
        )
        spy = _Frozen(tau=1.0)
        world = WorldBP(
            (encoder, step, readout),
            act_spec=inp.act_spec,
            kappa=kappa,
            directions=8,
            capacity=10,
            tau=TAU,
            optimizer=spy,
            learn_interval=25e-3,
            min_samples=4,
            seed=1,
        )
        net: Any = mujou.Network()
        net.inp, net.world = inp, world
        inp.next.connect(world.prev)
        g = torch.Generator().manual_seed(2)
        xs = torch.randn(5, b, d_x, generator=g)
        with mujou.testing.run(net) as run:
            host = Host(run, frame=FRAME, period=TAU.period)
            for k in range(5):
                reset = torch.tensor([k == 0, k in (0, 4)])
                sample = pc.msg.Sample(
                    x=xs[k], y=torch.zeros(b, 1), mask=~reset, reset=reset
                )
                spy.steps.clear()
                host.show({inp.prev: sample}, world.next)
        assert spy.steps and world.count + b >= 4  # SIGReg is on
        grads, _ = spy.steps[-1]
        params = [*encoder.parameters(), *step.parameters(), *readout.parameters()]
        z = encoder(xs[4])
        s = step(world.z_prev, world.s_prev)
        err = compare_energy(nn.Sequential(Top(step.d_s), readout), s, z, kappa)
        # Stream 1 reset: its error counts nothing and SIGReg gets no credit
        # through its z.
        z_sig = torch.stack([z[0], z[1].detach()])
        loss = err[0] / b + kappa * sigreg(
            z_sig, world.buffer[: world.count], world.directions
        )
        want = torch.autograd.grad(loss, params)
        for got, w in zip(grads, want, strict=True):
            torch.testing.assert_close(got, w, rtol=1e-4, atol=1e-7)
        assert any(w.abs().max() > 1e-4 for w in want)

    def test_the_reconstruction_adds_its_mean_per_sample_energy(self):
        """With a decoder of weight ``recon``, the loss also has the mean over
        the samples with a target of ``recon_energy``, whose gradient reaches
        the encoder and the decoder."""
        b, d_z, d_s, kappa, recon = 2, 4, 6, 0.05, 0.7
        torch.manual_seed(0)
        encoder = nn.Sequential(nn.Flatten(-3), nn.Linear(64, d_z))
        step = DynamicsStack(MinGRU(d_z, d_s), [])
        readout = nn.Linear(d_s, d_z)
        dec = decoder(d_z, (1, 8, 8))
        inp = pc.Input(
            x=TensorSpec((b, 1, 8, 8)),
            y=TensorSpec((b, 1)),
            period=TAU.period,
            batch=b,
        )
        spy = _Frozen(tau=1.0)
        world = WorldBP(
            (encoder, step, readout),
            act_spec=inp.act_spec,
            kappa=kappa,
            directions=8,
            capacity=10,
            tau=TAU,
            optimizer=spy,
            learn_interval=25e-3,
            min_samples=100,  # no SIGReg
            seed=1,
            decoder=dec,
            recon=recon,
        )
        net: Any = mujou.Network()
        net.inp, net.world = inp, world
        inp.next.connect(world.prev)
        g = torch.Generator().manual_seed(2)
        xs = torch.rand(4, b, 1, 8, 8, generator=g)
        with mujou.testing.run(net) as run:
            host = Host(run, frame=FRAME, period=TAU.period)
            for k in range(4):
                reset = torch.tensor([k == 0, k in (0, 3)])
                sample = pc.msg.Sample(
                    x=xs[k], y=torch.zeros(b, 1), mask=~reset, reset=reset
                )
                spy.steps.clear()
                host.show({inp.prev: sample}, world.next)
        grads, _ = spy.steps[-1]
        params = [
            *encoder.parameters(),
            *step.parameters(),
            *readout.parameters(),
            *dec.parameters(),
        ]
        z = encoder(xs[3])
        s = step(world.z_prev, world.s_prev)
        err = compare_energy(nn.Sequential(Top(step.d_s), readout), s, z, kappa)
        rec = recon_energy(dec, z, xs[3], recon)
        loss = (err[0] + rec[0]) / b  # stream 1 reset: no target
        want = torch.autograd.grad(loss, params)
        for got, w in zip(grads, want, strict=True):
            torch.testing.assert_close(got, w, rtol=1e-4, atol=1e-7)
        assert any(w.abs().max() > 1e-4 for w in want[-len(list(dec.parameters())) :])
