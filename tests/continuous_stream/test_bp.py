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

from experiments.continuous_stream.bp import bp_chain
from experiments.continuous_stream.host import Host
from experiments.continuous_stream.stream0 import LinearStream
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
