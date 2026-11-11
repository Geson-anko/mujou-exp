import math

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import nn

from exp.video.dyn.bp import DynamicsBP
from exp.video.dyn.modules import build_dynamics
from exp.video.dyn.nets import StateWorldBP
from exp.video.dyn.nodes import is_open
from exp.video.schema.data import DataConf
from exp.video.schema.dyn import DynamicsConf
from exp.video.schema.learn import LearnerConf, OptimConf
from exp.video.schema.run import RunConf, TimeConf

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
