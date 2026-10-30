import math

import mujou.pc as pc
import mujou.testing
import torch
from torch import nn

from experiments.continuous_stream.host import Host
from mujou import TensorSpec

TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)


def _chain() -> pc.Chain:
    torch.manual_seed(0)
    return pc.Chain(
        nn.Linear(2, 1),
        x=TensorSpec((2,)),
        y=TensorSpec((1,)),
        loss=pc.loss.MSELoss(),
        tau=TAU,
        optimizer=pc.optim.SGD(tau=1.0),
        learn_interval=math.inf,
    )


def _sample(x: torch.Tensor) -> pc.msg.Sample:
    yes = torch.tensor(True)
    return pc.msg.Sample(x=x, y=torch.zeros(1), mask=yes, reset=yes)


class TestHost:
    def test_the_frames_stay_on_the_grid(self):
        net = _chain()
        with mujou.testing.run(net) as run:
            host = Host(run, frame=1 / 30, period=TAU.period)
            for _ in range(30):
                host.show({net.prev: _sample(torch.ones(2))}, net.next)
            assert run.timer.now().ns == 1_000_000_000

    def test_it_returns_the_prediction_of_the_frame_written(self):
        net = _chain()
        x = torch.tensor([1.0, -2.0])
        with mujou.testing.run(net) as run:
            host = Host(run, frame=1 / 30, period=TAU.period)
            p, written = host.show({net.prev: _sample(x)}, net.next)
        assert p["time"] == written
        with torch.no_grad():
            torch.testing.assert_close(p["y_hat"], net.output.module(x))
