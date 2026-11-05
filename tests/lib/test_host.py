import math

import mujou.pc as pc
import mujou.testing
import torch
from torch import nn

import mujou
from exp.lib.host import Host
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

    def test_show_all_returns_the_first_message_of_each_port(self):
        net = _chain()
        x = torch.tensor([1.0, -2.0])
        with mujou.testing.run(net) as run:
            host = Host(run, frame=1 / 30, period=TAU.period)
            (a, b), written = host.show_all(
                {net.prev: _sample(x)}, [net.next, net.next]
            )
        assert a["time"] == b["time"] == written
        assert torch.equal(a["y_hat"], b["y_hat"])

    def test_a_host_resumed_from_its_state_steps_the_same_grid(self):
        """Frames of 1/30 s are 16 or 17 steps of 2 ms (the boundaries of
        ``floor(k frame / period)``): a host given ``frames = 4`` steps frame 4
        as a host that showed frames 0..3 does, not as frame 0."""
        net = _chain()
        with mujou.testing.run(net) as run:
            host = Host(run, frame=1 / 30, period=TAU.period)
            steps = []
            for _ in range(6):
                before = run.timer.now()
                host.show({net.prev: _sample(torch.ones(2))}, net.next)
                steps.append(round((run.timer.now() - before) / TAU.period))
            state = host.state_dict()
        assert len(set(steps)) == 2  # the grid is uneven, so frame k matters
        assert state == {"frames": 6}
        for k in (3, 4, 5):  # 16, 17, 17 steps
            other = _chain()
            with mujou.testing.run(other) as run:
                resumed = Host(run, frame=1 / 30, period=TAU.period)
                resumed.load_state_dict({"frames": k})
                before = run.timer.now()
                resumed.show({other.prev: _sample(torch.ones(2))}, other.next)
                got = round((run.timer.now() - before) / TAU.period)
            assert got == steps[k]
            assert resumed.frames == k + 1
