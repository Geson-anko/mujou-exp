import math
from typing import Any

import mujou.pc as pc
import mujou.testing
import torch
from torch import nn

import mujou
from exp.lib.host import Host
from exp.lib.mingru import MinGRU
from exp.lib.sigreg import SIGReg
from exp.video.nodes import (
    Compare,
    Dynamics,
    compare_energy,
)
from mujou import TensorSpec

TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
FRAME = 1 / 30
D, S = 6, 5
KAPPA = 0.05


def _fd(f, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The central finite differences of the scalar ``f`` at ``x``."""
    out = torch.zeros_like(x)
    for i in range(x.numel()):
        e = torch.zeros_like(x).flatten()
        e[i] = eps
        e = e.view_as(x)
        out.view(-1)[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return out


def _inputs(*names_dims: tuple[str, int]) -> dict[str, pc.Input]:
    return {
        name: pc.Input(x=TensorSpec((d,)), y=TensorSpec((1,)), period=TAU.period)
        for name, d in names_dims
    }


def _sample(
    x: torch.Tensor, *, reset: bool = False, mask: bool = True
) -> pc.msg.Sample:
    return pc.msg.Sample(
        x=x, y=torch.zeros(1), mask=torch.tensor(mask), reset=torch.tensor(reset)
    )


class _CompareNet(mujou.Network):
    def __init__(self, target_weight: float = 1.0) -> None:
        super().__init__()
        ins = _inputs(("s", S), ("z", D))
        self.s_in, self.z_in = ins["s"], ins["z"]
        torch.manual_seed(0)
        self.compare = Compare(
            nn.Linear(S, D),
            s_spec=self.s_in.act_spec,
            z_spec=self.z_in.act_spec,
            kappa=KAPPA,
            tau=TAU,
            optimizer=pc.optim.SGD(tau=1.0),
            learn_interval=math.inf,
            target_weight=target_weight,
        )
        self.s_in.next.connect(self.compare.s)
        self.z_in.next.connect(self.compare.z)


class TestCompare:
    def test_both_gradients_match_finite_differences_of_the_energy(self):
        net = _CompareNet()
        g = torch.Generator().manual_seed(3)
        s, z = torch.randn(S, generator=g), torch.randn(D, generator=g)
        writes = {net.s_in.prev: _sample(s), net.z_in.prev: _sample(z)}
        with mujou.testing.run(net, inputs=writes) as run:
            run.step(TAU.period)
            g_s, g_z = run.read(net.compare.s), run.read(net.compare.z)
            p = run.read(net.compare.next)
        readout = net.compare.module.double()
        s, z = s.double(), z.double()
        want_s = _fd(lambda v: compare_energy(readout, v, z, KAPPA), s)
        want_z = _fd(lambda v: compare_energy(readout, s, v, KAPPA), z)
        torch.testing.assert_close(g_s.double(), want_s, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(g_z.double(), want_z, rtol=1e-4, atol=1e-6)
        assert bool(p["mask"])

    def test_the_target_gets_target_weight_times_its_gradient(self):
        g = torch.Generator().manual_seed(3)
        s, z = torch.randn(S, generator=g), torch.randn(D, generator=g)
        sent = {}
        for w in (1.0, 0.25, 0.0):
            net = _CompareNet(target_weight=w)
            writes = {net.s_in.prev: _sample(s), net.z_in.prev: _sample(z)}
            with mujou.testing.run(net, inputs=writes) as run:
                run.step(TAU.period)
                sent[w] = (run.read(net.compare.s), run.read(net.compare.z))
        torch.testing.assert_close(sent[0.25][1], 0.25 * sent[1.0][1])
        assert not sent[0.0][1].any()
        torch.testing.assert_close(sent[0.0][0], sent[1.0][0])  # s is unchanged

    def test_without_a_prediction_it_sends_no_credit(self):
        net = _CompareNet()
        writes = {
            net.s_in.prev: _sample(torch.ones(S), mask=False),
            net.z_in.prev: _sample(torch.ones(D)),
        }
        with mujou.testing.run(net, inputs=writes) as run:
            run.step(TAU.period)
            assert not run.read(net.compare.s).any()
            assert not run.read(net.compare.z).any()
            assert not bool(run.read(net.compare.next)["mask"])


class _DynamicsNet(mujou.Network):
    def __init__(self, learn_interval: float = math.inf) -> None:
        super().__init__()
        self.z_in = _inputs(("z", D))["z"]
        torch.manual_seed(0)
        self.dynamics = Dynamics(
            MinGRU(D, S),
            act_spec=self.z_in.act_spec,
            tau=TAU,
            optimizer=pc.optim.SGD(tau=1.0),
            learn_interval=learn_interval,
        )
        self.z_in.next.connect(self.dynamics.z)


class TestDynamics:
    def test_the_state_carried_is_that_of_the_sequential_mingru(self):
        net = _DynamicsNet()
        g = torch.Generator().manual_seed(4)
        zs = torch.randn(12, D, generator=g)
        resets = {0, 7}
        got, masks = [], []
        with mujou.testing.run(net) as run:
            host = Host(run, frame=FRAME, period=TAU.period)
            for k, z in enumerate(zs):
                writes = {net.z_in.prev: _sample(z, reset=k in resets)}
                host.show(writes, net.dynamics.next)
                act = run.read(net.dynamics.next)
                got.append(act["h"])
                masks.append(bool(act["mask"]))
        module = net.dynamics.module
        s = torch.zeros(S)
        with torch.no_grad():
            for k, z in enumerate(zs):
                if k in resets:
                    s = torch.zeros(S)
                else:
                    s = module(zs[k - 1], s)
                torch.testing.assert_close(got[k], s)
                assert masks[k] == (k not in resets)

    def test_it_sends_no_credit_to_the_past(self):
        net = _DynamicsNet(learn_interval=25e-3)
        with mujou.testing.run(net) as run:
            host = Host(run, frame=FRAME, period=TAU.period)
            for k in range(5):
                host.show(
                    {net.z_in.prev: _sample(torch.ones(D), reset=k == 0)},
                    net.dynamics.next,
                )
            assert not run.read(net.dynamics.z).any()


class _World(mujou.Network):
    """Input -> Z -> (Dynamics -> Compare, SIGReg): the shape of the model."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        opt, li = pc.optim.SGD(tau=1.0), 25e-3
        self.inp = _inputs(("x", 3))["x"]
        self.z = pc.Layer(
            nn.Linear(3, D),
            act_spec=self.inp.act_spec,
            tau=TAU,
            optimizer=opt,
            learn_interval=li,
        )
        self.dynamics = Dynamics(
            MinGRU(D, S),
            act_spec=self.z.act_spec,
            tau=TAU,
            optimizer=opt,
            learn_interval=li,
        )
        self.sigreg = SIGReg(
            act_spec=self.z.act_spec,
            kappa=KAPPA,
            directions=8,
            capacity=10,
            tau=TAU,
            min_samples=4,
        )
        self.compare = Compare(
            nn.Linear(S, D),
            s_spec=self.dynamics.act_spec,
            z_spec=self.z.act_spec,
            kappa=KAPPA,
            tau=TAU,
            optimizer=opt,
            learn_interval=li,
        )
        self.inp.next.connect(self.z.prev)
        self.z.next.connect(self.dynamics.z)
        self.z.next.connect(self.sigreg.prev)
        self.z.next.connect(self.compare.z)
        self.dynamics.next.connect(self.compare.s)


class TestLocality:
    def test_no_node_puts_gradients_on_parameters_and_learning_happens(self):
        net = _World()
        params = {
            name: [p.detach().clone() for p in m.parameters()]
            for name, m in (
                ("z", net.z.module),
                ("d", net.dynamics.module),
                ("c", net.compare.module),
            )
        }
        g = torch.Generator().manual_seed(7)
        with mujou.testing.run(net) as run:
            host = Host(run, frame=FRAME, period=TAU.period)
            for k in range(20):
                x = torch.randn(3, generator=g)
                host.show(
                    {net.inp.prev: _sample(x, reset=k == 0, mask=k > 0)},
                    net.compare.next,
                )
            assert not run.read(net.dynamics.z).any()
        for m in (net.z.module, net.dynamics.module, net.compare.module):
            assert all(p.grad is None for p in m.parameters())
        moved = zip(
            params.values(),
            (net.z.module, net.dynamics.module, net.compare.module),
            strict=True,
        )
        for before, m in moved:
            assert any(not torch.equal(b, p) for b, p in zip(before, m.parameters()))


B = 3


def _batch_inputs(*names_dims: tuple[str, int]) -> dict[str, pc.Input]:
    return {
        name: pc.Input(
            x=TensorSpec((B, d)), y=TensorSpec((B, 1)), period=TAU.period, batch=B
        )
        for name, d in names_dims
    }


def _batch_sample(x: torch.Tensor, mask: list[bool], reset: list[bool]):
    return pc.msg.Sample(
        x=x, y=torch.zeros(B, 1), mask=torch.tensor(mask), reset=torch.tensor(reset)
    )


class TestBatchCompare:
    def test_each_sample_gets_the_gradient_of_its_own_energy(self):
        """Per sample: g_s, g_z are finite differences of its own energy, and
        zeros for the sample without a prediction; the error is the mean over
        the samples that have one."""
        ins = _batch_inputs(("s", S), ("z", D))
        net: Any = mujou.Network()
        net.s_in, net.z_in = ins["s"], ins["z"]
        torch.manual_seed(0)
        net.compare = Compare(
            nn.Linear(S, D),
            s_spec=net.s_in.act_spec,
            z_spec=net.z_in.act_spec,
            kappa=KAPPA,
            tau=TAU,
            optimizer=pc.optim.SGD(tau=1.0),
            learn_interval=math.inf,
        )
        net.s_in.next.connect(net.compare.s)
        net.z_in.next.connect(net.compare.z)
        g = torch.Generator().manual_seed(3)
        s, z = torch.randn(B, S, generator=g), torch.randn(B, D, generator=g)
        mask = [True, False, True]
        no = [False] * B
        writes = {
            net.s_in.prev: _batch_sample(s, mask, no),
            net.z_in.prev: _batch_sample(z, [True] * B, no),
        }
        with mujou.testing.run(net, inputs=writes) as run:
            run.step(TAU.period)
            g_s, g_z = run.read(net.compare.s), run.read(net.compare.z)
            p = run.read(net.compare.next)
            error = run.read(net.compare.error)
        readout = net.compare.module.double()
        for b in range(B):
            want_s = torch.zeros(S, dtype=torch.float64)
            want_z = torch.zeros(D, dtype=torch.float64)
            if mask[b]:
                sb, zb = s[b].double(), z[b].double()
                want_s = _fd(lambda v: compare_energy(readout, v, zb, KAPPA), sb)
                want_z = _fd(lambda v: compare_energy(readout, sb, v, KAPPA), zb)
            torch.testing.assert_close(g_s[b].double(), want_s, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(g_z[b].double(), want_z, rtol=1e-4, atol=1e-6)
        assert p["mask"].tolist() == mask
        err = ((p["y_hat"] - p["y"]) ** 2).mean(-1)[torch.tensor(mask)].mean()
        torch.testing.assert_close(error, err)


class _BatchDynamicsNet(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.z_in = _batch_inputs(("z", D))["z"]
        torch.manual_seed(0)
        self.dynamics = Dynamics(
            MinGRU(D, S),
            act_spec=self.z_in.act_spec,
            tau=TAU,
            optimizer=pc.optim.SGD(tau=1.0),
            learn_interval=math.inf,
        )
        self.z_in.next.connect(self.dynamics.z)


class TestBatchDynamics:
    def test_a_stream_that_resets_does_not_touch_the_others(self):
        """Only stream 1 resets at frame 4: stream 0 and 2 go on as the
        sequential minGRU of their own z, stream 1 restarts from s = 0."""
        net = _BatchDynamicsNet()
        g = torch.Generator().manual_seed(4)
        zs = torch.randn(9, B, D, generator=g)
        resets = torch.zeros(9, B, dtype=torch.bool)
        resets[0] = True
        resets[4, 1] = True
        got, masks, counts = [], [], []
        with mujou.testing.run(net) as run:
            host = Host(run, frame=FRAME, period=TAU.period)
            for k in range(9):
                sample = _batch_sample(zs[k], (~resets[k]).tolist(), resets[k].tolist())
                host.show({net.z_in.prev: sample}, net.dynamics.next)
                act = run.read(net.dynamics.next)
                got.append(act["h"])
                masks.append(act["mask"])
                counts.append(net.dynamics.predictions.clone())
        module = net.dynamics.module
        s = torch.zeros(B, S)
        with torch.no_grad():
            for k in range(9):
                nxt = module(zs[k - 1], s)
                s = torch.where(resets[k][:, None], torch.zeros(B, S), nxt)
                torch.testing.assert_close(got[k], s)
                assert masks[k].tolist() == (~resets[k]).tolist()
        assert counts[3].tolist() == [3, 3, 3]
        assert counts[4].tolist() == [4, 0, 4]  # stream 1 counts anew
        assert counts[5].tolist() == [5, 1, 5]
