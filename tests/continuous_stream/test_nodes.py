import math

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import nn

import mujou
from experiments.continuous_stream.host import Host
from experiments.continuous_stream.nodes import (
    Compare,
    Dynamics,
    MinGRU,
    SIGReg,
    compare_energy,
    epps_pulley,
    insert_,
    sigreg,
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


def _closed_ep(x: torch.Tensor) -> torch.Tensor:
    r"""EP of each column of ``x`` with the integral over the real line in closed
    form (the weight $e^{-t^2/2}$ makes every term a Gaussian integral)."""
    n = x.shape[0]
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N, M)
    ecf2 = math.sqrt(2 * math.pi) * torch.exp(-(diff**2) / 2).sum((0, 1)) / n**2
    cross = math.sqrt(math.pi) * torch.exp(-(x**2) / 4).sum(0) / n
    phi2 = math.sqrt(2 * math.pi / 3)
    return n * (ecf2 - 2 * cross + phi2)


class TestEppsPulley:
    def test_the_trapezoid_rule_matches_the_closed_form(self):
        g = torch.Generator().manual_seed(0)
        near = torch.randn(256, 4, generator=g, dtype=torch.float64)
        wide = 2 * torch.randn(64, 4, generator=g, dtype=torch.float64) + 0.5
        fine = torch.linspace(-5, 5, 4001, dtype=torch.float64)
        for x in (near, wide):  # the formula, on a fine grid
            torch.testing.assert_close(
                epps_pulley(x, fine), _closed_ep(x), rtol=1e-4, atol=0
            )
        # The 17 points: close near N(0, 1), aliased by a few % far from it.
        torch.testing.assert_close(
            epps_pulley(near), _closed_ep(near), rtol=1e-3, atol=0
        )
        torch.testing.assert_close(
            epps_pulley(wide), _closed_ep(wide), rtol=5e-2, atol=0
        )

    @pytest.mark.parametrize("change", ["wide", "narrow", "shifted"])
    def test_it_grows_away_from_the_standard_normal(self, change):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(256, 64, generator=g, dtype=torch.float64)
        moved = {"wide": 2 * x, "narrow": 0.5 * x, "shifted": x + 1}[change]
        assert epps_pulley(moved).mean() > 5 * epps_pulley(x).mean()


class TestSIGRegFunction:
    def test_the_gradient_matches_finite_differences(self):
        g = torch.Generator().manual_seed(2)
        z = torch.randn(D, generator=g, dtype=torch.float64)
        buffer = torch.randn(31, D, generator=g, dtype=torch.float64)
        a = torch.randn(16, D, generator=g, dtype=torch.float64)
        a = a / a.norm(dim=1, keepdim=True)
        leaf = z.clone().requires_grad_()
        (grad,) = torch.autograd.grad(sigreg(leaf, buffer, a), [leaf])
        torch.testing.assert_close(grad, _fd(lambda v: sigreg(v, buffer, a), z))


class TestInsert:
    def test_it_fills_in_order_then_replaces(self):
        buffer = torch.zeros(4, 1)
        g = torch.Generator().manual_seed(0)
        count = 0
        for i in range(4):
            count = insert_(buffer, count, torch.tensor([float(i + 1)]), g)
        assert count == 4
        assert buffer.flatten().tolist() == [1, 2, 3, 4]
        count = insert_(buffer, count, torch.tensor([9.0]), g)
        assert count == 4 and int((buffer == 9).sum()) == 1

    def test_ages_are_geometric_with_mean_capacity(self):
        """Each row is replaced with probability 1 / capacity per insert, so the
        age of a row is geometric: P(a) = p (1 - p)^a, of mean 1 / p - 1."""
        capacity = 255
        buffer = torch.zeros(capacity, 1)
        g = torch.Generator().manual_seed(0)
        count = 0
        means = []
        for k in range(capacity + 60_000):
            count = insert_(buffer, count, torch.tensor([float(k)]), g)
            if k > 2 * capacity and k % capacity == 0:
                means.append(float((k - buffer).mean()))
        assert abs(sum(means) / len(means) - (capacity - 1)) < 6

    def test_the_same_seed_replaces_the_same_rows(self):
        def fill(seed: int) -> torch.Tensor:
            buffer, g, count = torch.zeros(8, 1), torch.Generator().manual_seed(seed), 0
            for k in range(50):
                count = insert_(buffer, count, torch.tensor([float(k)]), g)
            return buffer

        assert torch.equal(fill(3), fill(3))


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
    def __init__(self) -> None:
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


class _SIGRegNet(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.z_in = _inputs(("z", D))["z"]
        self.sigreg = SIGReg(
            act_spec=self.z_in.act_spec,
            kappa=KAPPA,
            directions=8,
            capacity=10,
            tau=TAU,
            min_samples=4,
            seed=5,
        )
        self.z_in.next.connect(self.sigreg.prev)


class TestSIGRegNode:
    def test_it_sends_kappa_times_the_gradient_and_keeps_the_past(self):
        net = _SIGRegNet()
        g = torch.Generator().manual_seed(6)
        zs = torch.randn(14, D, generator=g)
        with mujou.testing.run(net) as run:
            for k in range(len(zs)):
                run.write({net.z_in.prev: _sample(zs[k], reset=k == 0)})
                for _ in range(16):
                    run.step(TAU.period)
                sent = run.read(net.sigreg.prev)
                node = net.sigreg
                if k < 3:  # fewer than min_samples with the current z
                    assert not sent.any()
                    continue
                assert node.count == min(k, 10)
                buffer = node.buffer[: node.count].double()
                assert not node.buffer.requires_grad
                a = node.directions.double()
                want = KAPPA * _fd(lambda v: sigreg(v, buffer, a), zs[k].double())
                torch.testing.assert_close(sent.double(), want, rtol=1e-4, atol=1e-7)
                if k == 6:  # before it is full: the frames before, in order
                    torch.testing.assert_close(node.buffer[:6], zs[:6])

    def test_the_directions_are_drawn_anew_with_each_frame_only(self):
        net = _SIGRegNet()
        with mujou.testing.run(net) as run:
            run.write({net.z_in.prev: _sample(torch.ones(D))})
            run.step(TAU.period)
            first = net.sigreg.directions.clone()
            run.step(TAU.period)
            assert torch.equal(net.sigreg.directions, first)
            run.write({net.z_in.prev: _sample(torch.ones(D))})
            run.step(TAU.period)
            assert not torch.equal(net.sigreg.directions, first)
        torch.testing.assert_close(first.norm(dim=1), torch.ones(8))


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
