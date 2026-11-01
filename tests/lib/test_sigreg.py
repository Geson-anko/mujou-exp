import math

import mujou.pc as pc
import mujou.testing
import pytest
import torch

import mujou
from exp.lib.sigreg import (
    SIGReg,
    ecf_sums,
    epps_pulley,
    insert_,
    sigreg,
    sigreg_from_sums,
)
from mujou import TensorSpec

TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
D = 6
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


class _SIGRegNet(mujou.Network):
    def __init__(self, every: int = 1) -> None:
        super().__init__()
        self.z_in = _inputs(("z", D))["z"]
        self.sigreg = SIGReg(
            act_spec=self.z_in.act_spec,
            kappa=KAPPA,
            directions=8,
            capacity=10,
            tau=TAU,
            min_samples=4,
            every=every,
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


class TestEvery:
    def test_it_keeps_the_z_of_one_frame_in_every(self):
        """With every=3, the buffer holds the z of frames 0, 3, 6, ...: by
        frame k, those before it."""
        net = _SIGRegNet(every=3)
        zs = torch.arange(10.0)[:, None].expand(10, D).contiguous()
        with mujou.testing.run(net) as run:
            for k in range(len(zs)):
                run.write({net.z_in.prev: _sample(zs[k], reset=k == 0)})
                for _ in range(16):
                    run.step(TAU.period)
                kept = net.sigreg.buffer[: net.sigreg.count, 0].tolist()
                assert kept == [float(j) for j in range(0, k, 3)]


class TestSums:
    def test_the_sums_give_the_value_and_gradient_of_sigreg(self):
        """Splitting the characteristic function into the buffer's sums and
        z's own term is the same sum, so the same value and gradient."""
        g = torch.Generator().manual_seed(7)
        buffer = torch.randn(40, D, generator=g, dtype=torch.float64)
        a = torch.randn(32, D, generator=g, dtype=torch.float64)
        a = a / a.norm(dim=1, keepdim=True)
        z = (2 * torch.randn(D, generator=g, dtype=torch.float64)).requires_grad_()
        want = sigreg(z, buffer, a)
        (g_want,) = torch.autograd.grad(want, z)
        got = sigreg_from_sums(z, ecf_sums(buffer, a), len(buffer) + 1, a)
        (g_got,) = torch.autograd.grad(got, z)
        torch.testing.assert_close(got, want)
        torch.testing.assert_close(g_got, g_want)
