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
    def __init__(self, every: int = 1, streams: int = 0) -> None:
        super().__init__()
        if streams:
            self.z_in = pc.Input(
                x=TensorSpec((streams, D)),
                y=TensorSpec((streams, 1)),
                period=TAU.period,
                batch=streams,
            )
        else:
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


class TestBatch:
    """B streams: the statistic of the B current z and the buffer together."""

    def test_the_statistic_is_that_of_the_concatenated_samples(self):
        g = torch.Generator().manual_seed(8)
        buffer = torch.randn(20, D, generator=g, dtype=torch.float64)
        z = 2 * torch.randn(3, D, generator=g, dtype=torch.float64)
        a = torch.randn(16, D, generator=g, dtype=torch.float64)
        a = a / a.norm(dim=1, keepdim=True)
        want = epps_pulley(torch.cat([z, buffer]) @ a.T).mean()
        torch.testing.assert_close(sigreg(z, buffer, a), want)

    def test_the_sums_give_the_value_and_gradient_of_the_batch(self):
        g = torch.Generator().manual_seed(9)
        buffer = torch.randn(40, D, generator=g, dtype=torch.float64)
        a = torch.randn(32, D, generator=g, dtype=torch.float64)
        a = a / a.norm(dim=1, keepdim=True)
        z = (2 * torch.randn(4, D, generator=g, dtype=torch.float64)).requires_grad_()
        want = sigreg(z, buffer, a)
        (g_want,) = torch.autograd.grad(want, z)
        got = sigreg_from_sums(z, ecf_sums(buffer, a), len(buffer) + 4, a)
        (g_got,) = torch.autograd.grad(got, z)
        torch.testing.assert_close(got, want)
        torch.testing.assert_close(g_got, g_want)

    def test_insert_fills_rows_in_order_then_replaces_drawn_rows(self):
        buffer = torch.zeros(5, 1)
        g = torch.Generator().manual_seed(0)
        z = torch.arange(1.0, 4.0)[:, None]
        count = insert_(buffer, 0, z, g)
        assert count == 3 and buffer[:3].flatten().tolist() == [1, 2, 3]
        count = insert_(buffer, count, z + 3, g)  # 2 fit, the third replaces
        assert count == 5
        kept = [v for v in buffer.flatten().tolist() if v != 6]
        assert len(kept) == 4  # the third row replaced one of the five
        assert set(kept) <= {1, 2, 3, 4, 5}

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="no CUDA"
                ),
            ),
        ],
    )
    def test_several_rows_replace_distinct_rows_the_same_way_each_time(self, device):
        capacity, k = 16, 12  # many rows: draws with replacement would collide

        def fill(seed: int) -> torch.Tensor:
            buffer = torch.zeros(capacity, 1, device=device)
            g = torch.Generator(device).manual_seed(seed)
            z = torch.arange(1.0, k + 1, device=device)[:, None]
            insert_(buffer, capacity, z, g)
            return buffer

        a = fill(3)
        assert torch.equal(a, fill(3))
        assert sorted(v for v in a.flatten().tolist() if v) == [
            float(i) for i in range(1, k + 1)
        ]  # every row landed, each in its own place

    def test_more_rows_than_the_capacity_keep_the_last_ones(self):
        buffer = torch.zeros(3, 1)
        g = torch.Generator().manual_seed(0)
        z = torch.arange(1.0, 8.0)[:, None]
        assert insert_(buffer, 0, z, g) == 3  # 1-3 fill it, 4-7 replace: 5-7 stay
        assert sorted(buffer.flatten().tolist()) == [5, 6, 7]

    def test_one_sample_draws_the_row_as_a_single_z_always_did(self):
        """The same generator state gives the same row to replace whether z
        is ``(D,)`` or ``(1, D)``, and the row is the one ``randint`` draws."""
        capacity = 8
        for z in (torch.ones(2), torch.ones(1, 2)):
            buffer = torch.zeros(capacity, 2)
            g = torch.Generator().manual_seed(4)
            insert_(buffer, capacity, z, g)
            want = int(
                torch.randint(capacity, (), generator=torch.Generator().manual_seed(4))
            )
            assert buffer[want].tolist() == [1, 1]
            assert int((buffer[:, 0] == 1).sum()) == 1

    def test_the_node_sends_b_times_kappa_times_the_gradient_per_sample(self):
        """Each sample gets kappa * B * dS/dz_b of the one statistic S of the
        current z and the buffer, and none when it has no target."""
        b = 3
        net = _SIGRegNet(streams=b)
        g = torch.Generator().manual_seed(10)
        zs = torch.randn(12, b, D, generator=g)
        with mujou.testing.run(net) as run:
            for k in range(len(zs)):
                mask = torch.tensor([True, True, k != 7])
                sample = pc.msg.Sample(
                    x=zs[k],
                    y=torch.zeros(b, 1),
                    mask=mask,
                    reset=torch.tensor([k == 0] * b),
                )
                run.write({net.z_in.prev: sample})
                for _ in range(16):
                    run.step(TAU.period)
                sent = run.read(net.sigreg.prev)
                node = net.sigreg
                assert node.count == min(b * k, 10)  # b rows per frame
                if node.count + b < 4:
                    assert not sent.any()
                    continue
                buffer = node.buffer[: node.count].double()
                a = node.directions.double()
                want = KAPPA * b * _fd(lambda v: sigreg(v, buffer, a), zs[k].double())
                want[~mask] = 0
                torch.testing.assert_close(sent.double(), want, rtol=1e-4, atol=1e-7)
                assert not sent[~mask].any()
                if k == 2:  # before it is full: the frames before, in order
                    torch.testing.assert_close(node.buffer[:6], zs[:2].reshape(6, D))
