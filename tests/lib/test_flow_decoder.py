r"""``exp.lib.flow_decoder``: the loss and the sampler against the optimal
velocity of 1-d Gaussian data (its closed-form minimum, the $\epsilon^2$ of a
shift, the exact ODE's map $x_0 \mapsto m + s x_0$), the data of one point
reached in the last Euler step, and the U-Net's shape, conditioning, seed,
parameter count by hand and a small fit that tells two images apart."""

import math

import pytest
import torch
from torch import Tensor

from exp.lib.flow_decoder import (
    FlowUNet,
    euler_sample,
    fit_flow,
    flow_loss,
    lr_at,
)

M, S = 1.5, 0.4  # the data N(M, S^2)


def v_star(x: Tensor, t: Tensor, c: Tensor) -> Tensor:
    """E[x1 - x0 | x_t = x] of x1 ~ N(M, S^2), x0 ~ N(0, 1) independent."""
    t = t.reshape(-1, *([1] * (x.dim() - 1))).to(x.dtype)
    var = (1 - t) ** 2 + t**2 * S**2
    return M + (t * S**2 - (1 - t)) / var * (x - t * M)


def min_loss(t: Tensor) -> Tensor:
    """Var(x1 - x0 | x_t) at t: (1 + S^2) - Cov(u, x_t)^2 / Var(x_t)."""
    return (1 + S**2) - (t * S**2 - (1 - t)) ** 2 / ((1 - t) ** 2 + t**2 * S**2)


def draws(n: int, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    x1 = M + S * torch.randn(n, 1, generator=g, dtype=torch.float64)
    x0 = torch.randn(n, 1, generator=g, dtype=torch.float64)
    return x1, x0, torch.rand(n, generator=g, dtype=torch.float64)


class TestFlowLoss:
    def test_the_optimal_velocity_reaches_the_closed_form_minimum(self):
        n = 400_000
        x1, x0, t = draws(n)
        loss = float(flow_loss(v_star, x1, x1[:, :0], t=t, x0=x0))
        # given t, the squared error has mean min_loss(t) and variance
        # 2 min_loss(t)^2 (the square of a Gaussian residual)
        want = float(min_loss(t).mean())
        se = math.sqrt(float((2 * min_loss(t) ** 2).mean()) / n)
        assert abs(loss - want) < 4 * se

    def test_a_shift_of_eps_adds_eps_squared(self):
        x1, x0, t = draws(400_000, seed=1)
        c = x1[:, :0]
        eps = 0.3
        base = flow_loss(v_star, x1, c, t=t, x0=x0)
        shifted = flow_loss(lambda x, s, c: v_star(x, s, c) + eps, x1, c, t=t, x0=x0)
        # the difference is eps^2 + 2 eps mean(v* - u), the mean 0 +- its se
        resid = v_star((1 - t[:, None]) * x0 + t[:, None] * x1, t, c) - (x1 - x0)
        se = float(resid.std()) / math.sqrt(len(t))
        assert abs(float(shifted - base) - eps**2) < 4 * 2 * eps * se


class TestEulerSample:
    def test_it_follows_the_exact_ode_to_m_plus_s_x0(self):
        x0 = torch.randn(1000, 1, generator=torch.Generator().manual_seed(2))
        x0 = x0.double()
        exact = M + S * x0  # x_t = t M + sqrt((1-t)^2 + t^2 S^2) x0 at t = 1
        err = {
            k: float((euler_sample(v_star, x0, x0[:, :0], k) - exact).abs().max())
            for k in (100, 1000)
        }
        assert err[1000] < 5e-3  # of |x0| up to about 3.3
        assert err[1000] < err[100] / 5  # first order: about 10 times smaller

    @pytest.mark.parametrize("steps", [1, 3, 100])
    def test_the_data_of_one_point_is_reached_at_the_last_step(self, steps):
        target = torch.tensor([[0.25, -0.75]], dtype=torch.float64)

        def v(x: Tensor, t: Tensor, c: Tensor) -> Tensor:
            return (target - x) / (1 - t[:, None])

        x0 = torch.randn(5, 2, generator=torch.Generator().manual_seed(3)).double()
        out = euler_sample(v, x0, x0[:, :0], steps)
        torch.testing.assert_close(out, target.expand(5, -1), rtol=0, atol=1e-12)


class TestLrAt:
    def test_the_warmup_and_the_cosine(self):
        assert lr_at(0, lr=1.0, warmup=4, steps=12) == 0.25
        assert lr_at(3, lr=1.0, warmup=4, steps=12) == 1.0
        assert lr_at(8, lr=1.0, warmup=4, steps=12) == pytest.approx(0.5)
        assert lr_at(12, lr=1.0, warmup=4, steps=12) == pytest.approx(0.0)


SMALL = {"channels": (8, 16), "attn_levels": (1,), "groups": 4, "embed": 32}
SMALL |= {"time_dim": 16, "head_dim": 8}


def small_net(code_dim: int = 4, seed: int = 0) -> FlowUNet:
    torch.manual_seed(seed)
    return FlowUNet(code_dim, **SMALL)


def hand_count(d: int, channels: tuple[int, ...], attn: tuple[int, ...]) -> int:
    """The parameters of FlowUNet from the shape of each layer."""
    e, td = SMALL["embed"], SMALL["time_dim"]

    def conv(a: int, b: int, k: int) -> int:
        return a * b * k * k + b

    def lin(a: int, b: int) -> int:
        return a * b + b

    def res(a: int, b: int) -> int:
        skip = 0 if a == b else conv(a, b, 1)
        return 2 * a + conv(a, b, 3) + lin(e, 2 * b) + 2 * b + conv(b, b, 3) + skip

    def att(c: int) -> int:
        return 2 * c + conv(c, 3 * c, 1) + conv(c, c, 1)

    n = lin(td, e) + lin(e, e) + lin(d, e) + lin(e, e) + conv(3, channels[0], 3)
    skips, c = [channels[0]], channels[0]
    for i, ch in enumerate(channels):
        for _ in range(2):
            n += res(c, ch) + (att(ch) if i in attn else 0)
            c = ch
            skips.append(c)
        if i < len(channels) - 1:
            n += conv(c, c, 3)
            skips.append(c)
    n += 2 * res(c, c) + att(c)
    for i, ch in reversed(list(enumerate(channels))):
        for _ in range(3):
            n += res(c + skips.pop(), ch) + (att(ch) if i in attn else 0)
            c = ch
        if i > 0:
            n += conv(c, c, 3)
    return n + 2 * c + conv(c, 3, 3)


class TestFlowUNet:
    def test_the_shape_and_the_code_reaches_the_output(self):
        net = small_net()
        x = torch.randn(3, 3, 16, 16, generator=torch.Generator().manual_seed(0))
        t = torch.tensor([0.1, 0.5, 0.9])
        c = torch.randn(
            3, 4, generator=torch.Generator().manual_seed(1), requires_grad=True
        )
        out = net(x, t, c)
        assert out.shape == x.shape
        assert not torch.allclose(out, net(x, t, c + 1))
        out.square().sum().backward()
        assert c.grad is not None and float(c.grad.abs().sum()) > 0

    def test_the_same_seed_gives_the_same_weights_but_for_the_code_input(self):
        a, b = small_net(4), small_net(4)
        assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
        wide = small_net(9)
        same = {k: v for k, v in a.state_dict().items() if not k.startswith("code.")}
        other = wide.state_dict()
        assert all(torch.equal(v, other[k]) for k, v in same.items())

    @pytest.mark.parametrize("d", [4, 1024])
    def test_the_parameters_counted_by_hand(self, d):
        net = small_net(d)
        assert sum(p.numel() for p in net.parameters()) == hand_count(
            d, SMALL["channels"], SMALL["attn_levels"]
        )

    def test_the_full_net_of_the_plan(self):
        net = FlowUNet(1024)
        x = torch.zeros(1, 3, 64, 64)
        assert net(x, torch.zeros(1), torch.zeros(1, 1024)).shape == x.shape
        # the count of the prototype (plans/jepa.md, 19.52 M at D = 1024)
        assert sum(p.numel() for p in net.parameters()) == 19_515_555


class TestFitFlow:
    @pytest.mark.slow
    def test_two_images_told_apart_by_their_codes(self):
        g = torch.Generator().manual_seed(0)
        frames = torch.zeros(2, 3, 16, 16, dtype=torch.uint8)
        frames[0, 0], frames[1, 2, 8:] = 255, 255  # red; blue below
        codes = torch.tensor([[1.0], [-1.0]])
        net = small_net(1)
        losses = fit_flow(
            net, codes, frames, steps=400, batch=16, lr=2e-3, warmup=20, gen=g
        )
        assert losses.shape == (400,) and losses[-50:].mean() < losses[:50].mean()
        x0 = torch.randn(8, 3, 16, 16, generator=torch.Generator().manual_seed(1))
        c = codes.repeat_interleave(4, 0)
        out = euler_sample(net, x0, c, 20)
        real = frames.float() * (2 / 255) - 1
        nearest = torch.cdist(out.flatten(1), real.flatten(1)).argmin(1)
        assert nearest.tolist() == [0] * 4 + [1] * 4
