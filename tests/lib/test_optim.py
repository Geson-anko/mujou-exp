import math

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from mujou.pc.optim import Optimizer
from torch import nn

from exp.lib.optim import AdamDirection, Grafted, Warmup, adam
from mujou import TensorSpec


def _params(*shapes: tuple[int, ...], seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=g, dtype=torch.float64) for s in shapes]


def _grads(params: list[torch.Tensor], steps: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return [
        [torch.randn(p.shape, generator=g, dtype=p.dtype) for p in params]
        for _ in range(steps)
    ]


def _trajectory(opt: Optimizer, params, grads, elapsed=0.03) -> list[torch.Tensor]:
    params = [p.clone() for p in params]
    bound = opt.bind(params)
    for gs in grads:
        bound.step_(gs, elapsed)
    return params


SHAPES = ((5, 3), (4, 2, 3, 3), (5,))


class TestAdamDirection:
    def test_stepped_by_itself_it_is_adamw_without_decay(self):
        params = _params(*SHAPES)
        grads = _grads(params, 10)
        want = pc.optim.AdamW(tau=2.0, tau_m=0.3, tau_v=3.0, weight_decay=0.0)
        got = AdamDirection(tau=2.0, tau_m=0.3, tau_v=3.0)
        for a, b in zip(
            _trajectory(got, params, grads),
            _trajectory(want, params, grads),
            strict=True,
        ):
            torch.testing.assert_close(a, b)


class TestGrafted:
    def test_the_step_has_the_size_of_sgd(self):
        params = _params(*SHAPES)
        moved = [p.clone() for p in params]
        inner = AdamDirection(tau=0.5, tau_m=0.3, tau_v=3.0)
        bound = Grafted(inner=inner).bind(moved)
        for gs in _grads(params, 5):
            before = [p.clone() for p in moved]
            bound.step_(gs, 0.03)
            for p, b, g in zip(moved, before, gs, strict=True):
                want = 0.03 / 0.5 * torch.linalg.vector_norm(g)
                torch.testing.assert_close(torch.linalg.vector_norm(p - b), want)


def test_state_dict_round_trip():
    opt = Grafted(inner=AdamDirection(tau=0.5, tau_m=0.3, tau_v=3.0))
    params = _params(*SHAPES)
    grads = _grads(params, 12)
    a = [p.clone() for p in params]
    bound_a = opt.bind(a)
    for gs in grads[:5]:
        bound_a.step_(gs, 0.03)
    b = [p.clone() for p in a]
    bound_b = opt.bind(b)
    bound_b.load_state_dict(bound_a.state_dict())
    for gs in grads[5:]:
        bound_a.step_(gs, 0.03)
        bound_b.step_(gs, 0.03)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_a_chain_learns_with_it():
    tau = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
    torch.manual_seed(0)
    net = pc.Chain(
        nn.Sequential(nn.Linear(3, 4), nn.Tanh()),
        nn.Linear(4, 1),
        x=TensorSpec((3,)),
        y=TensorSpec((1,)),
        loss=pc.loss.MSELoss(),
        tau=tau,
        optimizer=Grafted(inner=AdamDirection(tau=1.0, tau_m=0.3, tau_v=3.0)),
        learn_interval=None,
    )
    before = net.layers[0].module.state_dict()
    before = {k: v.clone() for k, v in before.items()}
    yes = torch.tensor(True)
    sample = pc.msg.Sample(x=torch.ones(3), y=torch.ones(1), mask=yes, reset=yes)
    with mujou.testing.run(net, inputs={net.prev: sample}) as run:
        for _ in range(20):
            run.step(tau.period, net.input, *net.layers, net.output)
    after = net.layers[0].module.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


class TestWarmup:
    @staticmethod
    def _moved(elapsed: list[float], tau=0.5, seconds=2.0) -> torch.Tensor:
        p = torch.zeros(3, dtype=torch.float64)
        bound = Warmup(inner=pc.optim.SGD(tau=tau), seconds=seconds).bind([p])
        g = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        for e in elapsed:
            bound.step_([g], e)
        return -p / g * tau  # the time SGD weighed in all

    @pytest.mark.parametrize(
        ("elapsed", "ramped"),
        [
            ([1.0], 1.0**2 / (2 * 2.0)),  # within the ramp: t^2 / 2T
            ([3.0], 3.0 - 2.0 / 2),  # past it: t - T / 2
            ([0.5] * 10, 5.0 - 2.0 / 2),
        ],
    )
    def test_the_time_weighed_is_the_integral_of_the_ramp(self, elapsed, ramped):
        torch.testing.assert_close(
            self._moved(elapsed), torch.full((3,), ramped, dtype=torch.float64)
        )

    def test_a_step_across_the_end_of_the_ramp(self):
        moved = self._moved([1.5, 1.0])  # 1.5 -> 2.5 with T = 2: 0.5625 + 0.9375
        torch.testing.assert_close(moved, torch.full((3,), 1.5, dtype=torch.float64))

    def test_a_step_of_no_time_moves_nothing(self):
        torch.testing.assert_close(self._moved([0.0, 1.0]), self._moved([1.0]))

    @pytest.mark.parametrize("seconds", [0.0, -1.0, math.inf])
    def test_seconds_must_be_finite_and_positive(self, seconds):
        with pytest.raises(ValueError, match="seconds"):
            Warmup(inner=pc.optim.SGD(tau=1.0), seconds=seconds)

    def test_it_does_not_depend_on_the_steps(self):
        torch.testing.assert_close(self._moved([0.03] * 100), self._moved([3.0]))

    def test_past_the_ramp_a_step_is_sgd(self):
        params = _params(*SHAPES)
        grads = _grads(params, 5)
        bound = Warmup(inner=pc.optim.SGD(tau=0.5), seconds=1.0).bind(params)
        bound.step_([torch.zeros_like(p) for p in params], 1.0)  # through the ramp
        expected = _trajectory(pc.optim.SGD(tau=0.5), params, grads)
        for gs in grads:
            bound.step_(gs, 0.03)
        for x, y in zip(params, expected, strict=True):
            torch.testing.assert_close(x, y)

    def test_state_dict_round_trip(self):
        opt = Warmup(
            inner=Grafted(inner=AdamDirection(tau=0.5, tau_m=0.3, tau_v=3.0)),
            seconds=0.2,
        )
        params = _params(*SHAPES)
        grads = _grads(params, 12)
        a = [p.clone() for p in params]
        bound_a = opt.bind(a)
        for gs in grads[:5]:
            bound_a.step_(gs, 0.03)
        b = [p.clone() for p in a]
        bound_b = opt.bind(b)
        bound_b.load_state_dict(bound_a.state_dict())
        for gs in grads[5:]:
            bound_a.step_(gs, 0.03)
            bound_b.step_(gs, 0.03)
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def _adam_by_hand(
    theta: torch.Tensor, grads: list[torch.Tensor], lr, b1, b2, eps
) -> torch.Tensor:
    """Adam of Kingma & Ba (2015), Algorithm 1, step by step."""
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        theta = theta - lr * m_hat / (v_hat.sqrt() + eps)
    return theta


class TestAdam:
    def test_one_step_is_lr_times_the_sign(self):
        """After one step, m_hat = g and v_hat = g^2: theta moves by
        -lr g / (|g| + eps), the closed form of Adam's first step."""
        (theta,) = _params((7,))
        g = torch.tensor([3.0, -0.5, 1e-3, -2e-6, 0.0, 40.0, -1.0], dtype=theta.dtype)
        moved = theta.clone()
        adam(0.01, 0.9, 0.999).bind([moved]).step_([g], 1.0)
        want = theta - 0.01 * g / (g.abs() + 1e-8)
        torch.testing.assert_close(moved, want, rtol=0, atol=1e-6)

    @pytest.mark.parametrize(("lr", "b1", "b2"), [(1e-3, 0.9, 0.999), (0.1, 0.5, 0.9)])
    def test_its_steps_are_those_of_adam(self, lr, b1, b2):
        params = _params(*SHAPES)
        grads = _grads(params, 20)
        got = [p.clone() for p in params]
        bound = adam(lr, b1, b2).bind(got)
        for gs in grads:
            bound.step_(gs, 1.0)
        for i, p in enumerate(params):
            want = _adam_by_hand(p, [gs[i] for gs in grads], lr, b1, b2, 1e-8)
            torch.testing.assert_close(got[i], want, rtol=0, atol=1e-6)

    @pytest.mark.parametrize(
        ("lr", "b1", "b2", "name"),
        [
            (0.0, 0.9, 0.999, "lr"),
            (1e-3, 0.0, 0.999, "beta1"),
            (1e-3, 0.9, 1.0, "beta2"),
        ],
    )
    def test_values_out_of_range_raise(self, lr, b1, b2, name):
        with pytest.raises(ValueError, match=name):
            adam(lr, b1, b2)
