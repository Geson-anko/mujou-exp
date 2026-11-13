import math

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from mujou.pc.optim import Optimizer
from torch import nn

from exp.lib.optim import (
    AdamDirection,
    Grafted,
    RMSPropDirection,
    StepAdamW,
    Warmup,
    adam,
)
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


class TestRMSPropDirection:
    FRAME, LR, TAU_V, EPS = 1 / 30, 1e-3, 30.0, 1e-8

    def _opt(self) -> RMSPropDirection:
        return RMSPropDirection(tau=self.FRAME / self.LR, tau_v=self.TAU_V)

    def test_steps_of_one_frame_are_those_of_adam_without_momentum(self):
        """Adam of beta1 = 0 has no first moment and no bias of it; its beta2
        per step of one frame is exp(-frame / tau_v)."""
        params = _params(*SHAPES)
        grads = _grads(params, 300)
        got = _trajectory(self._opt(), params, grads, elapsed=self.FRAME)
        want = [p.clone() for p in params]
        beta2 = math.exp(-self.FRAME / self.TAU_V)
        torch_opt = torch.optim.Adam(want, lr=self.LR, betas=(0.0, beta2), eps=self.EPS)
        for gs in grads:
            for p, g in zip(want, gs, strict=True):
                p.grad = g.clone()
            torch_opt.step()
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_state_round_trips_and_resuming_is_not_interrupting(self):
        params = _params(*SHAPES)
        grads = _grads(params, 12)
        whole = _trajectory(self._opt(), params, grads, elapsed=self.FRAME)
        first = [p.clone() for p in params]
        bound = self._opt().bind(first)
        for gs in grads[:5]:
            bound.step_(gs, self.FRAME)
        state = bound.state_dict()
        second = [p.clone() for p in first]
        resumed = self._opt().bind(second)
        resumed.load_state_dict(state)
        again = resumed.state_dict()
        assert again["c_v"] == state["c_v"]
        for x, y in zip(again["v"], state["v"], strict=True):
            assert x is not y and torch.equal(x, y)
        for gs in grads[5:]:
            resumed.step_(gs, self.FRAME)
        for x, y in zip(second, whole, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


BAD = [0.0, -1.0, math.inf, math.nan]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("name", ["tau", "tau_m", "tau_v", "eps"])
def test_adam_direction_refuses_what_adamw_refuses(name, bad):
    kw = {"tau": 2.0, "tau_m": 0.3, "tau_v": 3.0, "eps": 1e-8, name: bad}
    with pytest.raises(ValueError, match=f"AdamDirection.{name} must be finite"):
        AdamDirection(**kw)
    with pytest.raises(ValueError, match=f"AdamW.{name} must be finite"):
        pc.optim.AdamW(**kw)


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("name", ["tau", "tau_v", "eps"])
def test_rmsprop_direction_refuses_a_constant_not_finite_and_positive(name, bad):
    kw = {"tau": 2.0, "tau_v": 3.0, "eps": 1e-8, name: bad}
    with pytest.raises(ValueError, match=f"RMSPropDirection.{name} must be finite"):
        RMSPropDirection(**kw)


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


class TestStepAdamW:
    """The steps of ``torch.optim.AdamW`` (the oracle), with the learning rate
    and the weight decay of each step set by hand, the decay only of the
    parameters of two or more dimensions, and ``clip_grad_norm_``."""

    @staticmethod
    def _torch(opt: StepAdamW, params, grads) -> list[torch.Tensor]:
        ps = [p.clone().requires_grad_() for p in params]
        groups = [
            {"params": [p for p in ps if p.dim() >= 2]},
            {"params": [p for p in ps if p.dim() < 2], "weight_decay": 0.0},
        ]
        ref = torch.optim.AdamW(groups, lr=opt.lr, betas=opt.betas, eps=opt.eps)
        for t, gs in enumerate(grads, start=1):
            for p, g in zip(ps, gs, strict=True):
                p.grad = g.clone()
            if opt.clip > 0:
                torch.nn.utils.clip_grad_norm_(ps, opt.clip)
            for group in ref.param_groups:
                group["lr"] = opt.lr_at(t)
            ref.param_groups[0]["weight_decay"] = opt.wd_at(t)
            ref.step()
        return [p.detach() for p in ps]

    @pytest.mark.parametrize(
        "opt",
        [
            StepAdamW(lr=1e-2),
            StepAdamW(lr=1e-2, betas=(0.8, 0.99), weight_decay=0.05),
            StepAdamW(
                lr=1e-2,
                warmup_steps=3,
                decay="cosine",
                decay_steps=8,
                min_lr=1e-3,
                weight_decay=0.04,
                weight_decay_end=0.4,
                clip=0.5,
            ),
        ],
        ids=["plain", "betas-wd", "scheduled-clipped"],
    )
    def test_the_steps_of_torch_adamw(self, opt):
        params = _params(*SHAPES)
        grads = _grads(params, 10)
        for a, b in zip(
            _trajectory(opt, params, grads),
            self._torch(opt, params, grads),
            strict=True,
        ):
            torch.testing.assert_close(a, b)

    def test_the_schedules_by_hand(self):
        opt = StepAdamW(
            lr=1.0,
            warmup_steps=4,
            decay="cosine",
            decay_steps=8,
            min_lr=0.1,
            weight_decay=0.0,
            weight_decay_end=0.8,
        )
        assert [opt.lr_at(t) for t in (1, 2, 4)] == [0.25, 0.5, 1.0]
        # half way down the cosine: the mean of 1 and 0.1; then 0.1 for good
        assert opt.lr_at(6) == pytest.approx(0.55)
        assert opt.lr_at(8) == pytest.approx(0.1) and opt.lr_at(20) == opt.lr_at(8)
        assert [opt.wd_at(t) for t in (0, 4, 8, 20)] == [0.0, 0.4, 0.8, 0.8]
        constant = StepAdamW(lr=1.0, warmup_steps=2)
        assert [constant.lr_at(t) for t in (1, 2, 3, 1000)] == [0.5, 1.0, 1.0, 1.0]

    def test_a_step_of_no_time_moves_nothing_and_the_state_resumes(self):
        opt = StepAdamW(lr=1e-2, warmup_steps=4)
        params = _params(*SHAPES)
        grads = _grads(params, 6)
        whole = _trajectory(opt, params, grads)
        moved = [p.clone() for p in params]
        bound = opt.bind(moved)
        for gs in grads[:3]:
            bound.step_(gs, 0.03)
            bound.step_(gs, 0.0)
        again = opt.bind(moved)
        again.load_state_dict(bound.state_dict())
        for gs in grads[3:]:
            again.step_(gs, 0.03)
        for a, b in zip(moved, whole, strict=True):
            torch.testing.assert_close(a, b)

    @pytest.mark.parametrize(
        "kw",
        [
            {"lr": 0.0},
            {"lr": 1e-3, "betas": (0.9, 1.0)},
            {"lr": 1e-3, "decay": "linear"},
            {"lr": 1e-3, "decay": "cosine", "decay_steps": 5, "warmup_steps": 5},
            {"lr": 1e-3, "weight_decay_end": 0.4},
        ],
    )
    def test_refused(self, kw):
        with pytest.raises(ValueError):
            StepAdamW(**kw)
