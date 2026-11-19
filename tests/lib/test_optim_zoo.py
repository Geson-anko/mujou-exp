"""The optimizers of ``exp.lib.optim_zoo`` against ``torch.optim`` stepped
with the same constant ``elapsed`` $\\delta$: the learning rate $\\delta/\\tau$
and the decays $\\beta = e^{-\\delta/\\tau_x}$ (float64). Where the time form
corrects the bias of a moment that starts at zero and torch does not, the
state is loaded with the mass 1 first, which is where the two agree; the
first step from a fresh state is checked by hand."""

import math

import pytest
import torch
from mujou.pc.optim import SGD, AdamW

from exp.lib.optim_zoo import (
    Adadelta,
    Muon,
    Nesterov,
    RMSPropCentered,
    Rprop,
    newton_schulz,
)

DT = 0.1  # the elapsed of every step
STEPS = 6


def _params(*shapes: tuple[int, ...], seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=g, dtype=torch.float64) for s in shapes]


def _grads(params: list[torch.Tensor], steps: int, seed: int = 1) -> list[list]:
    """``steps`` lists of gradients of ``params``' shapes, of mixed signs."""
    g = torch.Generator().manual_seed(seed)
    return [
        [torch.randn(p.shape, generator=g, dtype=torch.float64) for p in params]
        for _ in range(steps)
    ]


def _torch_run(opt_cls, params, grads, **kw) -> list[torch.Tensor]:
    """``params`` after the steps of ``torch.optim`` with ``grads``."""
    ps = [p.clone().requires_grad_() for p in params]
    opt = opt_cls(ps, **kw)
    for gs in grads:
        for p, g in zip(ps, gs, strict=True):
            p.grad = g.clone()
        opt.step()
    return [p.detach() for p in ps]


def _ours(optimizer, params, grads, *, full_mass: tuple[str, ...] = ()):
    """``params`` after the steps of ``optimizer`` of ``DT`` each, the masses
    ``full_mass`` of the fresh state set to 1 first."""
    ps = [p.clone() for p in params]
    bound = optimizer.bind(ps)
    if full_mass:
        state = bound.state_dict()
        for key in full_mass:
            state[key] = 1.0
        bound.load_state_dict(state)
    for gs in grads:
        bound.step_(gs, DT)
    return ps


def _assert_all_close(got, want, **kw) -> None:
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, **kw)


class TestNesterov:
    def test_is_torchs_nesterov_sgd_once_the_mass_is_full(self):
        tau, tau_m = 2.0, 0.5
        beta = math.exp(-DT / tau_m)
        params = _params((3, 4), (5,))
        grads = _grads(params, STEPS)
        want = _torch_run(
            torch.optim.SGD,
            params,
            grads,
            lr=DT / tau * (1 - beta),
            momentum=beta,
            nesterov=True,
        )
        got = _ours(Nesterov(tau=tau, tau_m=tau_m), params, grads, full_mass=("c_m",))
        _assert_all_close(got, want, rtol=1e-12, atol=1e-14)

    def test_a_constant_gradient_steps_as_sgd(self):
        """From a fresh state: $\\hat m = g$, so each step is SGD's."""
        params = _params((4,))
        g = [torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)]
        got = _ours(Nesterov(tau=2.0, tau_m=0.5), params, [g] * 3)
        want = _ours(SGD(tau=2.0), params, [g] * 3)
        _assert_all_close(got, want, rtol=1e-14, atol=1e-15)


class TestRMSPropCentered:
    def test_is_torchs_centered_rmsprop_once_the_mass_is_full(self):
        tau, tau_v, eps = 3.0, 0.7, 1e-8
        params = _params((3, 4), (5,))
        grads = _grads(params, STEPS)
        want = _torch_run(
            torch.optim.RMSprop,
            params,
            grads,
            lr=DT / tau,
            alpha=math.exp(-DT / tau_v),
            eps=eps,
            centered=True,
        )
        opt = RMSPropCentered(tau=tau, tau_v=tau_v, eps=eps)
        got = _ours(opt, params, grads, full_mass=("c",))
        _assert_all_close(got, want, rtol=1e-12, atol=1e-14)

    def test_the_first_step_is_of_the_size_of_rmsprops(self):
        """Fresh, $c = a = 1 - e^{-\\delta/\\tau_v}$, $v = a g^2$, $m = a g$:
        $(v - m^2) / c = (1 - a) g^2$, so the step is
        $-\\eta\\, g / (|g| \\sqrt{1 - a} + \\varepsilon)$."""
        tau, tau_v, eps = 3.0, 0.7, 1e-8
        a = -math.expm1(-DT / tau_v)
        p0 = torch.zeros(3, dtype=torch.float64)
        g = torch.tensor([2.0, -0.5, 1e-3], dtype=torch.float64)
        got = _ours(RMSPropCentered(tau=tau, tau_v=tau_v, eps=eps), [p0], [[g]])[0]
        want = -DT / tau * g / (g.abs() * math.sqrt(1 - a) + eps)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=0.0)


class TestAdadelta:
    def test_is_torchs_adadelta_once_the_mass_is_full(self):
        tau, tau_v, eps = 0.5, 0.7, 1e-6
        params = _params((3, 4), (5,))
        grads = _grads(params, STEPS)
        want = _torch_run(
            torch.optim.Adadelta,
            params,
            grads,
            lr=DT / tau,
            rho=math.exp(-DT / tau_v),
            eps=eps,
        )
        opt = Adadelta(tau=tau, tau_v=tau_v, eps=eps)
        got = _ours(opt, params, grads, full_mass=("c_v", "c_u"))
        _assert_all_close(got, want, rtol=1e-12, atol=1e-14)

    def test_the_first_step(self):
        """Fresh: $\\hat v = g^2$ and no $\\hat u$ yet (0), so
        $\\Delta = -\\eta \\sqrt{\\varepsilon} / \\sqrt{g^2 + \\varepsilon}\\, g$."""
        tau, eps = 0.5, 1e-6
        p0 = torch.zeros(2, dtype=torch.float64)
        g = torch.tensor([3.0, -1e-4], dtype=torch.float64)
        got = _ours(Adadelta(tau=tau, tau_v=0.7, eps=eps), [p0], [[g]])[0]
        want = -DT / tau * math.sqrt(eps) / (g * g + eps).sqrt() * g
        torch.testing.assert_close(got, want, rtol=1e-12, atol=0.0)


class TestRprop:
    def test_is_torchs_rprop(self):
        """$s$ in seconds$^{-1}$: torch's step sizes are $\\delta s$, from
        $\\delta s_{\\max} / 10$ within $\\delta s_{\\max} [10^{-3}, 1]$."""
        tau = 4.0
        s_max = 1 / tau
        params = _params((3, 4), (5,))
        g = torch.Generator().manual_seed(1)
        signs = [torch.tensor(1.0)] * 3 + [torch.tensor(-1.0)] * 2 + [torch.tensor(1.0)]
        grads = [  # partly the same sign in a row, so both factors act
            [
                s * torch.randn(p.shape, generator=g, dtype=torch.float64).abs()
                for p in params
            ]
            for s in signs
        ]
        grads[2][0][0, 0] = 0.0  # a zero gradient
        want = _torch_run(
            torch.optim.Rprop,
            params,
            grads,
            lr=DT * s_max / 10,
            etas=(0.5, 1.2),
            step_sizes=(DT * s_max * 1e-3, DT * s_max),
        )
        got = _ours(Rprop(tau=tau), params, grads)
        _assert_all_close(got, want, rtol=1e-12, atol=1e-15)

    def test_the_step_is_elapsed_times_the_rate(self):
        """The first step is $-\\text{elapsed}\\, s_{\\max}/10\\, \\mathrm{sign}(g)$
        whatever ``elapsed``."""
        p = torch.zeros(2, dtype=torch.float64)
        bound = Rprop(tau=4.0).bind([p])
        bound.step_([torch.tensor([5.0, -1e-9], dtype=torch.float64)], 0.3)
        torch.testing.assert_close(p, torch.tensor([-1.0, 1.0]).double() * 0.3 / 40)


class TestNewtonSchulz:
    @pytest.mark.parametrize("shape", [(4, 7), (7, 4), (5, 5)])
    def test_is_torchs(self, shape):
        from torch.optim._muon import (  # the oracle only
            DEFAULT_A,
            DEFAULT_B,
            DEFAULT_C,
            EPS,
            _zeropower_via_newtonschulz,
        )

        x = torch.randn(shape, generator=torch.Generator().manual_seed(2))
        want = _zeropower_via_newtonschulz(x, (DEFAULT_A, DEFAULT_B, DEFAULT_C), 5, EPS)
        assert torch.equal(newton_schulz(x), want)

    def test_moves_the_singular_values_towards_one(self):
        """Its quintic leaves them in about (0.5, 1.5) (torch's docstring)."""
        x = torch.randn(16, 32, generator=torch.Generator().manual_seed(3))
        x[0] *= 100  # one large singular value
        s = torch.linalg.svdvals(newton_schulz(x).float())
        assert float(s.min()) > 0.3 and float(s.max()) < 1.6


class TestMuon:
    def test_is_torchs_muon_on_matrices_and_adamw_on_the_rest(self):
        """A Linear weight and a Conv2d weight (flattened to ``(out, in k
        k)``) by ``torch.optim.Muon(adjust_lr_fn="match_rms_adamw",
        nesterov=True)``, a bias by ``torch.optim.AdamW``; the weight decay
        0.01 of both."""
        tau, tau_m, tau_v, eps, wd = 2.0, 0.5, 0.7, 1e-8, 0.01
        lin, conv, bias = _params((5, 3), (4, 2, 3, 3), (4,))
        grads = _grads([lin, conv, bias], STEPS)
        lr, b1, b2 = DT / tau, math.exp(-DT / tau_m), math.exp(-DT / tau_v)
        want_lin, want_conv = _torch_run(
            torch.optim.Muon,
            [lin, conv.reshape(4, 18)],
            [[g[0], g[1].reshape(4, 18)] for g in grads],
            lr=lr,
            weight_decay=wd,
            momentum=b1,
            nesterov=True,
            adjust_lr_fn="match_rms_adamw",
        )
        (want_bias,) = _torch_run(
            torch.optim.AdamW,
            [bias],
            [[g[2]] for g in grads],
            lr=lr,
            betas=(b1, b2),
            eps=eps,
            weight_decay=wd,
        )
        opt = Muon(tau=tau, tau_m=tau_m, tau_v=tau_v, eps=eps, weight_decay=wd)
        got = _ours(opt, [lin, conv, bias], grads, full_mass=("c_m",))
        want = [want_lin, want_conv.reshape(conv.shape), want_bias]
        _assert_all_close(got, want, rtol=1e-12, atol=1e-14)

    def test_the_rest_is_mujous_adamw(self):
        """Without a matrix, Muon is ``pc.optim.AdamW`` bit for bit."""
        params = _params((4,), (3,))
        grads = _grads(params, 3)
        kw = dict(tau=2.0, tau_m=0.5, tau_v=0.7, eps=1e-8, weight_decay=0.01)
        got = _ours(Muon(**kw), params, grads)
        want = _ours(AdamW(**kw), params, grads)
        _assert_all_close(got, want, rtol=0.0, atol=0.0)


ALL = [
    Nesterov(tau=2.0, tau_m=0.5),
    RMSPropCentered(tau=2.0, tau_v=0.7),
    Adadelta(tau=0.5, tau_v=0.7),
    Rprop(tau=4.0),
    Muon(tau=2.0, tau_m=0.5, tau_v=0.7),
]


@pytest.mark.parametrize("opt", ALL, ids=lambda o: type(o).__name__)
class TestEvery:
    def test_a_step_of_no_time_moves_nothing(self, opt):
        params = _params((3, 4), (4,))
        before = [p.clone() for p in params]
        bound = opt.bind(params)
        bound.step_(_grads(params, 1)[0], 0.0)
        _assert_all_close(params, before, rtol=0.0, atol=0.0)

    def test_its_state_resumes_the_same_steps(self, opt):
        """Stopped after 3 steps and loaded into a fresh binding of copies of
        the parameters, the next 3 steps are those of one never stopped."""
        params = _params((3, 4), (4,))
        grads = _grads(params, 6)
        whole = _ours(opt, params, grads)
        first = [p.clone() for p in params]
        bound = opt.bind(first)
        for gs in grads[:3]:
            bound.step_(gs, DT)
        state = bound.state_dict()
        second = [p.clone() for p in first]
        resumed = opt.bind(second)
        resumed.load_state_dict(state)
        for gs in grads[3:]:
            resumed.step_(gs, DT)
        _assert_all_close(second, whole, rtol=0.0, atol=0.0)

    def test_a_tau_that_is_not_positive_fails(self, opt):
        with pytest.raises(ValueError, match="tau"):
            type(opt)(**{**opt.__dict__, "tau": 0.0})
