import math
from typing import Any

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from mujou.pc.optim import Optimizer
from torch import nn

from experiments.continuous_stream.optim import (
    AdamDirection,
    Grafted,
    Orthogonalized,
    Projected,
    Whitened,
    _eigh,
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


class TestWhitened:
    def test_a_b_zero_is_sgd(self):
        params = _params(*SHAPES)
        grads = _grads(params, 10)
        opt = Whitened(tau=0.5, tau_l=1.0, tau_r=1.0, a=0.0, b=0.0)
        want = _trajectory(pc.optim.SGD(tau=0.5), params, grads)
        for got, w in zip(_trajectory(opt, params, grads), want, strict=True):
            torch.testing.assert_close(got, w)

    def test_no_side_is_sgd(self):
        params = _params(*SHAPES)
        grads = _grads(params, 5)
        want = _trajectory(pc.optim.SGD(tau=0.5), params, grads)
        got = _trajectory(Whitened(tau=0.5), params, grads)
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b)

    @pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
    def test_tau_to_zero_is_the_polar_factor(self, shape):
        (w,) = _params(shape)
        (g,) = _grads([w], 1)[0]
        opt = Whitened(tau=1.0, tau_l=1e-9, tau_r=1e-9, eps=1e-14)
        moved = w.clone()
        opt.bind([moved]).step_([g], 0.5)
        u, _, vh = torch.linalg.svd(g, full_matrices=False)
        torch.testing.assert_close((w - moved) / 0.5, u @ vh, atol=1e-6, rtol=1e-6)

    def test_ema_does_not_depend_on_the_steps(self):
        params = _params((5, 3))
        g = _grads(params, 1)[0]
        opt = Whitened(tau=1.0, tau_l=0.7, tau_r=0.3)
        one, two = opt.bind([params[0].clone()]), opt.bind([params[0].clone()])
        one.step_(g, 0.2)
        two.step_(g, 0.1)
        two.step_(g, 0.1)
        for key in ("l", "r"):
            for a, b in zip(one.state_dict()[key], two.state_dict()[key], strict=True):
                torch.testing.assert_close(a, b)

    def test_ema_with_g_held_reaches_g_g_t(self):
        (w,) = _params((5, 3))
        (g,) = _grads([w], 1)[0]
        bound = Whitened(tau=1.0, tau_l=0.5).bind([w.clone()])
        bound.step_([g], 0.5)
        want = -math.expm1(-1.0) * (g @ g.T)
        torch.testing.assert_close(bound.state_dict()["l"][0], want)


class TestProjected:
    def test_the_update_is_orthogonal_to_the_top_directions(self):
        (w,) = _params((6, 5))
        grads = _grads([w], 8)
        opt = Projected(tau=1.0, tau_l=1.0, tau_r=1.0, rank=2, refresh=0.0)
        moved = w.clone()
        bound = opt.bind([moved])
        for gs in grads[:-1]:
            bound.step_(gs, 0.1)
        state = bound.state_dict()
        l, r = state["l"][0].clone(), state["r"][0].clone()
        before = moved.clone()
        bound.step_(grads[-1], 0.1)
        dw = moved - before
        u_l = torch.linalg.eigh(l).eigenvectors[:, -2:]
        u_r = torch.linalg.eigh(r).eigenvectors[:, -2:]
        assert dw.abs().max() > 1e-3  # something is left to move
        torch.testing.assert_close(u_l.T @ dw, torch.zeros(2, 5, dtype=w.dtype))
        torch.testing.assert_close(dw @ u_r, torch.zeros(6, 2, dtype=w.dtype))

    @pytest.mark.parametrize("rank1", [True, False])
    def test_tau_to_zero_the_same_g_again_does_not_move(self, rank1):
        (w,) = _params((4, 3))
        g = _grads([w], 1)[0][0]
        if rank1:  # the gradient of a Linear with one sample
            g = torch.outer(g[:, 0], g[0])
        opt = Projected(tau=1.0, tau_l=1e-9, tau_r=1e-9, rank=4, refresh=0.0)
        moved = w.clone()
        bound = opt.bind([moved])
        bound.step_([g], 0.1)
        assert not torch.equal(moved, w)  # nothing to project out at first
        before = moved.clone()
        bound.step_([g], 0.1)
        torch.testing.assert_close(moved, before)

    def test_without_l_and_r_it_is_sgd(self):
        params = _params(*SHAPES)
        grads = _grads(params, 4)
        want = _trajectory(pc.optim.SGD(tau=0.5), params, grads)
        got = _trajectory(Projected(tau=0.5, refresh=0.0), params, grads)
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b)


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


INNERS = [
    AdamDirection(tau=0.5, tau_m=0.3, tau_v=3.0),
    Whitened(tau=0.5, tau_l=1.0, tau_r=1.0),
    Projected(tau=0.5, tau_l=1.0, tau_r=1.0, rank=1, refresh=0.0),
]


class TestGrafted:
    @pytest.mark.parametrize("inner", INNERS, ids=lambda o: type(o).__name__)
    def test_the_step_has_the_size_of_sgd(self, inner):
        params = _params(*SHAPES)
        moved = [p.clone() for p in params]
        bound = Grafted(inner=inner).bind(moved)
        for gs in _grads(params, 5):
            before = [p.clone() for p in moved]
            bound.step_(gs, 0.03)
            for p, b, g in zip(moved, before, gs, strict=True):
                want = 0.03 / 0.5 * torch.linalg.vector_norm(g)
                torch.testing.assert_close(torch.linalg.vector_norm(p - b), want)


@pytest.mark.parametrize(
    "opt",
    [
        Whitened(tau=0.5, tau_l=1.0, tau_r=0.3),
        Projected(tau=0.5, tau_l=1.0, tau_r=0.3, rank=3, refresh=0.0),
        Grafted(inner=Whitened(tau=0.5, tau_l=1.0, tau_r=0.3)),
    ],
    ids=["whitened", "projected", "grafted"],
)
def test_a_convolution_steps_as_its_matrix(opt):
    (w,) = _params((4, 2, 3, 3))
    grads = _grads([w], 6)
    conv = _trajectory(opt, [w], grads)[0]
    flat = _trajectory(
        opt, [w.reshape(4, -1)], [[g.reshape(4, -1) for g in gs] for gs in grads]
    )[0]
    torch.testing.assert_close(conv.reshape(4, -1), flat)


@pytest.mark.parametrize(
    "opt",
    [
        Whitened(tau=0.5, tau_l=1.0, tau_r=0.3, refresh=0.1),
        Projected(tau=0.5, tau_l=1.0, tau_r=0.3, rank=2, refresh=0.1),
        Grafted(inner=AdamDirection(tau=0.5, tau_m=0.3, tau_v=3.0)),
    ],
    ids=["whitened", "projected", "grafted-adam"],
)
def test_state_dict_round_trip(opt):
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
        optimizer=Grafted(inner=Whitened(tau=1.0, tau_l=1.0, tau_r=1.0)),
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


class TestOrthogonalized:
    @pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
    def test_it_is_the_limit_of_whitened(self, shape):
        (w,) = _params(shape)
        (g,) = _grads([w], 1)[0]
        limit = Whitened(tau=1.0, tau_l=1e-9, tau_r=1e-9, eps=1e-14)
        want = _trajectory(limit, [w], [[g]], elapsed=0.5)[0]
        got = _trajectory(Orthogonalized(tau=1.0), [w], [[g]], elapsed=0.5)[0]
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)

    def test_a_rank_one_gradient_moves_along_u_v(self):
        u, v = torch.tensor([3.0, 4.0]), torch.tensor([1.0, 0.0, 0.0])
        g = torch.outer(u, v).double()
        got = _trajectory(
            Orthogonalized(tau=1.0), [torch.zeros(2, 3).double()], [[g]], 1.0
        )
        torch.testing.assert_close(got[0], -torch.outer(u / 5, v).double())


class TestEigh:
    def _failing_once(self, monkeypatch) -> list[torch.Tensor]:
        """Make the next ``torch.linalg.eigh`` fail as LAPACK does; return the
        matrices it is called with."""
        eigh, calls = torch.linalg.eigh, []

        def fake(a: torch.Tensor) -> Any:
            calls.append(a)
            if len(calls) == 1:
                raise RuntimeError("failed to converge")  # as LinAlgError
            return eigh(a)

        monkeypatch.setattr(torch.linalg, "eigh", fake)
        return calls

    def test_a_failure_on_a_finite_matrix_decomposes_it_perturbed(self, monkeypatch):
        g = torch.randn(64, 4, dtype=torch.float64)
        a = g @ g.T  # rank 4: 60 repeated zero eigenvalues
        want, _ = torch.linalg.eigh(a)
        calls = self._failing_once(monkeypatch)
        values, vectors = _eigh(a)
        assert len(calls) == 2
        torch.testing.assert_close(values, want, rtol=0, atol=1e-9)
        recon = vectors @ torch.diag(values) @ vectors.T
        torch.testing.assert_close(recon, a, rtol=0, atol=1e-9)

    def test_a_failure_on_a_matrix_with_nan_raises(self, monkeypatch):
        a = torch.eye(3, dtype=torch.float64)
        a[0, 1] = a[1, 0] = math.nan
        self._failing_once(monkeypatch)
        with pytest.raises(RuntimeError):
            _eigh(a)
