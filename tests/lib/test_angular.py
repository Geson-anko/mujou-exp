"""``exp.lib.angular.Angular``: each matrix moves by ``elapsed / tau`` of its
norm along its momentum, whatever the gradient's scale; the rest is
``pc.optim.RMSProp`` (the vectors of ``vec_scale * tau``, the scalars of
``tau_scalar``) bit for bit."""

import math

import pytest
import torch
from mujou.pc.optim import RMSProp

from exp.lib.angular import Angular

DT = 0.1
OPT = Angular(tau=50.0, tau_m=0.3, tau_v=0.5, eps=1e-8, vec_scale=10.0, tau_scalar=7.0)
SHAPES = [(3, 4), (2, 3, 2, 2), (5,), ()]  # a Linear, a Conv, a bias, ln sigma


def _params(seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(s, generator=g, dtype=torch.float64) for s in SHAPES]


def _grads(steps: int, seed: int = 1, scale: float = 1.0) -> list[list[torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [
        [scale * torch.randn(s, generator=g, dtype=torch.float64) for s in SHAPES]
        for _ in range(steps)
    ]


class TestMatrices:
    def test_the_first_step_by_hand(self):
        r"""The first momentum is $g$ (its mass corrected): $\Delta\theta =
        -(\delta/\tau)\, \lVert\theta\rVert\, g / \lVert g\rVert$."""
        params = _params()
        before = [p.clone() for p in params]
        (g,) = _grads(1)
        OPT.bind(params).step_(g, DT)
        for i in (0, 1):
            want = -(DT / 50.0) * before[i].norm() * g[i] / g[i].norm()
            torch.testing.assert_close(params[i] - before[i], want, rtol=1e-12, atol=0)

    def test_the_second_step_by_hand(self):
        r"""$m_2 = m_1 + a (g_2 - m_1)$ from $m_1 = a g_1$, $c_2 = a + a(1 -
        a)$, $a = 1 - e^{-\delta/\tau_m}$; the step is along $m_2 / c_2$."""
        params = _params()
        g1, g2 = _grads(2)
        bound = OPT.bind(params)
        bound.step_(g1, DT)
        mid = [p.clone() for p in params]
        bound.step_(g2, DT)
        a = -math.expm1(-DT / 0.3)
        for i in (0, 1):
            m = a * g1[i] + a * (g2[i] - a * g1[i])
            want = -(DT / 50.0) * mid[i].norm() * m / m.norm()
            torch.testing.assert_close(params[i] - mid[i], want, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
    def test_the_relative_step_is_elapsed_over_tau_whatever_the_scale(self, scale):
        params = _params()
        bound = OPT.bind(params)
        for g in _grads(5, scale=scale):
            before = [p.clone() for p in params]
            bound.step_(g, DT)
            for i in (0, 1):
                rel = (params[i] - before[i]).norm() / before[i].norm()
                assert float(rel) == pytest.approx(DT / 50.0, rel=1e-12)

    def test_a_zero_momentum_does_not_move_a_matrix(self):
        params = _params()
        before = [p.clone() for p in params]
        OPT.bind(params).step_([torch.zeros_like(p) for p in params], DT)
        for i in (0, 1):
            assert torch.equal(params[i], before[i])


def test_the_vectors_and_scalars_are_rmsprop_bit_for_bit():
    """The bias by ``RMSProp(vec_scale * tau)``, ln sigma by
    ``RMSProp(tau_scalar)``, both of ``tau_v`` and ``eps``."""
    mine, vec, sca = _params(), _params(), _params()
    bound = OPT.bind(mine)
    rv = RMSProp(tau=10.0 * 50.0, tau_v=0.5, eps=1e-8).bind([vec[2]])
    rs = RMSProp(tau=7.0, tau_v=0.5, eps=1e-8).bind([sca[3]])
    for g in _grads(4):
        bound.step_(g, DT)
        rv.step_([g[2]], DT)
        rs.step_([g[3]], DT)
    assert torch.equal(mine[2], vec[2]) and torch.equal(mine[3], sca[3])


def test_a_step_of_no_time_moves_nothing():
    params = _params()
    before = [p.clone() for p in params]
    OPT.bind(params).step_(_grads(1)[0], 0.0)
    assert all(torch.equal(p, b) for p, b in zip(params, before, strict=True))


def test_a_resumed_copy_steps_as_the_original():
    a = _params()
    gs = _grads(6)
    one = OPT.bind(a)
    for g in gs[:3]:
        one.step_(g, DT)
    b = [p.clone() for p in a]  # the same parameters, then the same state
    two = OPT.bind(b)
    two.load_state_dict(one.state_dict())
    for g in gs[3:]:
        one.step_(g, DT)
        two.step_(g, DT)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize(
    "kwargs",
    [{"tau": 0.0}, {"tau_m": -1.0}, {"vec_scale": 0.0}, {"tau_scalar": math.inf}],
)
def test_bad_settings_are_refused(kwargs):
    base = {"tau": 1.0, "tau_m": 1.0, "tau_v": 1.0, "eps": 1e-8}
    with pytest.raises(ValueError):
        Angular(**(base | kwargs))
