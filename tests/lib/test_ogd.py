"""``exp.lib.ogd.OGD``: each step is orthogonal to the kept gradients in the
metric of the base, the base itself while nothing is kept, and the kept
gradients fall on the steps nearest to the multiples of ``every``."""

import math

import pytest
import torch
from mujou.pc.optim import SGD, AdamW, RMSProp

from exp.lib.ogd import OGD

DT = 0.1
SHAPES = [(2, 3), (4,)]  # D = 10 over two tensors
BASES = [
    SGD(tau=2.0),
    RMSProp(tau=2.0, tau_v=0.5, eps=1e-8),
    AdamW(tau=2.0, tau_m=0.3, tau_v=0.5, eps=1e-8, weight_decay=0.1),
]


def _params() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return [torch.randn(s, generator=g, dtype=torch.float64) for s in SHAPES]


def _grads(steps: int, seed: int = 1) -> list[list[torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [
        [torch.randn(s, generator=g, dtype=torch.float64) for s in SHAPES]
        for _ in range(steps)
    ]


def _flat(ts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts])


def _metric(bound, base) -> torch.Tensor:
    r"""The diagonal of $P$ after a step: 1 for SGD, $1 / (\sqrt{\hat v} +
    \varepsilon)$ of the state for RMSProp and AdamW."""
    if isinstance(base, SGD):
        return torch.ones(sum(math.prod(s) for s in SHAPES), dtype=torch.float64)
    state = bound.state_dict()
    v = _flat(state["v"]) / state["c_v"]
    return 1 / (v.sqrt() + base.eps)


@pytest.mark.parametrize("base", BASES, ids=["sgd", "rmsprop", "adamw"])
class TestProjection:
    def test_while_nothing_is_kept_it_is_the_base_bit_for_bit(self, base):
        """The first step (nothing kept yet) moves the parameters and the
        moments exactly as the base does."""
        mine, theirs = _params(), _params()
        ogd, plain = OGD(base=base, n=4, every=DT).bind(mine), base.bind(theirs)
        (g,) = _grads(1)
        ogd.step_(g, DT)
        plain.step_(g, DT)
        for a, b in zip(mine, theirs, strict=True):
            assert torch.equal(a, b)
        for key, value in plain.state_dict().items():
            got = ogd.state_dict()[key]
            if isinstance(value, list):
                assert all(torch.equal(x, y) for x, y in zip(got, value, strict=True))
            else:
                assert got == value

    def test_the_moments_are_of_the_unprojected_gradients(self, base):
        """After steps that project, the moments (v of RMSProp, m and v of
        AdamW) and their masses are the base's from the same gradients, bit
        for bit: only the step is projected."""
        ogd, plain = OGD(base=base, n=3, every=DT).bind(_params()), base.bind(_params())
        for g in _grads(6):
            ogd.step_(g, DT)
            plain.step_(g, DT)
        for key, value in plain.state_dict().items():
            got = ogd.state_dict()[key]
            if isinstance(value, list):
                assert all(torch.equal(x, y) for x, y in zip(got, value, strict=True))
            else:
                assert got == value

    def test_each_step_is_orthogonal_to_the_kept_gradients(self, base):
        r"""$G^\top \Delta = 0$ for the gradients kept before the step, with
        AdamW's weight decay $-\eta w \theta$ taken out (it is outside the
        projection); to 1e-12 of $\lVert G\rVert \lVert\Delta\rVert$."""
        params = _params()
        bound = OGD(base=base, n=3, every=DT).bind(params)
        eta = DT / base.tau
        wd = base.weight_decay if isinstance(base, AdamW) else 0.0
        for g in _grads(7):
            kept = bound.kept_gradients()
            before = _flat(params).clone()
            bound.step_(g, DT)
            step = _flat(params) - before + eta * wd * before
            if len(kept):
                scale = float(kept.norm() * step.norm())
                assert float((kept @ step).abs().max()) <= 1e-12 * scale
                assert float(step.norm()) > 0


class TestIdentityMetric:
    def test_sgd_steps_along_the_gradient_off_the_span_of_the_kept(self):
        r"""$P = I$: $\Delta = -\eta (I - QQ^\top) g$, $Q$ an orthonormal basis
        of the last ``n`` = 2 kept gradients (QR in float64): step 2 against
        $g_1$, step 3 against $g_1, g_2$, step 4 against $g_2, g_3$."""
        params = _params()
        bound = OGD(base=SGD(tau=2.0), n=2, every=DT).bind(params)
        gs = _grads(4)
        eta = DT / 2.0
        for i, g in enumerate(gs):
            before = _flat(params).clone()
            bound.step_(g, DT)
            got = _flat(params) - before
            flat = _flat(g)
            if i == 0:
                want = -eta * flat
            else:
                q, _ = torch.linalg.qr(
                    torch.stack([_flat(x) for x in gs[max(0, i - 2) : i]]).T
                )
                want = -eta * (flat - q @ (q.T @ flat))
            torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-14)

    def test_parallel_kept_gradients_add_one_direction(self):
        """Kept g and 2 g (a Gram matrix of rank 1): the step is the gradient
        off g alone, finite."""
        params = _params()
        bound = OGD(base=SGD(tau=1.0), n=4, every=DT).bind(params)
        (g, h) = _grads(2)
        bound.step_(g, DT)
        bound.step_([2 * x for x in g], DT)
        before = _flat(params).clone()
        bound.step_(h, DT)
        u = _flat(g) / _flat(g).norm()
        want = -DT * (_flat(h) - u * (u @ _flat(h)))
        torch.testing.assert_close(_flat(params) - before, want, rtol=1e-9, atol=1e-12)


class TestKeeping:
    @staticmethod
    def _times(cut: list[float], total: float, every: float = 1.0) -> list[float]:
        bound = OGD(base=SGD(tau=1.0), n=16, every=every).bind(_params())
        g = _grads(1)[0]
        t, i = 0.0, 0
        while t < total - 1e-9:
            dt = cut[i % len(cut)]
            bound.step_(g, dt)
            t, i = t + dt, i + 1
        return bound.kept_times()

    @pytest.mark.parametrize("dt", [0.1, 0.05, 0.25])
    def test_kept_at_the_first_step_then_nearest_each_multiple(self, dt):
        """Kept at the first step, then at the step that ends nearest to 1, 2
        and 3 s, however the time is cut."""
        times = self._times([dt], 3.2)
        assert times[0] == pytest.approx(dt)
        assert times[1:] == pytest.approx([1.0, 2.0, 3.0], abs=1e-9)

    def test_an_uneven_cut_keeps_within_half_a_step_of_each_multiple(self):
        """Steps of 0.3, 0.2, 0.07 s: each kept time after the first is within
        half its step of a different multiple of 1 s."""
        times = self._times([0.3, 0.2, 0.07], 4.0)
        assert [round(t) for t in times[1:]] == [1, 2, 3, 4][: len(times) - 1]
        assert len(times) == 5
        assert all(abs(t - round(t)) <= 0.15 + 1e-9 for t in times[1:])

    def test_only_the_last_n_are_kept(self):
        bound = OGD(base=SGD(tau=1.0), n=3, every=DT).bind(_params())
        for g in _grads(5):
            bound.step_(g, DT)
        assert bound.kept_times() == pytest.approx([0.3, 0.4, 0.5])
        assert len(bound.kept_gradients()) == 3

    def test_a_step_of_no_time_moves_and_keeps_nothing(self):
        params = _params()
        bound = OGD(base=SGD(tau=1.0), n=3, every=DT).bind(params)
        before = [p.clone() for p in params]
        bound.step_(_grads(1)[0], 0.0)
        assert all(torch.equal(a, b) for a, b in zip(params, before, strict=True))
        assert bound.kept_times() == []


class TestState:
    @pytest.mark.parametrize("base", BASES, ids=["sgd", "rmsprop", "adamw"])
    def test_a_resumed_copy_steps_as_the_original(self, base):
        """Three steps, the state loaded into a fresh binding of a copy of the
        parameters, then three more on each: the same parameters bit for
        bit."""
        params = _params()
        opt = OGD(base=base, n=2, every=0.2)
        bound = opt.bind(params)
        gs = _grads(6)
        for g in gs[:3]:
            bound.step_(g, DT)
        copy = [p.clone() for p in params]
        other = opt.bind(copy)
        other.load_state_dict(bound.state_dict())
        for g in gs[3:]:
            bound.step_(g, DT)
            other.step_(g, DT)
        assert all(torch.equal(a, b) for a, b in zip(params, copy, strict=True))


class TestSettings:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"base": SGD(tau=1.0), "n": 0, "every": 1.0}, "n"),
            ({"base": SGD(tau=1.0), "n": 2, "every": 0.0}, "every"),
            ({"base": SGD(tau=1.0, tau_m=1.0), "n": 2, "every": 1.0}, "momentum"),
        ],
    )
    def test_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            OGD(**kwargs)

    def test_another_base_is_refused(self):
        from exp.lib.optim_zoo import Rprop

        with pytest.raises(TypeError, match="base"):
            OGD(base=Rprop(tau=1.0), n=2, every=1.0)  # pyright: ignore[reportArgumentType]
