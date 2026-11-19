"""``exp.lib.grad_stats.GradStats``: it leaves the learning as it is, and its
rows are those of a small case computed by hand."""

import math
import subprocess
import sys

import pytest
import torch
from mujou.pc.optim import SGD, AdamW

from exp.lib.grad_stats import GradStats, residual_share

DT = 0.1


def _f64(*xs: float) -> torch.Tensor:
    return torch.tensor(xs, dtype=torch.float64)


class TestResidualShare:
    def test_the_share_of_g_off_the_span(self):
        """span{(3, 4, 0), (0, 0, 1)}: (0, 3, 4) keeps (-1.44, 1.08, 0), of
        norm 1.8 of 5."""
        G = torch.stack([_f64(3, 4, 0), _f64(0, 0, 1)])
        assert residual_share(G, _f64(0, 3, 4)) == pytest.approx(0.36, rel=1e-12)

    def test_a_basis_of_repeated_directions(self):
        """Parallel rows add no direction: (1, 1, 1) keeps (0, 0, 1)."""
        G = torch.stack([_f64(1, 0, 0), _f64(2, 0, 0), _f64(0, 1, 0)])
        got = residual_share(G, _f64(1, 1, 1))
        assert got == pytest.approx(1 / math.sqrt(3), rel=1e-12)

    def test_g_in_the_span_keeps_nothing(self):
        G = torch.stack([_f64(1, 2, 0), _f64(0, 1, 1)])
        assert residual_share(G, _f64(2, 5, 1)) == pytest.approx(0.0, abs=1e-12)

    def test_nearly_parallel_rows_in_float32(self):
        r"""Rows of cosine 0.9999 in float32, $D = 10^6$, and $g = a +
        10^{-4} h$ nearly in their span: the share is the float64 one to
        1e-3 (float32 sums of $GG^\top$ and $Gg$ miss it by 18%)."""
        size, cos = 1_000_000, 0.9999
        gen = torch.Generator().manual_seed(0)
        a, b, h = (torch.randn(size, generator=gen) for _ in range(3))
        b = b - (b @ a) / (a @ a) * a
        b = b * (a.norm() / b.norm())
        G = torch.stack([a, cos * a + math.sqrt(1 - cos**2) * b])
        g = a + 1e-4 * h
        q, _ = torch.linalg.qr(G.double().T)
        r = g.double() - q @ (q.T @ g.double())
        want = float(r.norm() / g.double().norm())
        assert residual_share(G, g) == pytest.approx(want, rel=1e-3)

    @pytest.mark.cuda
    def test_many_threads_at_once_on_cuda_in_a_fresh_process(self):
        """The nodes' threads call it at once from the first step: in a fresh
        process (no linalg of CUDA loaded yet) 8 threads released together all
        get the share, none the error of torch's lazy load of its CUDA linalg
        ("lazy wrapper should be called at most once")."""
        code = """
import threading, torch
from exp.lib.grad_stats import residual_share
G = torch.eye(3, device="cuda")[:2]
g = torch.ones(3, device="cuda")
gate, out = threading.Barrier(8), []
def go():
    gate.wait()
    out.append(residual_share(G, g))
ts = [threading.Thread(target=go) for _ in range(8)]
for t in ts: t.start()
for t in ts: t.join()
assert len(out) == 8 and all(abs(x - 3 ** -0.5) < 1e-6 for x in out), out
"""
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]


def _hand() -> tuple[list[torch.Tensor], list[dict]]:
    """Four steps of 0.1 s of SGD (tau 1) of the parameters a = (1, 2) and
    b = (2,), with every = 0.2, tensors_every = 0.4, an anchor of 0.2 s and
    the last 2 gradients kept every 0.1 s, stored in float64."""
    a, b = _f64(1, 2), _f64(2)
    opt = GradStats(
        inner=SGD(tau=1.0),
        node="n",
        names={a: "a", b: "b"},
        store=torch.float64,
        every=0.2,
        tensors_every=0.4,
        anchors=(0.2,),
        spans=(0.1,),
        keep=2,
    )
    bound = opt.bind([a, b])
    gs = [_f64(1, 0, 0), _f64(3, 4, 0), _f64(0, 0, 2), _f64(0, 3, 4)]
    for g in gs:
        bound.step_([g[:2], g[2:]], DT)
    return [a, b], bound.drain()


class TestRows:
    def test_a_row_every_second_of_the_nodes_time(self):
        _, rows = _hand()
        assert [r["t"] for r in rows] == pytest.approx([0.2, 0.4])
        assert all(r["node"] == "n" for r in rows)

    def test_the_first_row(self):
        """At 0.2 s, g = (3, 4, 0): the previous gradient and the anchor are
        g1 = (1, 0, 0) (cos 0.6, 0.1 s ago), the kept one too (keeps (0, 4, 0)
        of 5). theta before the step is (0.9, 2, 2) and moves by 0.1 g."""
        r = _hand()[1][0]
        assert r["g"] == pytest.approx(5.0)
        assert r["cos_prev"] == pytest.approx(0.6)
        assert r["age_prev"] == pytest.approx(0.1)
        assert r["cos_0.2"] == pytest.approx(0.6)
        assert r["age_0.2"] == pytest.approx(0.1)
        assert r["rho_0.1"] == pytest.approx(0.8)
        assert r["theta"] == pytest.approx(math.sqrt(0.81 + 8))
        assert r["dtheta"] == pytest.approx(0.5)
        assert "rel" not in r

    def test_the_second_row(self):
        """At 0.4 s, g = (0, 3, 4): the previous g3 = (0, 0, 2) (cos 0.8), the
        anchor g2 = (3, 4, 0) of 0.2 s ago (cos 0.48), kept g2 and g3 (0.36);
        per tensor |dtheta| / |theta| from theta (0.6, 1.6 | 1.8)."""
        r = _hand()[1][1]
        assert r["cos_prev"] == pytest.approx(0.8)
        assert r["cos_0.2"] == pytest.approx(0.48)
        assert r["age_0.2"] == pytest.approx(0.2)
        assert r["rho_0.1"] == pytest.approx(0.36)
        assert r["rel"]["a"] == pytest.approx(0.3 / math.sqrt(2.92))
        assert r["rel"]["b"] == pytest.approx(0.4 / 1.8)

    def test_drain_empties_the_rows(self):
        a = _f64(1.0)
        bound = GradStats(inner=SGD(tau=1.0), node="n", every=0.1).bind([a])
        bound.step_([_f64(1.0)], DT)
        assert len(bound.drain()) == 1 and bound.drain() == []

    def test_before_any_anchor_or_kept_gradient_they_are_none(self):
        a = _f64(1.0, 0.0)
        bound = GradStats(inner=SGD(tau=1.0), node="n", every=0.1).bind([a])
        bound.step_([_f64(1.0, 1.0)], DT)
        (r,) = bound.drain()
        assert r["cos_prev"] is None and r["cos_1"] is None and r["rho_0.1"] is None


def _run(stats: bool, steps: int = 40) -> list[torch.Tensor]:
    params = [
        torch.randn(s, generator=torch.Generator().manual_seed(i))
        for i, s in enumerate([(6, 5), (5,)])
    ]
    inner = AdamW(tau=10.0, tau_m=0.3, tau_v=3.0)
    opt = (
        GradStats(inner=inner, node="n", every=0.5, tensors_every=1.0)
        if stats
        else inner
    )
    bound = opt.bind(params)
    g = torch.Generator().manual_seed(9)
    for _ in range(steps):
        bound.step_([torch.randn(p.shape, generator=g) for p in params], DT)
    return params


class TestLearning:
    def test_is_the_same_bit_for_bit(self):
        for a, b in zip(_run(True), _run(False), strict=True):
            assert torch.equal(a, b)

    def test_a_resumed_binding_writes_the_rows_of_one_never_stopped(self):
        a = torch.zeros(4)
        opt = GradStats(inner=SGD(tau=10.0), node="n", every=0.3, anchors=(0.5,))
        g = torch.Generator().manual_seed(5)
        grads = [[torch.randn(4, generator=g)] for _ in range(12)]
        whole = opt.bind([a.clone()])
        for gs in grads:
            whole.step_(gs, DT)
        first_p = a.clone()
        first = opt.bind([first_p])
        for gs in grads[:5]:
            first.step_(gs, DT)
        rows = first.drain()
        resumed = opt.bind([first_p.clone()])
        resumed.load_state_dict(first.state_dict())
        for gs in grads[5:]:
            resumed.step_(gs, DT)
        assert rows + resumed.drain() == whole.drain()

    def test_holds_the_kept_gradients_in_bf16(self):
        """The previous gradient, 3 anchors and 3 x 16 kept ones: 52 vectors
        of 2 bytes a parameter."""
        params = [torch.zeros(7, 3), torch.zeros(5)]
        bound = GradStats(inner=SGD(tau=1.0), node="n").bind(params)
        state = bound.state_dict()
        held = [v for k, v in state.items() if k != "inner" and torch.is_tensor(v)]
        vectors = [v for v in held if v.dtype == torch.bfloat16]
        assert sum(v.numel() * v.element_size() for v in vectors) == 52 * 26 * 2
