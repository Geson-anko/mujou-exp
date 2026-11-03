import pytest
import torch

from exp.lib.mingru import MinGRU, MinGRUTrace, ResidualMinGRU


def _bptt(cell: MinGRU, xs: torch.Tensor, deltas: torch.Tensor) -> list[torch.Tensor]:
    """The gradient of sum_k deltas_k . s_k by BP through the state, with the
    inputs held: the oracle of the trace."""
    s = torch.zeros(*xs.shape[1:-1], cell.gate.out_features)
    total = torch.zeros(())
    for x, d in zip(xs, deltas, strict=True):
        s = cell(x, s)
        total = total + (d * s).sum()
    return list(torch.autograd.grad(total, list(cell.parameters())))


class TestMinGRUTrace:
    @pytest.mark.parametrize("batch", [(), (3,)], ids=["one", "batch"])
    def test_it_is_bp_through_the_state(self, batch):
        """With fixed weights, the sum over steps of grads(delta_k) after each
        step is the gradient of sum_k delta_k . s_k by BP through time, the
        inputs held (the recurrence is diagonal in s, so RTRL is exact)."""
        torch.manual_seed(0)
        cell = MinGRU(4, 5)
        xs = torch.randn(6, *batch, 4)
        deltas = torch.randn(6, *batch, 5)
        trace = MinGRUTrace(cell, batch)
        s = torch.zeros(*batch, 5)
        got = [torch.zeros_like(p) for p in cell.parameters()]
        for x, d in zip(xs, deltas, strict=True):
            with torch.no_grad():
                trace.step_(x, s)
                s = cell(x, s)
            got = [g + h for g, h in zip(got, trace.grads(d), strict=True)]
        want = _bptt(cell, xs, deltas)
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6)

    def test_a_reset_forgets_the_past(self):
        """A stream that resets starts from s = 0: its trace after the step is
        that of a sequence that starts there."""
        torch.manual_seed(1)
        cell = MinGRU(3, 4)
        xs = torch.randn(4, 2, 3)
        trace = MinGRUTrace(cell, (2,))
        s = torch.zeros(2, 4)
        with torch.no_grad():
            for x in xs[:3]:
                trace.step_(x, s)
                s = cell(x, s)
            reset = torch.tensor([True, False])
            trace.reset_(reset)
            s = torch.where(reset[:, None], 0.0, s)
            trace.step_(xs[3], s)
        d = torch.tensor([[1.0, -1.0, 0.5, 2.0], [0.0, 0.0, 0.0, 0.0]])
        want = _bptt(cell, xs[3:, :1], d[None, :1])  # stream 0 from xs[3] only
        for g, w in zip(trace.grads(d), want, strict=True):
            torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6)

    def test_the_residual_layer_traces_its_cell_on_gelu_of_its_input(self):
        """For y = x + minGRU(GELU(x), h), dy/dtheta = dh/dtheta: the trace of
        the cell on GELU(x) is BP through h with x held."""
        torch.manual_seed(2)
        layer = ResidualMinGRU(4)
        xs = torch.randn(5, 4)
        deltas = torch.randn(5, 4)
        trace = MinGRUTrace(layer.cell, ())
        h = torch.zeros(4)
        got = [torch.zeros_like(p) for p in layer.parameters()]
        total = torch.zeros(())
        h_bp = torch.zeros(4)
        for x, d in zip(xs, deltas, strict=True):
            with torch.no_grad():
                trace.step_(torch.nn.functional.gelu(x), h)
                h = layer(x, h) - x
            got = [g + t for g, t in zip(got, trace.grads(d), strict=True)]
            y = layer(x, h_bp)
            h_bp = y - x
            total = total + (d * y).sum()
        want = torch.autograd.grad(total, list(layer.parameters()))
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6)

    def test_a_trace_restored_from_its_state_goes_on_as_the_original(self):
        """Traced 3 steps, its ``state_dict`` copied (as a run copies a node's
        state) into a new trace of the same cell, then both traced 3 more
        steps: the gradients are the same bits; a trace started anew is not."""
        torch.manual_seed(3)
        cell = MinGRU(3, 4)
        xs = torch.randn(6, 2, 3)
        delta = torch.randn(2, 4)
        trace = MinGRUTrace(cell, (2,))
        s = torch.zeros(2, 4)
        with torch.no_grad():
            for x in xs[:3]:
                trace.step_(x, s)
                s = cell(x, s)
        saved = {k: v.clone() for k, v in trace.state_dict().items()}
        restored = MinGRUTrace(cell, (2,))
        restored.load_state_dict(saved)
        anew = MinGRUTrace(cell, (2,))
        with torch.no_grad():
            for x in xs[3:]:
                for t in (trace, restored, anew):
                    t.step_(x, s)
                s = cell(x, s)
        for a, b in zip(trace.grads(delta), restored.grads(delta), strict=True):
            assert torch.equal(a, b)
        assert not all(
            torch.equal(a, b)
            for a, b in zip(trace.grads(delta), anew.grads(delta), strict=True)
        )

    def test_loading_keeps_none_of_the_tensors_given(self):
        """``load_state_dict`` copies: changing the given tensors afterwards
        does not change the trace."""
        cell = MinGRU(2, 3)
        trace = MinGRUTrace(cell, ())
        given = {k: torch.ones_like(v) for k, v in trace.state_dict().items()}
        trace.load_state_dict(given)
        for v in given.values():
            v.zero_()
        assert all(bool((v == 1).all()) for v in trace.state_dict().values())
