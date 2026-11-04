"""The functions replayed from CUDA graphs are the eager ones bit for bit."""

import pytest
import torch
from torch import nn

from exp.lib.cuda_graph import Replayed, graphed_vjp

pytestmark = pytest.mark.cuda


def _module() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3)).cuda()


class TestReplayed:
    def test_each_key_replays_its_own_computation_on_new_inputs(self):
        def fn(key: str, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
            return [x * y + 1] if key == "mul" else [x - y, x.sum()]

        replayed = Replayed(fn)
        for seed in range(3):
            g = torch.Generator(device="cuda").manual_seed(seed)
            x = torch.randn(4, 3, device="cuda", generator=g)
            y = torch.randn(4, 3, device="cuda", generator=g)
            for key in ("mul", "sub"):
                for a, b in zip(replayed(key, x, y), fn(key, x, y), strict=True):
                    assert torch.equal(a, b)

    def test_it_returns_copies_that_the_next_replay_leaves_alone(self):
        replayed = Replayed(lambda _, x: [x * 2])
        first = replayed(0, torch.ones(3, device="cuda"))[0]
        replayed(0, torch.full((3,), 5.0, device="cuda"))
        assert torch.equal(first, torch.full((3,), 2.0, device="cuda"))


class TestGraphedVjp:
    @pytest.mark.parametrize("credit", [False, True])
    @pytest.mark.parametrize("learn", [False, True])
    def test_it_is_the_eager_vjp(self, credit: bool, learn: bool):
        # g_in = -J^T delta and g_theta = -J_theta^T delta, as pc.functional.vjp.
        module = _module()
        params = list(module.parameters())
        vjp = graphed_vjp(module, params)
        for seed in range(3):
            g = torch.Generator(device="cuda").manual_seed(seed)
            x = torch.randn(4, 5, device="cuda", generator=g)
            delta = torch.randn(4, 3, device="cuda", generator=g)
            out, g_in, g_theta = vjp(x, delta, credit=credit, learn=learn)
            leaf = x.clone().requires_grad_()
            ref = module(leaf)
            wrt = [leaf] * credit + params * learn
            grads: list[torch.Tensor] = []
            if wrt:
                grads = list(torch.autograd.grad(ref, wrt, -delta))
            assert torch.equal(out, ref.detach())
            assert (g_in is None) is not credit
            if credit:
                assert g_in is not None and torch.equal(g_in, grads[0])
            assert len(g_theta) == len(params) * learn
            for a, b in zip(g_theta, grads[credit:], strict=True):
                assert torch.equal(a, b)
            with torch.no_grad():  # the parameters move in place, as pc.optim does
                for p in params:
                    p.mul_(0.9)
