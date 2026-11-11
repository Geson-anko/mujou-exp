import pytest
import torch
from torch import nn

from exp.lib.mingru import MinGRU, ResidualMinGRU
from exp.video.dyn.modules import Top, build_dynamics
from exp.video.schema.dyn import DynamicsConf


def _params(mods) -> list[torch.Tensor]:
    return [*mods.step.parameters(), *mods.readout.parameters()]


class TestBuildDynamics:
    @pytest.mark.parametrize("nodes", [1, 3])
    def test_it_draws_the_modules_in_order_after_the_seed(self, nodes):
        """The minGRU, the layers and the readout, in this order, after
        ``torch.manual_seed(seed)``: the same weights as made by hand."""
        conf = DynamicsConf(d_s=6, nodes=nodes)
        mods = build_dynamics(conf, 4, 5, seed=11)
        torch.manual_seed(11)
        first = MinGRU(4, 6)
        rest = [ResidualMinGRU(6) for _ in range(nodes - 1)]
        readout = nn.Linear(6, 5)
        want = [*first.parameters()]
        for layer in rest:
            want += layer.parameters()
        want += readout.parameters()
        got = _params(mods)
        assert len(got) == len(want)
        for a, b in zip(got, want, strict=True):
            assert torch.equal(a, b)

    def test_another_seed_draws_other_weights(self):
        conf = DynamicsConf(d_s=6)
        a = build_dynamics(conf, 4, 4, seed=0)
        b = build_dynamics(conf, 4, 4, seed=1)
        assert not torch.equal(a.step.first.gate.weight, b.step.first.gate.weight)

    def test_the_residual_readout_starts_at_zero(self):
        """W_o = 0, so the residual prediction starts as the input before; the
        dynamics is drawn as without ``residual``."""
        res = build_dynamics(DynamicsConf(d_s=6, residual=True), 4, 4, seed=0)
        plain = build_dynamics(DynamicsConf(d_s=6), 4, 4, seed=0)
        assert not any(p.any() for p in res.readout.parameters())
        for a, b in zip(res.step.parameters(), plain.step.parameters(), strict=True):
            assert torch.equal(a, b)

    def test_it_leaves_the_global_rng_as_it_was(self):
        """What is drawn after it is what would be drawn without it (design
        0.3 item 6: the encoder's weights do not depend on the dynamics)."""
        torch.manual_seed(5)
        want = torch.rand(3)
        torch.manual_seed(5)
        build_dynamics(DynamicsConf(d_s=6, nodes=2), 4, 4, seed=0)
        assert torch.equal(torch.rand(3), want)

    def test_residual_needs_the_prediction_of_the_input(self):
        with pytest.raises(ValueError, match="must equal d_out"):
            build_dynamics(DynamicsConf(residual=True), 4, 6, seed=0)

    @pytest.mark.cuda
    def test_the_weights_do_not_depend_on_the_device(self):
        conf = DynamicsConf(d_s=6, nodes=2)
        cpu = build_dynamics(conf, 4, 4, seed=3)
        cuda = build_dynamics(conf, 4, 4, seed=3, device="cuda")
        for a, b in zip(_params(cpu), _params(cuda), strict=True):
            assert b.is_cuda
            assert torch.equal(a, b.cpu())


class TestDynamicsStack:
    def test_it_is_the_layers_one_after_the_other(self):
        """The state (s_1, h_2, h_3) moves as the minGRU and the residual layers
        on its parts, and ``Top`` of it is the top output y_3 = s_1 + h_2 +
        h_3."""
        mods = build_dynamics(DynamicsConf(d_s=5, nodes=3), 4, 4, seed=0)
        st = mods.step
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2, 4, generator=g)
        state = torch.randn(2, 15, generator=g)
        s1, h2, h3 = state.split(5, -1)
        with torch.no_grad():
            y1 = st.first(x, s1)
            y2 = st.rest[0](y1, h2)
            y3 = st.rest[1](y2, h3)
            new = st(x, state)
            top = Top(5)(new)
        assert st.d_state == 15
        torch.testing.assert_close(new, torch.cat([y1, y2 - y1, y3 - y2], -1))
        torch.testing.assert_close(top, y3)

    def test_one_node_is_the_mingru(self):
        mods = build_dynamics(DynamicsConf(d_s=5), 4, 4, seed=0)
        g = torch.Generator().manual_seed(2)
        x, s = torch.randn(3, 4, generator=g), torch.randn(3, 5, generator=g)
        with torch.no_grad():
            assert torch.equal(mods.step(x, s), mods.step.first(x, s))
            assert torch.equal(Top(5)(s), s)
