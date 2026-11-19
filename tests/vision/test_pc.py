"""The VAE of PC-ALM against BP's float64 gradient of the same mean -ELBO
(the same weights, stacks and noise), under the host's Jacobi steps."""

import inspect
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any, override

import mujou.pc as pc
import pytest
import torch
from torch import Tensor

import mujou
from exp.lib.elbo import dequantize, gauss_kl, gauss_nll
from exp.lib.grad_stats import GradStats
from exp.lib.host import Host
from exp.lib.optim_zoo import Adadelta, Muon, Nesterov, RMSPropCentered, Rprop
from exp.vision.models import Block, VAEModules, build_vae, early_fusion
from exp.vision.pc import (
    HOPS,
    LEARN_INTERVAL,
    VAEPC,
    GaussOut,
    Latent,
    make_optimizer,
    make_optimizers,
)
from exp.vision.schema import OptimConf, RunConf, TimeConf
from exp.vision.schema.run import check_learn_interval
from mujou import TensorSpec
from tests.vision.test_models import D, stacks, tiny

FRAME = 0.1
ARCHS = ["vit", "resnet"]
NODES = ("enc1", "enc2", "enc3", "enc4", "dec1", "dec2", "out")


def _conf(
    arch: str,
    learn_interval: float = LEARN_INTERVAL,
    b: int = 2,
    learn_sigma: bool = True,
) -> RunConf:
    """The tiny run with ``learn_interval`` set after RunConf's check: these
    tests probe intervals outside its window (30 ms, inf) on purpose."""
    conf = tiny(arch, streams=b)
    conf.time = TimeConf(learn_interval=learn_interval)
    conf.model.learn_sigma = learn_sigma
    return conf


def _mods(conf: RunConf) -> VAEModules:
    mods = build_vae(conf, seed=conf.seed)
    for m in mods.modules().values():
        m.double()
    return mods


def _noise(x: Tensor, seed: int) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(x.shape, generator=g, dtype=torch.float64)
    eps = torch.randn(len(x), D, generator=g, dtype=torch.float64)
    return u, eps


def _oracle(
    mods: VAEModules, x: Tensor, u: Tensor, eps: Tensor, valid: Tensor
) -> tuple[Tensor, list[Tensor]]:
    r"""BP's loss, the mean $-\mathcal L$ over ``valid``, in float64, and the
    activities $h_1 .. h_7$ of its graph (enc1-4, $z$, dec1-2), composed here
    from the modules."""
    h = early_fusion(x).double()
    acts = []
    for part in mods.encoder.parts:
        h = part(h)
        acts.append(h)
    d = mods.latent
    mu, logvar = h[:, :d], h[:, d:]
    z = mu + (0.5 * logvar).exp() * eps
    acts.append(z)
    for part in mods.decoder.parts:
        z = part(z)
        acts.append(z)
    y_hat = mods.decoder.readout(z)
    y = dequantize(x, u).reshape(y_hat.shape)
    loss = gauss_nll(y, y_hat, mods.log_sigma) + gauss_kl(mu, logvar).sum(1)
    w = valid.double()
    return (loss * w).sum() / w.sum(), acts


def _bp_grads(
    mods: VAEModules, x: Tensor, u: Tensor, eps: Tensor, valid: Tensor
) -> dict[Tensor, Tensor]:
    loss, _ = _oracle(mods, x, u, eps, valid)
    params = mods.parameters()
    return dict(zip(params, torch.autograd.grad(loss, params), strict=True))


def _key_biases(mods: VAEModules) -> dict[Tensor, slice]:
    """The elements of the keys in each block's ``qkv.bias``: their exact
    gradient is 0 (a bias of the keys adds the same $q^\top b$ to a query's
    scores of every key, which the softmax removes), so both gradients are
    rounding there."""
    return {
        m.qkv.bias: slice(m.proj.in_features, 2 * m.proj.in_features)
        for m in mods.encoder.modules()
        if isinstance(m, Block)
    }


def _assert_bps(mods: VAEModules, pcs: dict, bps: dict) -> None:
    """Each of ``pcs`` is BP's to 1e-9 of its scale; the keys' biases are 0 to
    1e-12 of the largest gradient."""
    keys = _key_biases(mods)
    top = max(float(g.abs().max()) for g in bps.values())
    for p, g in bps.items():
        got, want = pcs[p].clone(), g.clone()
        if p in keys:
            k = keys[p]
            assert float(got[k].abs().max()) <= 1e-12 * top
            assert float(want[k].abs().max()) <= 1e-12 * top
            got[k] = want[k] = 0.0
        torch.testing.assert_close(
            got, want, rtol=1e-9, atol=1e-12 * float(want.abs().max())
        )


class GradRecorder(pc.optim.Optimizer):
    """Moves nothing; records each parameter's first gradient since
    ``grads`` was cleared, and each binding's steps (``steps``: per binding,
    the ``elapsed`` of each, in order). A node writes only its own keys."""

    def __init__(self) -> None:
        self.grads: dict[Tensor, Tensor] = {}
        self.steps: list[list[float]] = []

    @override
    def bind(self, params: Iterable[Tensor]) -> pc.optim.BoundOptimizer:
        mine, rec, steps = list(params), self, []
        self.steps.append(steps)

        class _Bound(pc.optim.BoundOptimizer):
            @override
            def step_(self, grads: Sequence[Tensor], elapsed: float) -> None:
                for p, g in zip(mine, grads, strict=True):
                    rec.grads.setdefault(p, g.detach().clone())
                steps.append(elapsed)

            @override
            def state_dict(self) -> dict[str, Any]:
                return {}

            @override
            def load_state_dict(self, state: Any, /) -> None:
                pass

        return _Bound()


class _Binds(pc.optim.Optimizer):
    """SGD that records the parameters of each binding (``bound``)."""

    def __init__(self) -> None:
        self.bound: list[list[Tensor]] = []

    @override
    def bind(self, params: Iterable[Tensor]) -> pc.optim.BoundOptimizer:
        self.bound.append(list(params))
        return pc.optim.SGD(tau=1.0).bind(self.bound[-1])


def _constrained(net: VAEPC) -> list[Any]:
    """The nodes with r and lam, from the input: enc1-4, latent, dec1-2."""
    enc = [getattr(net, n) for n in net.names if n.startswith("enc")]
    dec = [getattr(net, n) for n in net.names if n.startswith("dec")]
    return [*enc, net.latent, *dec]


def _frames(
    arch: str,
    learn_interval: float,
    valid: Tensor | None = None,
    frames: int = 2,
    learn_sigma: bool = True,
) -> tuple[VAEModules, GradRecorder, list[tuple[dict, dict]]]:
    """``frames`` frames of different stacks shown by the host, reset in the
    first only (the next start from the r and lam the first left, as in a
    video); per frame the first gradient each parameter was stepped by
    (nothing moves) and BP's. The noise of a stack that does not count is
    NaN: it must add nothing."""
    b = 2 if valid is None else len(valid)
    valid = torch.ones(b, dtype=torch.bool) if valid is None else valid
    conf = _conf(arch, learn_interval, b, learn_sigma)
    mods = _mods(conf)
    rec = GradRecorder()
    net = VAEPC(mods, conf=conf, optimizer=rec)
    host = Host(mujou.run_manually(net), frame=FRAME, period=net.tau.period)
    out = []
    try:
        for k in range(frames):
            x = stacks(b, seed=k)
            u, eps = _noise(x, seed=10 + k)
            bp = _bp_grads(mods, x[valid], u[valid], eps[valid], valid[valid])
            u[~valid], eps[~valid] = math.nan, math.nan
            rec.grads.clear()
            reset = torch.full((b,), k == 0)
            host.show_all(
                {net.prev: net.sample(x, u, eps, valid, reset)}, [net.stats, net.kl]
            )
            out.append((dict(rec.grads), bp))
    finally:
        host.run.shutdown()
    return mods, rec, out


@pytest.mark.parametrize("arch", ARCHS)
class TestBPGradient:
    def test_every_node_learns_by_bps_gradient(self, arch):
        """At ``LEARN_INTERVAL`` each node's g_theta / B is the float64 BP
        gradient of the mean -ELBO for every parameter (ln sigma and the rows
        of ln s^2 that start at 0 included), in the first frame and in the
        next (not reset)."""
        mods, _, frames = _frames(arch, LEARN_INTERVAL)
        for pcs, bps in frames:
            assert set(pcs) == set(mods.parameters())
            assert float(bps[mods.log_sigma].abs()) > 0
            assert float(bps[mods.encoder.head.weight][D:].abs().max()) > 0
            _assert_bps(mods, pcs, bps)

    def test_a_stack_that_does_not_count_adds_nothing(self, arch):
        """B = 3 with the middle stack masked, its noise NaN: the gradient is
        BP's mean over the other two (the energies weighed by B / n)."""
        valid = torch.tensor([True, False, True])
        mods, _, frames = _frames(arch, LEARN_INTERVAL, valid)
        for pcs, bps in frames:
            assert all(torch.isfinite(g).all() for g in pcs.values())
            _assert_bps(mods, pcs, bps)

    def test_a_fixed_sigma_is_not_learned_and_the_rest_is_bps(self, arch):
        """``learn_sigma=False``: no node steps ln sigma, which stays at
        1/2 ln sigma0_sq, and every other parameter still learns by BP's
        gradient of the -ELBO at that sigma."""
        mods, _, frames = _frames(arch, LEARN_INTERVAL, learn_sigma=False)
        assert float(mods.log_sigma) == pytest.approx(0.5 * math.log(0.05), rel=1e-6)
        for pcs, bps in frames:
            assert mods.log_sigma not in pcs
            assert set(pcs) == set(mods.parameters())
            _assert_bps(mods, pcs, bps)


class TestCreditArrival:
    """The derivation of ``LEARN_INTERVAL`` (exp.vision.pc's docstring): BP's
    credit reaches enc1 in step 2 * HOPS = 16 after the write, not before."""

    def _errors(self, learn_interval: float) -> list[tuple[float, bool]]:
        mods, _, frames = _frames("vit", learn_interval, frames=1)
        enc1 = set(mods.encoder.parts[0].parameters())
        keys = _key_biases(mods)
        pcs, bps = frames[0]
        return [
            (float((pcs[p] - g).norm() / g.norm()), p in enc1)
            for p, g in bps.items()
            if p not in keys
        ]

    def test_at_step_16_every_node_is_exact(self):
        assert all(err < 1e-9 for err, _ in self._errors(32e-3))

    def test_at_step_15_only_enc1_is_off(self):
        errors = self._errors(30e-3)
        assert all((err > 1e-2) if first else (err < 1e-9) for err, first in errors)
        assert any(first for _, first in errors)


class TestOncePerFrame:
    def test_every_node_learns_exactly_once_a_frame_over_one_frame(self):
        """Over 4 frames each of the 7 nodes with parameters steps once a
        frame, the latent never. A step's elapsed is the time since the last:
        in the steady state the frame (so SGD(tau) steps by 0.1 / tau of the
        gradient a frame); in frame 1 a period less, as the host's first write
        came a period after time 0."""
        _, rec, _ = _frames("resnet", LEARN_INTERVAL, frames=4)
        period = 2e-3
        assert len(rec.steps) == 7  # enc1-4, dec1-2, out
        for steps in rec.steps:
            assert len(steps) == 4
            assert steps[0] < FRAME - period
            assert steps[1] == pytest.approx(FRAME - period, rel=1e-12)
            for elapsed in steps[2:]:
                assert elapsed == pytest.approx(FRAME, rel=1e-12)


@pytest.mark.parametrize("arch", ARCHS)
class TestFixedPoint:
    def _run(self, arch: str, order: str | None) -> None:
        """With nothing learning and one write held, each constrained node's
        lam is -dE/dh of BP's E = B L at the forward values (the KL's at
        enc4) and r is 0, to the rounding of float64, whatever the order of
        the updates (Jacobi, or reversed one by one)."""
        conf = _conf(arch, learn_interval=math.inf)
        mods = _mods(conf)
        net = VAEPC(mods, conf=conf, optimizer=pc.optim.SGD(tau=1.0))
        x, valid = stacks(2), torch.ones(2, dtype=torch.bool)
        u, eps = _noise(x, seed=5)
        every = [net.inp, *_constrained(net), net.out]
        with mujou.run_manually(net) as run:
            run.step(net.tau.period)  # not at the time of the zeros
            run.write({net.prev: net.sample(x, u, eps, valid, valid)})
            for _ in range(60):
                if order is None:
                    run.step(net.tau.period)
                else:
                    run.step(net.tau.period, *every[::-1])
            loss, acts = _oracle(mods, x, u, eps, valid)
            credits = torch.autograd.grad(2 * loss, acts)
            for node, c in zip(_constrained(net), credits, strict=True):
                scale = float(c.abs().max())
                torch.testing.assert_close(node.lam, -c, rtol=0, atol=1e-10 * scale)
                assert float(node.r.abs().max()) <= 1e-10 * scale

    def test_jacobi(self, arch):
        self._run(arch, None)

    def test_reversed_one_by_one(self, arch):
        self._run(arch, "reversed")


class _LatentNet(mujou.Network):
    """``pc.Input -> Latent`` of h ``(B, 2d)`` and y ``(B, P + d)``."""

    def __init__(self, b: int, d: int, values: int) -> None:
        super().__init__()
        self.tau = tau = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
        self.inp = pc.Input(
            x=TensorSpec((b, 2 * d), torch.float64),
            y=TensorSpec((b, values + d), torch.float64),
            period=tau.period,
            batch=b,
        )
        self.latent = Latent(act_spec=self.inp.act_spec, tau=tau)
        self.inp.next.connect(self.latent.prev)


class TestLatent:
    B, D, P = 3, 2, 5

    def _sample(self, seed: int, valid: Tensor, reset: Tensor):
        g = torch.Generator().manual_seed(seed)
        h = torch.randn(self.B, 2 * self.D, generator=g, dtype=torch.float64)
        y = torch.randn(self.B, self.P + self.D, generator=g, dtype=torch.float64)
        return pc.msg.Sample(x=h, y=y, mask=valid, reset=reset), h, y[:, self.P :]

    def test_z_and_the_credit_of_the_kl_are_the_hand_derivatives(self):
        r"""z = mu + e^{l/2} eps goes on with the y it came with; with no next
        node the relaxed delta is 0, so the credit is (B / n) dKL/dh with
        dKL/dmu = mu, dKL/dl = (e^l - 1) / 2; a stack that does not count
        sends none and its eps (NaN here) is not read; stats is the mean KL
        over the stacks that count."""
        net = _LatentNet(self.B, self.D, self.P)
        valid = torch.tensor([True, False, True])
        sample, h, eps = self._sample(0, valid, valid)
        sample["y"][1] = math.nan
        d, w = self.D, self.B / 2
        mu, ell = h[:, :d], h[:, d:]
        with mujou.run_manually(net) as run:
            run.step(net.tau.period)
            run.write({net.inp.prev: sample})
            run.step(net.tau.period)  # inp sends
            run.step(net.tau.period)  # latent receives
            act = run.read(net.latent.next)
            on = valid[:, None]
            z = mu + (0.5 * ell).exp() * torch.where(on, eps, 0.0)
            torch.testing.assert_close(act["h"], z)
            assert torch.equal(act["y"].isnan(), sample["y"].isnan())
            kl_grad = torch.cat([mu, 0.5 * (ell.exp() - 1)], -1)
            want = torch.where(on, w * kl_grad, 0.0)
            torch.testing.assert_close(run.read(net.latent.prev), want)
            kl = 0.5 * (mu.square() + ell.exp() - 1 - ell).sum(1)
            stats = run.read(net.latent.stats)
            assert float(stats["kl"]) == pytest.approx(float(kl[valid].mean()))
            assert float(stats["n"]) == 2.0

    def test_a_new_input_resets_the_rows_it_says(self):
        """r and lam of the stacks to reset are 0 after the update that first
        sees a write (no credit comes from next), the others' only relax."""
        net = _LatentNet(self.B, self.D, self.P)
        on = torch.ones(self.B, dtype=torch.bool)
        with mujou.run_manually(net) as run:
            run.step(net.tau.period)
            run.write({net.inp.prev: self._sample(0, on, on)[0]})
            run.step(net.tau.period)
            run.step(net.tau.period)
            net.latent.r.fill_(1.0)
            net.latent.lam.fill_(1.0)
            reset = torch.tensor([True, False, False])
            run.write({net.inp.prev: self._sample(1, on, reset)[0]})
            run.step(net.tau.period)
            run.step(net.tau.period)
            r, lam = net.latent.r, net.latent.lam
            assert not r[0].any() and not lam[0].any()
            assert r[1:].abs().min() > 0 and lam[1:].abs().min() > 0


class TestMakeOptimizer:
    @pytest.mark.parametrize(
        ("conf", "want"),
        [
            (OptimConf(kind="sgd", tau=2.0), pc.optim.SGD(tau=2.0)),
            (
                OptimConf(kind="sgd_momentum", tau=2.0, tau_m=0.5),
                pc.optim.SGD(tau=2.0, tau_m=0.5),
            ),
            (
                OptimConf(kind="rmsprop", tau=2.0, tau_v=7.0, eps=1e-6),
                pc.optim.RMSProp(tau=2.0, tau_v=7.0, eps=1e-6),
            ),
            (
                OptimConf(
                    kind="adamw", tau=2.0, tau_m=0.5, tau_v=7.0, weight_decay=0.1
                ),
                pc.optim.AdamW(tau=2.0, tau_m=0.5, tau_v=7.0, weight_decay=0.1),
            ),
        ],
    )
    def test_each_kind_is_mujous_optimizer_with_its_settings(self, conf, want):
        assert make_optimizer(conf) == want

    @pytest.mark.parametrize(
        ("conf", "want"),
        [
            (
                OptimConf(kind="nesterov", tau=2.0, tau_m=0.5),
                Nesterov(tau=2.0, tau_m=0.5),
            ),
            (
                OptimConf(kind="rmsprop_centered", tau=2.0, tau_v=7.0, eps=1e-6),
                RMSPropCentered(tau=2.0, tau_v=7.0, eps=1e-6),
            ),
            (
                OptimConf(kind="adadelta", tau=0.1, tau_v=7.0, eps=1e-6),
                Adadelta(tau=0.1, tau_v=7.0, eps=1e-6),
            ),
            (OptimConf(kind="rprop", tau=2.0), Rprop(tau=2.0)),
            (
                OptimConf(
                    kind="muon", tau=2.0, tau_m=0.5, tau_v=7.0, weight_decay=0.01
                ),
                Muon(tau=2.0, tau_m=0.5, tau_v=7.0, weight_decay=0.01),
            ),
        ],
    )
    def test_each_kind_of_the_zoo(self, conf, want):
        assert make_optimizer(conf) == want

    def test_scale_multiplies_tau(self):
        c = OptimConf(kind="rmsprop", tau=2.0, tau_v=7.0)
        assert make_optimizer(c, scale=10.0) == pc.optim.RMSProp(tau=20.0, tau_v=7.0)


class TestNodeOptimizers:
    def test_the_decoders_tau_is_scaled_and_the_rest_kept(self):
        """``optim.dec_scale``: dec1, dec2 and out learn with tau * dec_scale,
        the encoder with tau (latent has no parameters)."""
        conf = _conf("vit")
        conf.optim = OptimConf(kind="rmsprop", tau=1e3, tau_v=30.0, dec_scale=10.0)
        opts = make_optimizers(conf, _mods(conf))
        assert sorted(opts) == sorted([*NODES])
        for name, opt in opts.items():
            want = 1e4 if name in ("dec1", "dec2", "out") else 1e3
            assert opt == pc.optim.RMSProp(tau=want, tau_v=30.0), name

    def test_grad_stats_wraps_each_node_by_its_name(self):
        conf = _conf("vit")
        conf.measure.grad_stats = True
        opts = make_optimizers(conf, _mods(conf))
        for name, opt in opts.items():
            assert isinstance(opt, GradStats) and opt.node == name
            assert opt.inner == make_optimizer(conf.optim)

    @pytest.mark.parametrize("arch", ARCHS)
    def test_each_node_binds_its_own_optimizer(self, arch):
        """Given a mapping, every node binds the optimizer of its name to its
        own parameters (the readout's and ln sigma for out)."""
        conf = _conf(arch)
        mods = _mods(conf)
        binds = {name: _Binds() for name in NODES}
        VAEPC(mods, conf=conf, optimizer=binds)
        parts = [*mods.encoder.parts, *mods.decoder.parts]
        own = {
            **{n: list(m.parameters()) for n, m in zip(NODES[:6], parts, strict=True)},
            "out": [*mods.decoder.readout.parameters(), mods.likelihood.log_sigma],
        }
        for name in NODES:
            got = [[id(p) for p in ps] for ps in binds[name].bound]
            assert got == [[id(p) for p in own[name]]], name

    def test_muons_matrices_are_the_linear_and_conv_weights(self):
        """Muon takes the parameters of 2 or more dimensions for matrices: in
        ViT-S and its decoder those are exactly the weights of the
        ``nn.Linear`` and ``nn.Conv2d`` layers."""
        from exp.vision.compose import compose

        mods = build_vae(compose(["+experiment=v0/base"]), seed=0)
        weights = {
            m.weight
            for module in mods.modules().values()
            for m in module.modules()
            if isinstance(m, torch.nn.Linear | torch.nn.Conv2d)
        }
        matrices = {
            p
            for module in mods.modules().values()
            for p in module.parameters()
            if p.ndim >= 2
        }
        assert matrices == weights


def test_the_new_nodes_hold_no_cross_node_backward_or_grad():
    """Locality (mujou/.claude/skills/locality-and-async): the nodes
    differentiate their own computation from their detached input only."""
    for obj in (Latent, GaussOut, VAEPC):
        src = inspect.getsource(obj)
        assert ".backward(" not in src
        assert re.search(r"\.grad\b(?!\()", src) is None, obj.__name__


class TestLearnWindow:
    """RunConf's window of ``learn_interval`` (``check_learn_interval``) is
    the derivation's, measured here by counting the learns over 4 frames."""

    def test_the_period_is_mujous(self):
        t = TimeConf()
        tau = pc.TimeConstants(h=t.h, lam=t.lam, r=t.r)
        assert t.period == tau.period

    def test_the_hops_are_the_nodes_after_the_input(self):
        conf = _conf("vit")
        net = VAEPC(_mods(conf), conf=conf, optimizer=GradRecorder())
        assert len(net.names) + 2 == HOPS  # the encoder, decoder, latent, out

    @pytest.mark.parametrize("learn_interval", [0.056, 0.102])
    def test_outside_it_a_node_does_not_learn_once_a_frame(self, learn_interval):
        _, rec, _ = _frames("resnet", learn_interval, frames=4)
        assert any(len(steps) != 4 for steps in rec.steps)
        with pytest.raises(ValueError, match="learn_interval"):
            check_learn_interval(TimeConf(learn_interval=learn_interval), HOPS)

    @pytest.mark.parametrize("learn_interval", [0.06, 0.1])
    def test_inside_it_every_node_does(self, learn_interval):
        _, rec, _ = _frames("resnet", learn_interval, frames=4)
        assert [len(steps) for steps in rec.steps] == [4] * 7
        check_learn_interval(TimeConf(learn_interval=learn_interval), HOPS)
