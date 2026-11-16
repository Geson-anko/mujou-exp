import copy
import inspect
import math
import re
from dataclasses import replace
from typing import Any

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import Tensor

import mujou
from exp.lib.elbo import gauss_kl, gauss_nll
from exp.lib.host import Host
from exp.video.diag.m18_pc_grad import GradRecorder
from exp.video.dyn.nets import make_optimizer
from exp.video.jepa.vit import Block
from exp.video.schema import AEConf, LearnerConf, OptimConf, RunConf
from exp.video.vae.bp import VAENet, VAESample
from exp.video.vae.elbo import elbo_terms
from exp.video.vae.loss import batch_loss
from exp.video.vae.modules import DecoderPart, VAEModules, build_vae
from exp.video.vae.pc import VAEPC, GaussOut, Latent, build_vae_pc_net
from mujou import TensorSpec
from tests.video.packed import packed_conf, stacks

FRAME = 1 / 30
# The 15th step of 2 ms: 4 encoder nodes, the latent and 2 decoder nodes put
# enc1 7th from the output (plans/m18-vae-stream.md 「有限の歩数での一致」).
AT_15 = 29e-3
AT_14 = 27e-3


def _conf(learn_interval: float = AT_15, streams: int = 2) -> RunConf:
    """The real layout (4 encoder nodes of 1 block, the latent, 2 decoder
    nodes of 1 block, the output) at a small width on 16 x 16 frames."""
    conf = packed_conf(
        "pc",
        streams=streams,
        learn_interval=learn_interval,
        token_dim=4,
        depth=4,
        nodes=4,
    )
    return replace(
        conf,
        program="vae",
        ae=AEConf(decoder_depth=2, sigma0_sq=0.05),
        optim_vision=OptimConf(kind="sgd", lr=0.1, warmup=0.0),
    )


def _mods(conf: RunConf, *, double: bool = True) -> VAEModules:
    mods = build_vae(conf, seed=conf.seed)
    if double:
        for m in mods.named().values():
            m.double()
    return mods


def _noise(
    mods: VAEModules, x: Tensor, seed: int = 5, dtype: torch.dtype = torch.float64
) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(x.shape, generator=g, dtype=dtype)
    eps = torch.randn(len(x), *mods.code_shape, generator=g, dtype=dtype)
    return u, eps


def _oracle(
    mods: VAEModules, x: Tensor, u: Tensor, eps: Tensor, valid: Tensor
) -> tuple[Tensor, list[Tensor]]:
    r"""BP's loss (the mean $-\mathcal L$ over ``valid``) in float64, and the
    activities $h_1 .. h_7$ in its graph: the encoder's stages, $z$, the
    decoder's two blocks, composed here from the modules themselves."""
    v, dec = mods.vision, mods.decoder
    h = v.whole(x).double()
    acts = []
    for stage in v.encoder.stages:
        h = stage(h, (v.tokens_per_view,))
        acts.append(h)
    d = h.shape[-1] // 2
    mu, logvar = h[..., :d], h[..., d:]
    z = mu + (0.5 * logvar).exp() * eps
    acts.append(z)
    out = dec.embed(z) + dec.pos
    for block in dec.stage.blocks:
        out = block(out)
        acts.append(out)
    assert dec.stage.norm is not None and dec.stage.out is not None
    y_hat = dec.stacks(dec.stage.out(dec.stage.norm(out)))
    y = (x.double() + u) / 256
    nll = gauss_nll(y, y_hat, mods.log_sigma)
    kl = gauss_kl(mu, logvar).flatten(1).sum(1)
    return batch_loss(nll + kl, valid), acts


def _bp_grads(
    mods: VAEModules, x: Tensor, u: Tensor, eps: Tensor, valid: Tensor
) -> dict[Tensor, Tensor]:
    loss, _ = _oracle(mods, x, u, eps, valid)
    params = mods.parameters()
    return dict(zip(params, torch.autograd.grad(loss, params), strict=True))


def _key_biases(mods: VAEModules) -> dict[Tensor, slice]:
    """The elements of the keys in each block's ``qkv.bias``: their exact
    gradient is 0 (a bias of the keys adds the same $q^\top b$ to a query's
    scores of every key, which the softmax removes)."""
    out = {}
    for module in [mods.vision.encoder, mods.decoder]:
        for m in module.modules():
            if isinstance(m, Block):
                d = m.proj.in_features
                out[m.qkv.bias] = slice(d, 2 * d)
    return out


def _nodes(net: VAEPC) -> list[Any]:
    """The nodes with a constraint, from the input: enc1-4, latent, dec1-2."""
    enc = [getattr(net, n) for n in net.names if n.startswith("enc")]
    dec = [getattr(net, n) for n in net.names if n.startswith("dec")]
    return [*enc, net.latent, *dec]


def _encoder_params(mods: VAEModules, i: int) -> set[Tensor]:
    return set(mods.vision.encoder.stages[i].parameters())


def _show(
    host: Host,
    net: VAEPC,
    x: Tensor,
    u: Tensor,
    eps: Tensor,
    valid: Tensor,
    reset: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Show one frame; ``reset`` is that of every stack (the program writes
    the stream's: True only where a video starts)."""
    reset_ = torch.full((len(x),), reset, dtype=torch.bool)
    (out, latent), _ = host.show_all(
        {net.prev: net.sample(x, u, eps, valid, reset_)}, [net.stats, net.kl]
    )
    return out, latent


class TestFixedPoint:
    def test_lam_is_bps_credit_and_r_vanishes(self):
        """With nothing learning and one write held, each node's lam is
        -dE/dh of BP's loss E = B L at the forward values (the KL's in
        enc4's), and r is 0, to the rounding of float64."""
        conf = _conf(learn_interval=math.inf)
        mods = _mods(conf)
        net = VAEPC(mods, conf=conf, optimizer=pc.optim.SGD(tau=1.0))
        x, valid = stacks(2), torch.ones(2, dtype=torch.bool)
        u, eps = _noise(mods, x)
        with mujou.testing.run(net) as run:
            run.write({net.prev: net.sample(x, u, eps, valid, valid)})
            for _ in range(60):
                run.step(net.tau.period)
            loss, acts = _oracle(mods, x, u, eps, valid)
            credits = torch.autograd.grad(2 * loss, acts)
            for node, c in zip(_nodes(net), credits, strict=True):
                scale = float(c.abs().max())
                torch.testing.assert_close(node.lam, -c, rtol=0, atol=1e-10 * scale)
                assert float(node.r.abs().max()) <= 1e-10 * scale

    @pytest.mark.parametrize("order", ["reversed", "random"])
    def test_other_schedules_reach_the_same_fixed_point(self, order):
        """The nodes updated in reverse, or in an order drawn each step from
        a seeded generator: lam reaches the same credit (the fixed point does
        not depend on the schedule)."""
        conf = _conf(learn_interval=math.inf)
        mods = _mods(conf)
        net = VAEPC(mods, conf=conf, optimizer=pc.optim.SGD(tau=1.0))
        x, valid = stacks(2), torch.ones(2, dtype=torch.bool)
        u, eps = _noise(mods, x)
        every = [net.inp, *_nodes(net), net.out]
        g = torch.Generator().manual_seed(3)
        with mujou.testing.run(net) as run:
            run.write({net.prev: net.sample(x, u, eps, valid, valid)})
            for _ in range(200):
                if order == "reversed":
                    nodes = every[::-1]
                else:
                    nodes = [every[i] for i in torch.randperm(len(every), generator=g)]
                run.step(net.tau.period, *nodes)
            loss, acts = _oracle(mods, x, u, eps, valid)
            credits = torch.autograd.grad(2 * loss, acts)
            for node, c in zip(_nodes(net), credits, strict=True):
                scale = float(c.abs().max())
                torch.testing.assert_close(node.lam, -c, rtol=0, atol=1e-9 * scale)


class TestStepFifteen:
    def _frames(
        self, learn_interval: float, valid: Tensor | None = None
    ) -> list[tuple[dict[Tensor, Tensor], dict[Tensor, Tensor], VAEModules]]:
        """Two frames of different stacks under the host's steps, reset only
        in the first (the second's r and lam start from the first's, as in a
        video of the stream); per frame the gradients the nodes stepped by
        (nothing moves) and BP's. The noise of a stack that does not count
        is NaN: it must add nothing."""
        b = 2 if valid is None else len(valid)
        conf = _conf(learn_interval=learn_interval, streams=b)
        mods = _mods(conf)
        rec = GradRecorder()
        net = VAEPC(mods, conf=conf, optimizer=rec)
        valid = torch.ones(b, dtype=torch.bool) if valid is None else valid
        host = Host(mujou.testing.run(net), frame=FRAME, period=net.tau.period)
        out = []
        for k in range(2):
            x = stacks(b, seed=k)
            u, eps = _noise(mods, x, seed=10 + k)
            bp = _bp_grads(mods, x, u, eps, valid) if valid.all() else {}
            u[~valid], eps[~valid] = math.nan, math.nan
            rec.grads.clear()
            _show(host, net, x, u, eps, valid, reset=k == 0)
            out.append((dict(rec.grads), bp, mods))
        host.run.shutdown()
        return out

    def test_every_node_steps_by_bps_gradient(self):
        """At 29 ms (the 15th step) each node's g_theta / B is the gradient
        of BP's mean -L for every parameter, ln sigma and the rows of ln s^2
        that start at 0 included, in the first frame and in the next (not
        reset: its r and lam start from those of the frame before)."""
        for pcs, bps, mods in self._frames(AT_15):
            assert set(pcs) == set(mods.parameters())
            assert float(bps[mods.log_sigma].abs()) > 0
            head = mods.head.weight
            d = head.shape[0] // 2
            assert float(bps[head][d:].abs().max()) > 0  # the ln s^2 rows learn
            for p, g in bps.items():
                torch.testing.assert_close(
                    pcs[p], g, rtol=1e-9, atol=1e-12 * float(g.abs().max())
                )

    def test_one_step_earlier_only_enc1_is_off(self):
        """At 27 ms (the 14th step) enc1's credit is still that of a step
        before it settles: its gradient differs from BP's, every other
        node's is BP's, in the first frame and in the next. The derived
        step count is not more than needed."""
        for pcs, bps, mods in self._frames(AT_14):
            first = _encoder_params(mods, 0)
            for p, g in bps.items():
                err = float((pcs[p] - g).norm() / g.norm())
                if p in first:
                    assert err > 1e-2
                else:
                    assert err < 1e-9

    def test_a_stack_that_does_not_count_adds_nothing(self):
        """B = 3 with the middle stack masked (its noise NaN): the gradient
        is BP's mean over the other two (the energies weighed by B / n), not
        over 3."""
        valid = torch.tensor([True, False, True])
        for k, (pcs, _, mods) in enumerate(self._frames(AT_15, valid)):
            x = stacks(3, seed=k)
            u, eps = _noise(mods, x, seed=10 + k)
            two = _bp_grads(mods, x[valid], u[valid], eps[valid], valid[valid])
            for p, g in two.items():
                torch.testing.assert_close(
                    pcs[p], g, rtol=1e-9, atol=1e-12 * float(g.abs().max())
                )


class TestAgainstVAEBP:
    """The float32 net against the BP node of stage 2 (``bp.VAEBP``) built
    from the same weights."""

    def _pair(self) -> tuple[RunConf, VAEModules, VAEModules]:
        rms = OptimConf(kind="rmsprop", lr=1e-3, warmup=0.0, scale="fan_in")
        conf = replace(_conf(), optim_vision=rms)
        return conf, _mods(conf, double=False), _mods(conf, double=False)

    def test_the_first_stats_are_vaebps(self):
        r"""A frame's first -L (out's nll plus latent's kl), KL, ln sigma and
        n are those VAEBP sends of the same stacks and noise."""
        conf, a, b = self._pair()
        x, valid = stacks(2), torch.tensor([True, False])
        u, eps = _noise(a, x, dtype=torch.float32)
        assert conf.optim_vision is not None
        mods = list(b.named().values())
        bp_net = VAENet(
            b, conf=conf, optimizer=make_optimizer(conf.optim_vision, FRAME, mods)
        )
        bp_host = Host(mujou.testing.run(bp_net), frame=FRAME, period=bp_net.tau.period)
        want, _ = bp_host.show(
            {bp_net.prev: VAESample(x=x, u=u, eps=eps, mask=valid)}, bp_net.stats
        )
        net = build_vae_pc_net(conf)
        host = Host(mujou.testing.run(net), frame=FRAME, period=net.tau.period)
        out, latent = _show(host, net, x, u, eps, valid)
        assert float(out["nll"] + latent["kl"]) == pytest.approx(
            float(want["loss"]), rel=1e-6
        )
        assert float(latent["kl"]) == pytest.approx(float(want["kl"]), rel=1e-6)
        assert float(out["log_sigma"]) == float(want["log_sigma"])
        assert float(out["n"]) == float(latent["n"]) == float(want["n"]) == 1.0
        host.run.shutdown()
        bp_host.run.shutdown()

    def test_rmsprop_fan_in_moves_the_weights_as_vaebp(self):
        """Two frames at 29 ms under RMSProp of the per-layer rate (one
        optimizer of all the modules, bound by each node to its own): the
        weights move as VAEBP's, to a small fraction of a step (eta / n a
        frame)."""
        conf, a, b = self._pair()
        assert conf.optim_vision is not None
        start = copy.deepcopy(a)
        net = VAEPC(
            a,
            conf=conf,
            optimizer=make_optimizer(
                conf.optim_vision, FRAME, list(a.named().values())
            ),
        )
        bp_net = VAENet(
            b,
            conf=conf,
            optimizer=make_optimizer(
                conf.optim_vision, FRAME, list(b.named().values())
            ),
        )
        host = Host(mujou.testing.run(net), frame=FRAME, period=net.tau.period)
        bp_host = Host(mujou.testing.run(bp_net), frame=FRAME, period=bp_net.tau.period)
        valid = torch.ones(2, dtype=torch.bool)
        for k in range(2):
            x = stacks(2, seed=k)
            u, eps = _noise(a, x, seed=20 + k, dtype=torch.float32)
            _show(host, net, x, u, eps, valid)
            bp_host.show(
                {bp_net.prev: VAESample(x=x, u=u, eps=eps, mask=valid)}, bp_net.stats
            )
        host.run.shutdown()
        bp_host.run.shutdown()
        keys = _key_biases(a)
        moved = 0.0
        for p0, pa, pb in zip(
            start.parameters(), a.parameters(), b.parameters(), strict=True
        ):
            da, db = (pa - p0).detach(), (pb - p0).detach()
            if pa in keys:  # RMSProp steps by the sign of the rounding there
                da[keys[pa]] = db[keys[pa]] = 0.0
            moved = max(moved, float(db.abs().max()))
            torch.testing.assert_close(da, db, rtol=0, atol=1e-3 * 1e-3)
        assert moved > 1e-4  # the steps are not below the tolerance


class _LatentNet(mujou.Network):
    """``pc.Input -> Latent`` of h ``(B, T, 2d)`` and y ``(B, P + T d)``."""

    def __init__(self, b: int, t: int, d: int, values: int) -> None:
        super().__init__()
        tau = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
        self.tau = tau
        self.inp = pc.Input(
            x=TensorSpec((b, t, 2 * d), torch.float64),
            y=TensorSpec((b, values + t * d), torch.float64),
            period=tau.period,
            batch=b,
        )
        self.latent = Latent(act_spec=self.inp.act_spec, tau=tau)
        self.inp.next.connect(self.latent.prev)


class TestLatent:
    B, T, D, P = 3, 2, 2, 5

    def _write(self, net: _LatentNet, seed: int, valid: Tensor, reset: Tensor):
        g = torch.Generator().manual_seed(seed)
        h = torch.randn(self.B, self.T, 2 * self.D, generator=g, dtype=torch.float64)
        y = torch.randn(
            self.B, self.P + self.T * self.D, generator=g, dtype=torch.float64
        )
        sample = pc.msg.Sample(x=h, y=y, mask=valid, reset=reset)
        return sample, h, y[:, self.P :].reshape(self.B, self.T, self.D)

    def test_the_credit_is_the_hand_derivatives(self):
        r"""z = mu + e^{l/2} eps is sent on with the y it came with; the
        credit is -dz/dh^T delta + (B / n) dKL/dh with dKL/dmu = mu, dKL/dl
        = (e^l - 1) / 2, dz/dmu = 1, dz/dl = e^{l/2} eps / 2; a stack that
        does not count sends none and its eps is not read (its z is mu);
        stats is the mean KL over the stacks that count."""
        net = _LatentNet(self.B, self.T, self.D, self.P)
        valid = torch.tensor([True, False, True])
        no = torch.zeros(self.B, dtype=torch.bool)
        sample, h, eps = self._write(net, 0, valid, no)
        d, w = self.D, self.B / 2
        mu, ell = h[..., :d], h[..., d:]
        with mujou.testing.run(net) as run:
            run.write({net.inp.prev: sample})
            run.step(net.tau.period)
            act = run.read(net.latent.next)
            on = valid[:, None, None]
            z = mu + (0.5 * ell).exp() * torch.where(on, eps, 0.0)
            torch.testing.assert_close(act["h"], z)
            assert torch.equal(act["y"], sample["y"])
            kl_grad = torch.cat([mu, 0.5 * (ell.exp() - 1)], -1)
            want = torch.where(on, w * kl_grad, 0.0)  # delta is 0 at the start
            torch.testing.assert_close(run.read(net.latent.prev), want)
            kl = 0.5 * (mu.square() + ell.exp() - 1 - ell).sum((1, 2))
            stats = run.read(net.latent.stats)
            assert float(stats["kl"]) == pytest.approx(float(kl[valid].mean()))
            assert float(stats["n"]) == 2.0
            # a delta of our own: the constraint's part of the credit
            delta = torch.randn(self.B, self.T, d, dtype=torch.float64)
            net.latent.r.copy_(delta)
            net.latent.lam.zero_()
            run.step(net.tau.period)
            jt = torch.cat([-delta, -0.5 * (0.5 * ell).exp() * eps * delta], -1)
            want = torch.where(on, jt + w * kl_grad, 0.0)
            torch.testing.assert_close(run.read(net.latent.prev), want)

    def test_a_new_input_resets_the_rows_it_says(self):
        """r and lam of the stacks to reset are 0 after the update that first
        sees a write (no credit comes from next), the others' only relax."""
        net = _LatentNet(self.B, self.T, self.D, self.P)
        on = torch.ones(self.B, dtype=torch.bool)
        with mujou.testing.run(net) as run:
            run.write({net.inp.prev: self._write(net, 0, on, on)[0]})
            run.step(net.tau.period)
            net.latent.r.fill_(1.0)
            net.latent.lam.fill_(1.0)
            reset = torch.tensor([True, False, False])
            run.write({net.inp.prev: self._write(net, 1, on, reset)[0]})
            run.step(net.tau.period)
            r, lam = net.latent.r, net.latent.lam
            assert not r[0].any() and not lam[0].any()
            assert r[1:].abs().min() > 0 and lam[1:].abs().min() > 0


class TestBuild:
    def test_build_needs_pc_and_binds_every_parameter_under_fan_in(self):
        """``learner=bp`` is refused; under RMSProp with ``scale=fan_in`` the
        net builds, its nodes in the order of the tree, the decoder at the
        encoder's blocks a node."""
        with pytest.raises(ValueError, match="learner=pc"):
            build_vae_pc_net(replace(_conf(), learner=LearnerConf(kind="bp")))
        rms = OptimConf(kind="rmsprop", lr=1e-3, warmup=0.0, scale="fan_in")
        net = build_vae_pc_net(replace(_conf(), optim_vision=rms))
        assert net.names == ("enc1", "enc2", "enc3", "enc4", "dec1", "dec2")

    def test_a_decoder_not_cut_by_the_encoders_blocks_a_node_is_refused(self):
        conf = _conf()
        assert conf.vision is not None
        conf = replace(
            conf,
            vision=replace(conf.vision, nodes=2),  # 2 blocks a node
            ae=AEConf(decoder_depth=3, sigma0_sq=0.05),
        )
        with pytest.raises(ValueError, match="cannot be cut"):
            build_vae_pc_net(conf)


def test_the_new_nodes_hold_no_cross_node_backward_or_grad():
    """Locality (mujou/.claude/skills/locality-and-async): the nodes
    differentiate their own computation from their detached input only."""
    for obj in (Latent, GaussOut, DecoderPart, VAEPC):
        src = inspect.getsource(obj)
        assert ".backward(" not in src
        assert re.search(r"\.grad\b(?!\()", src) is None, obj.__name__


def test_bp_elbo_terms_is_the_oracles_loss():
    """The float64 oracle of these tests is ``elbo_terms``'s loss (the BP
    path of stage 2) at the same weights and noise, to float32's rounding."""
    conf = _conf()
    mods = _mods(conf, double=False)
    x, valid = stacks(2), torch.ones(2, dtype=torch.bool)
    u, eps = _noise(mods, x, dtype=torch.float32)
    want = batch_loss(elbo_terms(mods, x, u, eps).loss, valid)
    double = _mods(conf)
    got, _ = _oracle(double, x, u.double(), eps.double(), valid)
    assert float(got.detach()) == pytest.approx(float(want.detach()), rel=1e-6)
