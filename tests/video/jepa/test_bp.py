from typing import override

import mujou.testing
import pytest
import torch
from mujou.pc.optim import BoundOptimizer, Optimizer

import mujou
from exp.video.jepa.bp import JEPABP
from exp.video.jepa.loss import jepa_loss
from exp.video.jepa.nets import VisionBP, VisionPC, build_vision_net
from exp.video.jepa.nodes import Chunker
from exp.video.jepa.vit import build_vision
from mujou import TensorSpec
from tests.video.jepa.test_nodes import TAU, _conf, _frames, _sample, _show


class _Recording(Optimizer):
    """Records the gradients of each step, and moves nothing."""

    def __init__(self) -> None:
        self.steps: list[list[torch.Tensor]] = []

    @override
    def bind(self, params) -> BoundOptimizer:
        steps = self.steps

        class Bound(BoundOptimizer):
            @override
            def step_(self, grads, elapsed):
                steps.append([g.clone() for g in grads])

            @override
            def state_dict(self):
                return {}

            @override
            def load_state_dict(self, state, /):
                pass

        return Bound()


class _Net(mujou.Network):
    def __init__(self, chunker: Chunker, vision: JEPABP) -> None:
        super().__init__()
        self.chunker, self.vision = chunker, vision


def _params(net) -> list[torch.Tensor]:
    mods = net.mods
    return [*mods.encoder.parameters(), *mods.predictor.parameters()]


def test_pc_and_bp_start_from_the_same_weights():
    pc_net = build_vision_net(_conf("pc", z_pool="cls"))
    bp_net = build_vision_net(_conf("bp", z_pool="cls"))
    assert isinstance(pc_net, VisionPC) and isinstance(bp_net, VisionBP)
    mine, theirs = _params(pc_net), _params(bp_net)
    assert len(mine) == len(theirs) > 0
    for a, b in zip(mine, theirs, strict=True):
        assert a is not b and torch.equal(a, b)


def test_it_learns_by_the_gradient_of_the_loss_and_sends_z():
    """When it learns, the gradients are dL/dtheta of jepa_loss of the views
    it holds through the same modules (encoder, then predictor); z_out is
    the pooled z of the target views at the weights."""
    conf = _conf("bp", streams=2, learn_interval=25e-3, frames=2)
    v = conf.vision
    assert v is not None
    mods = build_vision(conf, seed=0)
    fresh = build_vision(conf, seed=0)
    chunker = Chunker(
        image=TensorSpec((2, 1, 8, 8)),
        frames=2,
        views=2,
        patch=4,
        ratio=0.5,
        blocks=2,
        period=TAU.period,
        seed=3,
    )
    rec = _Recording()
    node = JEPABP(
        mods,
        act_spec=chunker.act_spec,
        views=2,
        lam=v.sigreg,
        sg=False,
        directions=v.sigreg_directions,
        tau=TAU,
        optimizer=rec,
        learn_interval=25e-3,
        seed=4,
    )
    chunker.next.connect(node.prev)
    net = _Net(chunker, node)
    with mujou.testing.run(net) as run:
        for k, x in enumerate(_frames(2, 2)):
            _show(run, chunker.prev, _sample(x, k == 0))
        act = run.read(chunker.next)
        z = run.read(node.z_out)
    assert rec.steps  # it learned in frame 1 (the chunk is full)
    params = [*fresh.encoder.parameters(), *fresh.predictor.parameters()]
    tokens = fresh.encoder(act["h"]).view(2, 2, -1, v.dim)
    t = jepa_loss(
        tokens,
        act["y"].view(2, 2, -1)[:, 1:],
        act["mask"].view(2, 2)[:, 0],
        node.directions,
        fresh.predictor,
        extra=0,
        z_pool="mean",
        lam=v.sigreg,
        sg=False,
    )
    expect = torch.autograd.grad(t.loss, params)
    for got, want in zip(rec.steps[-1], expect, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)
    with torch.no_grad():
        torch.testing.assert_close(z["h"], fresh.encode(act["h"][::2, :, :1]))


def test_no_parameter_gets_a_grad_and_only_the_nodes_move_the_weights():
    """Locality: after frames that learn, no parameter of PC-ALM's nodes has a
    .grad (each node differentiates detached inputs) and the weights moved."""
    net = build_vision_net(_conf("pc", learn_interval=25e-3))
    before = [p.clone() for p in _params(net)]
    with mujou.testing.run(net) as run:
        for k, x in enumerate(_frames(3, 2)):
            _show(run, net.prev, _sample(x, k == 0))
    after = _params(net)
    assert all(p.grad is None for p in after)
    moved = [not torch.equal(a, b) for a, b in zip(before, after, strict=True)]
    assert sum(moved) > len(moved) // 2
    assert all(bool(p.isfinite().all()) for p in after)


@pytest.mark.parametrize("decoder_kind", ["mlp", "broadcast"])
def test_some_hundred_adam_steps_reconstruct_the_balls_better_than_the_mean_image(
    decoder_kind,
):
    """Sanity of the reconstruction term: BP on L (lam small, mu 1) on random
    positions of two balls, 16 x 16 pixels. The decoder of z, on fresh
    positions, ends with a squared error per pixel (standardized) below that
    of the mean image of the training set, which is the best a decoder that
    sees nothing of the position can do."""
    import math as _math

    from exp.video.data.physics import render
    from exp.video.jepa.loss import Reconstruction
    from exp.video.jepa.masks import block_masks, make_views
    from exp.video.jepa.nodes import draw_directions
    from exp.video.schema import (
        BallsConf,
        DataConf,
        LearnerConf,
        OptimConf,
        TimeConf,
        VisionConf,
    )
    from exp.video.schema.run import RunConf

    side, radii, mean, std = 16, (0.15, 0.2), 0.1, 0.2
    conf = RunConf(
        name="t",
        program="vision",
        learner=LearnerConf(kind="bp"),
        data=DataConf(streams=16, balls=BallsConf(obs="frames", size=side)),
        vision=VisionConf(
            patch=4,
            dim=32,
            depth=2,
            heads=2,
            nodes=1,
            pred_depth=1,
            sigreg=0.05,
            recon=1.0,
            recon_hidden=64,
            recon_decoder=decoder_kind,
            pixel_mean=mean,
            pixel_std=std,
        ),
        optim_vision=OptimConf(kind="sgd", lr=0.1, warmup=0.0),  # unused: Adam below
        time=TimeConf(),
    )
    v = conf.vision
    assert v is not None
    mods = build_vision(conf, seed=0)
    decoder = mods.decoder
    assert decoder is not None
    params = [*mods.encoder.parameters(), *mods.predictor.parameters()]
    params += decoder.parameters()
    opt = torch.optim.Adam(params, lr=3e-3)
    g = torch.Generator().manual_seed(0)
    dirs = draw_directions(32, v.dim, g)

    def batch(n: int):
        states = torch.zeros(n, 8)
        states[:, [0, 1, 4, 5]] = torch.rand(n, 4, generator=g) * 0.8 + 0.1
        return render(states, side, radii)[:, None]  # (n, 1, 1, H, W)

    def terms(chunks: torch.Tensor):
        n = len(chunks)
        masks = block_masks(n, (4, 4), ratio=0.5, blocks=2, generator=g)
        h, y = make_views(chunks, masks.unflatten(0, (n, 1)), 4)
        rec = Reconstruction(v.recon, decoder, mean, std, chunks[:, -1])
        return jepa_loss(
            mods.encoder(h).unflatten(0, (n, 2)),
            y.unflatten(0, (n, 2))[:, 1:],
            torch.ones(n, dtype=torch.bool),
            dirs,
            mods.predictor,
            extra=mods.encoder.extra,
            z_pool="mean",
            lam=v.sigreg,
            sg=False,
            recon=rec,
        )

    train_mean = ((batch(256)[:, 0] - mean) / std).mean(0)
    for _ in range(800):
        opt.zero_grad()
        terms(batch(16)).loss.backward()
        opt.step()
    with torch.no_grad():
        test = batch(128)
        got = terms(test).recon
        base = (((test[:, 0] - mean) / std - train_mean) ** 2).mean()
    assert _math.isfinite(float(got)) and got < 0.8 * base
