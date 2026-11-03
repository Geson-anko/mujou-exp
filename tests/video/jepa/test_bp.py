from typing import override

import mujou.testing
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
