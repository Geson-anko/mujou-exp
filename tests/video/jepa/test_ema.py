"""The target encoder of I-JEPA (``vision.ema``) in BP and PC-ALM: the update
formula, its two ends, no gradient into it, and the same targets, loss and
credit in both learners."""

import inspect
import json
import re
from typing import Any, override

import mujou.testing
import pytest
import torch
from mujou.pc.optim import BoundOptimizer
from torch import nn

from exp.lib import ckpt
from exp.video.compose import compose, from_dict, to_dict
from exp.video.dyn.nets import make_optimizer
from exp.video.jepa.ema import Following, TargetStage, ema_
from exp.video.jepa.loss import jepa_loss_packed
from exp.video.jepa.masks import spatial_keep
from exp.video.jepa.nets import VisionBP, VisionPC, build_vision_net
from exp.video.jepa.nodes import PackedJEPAHead
from exp.video.jepa.vit import build_vision
from exp.video.programs.vision import rebuild
from tests.video.jepa.test_nodes import _hold, _show, _stack_sample
from tests.video.packed import packed_conf, stacks


def _clone(module: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


class TestEmaOf:
    def test_it_is_the_formula(self):
        g = torch.Generator().manual_seed(0)
        target, online = nn.Linear(3, 2), nn.Linear(3, 2)
        for p in [*target.parameters(), *online.parameters()]:
            p.data = torch.randn(p.shape, generator=g)
        before, m = _clone(target), 0.7
        ema_(target, online, m)
        for got, e, p in zip(target.parameters(), before, online.parameters()):
            torch.testing.assert_close(got.detach(), m * e + (1 - m) * p.detach())

    def test_m_1_keeps_the_target_and_m_0_makes_it_the_online(self):
        target, online = nn.Linear(3, 2), nn.Linear(3, 2)
        before = _clone(target)
        ema_(target, online, 1.0)
        assert all(torch.equal(a, b) for a, b in zip(target.parameters(), before))
        ema_(target, online, 0.0)
        assert all(
            torch.equal(a, b) for a, b in zip(target.parameters(), online.parameters())
        )


class _After(BoundOptimizer):
    """``inner``, recording the online and the target parameters after each
    of its steps (the target's move included)."""

    def __init__(self, inner: BoundOptimizer, online: nn.Module, target: nn.Module):
        self.inner, self.online, self.target = inner, online, target
        self.after: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []

    @override
    def step_(self, grads, elapsed):
        self.inner.step_(grads, elapsed)
        self.after.append((_clone(self.online), _clone(self.target)))

    @override
    def state_dict(self):
        return self.inner.state_dict()

    @override
    def load_state_dict(self, state, /):
        self.inner.load_state_dict(state)


def _learn(kind: str, m: float, frames: int = 3) -> tuple[Any, list[_After]]:
    """A net of ``kind`` with the target encoder of momentum ``m`` that learns
    once a frame of ``frames`` frames, each node's optimizer recorded."""
    net = build_vision_net(packed_conf(kind, learn_interval=25e-3, ema=m))
    mods = net.mods
    assert mods.target is not None
    if isinstance(net, VisionBP):
        node = net.vision
        recs = [_After(node.optimizer, node.encoder, mods.target)]
        node.optimizer = recs[0]
    else:
        assert isinstance(net, VisionPC)
        recs = []
        for i, name in enumerate(net.names):
            node = getattr(net, name)
            rec = _After(node.optimizer, mods.encoder.stages[i], mods.target.stages[i])
            node.optimizer = rec
            recs.append(rec)
    with mujou.testing.run(net) as run:
        for k in range(frames):
            _show(run, net.prev, _stack_sample(stacks(2, seed=k), k == 0))
    return net, recs


@pytest.mark.parametrize("kind", ["bp", "pc"])
class TestTheTargetEncoder:
    def test_it_moves_by_the_formula_after_each_step(self, kind):
        """theta_bar_k = m theta_bar_{k-1} + (1 - m) theta_k, theta_k the
        online weights after step k, from the initial copy."""
        m = 0.7
        net, recs = _learn(kind, m)
        init = build_vision(packed_conf(kind, ema=m), seed=0).target
        assert init is not None
        stages = [init] if kind == "bp" else list(init.stages)
        for rec, start in zip(recs, stages, strict=True):
            assert len(rec.after) >= 2
            prev = _clone(start)
            for online, target in rec.after:
                want = [m * e + (1 - m) * p for e, p in zip(prev, online)]
                for got, w in zip(target, want, strict=True):
                    torch.testing.assert_close(got, w)
                prev = target

    def test_m_1_keeps_the_initial_weights(self, kind):
        init = build_vision(packed_conf(kind, ema=1.0), seed=0).encoder
        net, _ = _learn(kind, 1.0)
        target, online = net.mods.target, net.mods.encoder
        assert all(torch.equal(a, b) for a, b in zip(target.parameters(), _clone(init)))
        moved = [
            not torch.equal(a, b) for a, b in zip(online.parameters(), _clone(init))
        ]
        assert any(moved)

    def test_no_gradient_reaches_it(self, kind):
        """Its parameters require no gradient, get none, and no node binds
        them to an optimizer."""
        net, _ = _learn(kind, 0.7)
        target = net.mods.target
        ids = {id(p) for p in target.parameters()}
        assert all(not p.requires_grad and p.grad is None for p in target.parameters())
        if kind == "bp":
            bound = net.vision._params
        else:
            bound = [p for n in net.names for p in getattr(net, n)._params]
            bound += net.head._params
        assert ids.isdisjoint(id(p) for p in bound)


def test_following_at_m_0_makes_the_target_the_online_after_each_step():
    """The limit m -> 0 (vision.ema 0 itself builds no target encoder): the
    move after the inner optimizer's step copies the online weights."""
    online, target = nn.Linear(3, 2), nn.Linear(3, 2)
    inner = packed_conf("bp").optim_vision
    assert inner is not None
    opt = Following(
        make_optimizer(inner, 1 / 30), online=online, target=target, momentum=0.0
    ).bind(list(online.parameters()))
    before = _clone(online)
    opt.step_([torch.ones_like(p) for p in online.parameters()], 1 / 30)
    assert any(not torch.equal(a, b) for a, b in zip(online.parameters(), before))
    assert all(
        torch.equal(a, b) for a, b in zip(target.parameters(), online.parameters())
    )


def test_ema_0_of_an_older_run_builds_no_target_and_loads_its_checkpoint(tmp_path):
    """A job.json from before 2026-11-13 holds "ema": 0.0: its modules are
    exactly the encoder and the predictor, and a checkpoint of those loads."""
    old = to_dict(compose(["+experiment=m17/wt_n16"]))
    assert old["vision"]["ema"] == 0.0
    conf = from_dict(json.loads(json.dumps(old)))
    mods = rebuild(conf)
    assert sorted(mods) == ["encoder", "predictor"]
    path = ckpt.save_eval(
        tmp_path, program="vision", group="g", t=300.0, frame=0, modules=mods
    )
    ckpt.load_eval(path, rebuild(conf), group="g")
    assert build_vision(conf, seed=0).target is None
    net = build_vision_net(packed_conf("bp", ema=0.0))
    assert isinstance(net, VisionBP) and net.vision.target is None


@pytest.mark.parametrize(
    "vision",
    [
        {"ema": 0.9},
        {
            "ema": 0.9,
            "token_dim": 8,
            "sigreg": 0.0,
            "sigreg_on": "tokens",
            "token_sigreg": "raw",
        },
    ],
    ids=["plain", "preset"],
)
def test_bp_and_pc_alm_have_the_same_targets_loss_and_credit(vision):
    """Hold a stack with learning off: the last encoder node of PC-ALM sends
    the packed tokens of JEPABP's encoder and then JEPABP's targets (the
    target encoder's tokens of the target view); the head's loss is JEPABP's;
    lam of each encoder node relaxes to -B dL/dy_i on the packed rows and
    stays 0 on the target's path, which gets no credit."""
    b = 2
    pc_net, bp_net = (
        build_vision_net(packed_conf(k, streams=b, **vision)) for k in ("pc", "bp")
    )
    assert isinstance(pc_net, VisionPC) and isinstance(bp_net, VisionBP)
    last = getattr(pc_net, pc_net.names[-1])
    with mujou.testing.run(pc_net) as r1, mujou.testing.run(bp_net) as r2:
        for k in range(2):
            for run, net in ((r1, pc_net), (r2, bp_net)):
                _show(run, net.prev, _stack_sample(stacks(b, seed=k), k == 0))
        _hold(r1, 2000)
        act = r2.read(bp_net.chunker.next)
        top = r1.read(last.next)
        g = r1.state_dict()["nodes"]["head"]["ports"]["prev"]["out"]["value"]
    mods = bp_net.mods
    assert mods.target is not None
    t, n = mods.segments[0], sum(mods.segments)
    with torch.no_grad():
        goal = mods.target(act["h"][:, :t], (t,))
    outputs: list[torch.Tensor] = []
    terms = bp_net.vision.terms(act, outputs)
    assert top["h"].shape == (b, n + t, goal.shape[-1])
    torch.testing.assert_close(top["h"][:, n:], goal)
    torch.testing.assert_close(top["h"][:, :n], outputs[-1].detach())
    head = pc_net.head
    assert isinstance(head, PackedJEPAHead)
    torch.testing.assert_close(head.values[0], terms.loss.detach())
    assert (g[:, n:] == 0).all()
    credit = [-b * x for x in torch.autograd.grad(terms.loss, outputs)]
    for name, expect in zip(pc_net.names, credit, strict=True):
        node = getattr(pc_net, name)
        scale = float(expect.abs().max())
        assert scale > 0
        torch.testing.assert_close(
            node.lam[:, :n], expect, rtol=1e-3, atol=1e-4 * scale
        )
        assert (node.lam[:, n:] == 0).all() and (node.r[:, n:] == 0).all()


def test_the_head_credit_is_that_of_b_times_the_loss_with_the_targets():
    b = 3
    net = build_vision_net(packed_conf(streams=b, ema=0.9))
    assert isinstance(net, VisionPC)
    with mujou.testing.run(net) as run:
        _show(run, net.prev, _stack_sample(stacks(b), True))
        act = run.read(net.enc2.next)
        g = run.state_dict()["nodes"]["head"]["ports"]["prev"]["out"]["value"]
    head = net.head
    assert isinstance(head, PackedJEPAHead)
    v = net.mods
    t = v.segments[0]
    h = act["h"].clone().requires_grad_()
    terms = jepa_loss_packed(
        h[:, :-t],
        act["y"],
        act["mask"],
        head.directions,
        None,
        head.module,
        keeps=list(v.segments[1:]),
        proj=None,
        lam=0.05,
        sg=False,
        sigreg_on="z",
        pred_loss="l2",
        goal=h[:, -t:],
    )
    (expect,) = torch.autograd.grad(b * terms.loss, [h])
    torch.testing.assert_close(g, expect)
    assert (g[:, -t:] == 0).all() and (g[:, :-t] != 0).any()


ARMS = [("m17/ema_t512", 512, 2), ("m17/ema_t128", 128, 8), ("m17/ema_t64", 64, 16)]


@pytest.mark.parametrize(("experiment", "tokens", "d"), ARMS)
def test_the_shapes_of_the_three_arms(experiment, tokens, d):
    """D = T d = 1024 in each arm; the target encoder starts as the encoder,
    and the packed sequence is T and the visible tokens of the two context
    views (spatial_keep per slot)."""
    conf = compose([f"+experiment={experiment}"])
    v = conf.vision
    assert v is not None
    mods = build_vision(conf, seed=0)
    assert mods.tokens_per_view == tokens and tokens * d == 1024
    slots = v.frames // v.tubelet
    keeps = [spatial_keep(mods.grid, m) * slots for m in v.context_masks]
    assert mods.segments == (tokens, *keeps)
    x = stacks(1, frames=16, seed=0)
    x = x.repeat_interleave(8, 3).repeat_interleave(8, 4)  # 128 x 128
    with torch.no_grad():
        out, goal = mods.tokens(x), mods.target_tokens(x)
    assert out.shape == (1, tokens, d)
    assert torch.equal(out, goal)


def test_the_nodes_hold_no_cross_node_backward_or_grad():
    """Locality (mujou/.claude/skills/locality-and-async): the copy and its
    move are each node's own."""
    for obj in (TargetStage, Following, ema_):
        src = inspect.getsource(obj)
        assert ".backward(" not in src
        assert re.search(r"\.grad\b(?!\()", src) is None, obj.__name__
