import inspect
import math
import re
from typing import Any, cast

import mujou.pc as pc
import mujou.testing
import pytest
import torch

from exp.video.jepa.loss import jepa_loss, jepa_loss_packed
from exp.video.jepa.masks import block_masks, make_views, multiblock_masks, pack_views
from exp.video.jepa.nets import CHUNKER_SEED, VisionBP, VisionPC, build_vision_net
from exp.video.jepa.nodes import (
    Chunker,
    FrameRing,
    JEPAHead,
    PackedJEPAHead,
    StackChunker,
    z_spec,
)
from exp.video.jepa.vit import Segmented, build_vision
from exp.video.schema import (
    BallsConf,
    DataConf,
    LearnerConf,
    OptimConf,
    RunConf,
    TimeConf,
    VisionConf,
)
from mujou import Port, TensorSpec
from tests.video.packed import masks as packed_masks, packed_conf, stacks

TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
STEPS = 17  # of TAU.period in a frame of 1/30 s
SIDE = 8
SMALL: dict[str, Any] = {
    "patch": 4,
    "dim": 16,
    "depth": 2,
    "heads": 2,
    "nodes": 2,
    "pred_depth": 1,
}


def _conf(
    learner: str = "pc",
    *,
    streams: int = 2,
    learn_interval: float | None = math.inf,
    **vision,
) -> RunConf:
    return RunConf(
        name="test",
        program="vision",
        learner=LearnerConf(kind=learner),
        data=DataConf(streams=streams, balls=BallsConf(obs="frames", size=SIDE)),
        vision=VisionConf(**(SMALL | vision)),
        optim_vision=OptimConf(kind="sgd", lr=0.1, warmup=0.0),
        time=TimeConf(learn_interval=learn_interval),
    )


def _sample(x: torch.Tensor, reset) -> pc.msg.Sample:
    b = x.shape[0]
    reset = torch.as_tensor(reset).expand(b).clone()
    ones = torch.ones(b, dtype=torch.bool)
    return pc.msg.Sample(x=x, y=torch.zeros(b, 1), mask=ones, reset=reset)


def _show(run, port, sample, steps: int = STEPS) -> None:
    run.write({port: sample})
    for _ in range(steps):
        run.step(TAU.period)


def _hold(run, steps: int) -> None:
    for _ in range(steps):
        run.step(TAU.period)


def _frames(n: int, b: int, seed: int = 0) -> torch.Tensor:
    return torch.rand(
        n, b, 1, SIDE, SIDE, generator=torch.Generator().manual_seed(seed)
    )


class TestFrameRing:
    def test_the_last_n_frames_oldest_first_and_ready_after_n(self):
        ring = FrameRing(TensorSpec((2, 1, 2, 2)), 3, "cpu")
        xs = _frames(5, 2)[..., :2, :2]
        for k, x in enumerate(xs):
            ring.push_(x, torch.tensor([k == 0, k in (0, 3)]))
            assert ring.ready.tolist() == [k >= 2, k >= 2 and k != 3 and k != 4]
        assert torch.equal(ring.chunk[0], xs[2:, 0])
        assert torch.equal(ring.chunk[1, 1:], xs[3:, 1])
        assert (ring.chunk[1, 0] == 0).all()  # stream 1 reset at frame 3

    def test_the_state_round_trips(self):
        ring = FrameRing(TensorSpec((2, 1, 2, 2)), 3, "cpu")
        for k, x in enumerate(_frames(2, 2)[..., :2, :2]):
            ring.push_(x, torch.tensor([k == 0, k == 0]))
        other = FrameRing(TensorSpec((2, 1, 2, 2)), 3, "cpu")
        other.load_state_dict(ring.state_dict())
        x = _frames(1, 2, seed=1)[0, ..., :2, :2]
        for r in (ring, other):
            r.push_(x, torch.zeros(2, dtype=torch.bool))
        assert torch.equal(ring.chunk, other.chunk)
        assert torch.equal(ring.ready, other.ready)


def _chunker(b: int, frames: int, views: int = 2, seed: int = 5) -> Chunker:
    return Chunker(
        image=TensorSpec((b, 1, SIDE, SIDE)),
        frames=frames,
        views=views,
        patch=4,
        ratio=0.5,
        blocks=2,
        period=TAU.period,
        seed=seed,
    )


class TestChunker:
    def test_it_sends_the_views_of_the_chunk_with_the_masks_of_its_generator(self):
        """Frame k's chunk is frames k-N+1..k; the masks are the k-th draw of
        block_masks from the node's generator (the same seed, the same);
        mask is False until N frames since the reset."""
        b, n = 2, 2
        node = _chunker(b, n)
        g = torch.Generator().manual_seed(5)
        xs = _frames(3, b)
        with mujou.testing.run(node) as run:
            for k, x in enumerate(xs):
                _show(run, node.prev, _sample(x, k == 0))
                act = run.read(node.next)
                masks = block_masks(b, (2, 2), ratio=0.5, blocks=2, generator=g)
                chunk = torch.stack([xs[max(k - 1, 0)], x], 1)
                if k == 0:
                    chunk[:, 0] = 0
                h, y = make_views(chunk, masks.unflatten(0, (b, 1)), 4)
                assert torch.equal(act["h"], h) and torch.equal(act["y"], y)
                assert act["mask"].tolist() == [k >= 1] * (2 * b)
                assert act["reset"].tolist() == [k == 0] * (2 * b)

    def test_it_sends_once_a_frame(self):
        node = _chunker(1, 1)
        with mujou.testing.run(node) as run:
            run.write({node.prev: _sample(_frames(1, 1)[0], True)})
            run.step(TAU.period)
            first = run.read(node.next)
            _hold(run, 5)
            assert torch.equal(run.read(node.next)["y"], first["y"])


class TestHead:
    def test_z_out_only_sends(self):
        """What reads z_out cannot send credit back (M16's dynamics)."""
        net = build_vision_net(_conf())
        assert isinstance(net, VisionPC)
        spec = z_spec(2, 16)
        back = Port(send=spec.field("h", TensorSpec), recv=spec)
        with pytest.raises(TypeError, match="do not match"):
            net.head.z_out.connect(cast(Any, back))  # the mismatch under test
        net.head.z_out.connect(Port(recv=spec))

    @pytest.mark.parametrize("sg", [False, True])
    def test_the_credit_is_that_of_bv_times_the_loss(self, sg):
        """The head sends dE/dh with E = B V L (loss.jepa_loss of what it
        received, at its directions); with sg and lam 0 the target views get
        exactly 0."""
        b, v = 3, 3
        conf = _conf(streams=b, views=v, sg=sg, sigreg=0.0 if sg else 0.3)
        net = build_vision_net(conf)
        assert isinstance(net, VisionPC)
        with mujou.testing.run(net) as run:
            _show(run, net.prev, _sample(_frames(1, b)[0], True))
            act = run.read(net.enc2.next)
            g = run.state_dict()["nodes"]["head"]["ports"]["prev"]["out"]["value"]
        head = net.head
        assert isinstance(head, JEPAHead)
        h = act["h"].clone().requires_grad_()
        t = jepa_loss(
            h.view(b, v, *h.shape[1:]),
            act["y"].view(b, v, -1)[:, 1:],
            act["mask"].view(b, v)[:, 0],
            head.directions,
            head.module,
            extra=0,
            z_pool="mean",
            lam=0.0 if sg else 0.3,
            sg=sg,
        )
        (expect,) = torch.autograd.grad(b * v * t.loss, [h])
        torch.testing.assert_close(g, expect)
        if sg:
            assert (g.view(b, v, -1)[:, 0] == 0).all()
            assert (g.view(b, v, -1)[:, 1:] != 0).any()


def _equilibrium(streams: int, frames: int, **vision):
    """Show ``frames`` frames to VisionPC and VisionBP of the same seed with
    learning off, hold the last for 4 s; returns the nets and BP's credit of
    each stage's output, -B V dL/dy_i, of the views it holds."""
    nets = [
        build_vision_net(_conf(k, streams=streams, frames=frames, **vision))
        for k in ("pc", "bp")
    ]
    pc_net, bp_net = nets
    assert isinstance(pc_net, VisionPC) and isinstance(bp_net, VisionBP)
    xs = _frames(frames, streams)
    with mujou.testing.run(pc_net) as r1, mujou.testing.run(bp_net) as r2:
        for k, x in enumerate(xs):
            _show(r1, pc_net.prev, _sample(x, k == 0))
            _show(r2, bp_net.prev, _sample(x, k == 0))
        _hold(r1, 2000)
        act = r2.read(bp_net.chunker.next)
        assert torch.equal(act["h"], r1.read(pc_net.chunker.next)["h"])
    assert torch.equal(pc_net.head.directions, bp_net.vision.directions)
    outputs: list[torch.Tensor] = []
    loss = bp_net.vision.terms(act, outputs).loss
    v = _conf(**vision).vision
    assert v is not None
    bv = streams * v.views
    credit = [-bv * g for g in torch.autograd.grad(loss, outputs)]
    return pc_net, credit


class TestEquilibrium:
    @pytest.mark.parametrize("streams", [1, 3])
    @pytest.mark.parametrize(
        ("frames", "vision"),
        [
            (1, {}),
            (2, {"views": 3, "z_pool": "cls", "sg": True, "sigreg_on": "tokens"}),
            (
                2,
                {
                    "nodes": 1,
                    "z_pool": "registers",
                    "registers": 2,
                    "sigreg_on": "both",
                    "pixel_mean": 0.5,
                    "pixel_std": 0.3,
                },
            ),
            (1, {"recon": 2.0, "recon_hidden": 8, "pixel_mean": 0.5, "pixel_std": 0.3}),
            (
                1,
                {
                    "recon": 2.0,
                    "recon_hidden": 8,
                    "recon_decoder": "broadcast",
                    "pixel_mean": 0.5,
                    "pixel_std": 0.3,
                },
            ),
            (
                2,
                {
                    "recon": 1.5,
                    "recon_hidden": 8,
                    "recon_decoder": "broadcast",
                    "sg": True,
                    "views": 3,
                },
            ),
            (
                2,
                {
                    "recon": 1.5,
                    "recon_hidden": 8,
                    "sg": True,
                    "views": 3,
                    "sigreg": 0.0,
                },
            ),
        ],
        ids=[
            "n1",
            "n2-v3-cls-sg-tokens",
            "n2-1node-registers-both",
            "n1-recon",
            "n1-recon-broadcast",
            "n2-v3-sg-recon-broadcast",
            "n2-v3-sg-recon-only",
        ],
    )
    def test_lam_of_each_encoder_node_is_the_credit_of_bp(
        self, streams, frames, vision
    ):
        """Hold a frame with learning off: each encoder node's lam relaxes to
        -B V dL/dy_i of JEPABP's loss L (the mean over the streams) at the
        output y_i of its stage, sample by sample and view by view; r to 0."""
        pc_net, credit = _equilibrium(streams, frames, **vision)
        for name, expect in zip(pc_net.names, credit, strict=True):
            node = getattr(pc_net, name)
            scale = expect.abs().max()
            assert scale > 0
            torch.testing.assert_close(
                node.lam, expect, rtol=1e-3, atol=1e-4 * float(scale)
            )
            assert node.r.abs().max() <= 1e-6 * scale


class TestViews:
    def test_a_streams_embeddings_do_not_depend_on_the_other_streams(self):
        """Change the frames of stream 1: the tokens the last encoder node
        sends for the views of stream 0 stay the same, bit for bit."""
        outs = []
        for change in (False, True):
            net = build_vision_net(_conf(streams=2))
            assert isinstance(net, VisionPC)
            x = _frames(1, 2)[0]
            if change:
                x[1] = _frames(1, 1, seed=1)[0, 0]
            with mujou.testing.run(net) as run:
                _show(run, net.prev, _sample(x, True))
                outs.append(run.read(net.enc2.next)["h"])
        assert torch.equal(outs[0][:2], outs[1][:2])
        assert not torch.equal(outs[0][2:], outs[1][2:])


def test_pc_and_bp_draw_the_same_masks():
    """The masks come from the Chunker's generator (seed + CHUNKER_SEED), the
    same in both."""
    acts = []
    for kind in ("pc", "bp"):
        net = build_vision_net(_conf(kind, streams=2, learn_interval=25e-3))
        with mujou.testing.run(net) as run:
            for k, x in enumerate(_frames(3, 2)):
                _show(run, net.prev, _sample(x, k == 0))
            acts.append(run.read(net.chunker.next)["y"])
    assert torch.equal(acts[0], acts[1])
    assert CHUNKER_SEED != 0  # not the modules' seed


def test_the_chunker_carries_the_last_frame_of_the_target_view_in_y():
    """Only with carry_frame: y is the T masks and then the raw last frame of
    the stream in the target view's row (zeros in the others')."""
    b, n, v = 2, 2, 3
    xs = _frames(n, b)
    ys = []
    for carry in (False, True):
        node = Chunker(
            image=TensorSpec((b, 1, SIDE, SIDE)),
            frames=n,
            views=v,
            patch=4,
            ratio=0.5,
            blocks=2,
            period=TAU.period,
            seed=5,
            carry_frame=carry,
        )
        with mujou.testing.run(node) as run:
            for k, x in enumerate(xs):
                _show(run, node.prev, _sample(x, k == 0))
            ys.append(run.read(node.next)["y"])
    plain, carried = ys
    assert plain.shape == (b * v, 4)
    assert carried.shape == (b * v, 4 + SIDE * SIDE)
    assert torch.equal(carried[:, :4], plain)
    frames = carried[:, 4:].view(b, v, -1)
    assert torch.equal(frames[:, 0], xs[-1].flatten(1))
    assert (frames[:, 1:] == 0).all()


def test_the_modules_are_those_of_build_vision():
    net = build_vision_net(_conf())
    mods = build_vision(_conf(), seed=0)
    assert isinstance(net, VisionPC)
    for a, b in zip(net.mods.encoder.parameters(), mods.encoder.parameters()):
        assert torch.equal(a, b)


class TestEquilibriumWithRecon:
    def test_the_reconstruction_changes_the_credit_both_nets_agree_on(self):
        """The wiring of ``recon`` is live: the credit of BP (which the other
        test equates with PC-ALM's lam) differs from that of ``recon`` 0."""
        kw = {"pixel_mean": 0.5, "pixel_std": 0.3}
        _, plain = _equilibrium(1, 1, **kw)
        _, with_rec = _equilibrium(1, 1, recon=2.0, recon_hidden=8, **kw)
        diff = max(float((a - b).abs().max()) for a, b in zip(plain, with_rec))
        assert diff > 1e-3


# --- the packed layout (M17) ---------------------------------------------------


def _stack_sample(x: torch.Tensor, reset, ready: Any = True) -> pc.msg.Sample:
    b = x.shape[0]
    reset = torch.as_tensor(reset).expand(b).clone()
    ready = torch.as_tensor(ready).expand(b).clone()
    return pc.msg.Sample(x=x, y=torch.zeros(b, 1), mask=ready, reset=reset)


class TestStackChunker:
    def test_it_sends_the_packed_views_of_the_masks_of_its_generator(self):
        """The k-th write is packed with the k-th draw of multiblock_masks
        (view by view) from the node's generator; mask and reset are the
        host's."""
        mods = build_vision(packed_conf(streams=2), seed=0)
        keeps = [k // 2 for k in mods.segments[1:]]
        node = StackChunker(
            stack=TensorSpec((2, 4, 3, 16, 16), torch.uint8),
            patch=4,
            tubelet=2,
            masks=packed_masks(),
            keeps=keeps,
            period=TAU.period,
            seed=5,
        )
        g = torch.Generator().manual_seed(5)
        with mujou.testing.run(node) as run:
            for k in range(2):
                x = stacks(2, seed=k)
                _show(run, node.prev, _stack_sample(x, k == 0, [True, k == 1]))
                act = run.read(node.next)
                views = [
                    multiblock_masks(2, (4, 4), m, keep=n, generator=g)
                    for m, n in zip(packed_masks(), keeps, strict=True)
                ]
                h, y = pack_views(x, views, keeps=keeps, patch=4, tubelet=2)
                assert torch.equal(act["h"], h) and torch.equal(act["y"], y)
                assert act["mask"].tolist() == [True, k == 1]
                assert act["reset"].tolist() == [k == 0] * 2


class TestPackedHead:
    @pytest.mark.parametrize(
        "vision",
        [
            {},
            {"sg": True, "sigreg": 0.0},
            {"z_dim": 8, "sigreg_on": "both"},
            {"sigreg_on": "tokens", "token_sigreg": "raw"},
        ],
        ids=["plain", "sg", "proj-both", "tokens-raw"],
    )
    def test_the_credit_is_that_of_b_times_the_loss(self, vision):
        b = 3
        conf = packed_conf(streams=b, **({"sigreg": 0.3} | vision))
        net = build_vision_net(conf)
        assert isinstance(net, VisionPC)
        with mujou.testing.run(net) as run:
            _show(run, net.prev, _stack_sample(stacks(b), True))
            act = run.read(net.enc2.next)
            g = run.state_dict()["nodes"]["head"]["ports"]["prev"]["out"]["value"]
        head = net.head
        assert isinstance(head, PackedJEPAHead)
        v = conf.vision
        assert v is not None
        h = act["h"].clone().requires_grad_()
        t = jepa_loss_packed(
            h,
            act["y"],
            act["mask"],
            head.directions,
            head.directions_t if v.projected and v.sigreg_on == "both" else None,
            head.module,
            keeps=list(net.mods.segments[1:]),
            proj=net.mods.proj,
            lam=v.sigreg,
            sg=v.sg,
            sigreg_on=v.sigreg_on,
            pred_loss=v.pred_loss,
            token_sigreg=v.token_sigreg,
        )
        (expect,) = torch.autograd.grad(b * t.loss, [h])
        torch.testing.assert_close(g, expect)
        if v.sg:
            assert (g[:, :32] == 0).all() and (g[:, 32:] != 0).any()


def _packed_equilibrium(streams: int, **vision):
    frames = vision.get("frames", 4)
    nets = [
        build_vision_net(packed_conf(k, streams=streams, **vision))
        for k in ("pc", "bp")
    ]
    pc_net, bp_net = nets
    assert isinstance(pc_net, VisionPC) and isinstance(bp_net, VisionBP)
    with mujou.testing.run(pc_net) as r1, mujou.testing.run(bp_net) as r2:
        for k in range(2):
            ready = [True] * (streams - 1) + [k == 1]
            _show(
                r1,
                pc_net.prev,
                _stack_sample(stacks(streams, frames, seed=k), k == 0, ready),
            )
            _show(
                r2,
                bp_net.prev,
                _stack_sample(stacks(streams, frames, seed=k), k == 0, ready),
            )
        _hold(r1, 2000)
        act = r2.read(bp_net.chunker.next)
        assert torch.equal(act["h"], r1.read(pc_net.chunker.next)["h"])
    assert torch.equal(pc_net.head.directions, bp_net.vision.directions)
    outputs: list[torch.Tensor] = []
    loss = bp_net.vision.terms(act, outputs).loss
    credit = [-streams * g for g in torch.autograd.grad(loss, outputs)]
    return pc_net, credit


class TestPackedEquilibrium:
    @pytest.mark.parametrize(
        "vision",
        [
            {},
            {"sg": True, "pred_loss": "l1"},
            {"z_dim": 8, "sigreg_on": "both", "pixel_mean": 0.5, "pixel_std": 0.3},
            {"frames": 1, "tubelet": 1, "nodes": 1},
            {"sigreg_on": "both", "token_sigreg": "clip", "sigreg": 0.3},
        ],
        ids=["n4", "sg-l1", "proj-both", "n1", "both-clip"],
    )
    def test_lam_of_each_encoder_node_is_the_credit_of_bp(self, vision):
        """Hold a stack with learning off: lam of each encoder node relaxes to
        -B dL/dy_i of JEPABP's loss at its stage's output (the views of a
        sequence included), r to 0."""
        pc_net, credit = _packed_equilibrium(2, **vision)
        for name, expect in zip(pc_net.names, credit, strict=True):
            node = getattr(pc_net, name)
            scale = expect.abs().max()
            assert scale > 0
            torch.testing.assert_close(
                node.lam, expect, rtol=1e-3, atol=1e-4 * float(scale)
            )
            assert node.r.abs().max() <= 1e-6 * scale


def test_pc_and_bp_draw_the_same_packed_masks():
    ys = []
    for kind in ("pc", "bp"):
        net = build_vision_net(packed_conf(kind, learn_interval=25e-3))
        with mujou.testing.run(net) as run:
            for k in range(3):
                _show(run, net.prev, _stack_sample(stacks(2, seed=k), k == 0))
            ys.append(run.read(net.chunker.next)["y"])
    assert torch.equal(ys[0], ys[1])


def test_the_packed_nodes_hold_no_cross_node_backward_or_grad():
    """Locality (mujou/.claude/skills/locality-and-async): the PC-ALM nodes of
    the packed layout differentiate their own computation only."""
    for obj in (StackChunker, PackedJEPAHead, Segmented):
        src = inspect.getsource(obj)
        assert ".backward(" not in src
        assert re.search(r"\.grad\b(?!\()", src) is None, obj.__name__
