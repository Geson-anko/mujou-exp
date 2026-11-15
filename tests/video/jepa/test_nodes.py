import inspect
import math
import re
from typing import Any, cast

import mujou.pc as pc
import mujou.testing
import pytest
import torch

from exp.lib.gauss_kl import buffer_moments
from exp.lib.sigreg import ecf_sums, sigreg
from exp.video.jepa.loss import jepa_loss, jepa_loss_packed
from exp.video.jepa.masks import block_masks, make_views, multiblock_masks, pack_views
from exp.video.jepa.nets import CHUNKER_SEED, VisionBP, VisionPC, build_vision_net
from exp.video.jepa.nodes import (
    Chunker,
    FrameRing,
    JEPAHead,
    PackedJEPAHead,
    PastPoints,
    PastTokens,
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
from exp.video.schema.vision import MaskConf
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


TOKENS = {
    "token_dim": 8,
    "sigreg_on": "tokens",
    "token_sigreg": "raw",
    "token_buffer": 20,
    "token_insert": 6,
    "sigreg": 0.3,
}  # the tokens of 8 as the representation, SIGReg of them with the past (per
# stream: of 2 streams 40 rows, 12 tokens a frame)


def _ids(streams: int, tokens: int, dim: int) -> torch.Tensor:
    """Target tokens ``(B, T, D)`` whose every entry is the token's index in
    the flattened ``(B, T)``: a row tells which token it is."""
    i = torch.arange(streams * tokens, dtype=torch.float32)
    return i.view(streams, tokens, 1).expand(-1, -1, dim).clone()


def _past(
    capacity: int = 10, insert: int = 4, streams: int = 3, tokens: int = 8
) -> PastTokens:
    """Past tokens of ``capacity`` and ``insert`` per stream: of 3 streams, 30
    rows and 12 tokens a frame."""
    return PastTokens(
        capacity=capacity,
        insert=insert,
        streams=streams,
        tokens=tokens,
        dim=4,
        directions=5,
        device="cpu",
    )


def _dirs() -> torch.Tensor:
    a = torch.randn(5, 4, generator=torch.Generator().manual_seed(9))
    return a / a.norm(dim=1, keepdim=True)


class TestPastTokens:
    @pytest.mark.parametrize("streams", [1, 4])
    def test_the_capacity_and_the_insertion_are_per_stream(self, streams):
        past, g = _past(streams=streams), torch.Generator().manual_seed(0)
        assert past.rows.shape == (10 * streams, 4)
        past.keep(_ids(streams, 8, 4), torch.ones(streams, dtype=torch.bool))
        past.next_frame(_dirs(), g)
        assert past.count == 4 * streams

    @pytest.mark.parametrize(
        ("capacity", "insert"), [(0, 4), (10, 0), (10, 9)], ids=["cap", "0", "T"]
    )
    def test_refused(self, capacity, insert):
        with pytest.raises(ValueError, match="insert <= the 8 tokens"):
            _past(capacity=capacity, insert=insert)

    def test_a_frame_inserts_k_distinct_tokens_of_the_valid_streams(self):
        past, g = _past(), torch.Generator().manual_seed(0)
        past.keep(_ids(3, 8, 4), torch.tensor([True, False, True]))
        past.next_frame(_dirs(), g)
        assert past.count == 12  # 3 streams x 4, of the 16 valid tokens
        ids = past.rows[:12, 0].long()
        assert len(set(ids.tolist())) == 12  # without replacement
        assert ((ids < 8) | (ids >= 16)).all()  # of streams 0 and 2 only
        assert (past.rows[:12] == past.rows[:12, :1]).all()  # whole rows

    def test_fewer_valid_tokens_than_k_insert_them_all_and_none_inserts_none(self):
        past, g = _past(streams=2, insert=8), torch.Generator().manual_seed(0)
        past.keep(_ids(2, 8, 4), torch.tensor([True, False]))  # 8 of 16 asked
        past.next_frame(_dirs(), g)
        assert past.count == 8
        assert sorted(past.rows[:8, 0].tolist()) == list(range(8))
        past.keep(_ids(2, 8, 4), torch.tensor([False, False]))
        past.next_frame(_dirs(), g)
        assert past.count == 8

    def test_insert_t_puts_every_token_of_each_stream(self):
        """At insert = T (the preset's), a frame's every target token goes in,
        one stream (B = 1) included."""
        for streams in (1, 3):
            past, g = _past(streams=streams, insert=8), torch.Generator()
            past.keep(_ids(streams, 8, 4), torch.ones(streams, dtype=torch.bool))
            past.next_frame(_dirs(), g.manual_seed(0))
            ids = sorted(past.rows[: past.count, 0].tolist())
            assert ids == list(range(8 * streams))

    def test_a_frame_is_inserted_once(self):
        past, g = _past(), torch.Generator().manual_seed(0)
        past.keep(_ids(3, 8, 4), torch.ones(3, dtype=torch.bool))
        past.next_frame(_dirs(), g)
        past.next_frame(_dirs(), g)  # no keep in between
        assert past.count == 12

    def test_once_full_a_frame_replaces_k_rows(self):
        """Random replacement (exp.lib.sigreg.insert_): the buffer stays full,
        and exactly the k rows drawn change, to the frame's tokens."""
        past, g = _past(capacity=7), torch.Generator().manual_seed(0)
        for k in range(2):
            past.keep(_ids(3, 8, 4) + 100 * k, torch.ones(3, dtype=torch.bool))
            past.next_frame(_dirs(), g)
        assert past.count == 21
        before = past.rows.clone()
        past.keep(_ids(3, 8, 4) + 1000, torch.ones(3, dtype=torch.bool))
        past.next_frame(_dirs(), g)
        changed = (past.rows != before).any(1)
        assert int(changed.sum()) == 12 and (past.rows[changed] >= 1000).all()

    @pytest.mark.parametrize("streams", [1, 4])
    def test_the_rows_survive_a_frame_with_probability_one_less_insert_over_capacity(
        self, streams
    ):
        """Once full, each frame replaces streams x insert of the streams x
        capacity rows drawn uniformly without replacement, independently of
        the frames before: a row survives k frames with probability (1 -
        insert / capacity)^k, (3/4)^k at insert = capacity / 4, whatever the
        streams. Over 4096 x 8 rows the fraction is within 0.02 of it."""
        tokens, capacity = 256, 1024
        past = _past(capacity=capacity, insert=tokens, streams=streams, tokens=tokens)
        g = torch.Generator().manual_seed(0)
        every = torch.ones(streams, dtype=torch.bool)
        frames = 0

        def show() -> None:
            """A frame of tokens whose second entry is the frame's number."""
            nonlocal frames
            frames += 1
            x = _ids(streams, tokens, 4)
            x[..., 1] = frames
            past.keep(x, every)
            past.next_frame(_dirs(), g)

        while past.count < len(past.rows):
            show()
        survived = torch.zeros(4)
        for _ in range(4096 // streams):  # 4096 x capacity rows in all of each k
            first = past.rows.clone()
            for k in range(4):
                show()
                survived[k] += float((past.rows == first).all(1).sum())
        fraction = survived / (4096 * capacity)
        expect = torch.tensor([0.75**k for k in range(1, 5)])
        torch.testing.assert_close(fraction, expect, atol=0.005, rtol=0)

    def test_the_sums_are_those_of_the_filled_rows_and_detached(self):
        past, g = _past(), torch.Generator().manual_seed(0)
        target = _ids(3, 8, 4).requires_grad_()
        past.keep(target, torch.ones(3, dtype=torch.bool))
        dirs = _dirs()
        past.next_frame(dirs, g)
        cos, sin = ecf_sums(past.rows[:12], dirs)
        torch.testing.assert_close(past.sums.cos, cos)
        torch.testing.assert_close(past.sums.sin, sin)
        assert float(past.sums.n) == 12
        assert not (past.rows.requires_grad or past.sums.cos.requires_grad)

    def test_the_state_round_trips(self):
        past, g = _past(), torch.Generator().manual_seed(0)
        past.keep(_ids(3, 8, 4), torch.ones(3, dtype=torch.bool))
        past.next_frame(_dirs(), g)
        past.keep(_ids(3, 8, 4) + 50, torch.tensor([True, False, False]))
        other = _past()
        other.load_state_dict(past.state_dict())
        for p in (past, other):
            p.next_frame(_dirs(), torch.Generator().manual_seed(1))
        assert other.count == past.count == 20
        for name, x in past.state_dict().items():
            if isinstance(x, torch.Tensor):
                assert torch.equal(getattr(other, name), x), name

    def test_a_state_of_other_rows_is_refused(self):
        """A state from before the sizes were per stream (capacity rows in
        all) would else replace the rows silently."""
        past = _past()  # 3 streams x 10 rows
        state = past.state_dict() | {"rows": past.rows[:10].clone()}  # 10 in all
        with pytest.raises(ValueError, match=r"rows is torch.Size\(\[30, 4\]\)"):
            _past().load_state_dict(state)


DATAPOINTS = {
    "token_dim": 8,
    "sigreg_on": "datapoints",
    "datapoint_buffer": 5,
    "sigreg": 0.3,
}  # SIGReg of the data points (2 slots x 16 tokens x 8 = 256 numbers each)
# with 5 past data points per stream

GAUSS = {
    "token_dim": 1,
    "sigreg_on": "datapoints",
    "datapoint_buffer": 20,
    "datapoint_law": "gauss",
    "sigreg": 0.3,
}  # the Gaussian moment matching of the data points (32 numbers each) with
# 20 past data points per stream (of 2 streams 40 rows, at least D + 5 = 37)


def _fill(points: PastPoints, seed: int = 0) -> None:
    """Fill every row of ``points`` with N(0, 1) values (as a full buffer):
    the moments are made at the next insertion."""
    g = torch.Generator().manual_seed(seed)
    points.rows.copy_(torch.randn(points.rows.shape, generator=g))
    points.count = len(points.rows)


def _points(capacity: int = 4, every: int = 1, streams: int = 1) -> PastPoints:
    return PastPoints(
        capacity=capacity,
        every=every,
        streams=streams,
        width=3,
        dtype=torch.float32,
        directions=5,
        device="cpu",
    )


def _point_dirs() -> torch.Tensor:
    a = torch.randn(5, 3, generator=torch.Generator().manual_seed(9))
    return a / a.norm(dim=1, keepdim=True)


class TestPastPoints:
    def test_one_insertion_every_k_frames_of_the_frame_before(self):
        """Frame j (from 0) inserts the data point kept in frame j - 1 when j
        is a multiple of k: of 10 frames at k = 3, those of frames 2, 5, 8;
        a stream without a stack inserts nothing."""
        points, g = _points(capacity=100, every=3, streams=2), torch.Generator()
        valid = torch.tensor([True, False])
        for j in range(10):
            points.next_frame(_point_dirs(), g)
            points.keep(torch.full((2, 3), float(j)), valid)
        assert points.count == 3 and points.frames == 10
        assert points.filled[:, 0].tolist() == [2.0, 5.0, 8.0]

    @pytest.mark.parametrize(("streams", "every"), [(1, 1), (1, 3), (4, 2)])
    def test_a_row_survives_s_frames_with_probability_1_less_1_over_n_to_s_over_k(
        self, streams, every
    ):
        """Once full, each insertion replaces B of the B N rows drawn
        uniformly without replacement, independently of the ones before: a
        row survives s frames (s a multiple of k) with probability (1 -
        1/N)^(s/k), whatever B, so its age is geometric of mean N k frames.
        Over 4096 x 8 rows the fraction is within 0.012 (4 sd) of it."""
        n = 8
        points, g = _points(capacity=n, every=every, streams=streams), torch.Generator()
        g.manual_seed(0)
        frames = 0
        every_stream = torch.ones(streams, dtype=torch.bool)

        def show() -> None:
            nonlocal frames
            points.next_frame(None, g)
            frames += 1
            x = torch.arange(streams, dtype=torch.float32)[:, None].expand(-1, 3)
            points.keep(x + 1000 * frames, every_stream)

        while points.count < len(points.rows) or points.frames % every != 1 % every:
            show()
        survived = torch.zeros(4)
        for _ in range(4096 // streams):
            first = points.rows.clone()
            for s in range(4):
                for _ in range(every):
                    show()
                survived[s] += float((points.rows == first).all(1).sum())
        fraction = survived / (4096 * n)
        expect = torch.tensor([(1 - 1 / n) ** s for s in range(1, 5)])
        torch.testing.assert_close(fraction, expect, atol=0.012, rtol=0)

    def test_the_sums_are_those_of_the_filled_rows(self):
        points, g = _points(), torch.Generator().manual_seed(0)
        points.next_frame(_point_dirs(), g)
        points.keep(
            torch.randn(1, 3, generator=g).requires_grad_(), torch.tensor([True])
        )
        points.next_frame(_point_dirs(), g)
        cos, sin = ecf_sums(points.filled, _point_dirs())
        torch.testing.assert_close(points.sums.cos, cos)
        torch.testing.assert_close(points.sums.sin, sin)
        assert float(points.sums.n) == 1 and not points.rows.requires_grad

    def test_the_state_round_trips_and_other_shapes_are_refused(self):
        points, g = _points(every=2), torch.Generator().manual_seed(0)
        for j in range(6):
            points.next_frame(_point_dirs(), g)
            points.keep(torch.full((1, 3), float(j)), torch.tensor([True]))
        other = _points(every=2)
        other.load_state_dict(points.state_dict())
        for p in (points, other):
            p.next_frame(_point_dirs(), torch.Generator().manual_seed(1))
        assert other.count == points.count == 3 and other.frames == points.frames
        for name, x in points.state_dict().items():
            if isinstance(x, torch.Tensor):
                assert torch.equal(getattr(other, name), x), name
        with pytest.raises(ValueError, match="rows"):
            _points(capacity=3).load_state_dict(points.state_dict())

    @pytest.mark.parametrize(("capacity", "every"), [(0, 1), (4, 0)])
    def test_refused(self, capacity, every):
        with pytest.raises(ValueError, match="capacity >= 1 and every >= 1"):
            _points(capacity=capacity, every=every)

    def test_the_moments_are_made_at_each_insertion_from_d_plus_5_rows(self):
        """With moments the buffer makes buffer_moments of its filled rows at
        each insertion (off until width + 5 = 8 rows), none in a frame
        without one; they are in the state and round-trip."""
        points = PastPoints(
            capacity=10,
            every=2,
            streams=1,
            width=3,
            dtype=torch.float32,
            directions=5,
            device="cpu",
            moments=True,
        )
        g = torch.Generator().manual_seed(0)
        for j in range(24):
            points.next_frame(None, g)
            if points.count < 8:
                assert float(points.on) == 0
            else:
                expect = buffer_moments(points.filled)
                for got, want in zip(points.moments, expect, strict=True):
                    torch.testing.assert_close(got, want)
            points.keep(torch.randn(1, 3, generator=g), torch.tensor([True]))
        assert points.count == 10 and float(points.on) == 1
        other = PastPoints(
            capacity=10,
            every=2,
            streams=1,
            width=3,
            dtype=torch.float32,
            directions=5,
            device="cpu",
            moments=True,
        )
        other.load_state_dict(points.state_dict())
        for got, want in zip(other.moments, points.moments, strict=True):
            assert torch.equal(got, want)

    def test_the_moments_need_d_plus_5_rows_in_all(self):
        with pytest.raises(ValueError, match="at least 8 rows"):
            PastPoints(
                capacity=2,
                every=1,
                streams=2,
                width=3,
                dtype=torch.float32,
                directions=5,
                device="cpu",
                moments=True,
            )


def _point_loss(net, act, h: torch.Tensor, **kw) -> Any:
    """The loss of the data points of ``net``'s head (PC-ALM) or JEPABP."""
    node = net.head if isinstance(net, VisionPC) else net.vision
    return jepa_loss_packed(
        h,
        act["y"],
        act["mask"],
        node.directions,
        None,
        net.mods.predictor,
        keeps=list(net.mods.segments[1:]),
        proj=None,
        lam=DATAPOINTS["sigreg"],
        sg=False,
        sigreg_on="datapoints",
        pred_loss="l2",
        **kw,
    )


class TestDataPointSIGReg:
    def test_the_head_credit_is_that_of_b_times_the_loss_with_its_points(self):
        """After three frames the head's buffer holds the data points of the
        first two (each stream's flattened target tokens); its credit is the
        gradient of B L, whose S is that of exp.lib.sigreg.sigreg of the
        current data points and the buffer."""
        b = 2
        net = build_vision_net(packed_conf(streams=b, **DATAPOINTS))
        assert isinstance(net, VisionPC)
        head = net.head
        assert isinstance(head, PackedJEPAHead) and head.points is not None
        t = net.mods.segments[0]
        assert head.directions.shape == (16, t * 8)
        with mujou.testing.run(net) as run:
            for k in range(3):
                _show(run, net.prev, _stack_sample(stacks(b, seed=k), k == 0))
            act = run.read(net.enc2.next)
            g = run.state_dict()["nodes"]["head"]["ports"]["prev"]["out"]["value"]
        assert head.points.count == 2 * b
        h = act["h"].clone().requires_grad_()
        terms = _point_loss(net, act, h, past=head.points.sums)
        (expect,) = torch.autograd.grad(b * terms.loss, [h])
        torch.testing.assert_close(g, expect)
        current = act["h"][:, :t].flatten(1)
        torch.testing.assert_close(
            terms.sigreg, sigreg(current, head.points.filled, head.directions)
        )

    def test_pc_alm_refuses_the_stored_inputs(self):
        conf = packed_conf(streams=1, **DATAPOINTS, datapoint_store="input")
        with pytest.raises(ValueError, match="learner=bp only"):
            build_vision_net(conf)

    def test_bp_stores_the_inputs_exactly_and_encodes_them_in_its_loss(self):
        """Mode (b): the buffer holds the u8 target view of each stack (the
        StackChunker's pixels times 255, exactly), and L sees the buffer
        encoded by the current encoder, in its graph: L and its gradient are
        those of jepa_loss_packed with points = the stored stacks' flattened
        tokens."""
        conf = packed_conf("bp", streams=1, **DATAPOINTS, datapoint_store="input")
        net = build_vision_net(conf)
        assert isinstance(net, VisionBP)
        bp = net.vision
        xs = [stacks(1, seed=k) for k in range(3)]
        with mujou.testing.run(net) as run:
            for k, x in enumerate(xs):
                _show(run, net.prev, _stack_sample(x, k == 0))
            act = run.read(net.chunker.next)
        assert bp.points is not None and bp.points.count == 2
        assert bp.points.rows.dtype == torch.uint8
        t = net.mods.segments[0]
        for row, x in zip(bp.points.filled, xs[:2], strict=True):
            pixels = row.view(t, -1).float() / 255
            torch.testing.assert_close(
                pixels, net.mods._whole(x)[0, :, :-1], rtol=0, atol=0
            )
        params = list(net.mods.encoder.parameters())
        loss = bp.terms(act).loss
        got = torch.autograd.grad(loss, params)
        stored = torch.cat(xs[:2])  # the stacks of the buffer, in its order
        points = net.mods.tokens(stored).flatten(1)
        h = net.mods.encoder(act["h"], net.mods.segments)
        expect = _point_loss(net, act, h, points=points).loss
        torch.testing.assert_close(loss, expect)
        for a, e in zip(got, torch.autograd.grad(expect, params), strict=True):
            torch.testing.assert_close(a, e)

    @pytest.mark.parametrize("store", ["z", "input"])
    def test_a_resumed_bp_run_is_the_one_never_stopped(self, store):
        """Stopped after 3 of 6 frames and restored from the run's state, the
        node goes on as one never stopped: the weights and the buffer."""
        vision = DATAPOINTS | {"datapoint_store": store, "datapoint_every": 2}

        def net() -> VisionBP:
            n = build_vision_net(
                packed_conf("bp", streams=1, learn_interval=25e-3, **vision)
            )
            assert isinstance(n, VisionBP)
            return n

        def show(run, n: VisionBP, frames: range) -> None:
            for k in frames:
                _show(run, n.prev, _stack_sample(stacks(1, seed=k), k == 0))

        whole = net()
        with mujou.testing.run(whole) as run:
            show(run, whole, range(6))
        first = net()
        with mujou.testing.run(first) as run:
            show(run, first, range(3))
            state = run.state_dict()
        second = net()
        with mujou.testing.run(second, state=state) as run:
            show(run, second, range(3, 6))
        a, b = whole.vision.points, second.vision.points
        assert a is not None and b is not None and a.count == b.count == 2
        assert a.frames == b.frames == 6
        assert torch.equal(a.rows, b.rows)
        for p, q in zip(
            whole.mods.encoder.parameters(),
            second.mods.encoder.parameters(),
            strict=True,
        ):
            assert torch.equal(p, q)


class TestDataPointGauss:
    def test_the_head_credit_is_that_of_b_times_the_loss_with_its_moments(self):
        """With a full buffer, after three frames the head's moments are those
        of its rows (made at the last insertion), and its credit is the
        gradient of B L with them: lam w_b g^_b / D at the target tokens."""
        b = 2
        net = build_vision_net(packed_conf(streams=b, **GAUSS))
        assert isinstance(net, VisionPC)
        head = net.head
        assert isinstance(head, PackedJEPAHead) and head.points is not None
        _fill(head.points)
        with mujou.testing.run(net) as run:
            for k in range(3):
                _show(run, net.prev, _stack_sample(stacks(b, seed=k), k == 0))
            act = run.read(net.enc2.next)
            g = run.state_dict()["nodes"]["head"]["ports"]["prev"]["out"]["value"]
        m = head.points.moments
        assert float(m.on) == 1
        for got, want in zip(m, buffer_moments(head.points.filled), strict=True):
            torch.testing.assert_close(got, want)
        h = act["h"].clone().requires_grad_()
        terms = _point_loss(net, act, h, datapoint_law="gauss", moments=m)
        (expect,) = torch.autograd.grad(b * terms.loss, [h])
        torch.testing.assert_close(g, expect)
        torch.testing.assert_close(terms.sigreg, m.kl / 32)

    def test_the_stored_inputs_are_refused(self):
        with pytest.raises(ValueError, match="datapoint_store 'z'"):
            packed_conf("bp", streams=2, **GAUSS, datapoint_store="input")

    def test_a_resumed_bp_run_is_the_one_never_stopped(self):
        """Stopped after 3 of 6 frames and restored, the node goes on as one
        never stopped: the weights, the buffer and its moments."""

        def net() -> VisionBP:
            n = build_vision_net(
                packed_conf("bp", streams=2, learn_interval=25e-3, **GAUSS)
            )
            assert isinstance(n, VisionBP) and n.vision.points is not None
            _fill(n.vision.points)
            return n

        def show(run, n: VisionBP, frames: range) -> None:
            for k in frames:
                _show(run, n.prev, _stack_sample(stacks(2, seed=k), k == 0))

        whole = net()
        with mujou.testing.run(whole) as run:
            show(run, whole, range(6))
        first = net()
        with mujou.testing.run(first) as run:
            show(run, first, range(3))
            state = run.state_dict()
        second = net()
        with mujou.testing.run(second, state=state) as run:
            show(run, second, range(3, 6))
        a, b = whole.vision.points, second.vision.points
        assert a is not None and b is not None and float(a.on) == 1
        assert torch.equal(a.rows, b.rows)
        for x, y in zip(a.moments, b.moments, strict=True):
            assert torch.equal(x, y)
        for p, q in zip(
            whole.mods.encoder.parameters(),
            second.mods.encoder.parameters(),
            strict=True,
        ):
            assert torch.equal(p, q)


def _token_loss(net: VisionPC, act, h: torch.Tensor, past) -> Any:
    v = net.mods  # the head's loss of the packed layout, without a projection
    head = cast(PackedJEPAHead, net.head)
    conf = TOKENS
    return jepa_loss_packed(
        h,
        act["y"],
        act["mask"],
        head.directions,
        None,
        head.module,
        keeps=list(v.segments[1:]),
        proj=None,
        lam=conf["sigreg"],
        sg=False,
        sigreg_on=conf["sigreg_on"],
        pred_loss="l2",
        token_sigreg=conf["token_sigreg"],
        past=past,
    )


class TestTokenRepresentation:
    def test_the_head_credit_with_the_past_is_that_of_b_times_the_loss(self):
        """After two frames the buffer holds tokens of the first; the credit
        is the gradient of B L with the head's own past sums, which count."""
        b = 2
        net = build_vision_net(packed_conf(streams=b, **TOKENS))
        assert isinstance(net, VisionPC)
        head = net.head
        assert isinstance(head, PackedJEPAHead) and head.past is not None
        with mujou.testing.run(net) as run:
            for k in range(2):
                _show(run, net.prev, _stack_sample(stacks(b, seed=k), k == 0))
            act = run.read(net.enc2.next)
            g = run.state_dict()["nodes"]["head"]["ports"]["prev"]["out"]["value"]
        assert act["h"].shape[-1] == 8  # the encoder's tokens are of token_dim
        assert head.past.count == 2 * 6  # per stream
        h = act["h"].clone().requires_grad_()
        t = _token_loss(net, act, h, head.past.sums)
        (expect,) = torch.autograd.grad(b * t.loss, [h])
        torch.testing.assert_close(g, expect)
        assert t.sigreg != _token_loss(net, act, h, None).sigreg

    def test_the_shapes_through_the_encoder_and_the_predictor(self):
        mods = build_vision(packed_conf(token_dim=8), seed=0)
        tokens = mods.tokens(stacks(3))
        assert tokens.shape == (3, mods.tokens_per_view, 8)
        torch.testing.assert_close(mods.z_of(tokens), tokens.mean(1))
        assert mods.proj is None
        k = mods.segments[1]
        ids = torch.arange(mods.tokens_per_view).expand(3, -1)
        pred = mods.predictor(tokens[:, :k], ids[:, :k], ids[:, k:])
        assert pred.shape == (3, mods.tokens_per_view - k, 8)

    def test_the_token_map_is_lecun_and_the_tokens_keep_a_unit_scale(self):
        """The last stage's map to token_dim: weights of std 1 / sqrt(dim)
        truncated at 2 std, zero bias, so the tokens of the LayerNorm's
        output (unit scale per dimension) are of about unit scale."""
        mods = build_vision(packed_conf(token_dim=8, dim=64, heads=2), seed=0)
        out = cast(Any, mods.encoder.stages[-1]).out
        assert isinstance(out, torch.nn.Linear) and out.out_features == 8
        w = out.weight.detach()
        assert (out.bias == 0).all()
        assert float(w.abs().max()) <= 2 / math.sqrt(64) + 1e-6
        # a normal of std s truncated at 2 s has std 0.88 s
        assert float(w.std()) == pytest.approx(0.88 / math.sqrt(64), rel=0.1)
        with torch.no_grad():
            rms = float(mods.tokens(stacks(8)).pow(2).mean().sqrt())
        assert 0.5 < rms < 2

    def test_without_token_dim_no_map_and_the_same_first_weights(self):
        """token_dim None builds no map: the older presets are unchanged, and
        the map comes after every weight of the stages before the last."""
        a = build_vision(packed_conf(), seed=0)
        b = build_vision(packed_conf(token_dim=8), seed=0)
        assert cast(Any, a.encoder.stages[-1]).out is None
        first_a, first_b = a.encoder.stages[0], b.encoder.stages[0]
        for (name, x), y in zip(
            first_a.state_dict().items(), first_b.state_dict().values(), strict=True
        ):
            assert torch.equal(x, y), name


def _packed_equilibrium(streams: int, **vision):
    frames = vision.get("frames", 4)
    nets = [
        build_vision_net(packed_conf(k, streams=streams, **vision))
        for k in ("pc", "bp")
    ]
    pc_net, bp_net = nets
    assert isinstance(pc_net, VisionPC) and isinstance(bp_net, VisionBP)
    if vision.get("datapoint_law") == "gauss":
        # full buffers: the moments exist from the first insertion
        for points in (cast(PackedJEPAHead, pc_net.head).points, bp_net.vision.points):
            assert points is not None
            _fill(points)
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
    points = cast(PackedJEPAHead, pc_net.head).points
    if points is not None:
        # As the past tokens below: BP takes the head's data points.
        assert bp_net.vision.points is not None and points.count > 0
        assert bp_net.vision.points.count == points.count
        bp_net.vision.points.load_state_dict(points.state_dict())
    past = cast(PackedJEPAHead, pc_net.head).past
    if past is not None:
        # The past tokens of each are of its own first frame (PC-ALM's carry
        # its relaxation): BP takes the head's, to compare the same loss.
        assert bp_net.vision.past is not None and past.count > 0
        assert bp_net.vision.past.count == past.count
        bp_net.vision.past.load_state_dict(past.state_dict())
    outputs: list[torch.Tensor] = []
    loss = bp_net.vision.terms(act, outputs).loss
    credit = [-streams * g for g in torch.autograd.grad(loss, outputs)]
    if vision.get("datapoint_law") == "gauss":
        # the gauss term is on and moves the credit well past the tolerance
        # of the comparison (rtol 1e-3): with it off the credit differs
        bp_points = bp_net.vision.points
        assert bp_points is not None and float(bp_points.on) == 1
        bp_points.on = torch.zeros_like(bp_points.on)
        outputs = []
        loss = bp_net.vision.terms(act, outputs).loss
        without = [-streams * g for g in torch.autograd.grad(loss, outputs)]
        for c, w in zip(credit, without, strict=True):
            assert (c - w).abs().max() > 1e-2 * c.abs().max()
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
            {
                "context_masks": [
                    MaskConf(blocks=2, scale=0.2, future=1),
                    MaskConf(blocks=0, scale=1.0, future=1),
                ]
            },
            {
                "z_weight": 1.0,
                "token_weight": 0.5,
                "projector": [12, 6],
                "zpred_hidden": 8,
                "sigreg_on": "views_tokens",
                "token_sigreg": "clip",
                "sigreg": 0.3,
                "z_dim": 8,
                "context_masks": [
                    MaskConf(blocks=2, scale=0.2),
                    MaskConf(blocks=1, scale=0.3, future=1),
                ],
            },
            {"z_weight": 1.0, "token_weight": 0.0, "sigreg_on": "views"},
            {"pred_dim": 32, "pred_heads": 4, "target_norm": True},
            TOKENS,
            DATAPOINTS,
            GAUSS,
        ],
        ids=[
            "n4",
            "sg-l1",
            "proj-both",
            "n1",
            "both-clip",
            "future",
            "space-time",
            "z-only",
            "wide-pred-norm",
            "tokens-past",
            "datapoints-past",
            "datapoints-gauss",
        ],
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


def test_the_projector_and_the_z_predictor_learn_in_the_head():
    vision = {"z_weight": 1.0, "projector": [12, 6], "zpred_hidden": 8}
    pc_net = build_vision_net(packed_conf("pc", **vision))
    bp_net = build_vision_net(packed_conf("bp", **vision))
    assert isinstance(pc_net, VisionPC) and isinstance(bp_net, VisionBP)
    for net, params in ((pc_net, pc_net.head._params), (bp_net, bp_net.vision._params)):
        mods = net.mods
        assert mods.projector is not None and mods.zpred is not None
        ids = {id(p) for p in params}
        assert {id(p) for p in mods.projector.parameters()} <= ids
        assert {id(p) for p in mods.zpred.parameters()} <= ids
        assert {"projector", "zpred"} <= set(mods.named())


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
    for obj in (StackChunker, PackedJEPAHead, PastTokens, PastPoints, Segmented):
        src = inspect.getsource(obj)
        assert ".backward(" not in src
        assert re.search(r"\.grad\b(?!\()", src) is None, obj.__name__
