import json
import math
from dataclasses import replace
from typing import Any

import mujou.pc as pc
import mujou.testing
import pytest
import torch

from exp.lib.sigreg import sigreg
from exp.video import model, run
from exp.video.balls import Video
from exp.video.bp import WorldBPNetwork
from exp.video.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TimeConfig,
)
from exp.video.evaluate import Snapshot, evaluate
from exp.video.model import Network
from exp.video.nodes import compare_energy, is_open

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
    ),
]

SMALL = Config(
    model=ModelConfig(d_z=8, d_s=12, directions=16, buffer=31),
    optim=OptimConfig(tau=1.0),
)


def _sample(image: torch.Tensor, reset: bool) -> pc.msg.Sample:
    return pc.msg.Sample(
        x=image,
        y=torch.zeros(1),
        mask=torch.tensor(not reset),
        reset=torch.tensor(reset),
    )


class TestEncoderWidth:
    def test_the_width_sets_the_channels(self):
        g1, z = model.encoder_modules(32, 16, width=64)
        convs = [
            m for s in (g1, z) for m in s.modules() if isinstance(m, torch.nn.Conv2d)
        ]
        assert [c.out_channels for c in convs] == [64, 128, 256]
        assert z(g1(torch.zeros(5, 1, 32, 32))).shape == (5, 16)

    def test_the_default_is_the_encoder_before(self):
        g1, z = model.encoder_modules(32, 64)
        count = sum(p.numel() for p in (*g1.parameters(), *z.parameters()))
        # 3x3 convolutions 1->32->64->128 and the linear map of 128*4*4 to 64.
        want = (9 * 32 + 32) + (9 * 32 * 64 + 64) + (9 * 64 * 128 + 128)
        assert count == want + 128 * 16 * 64 + 64

    def test_the_config_reaches_the_network(self):
        config = replace(SMALL, model=replace(SMALL.model, width=8))
        stages = model.encoder_stages(config)
        convs = [
            m for s in stages for m in s.modules() if isinstance(m, torch.nn.Conv2d)
        ]
        assert [c.out_channels for c in convs] == [8, 16, 32]


class TestNetwork:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        ("stop", "weight", "z_norm", "residual"),
        [
            (False, 0.0, False, False),
            (True, 0.0, False, False),
            (True, 0.3, True, False),
            (True, 0.0, True, True),
        ],
        ids=["", "stop_target", "target_weight-z_norm", "residual"],
    )
    def test_lam_is_the_credit_of_bp_at_the_fixed_point(
        self, device, stop, weight, z_norm, residual
    ):
        """Hold a frame with learning off: each node's lam relaxes to the
        credit -dL/dh of L_k (the prediction error and SIGReg), computed by
        autograd through the same modules, memory, buffer and directions;
        with ``stop_target``, through a target of the prediction whose
        gradient is ``target_weight`` times its own (0: detached)."""
        config = replace(
            SMALL,
            time=TimeConfig(learn_interval=math.inf),
            model=replace(
                SMALL.model,
                stop_target=stop,
                target_weight=weight,
                z_norm=z_norm,
                residual=residual,
            ),
            device=device,
        )
        net = Network(config)
        if residual:  # W_o starts at 0: give it weights to carry credit back
            torch.nn.init.normal_(net.compare.module.weight, std=0.3)
        video = Video(seed=0)
        f = video.next()
        with mujou.testing.run(net) as run_:
            for k in range(20):  # fills the buffer past min_samples
                if k:
                    f = video.next()
                run_.write({net.prev: _sample(f.image, k == 0)})
                for _ in range(17):
                    run_.step(net.tau.period)
            for _ in range(3000):  # hold the last frame: 6 s
                run_.step(net.tau.period)
        x = f.image.to(device)
        g1, z_mod = net.g1.module, net.z.module
        h1 = g1(x).detach().requires_grad_()
        z = z_mod(h1)
        z_leaf = z.detach().requires_grad_()
        d = net.dynamics
        s = d.module(d.z_prev, d.s_prev).detach().requires_grad_()
        sr = net.sigreg
        w = weight if stop else 1.0
        target = w * z_leaf + (1 - w) * z_leaf.detach()
        base = d.z_prev if residual else None
        loss = compare_energy(
            net.compare.module, s, target, 0.05, base
        ) + 0.05 * sigreg(z_leaf, sr.buffer[: sr.count], sr.directions)
        g_z, g_s = torch.autograd.grad(loss, [z_leaf, s])
        (g_h1,) = torch.autograd.grad(z, h1, g_z)
        torch.testing.assert_close(net.z.lam, -g_z, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(d.lam, -g_s, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(net.g1.lam, -g_h1, rtol=1e-3, atol=1e-7)
        assert g_z.abs().max() > 1e-4  # the credit is not trivially zero
        for r in (net.g1.r, net.z.r, d.r):
            assert r.abs().max() < 1e-6

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
    @pytest.mark.parametrize("bp", [False, True], ids=["pc", "bp"])
    def test_cuda_streams_as_the_cpu_does(self, tmp_path, monkeypatch, bp):
        """The same run on the CPU and on CUDA (without TF32) predicts the same
        first 10 frames up to rounding: the same initial weights and steps.
        SIGReg is not on yet (``min_samples``); after it, the directions differ,
        as the generators of the CPU and CUDA draw differently."""
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        errors = {}
        for device in ("cpu", "cuda"):
            records = _video_run(tmp_path, f"{device}-{bp}", bp=bp, device=device)
            errors[device] = torch.tensor(
                [r["error"] for r in records if "error" in r][:10]
            )
        assert len(errors["cpu"]) == 10
        torch.testing.assert_close(errors["cuda"], errors["cpu"], rtol=1e-3, atol=1e-6)

    def test_bp_starts_from_the_same_weights(self):
        pcn, bpn = Network(SMALL), WorldBPNetwork(SMALL)
        mine = [
            *pcn.g1.module.parameters(),
            *pcn.z.module.parameters(),
            *pcn.dynamics.module.parameters(),
            *pcn.compare.module.parameters(),
        ]
        w = bpn.world
        theirs = [
            *w.encoder.parameters(),
            *w.step.parameters(),
            *w.readout.parameters(),
        ]
        for a, b in zip(mine, theirs, strict=True):
            assert torch.equal(a, b)


class TestMoreNodes:
    """The network cut into more nodes (``enc_nodes``, ``dyn_nodes``)."""

    def test_lam_of_each_dynamics_node_is_the_credit_of_bp(self):
        """With three dynamics nodes, held on a frame with learning off, each
        node's lam relaxes to -dL/dy of its output, by autograd through the
        stacked modules and the states they remembered."""
        config = replace(
            SMALL,
            time=TimeConfig(learn_interval=math.inf),
            model=replace(SMALL.model, dyn_nodes=3, stop_target=True, z_norm=True),
        )
        net = Network(config)
        video = Video(seed=0)
        f = video.next()
        with mujou.testing.run(net) as run_:
            for k in range(20):
                if k:
                    f = video.next()
                run_.write({net.prev: _sample(f.image, k == 0)})
                for _ in range(17):
                    run_.step(net.tau.period)
            for _ in range(500):  # hold the last frame: 1 s
                run_.step(net.tau.period)
        d, d2, d3 = net.dynamics, net.dynamics2, net.dynamics3
        z = net.z.module(net.g1.module(f.image)).detach()
        y1 = d.module(d.z_prev, d.s_prev).detach().requires_grad_()
        y2 = d2.module(y1, d2.h_prev)
        y3 = d3.module(y2, d3.h_prev)
        y2.retain_grad()
        y3.retain_grad()
        compare_energy(net.compare.module, y3, z, 0.05).backward()
        for node, y in ((d, y1), (d2, y2), (d3, y3)):
            assert y.grad is not None
            torch.testing.assert_close(node.lam, -y.grad, rtol=1e-3, atol=1e-7)
            assert node.r.abs().max() < 1e-6
        assert y1.grad is not None and y1.grad.abs().max() > 1e-5

    @pytest.mark.parametrize("residual", [False, True], ids=["", "residual"])
    def test_the_states_carried_are_those_of_the_stack(self, residual):
        """Over frames of two videos with learning off, the top node predicts
        what ``DynamicsStack`` computes from the same z, frame by frame, from
        s = 0 at each video and through a frame without a target."""
        config = replace(
            SMALL,
            time=TimeConfig(learn_interval=math.inf),
            model=replace(SMALL.model, dyn_nodes=3, residual=residual),
        )
        net = Network(config)
        torch.nn.init.normal_(net.compare.module.weight, std=0.3)  # not 0
        snap = Snapshot.of(net)
        video = Video(seed=1, episode=(100.0, 100.0))
        zs, preds = [], []
        resets = (0, 6)  # a second video from frame 6
        with mujou.testing.run(net) as run_:
            for k in range(10):
                f = video.next()
                sample = pc.msg.Sample(  # frame 3 has no target, yet moves on
                    x=f.image,
                    y=torch.zeros(1),
                    mask=torch.tensor(k != 3 and k not in resets),
                    reset=torch.tensor(k in resets),
                )
                run_.write({net.prev: sample})
                for _ in range(17):
                    run_.step(net.tau.period)
                zs.append(run_.read(net.z.next)["h"])
                preds.append(run_.read(net.compare.next)["y_hat"])
        state = torch.zeros(3 * config.model.d_s)
        with torch.no_grad():
            for k in range(10):
                if k in resets:
                    state = torch.zeros(3 * config.model.d_s)
                    continue
                state = snap.step(zs[k - 1], state)
                base = zs[k - 1] if residual else 0
                torch.testing.assert_close(preds[k], base + snap.readout(state))

    def test_how_the_encoder_is_cut_does_not_change_the_function(self, monkeypatch):
        """The stages of EgoDex in 1, 2 or 3 nodes: the same weights, so the
        same z of the same image."""
        g = torch.Generator().manual_seed(0)
        images = torch.rand(256, 3, 64, 64, generator=g)
        monkeypatch.setattr(model, "calibration_images", lambda c: images)
        zs = []
        for n in (1, 2, 3):
            config = replace(
                SMALL,
                model=replace(SMALL.model, enc_nodes=n, z_norm=True),
                data=DataConfig(kind="egodex", size=64),
            )
            net = Network(config)
            enc = Snapshot.of(net).encoder
            assert len(net.encoder_names) == n
            with torch.no_grad():
                zs.append(enc(images[:4]))
        torch.testing.assert_close(zs[0], zs[1])
        torch.testing.assert_close(zs[0], zs[2])
        assert torch.allclose(zs[0].pow(2).mean(-1), torch.ones(4), atol=1e-4)


class TestBlur:
    def test_an_impulse_spreads_as_the_gaussian(self):
        """The blur of one lit pixel, away from the edges, is the Gaussian of
        the distance to it, normalized to sum to 1."""
        sigma = 1.5
        x = torch.zeros(1, 1, 21, 21)
        x[0, 0, 10, 10] = 1.0
        y = model.GaussianBlur(sigma)(x)[0, 0]
        i = torch.arange(21.0) - 10
        g = torch.exp(-(i[:, None] ** 2 + i**2) / (2 * sigma**2))
        g = g * (i.abs()[:, None] <= 5) * (i.abs() <= 5)  # cut at 3 sigma
        torch.testing.assert_close(y, g / g.sum())

    def test_it_adds_no_parameters_to_learn(self):
        def shapes(blur: float) -> list[torch.Size]:
            config = replace(SMALL, model=replace(SMALL.model, blur=blur))
            return [p.shape for p in Network(config).g1.module.parameters()]

        assert shapes(2.0) == shapes(0.0)


class TestResidual:
    def test_the_first_prediction_is_the_frame_before(self):
        """W_o starts at 0, so the residual prediction starts as z_{k-1}."""
        config = replace(SMALL, model=replace(SMALL.model, residual=True))
        readouts = [
            Network(config).compare.module,
            WorldBPNetwork(config).world.readout,
        ]
        for readout in readouts:
            assert not any(p.any() for p in readout.parameters())


class TestOpenLoop:
    def test_the_last_part_of_each_cycle_is_open(self):
        opened = [is_open(c, 0.5, 4) for c in range(1, 10)]
        assert opened == [False, False, True, True, False, False, True, True, False]
        assert not any(is_open(c, 0.0, 4) for c in range(1, 10))

    @pytest.mark.parametrize("bp", [False, True], ids=["pc", "bp"])
    def test_an_open_prediction_starts_from_the_one_before(self, bp):
        """In a cycle of 4 with half open, the 3rd and 4th predictions of a
        video remember the prediction of the frame before as $z_{k-1}$, the
        others the observed $z_{k-1}$."""
        config = replace(
            SMALL,
            time=TimeConfig(learn_interval=math.inf),
            model=replace(SMALL.model, open_loop=0.5, open_cycle=4),
        )
        net: Any = WorldBPNetwork(config) if bp else Network(config)
        video = Video(seed=2, episode=(100.0, 100.0))
        z_of = (
            (lambda: net.world.z_last) if bp else (lambda: run_.read(net.z.next)["h"])
        )
        node = net.world if bp else net.dynamics
        last: dict[str, torch.Tensor] = {}
        with mujou.testing.run(net) as run_:
            for k in range(9):
                f = video.next()
                run_.write({net.prev: _sample(f.image, k == 0)})
                for _ in range(17):
                    run_.step(net.tau.period)
                if k >= 2:  # z_prev was set at the start of frame k (count k)
                    want = last["z_hat"] if is_open(k, 0.5, 4) else last["z"]
                    torch.testing.assert_close(node.z_prev, want)
                last = {"z": z_of(), "z_hat": run_.read(net.next)["y_hat"]}


def _batch_sample(images: torch.Tensor, resets: list[bool]) -> pc.msg.Sample:
    reset = torch.tensor(resets)
    return pc.msg.Sample(
        x=images, y=torch.zeros(len(resets), 1), mask=~reset, reset=reset
    )


def _single_sample(f, reset: bool) -> pc.msg.Sample:
    return pc.msg.Sample(
        x=f.image,
        y=torch.zeros(1),
        mask=torch.tensor(not reset),
        reset=torch.tensor(reset),
    )


def _hold(run_, net, sample, steps: int = 17) -> None:
    run_.write({net.prev: sample})
    for _ in range(steps):
        run_.step(net.tau.period)


class TestStreams:
    """``data.streams`` = B: a batch of B independent streams."""

    @pytest.mark.parametrize(
        ("stop", "weight", "z_norm", "residual"),
        [(False, 0.0, False, False), (True, 0.3, True, True)],
        ids=["", "target_weight-z_norm-residual"],
    )
    def test_lam_is_the_credit_of_bp_per_sample_at_the_fixed_point(
        self, stop, weight, z_norm, residual
    ):
        """Hold a frame of two different streams with learning off: each node's
        lam, per sample, is -B dL/dh_b of the loss L of the batch (the mean over
        the samples of the prediction error, plus the one SIGReg of the two
        current z with the buffer), by autograd through the same modules."""
        b = 2
        config = replace(
            SMALL,
            time=TimeConfig(learn_interval=math.inf),
            model=replace(
                SMALL.model,
                stop_target=stop,
                target_weight=weight,
                z_norm=z_norm,
                residual=residual,
            ),
            data=DataConfig(streams=b),
        )
        net = Network(config)
        if residual:  # W_o starts at 0: give it weights to carry credit back
            torch.nn.init.normal_(net.compare.module.weight, std=0.3)
        videos = [Video(seed=0), Video(seed=1)]
        fs = [v.next() for v in videos]  # the frames shown, the last of them held
        with mujou.testing.run(net) as run_:
            for k in range(12):  # fills the buffer past min_samples
                if k:
                    fs = [v.next() for v in videos]
                _hold(
                    run_,
                    net,
                    _batch_sample(torch.stack([f.image for f in fs]), [k == 0] * b),
                )
            for _ in range(3000):  # hold the last frame: 6 s
                run_.step(net.tau.period)
        x = torch.stack([f.image for f in fs])
        h1 = net.g1.module(x).detach().requires_grad_()
        z = net.z.module(h1)
        z_leaf = z.detach().requires_grad_()
        d = net.dynamics
        s = d.module(d.z_prev, d.s_prev).detach().requires_grad_()
        sr = net.sigreg
        assert sr.count + b >= 16  # SIGReg is on
        w = weight if stop else 1.0
        target = w * z_leaf + (1 - w) * z_leaf.detach()
        base = d.z_prev if residual else None
        energy = compare_energy(net.compare.module, s, target, 0.05, base)
        sig = sigreg(z_leaf, sr.buffer[: sr.count], sr.directions)
        loss = energy.mean() + 0.05 * sig
        g_z, g_s = torch.autograd.grad(b * loss, [z_leaf, s])
        (g_h1,) = torch.autograd.grad(z, h1, g_z)
        torch.testing.assert_close(net.z.lam, -g_z, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(d.lam, -g_s, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(net.g1.lam, -g_h1, rtol=1e-3, atol=1e-7)
        assert g_z.abs().max() > 1e-4  # the credit is not trivially zero
        for r in (net.g1.r, net.z.r, d.r):
            assert r.abs().max() < 1e-6

    @pytest.mark.parametrize(
        "model",
        [{}, {"open_loop": 0.5, "open_cycle": 3, "residual": True, "dyn_nodes": 2}],
        ids=["", "open_loop-residual-dyn_nodes"],
    )
    def test_identical_streams_are_one_stream(self, model):
        """B streams of the same frames and resets, with learning on: every
        sample has the activities of the one stream, and the parameters have
        moved as for the one stream (the mean of identical gradients). Few
        enough frames (B * frames < 16) that SIGReg, which sees the batch as more samples, is
        not on yet."""
        config = replace(
            SMALL,
            model=replace(SMALL.model, **model),
            time=TimeConfig(learn_interval=25e-3),
        )
        streams = 3
        one = Network(config)
        many = Network(replace(config, data=DataConfig(streams=streams)))
        video = Video(seed=3, episode=(100.0, 100.0))
        frames = [video.next() for _ in range(5)]
        with mujou.testing.run(one) as r1, mujou.testing.run(many) as r2:
            for k, f in enumerate(frames):
                reset = k in (0, 3)  # a second video from frame 3
                _hold(r1, one, _single_sample(f, reset))
                images = f.image.expand(streams, -1, -1, -1)
                _hold(r2, many, _batch_sample(images, [reset] * streams))
            z1, z2 = r1.read(one.z.next)["h"], r2.read(many.z.next)["h"]
            s1, s2 = r1.read(one.dynamics.next)["h"], r2.read(many.dynamics.next)["h"]
            p1, p2 = r1.read(one.compare.next), r2.read(many.compare.next)
        for b in range(streams):
            torch.testing.assert_close(z2[b], z1, rtol=1e-3, atol=1e-5)
            torch.testing.assert_close(s2[b], s1, rtol=1e-3, atol=1e-5)
            torch.testing.assert_close(
                p2["y_hat"][b], p1["y_hat"], rtol=1e-3, atol=1e-5
            )
        assert p1["mask"] and p2["mask"].all()
        for name in ("g1", "z", "dynamics", "compare"):
            got, want = getattr(many, name).module, getattr(one, name).module
            for a, b_ in zip(got.parameters(), want.parameters(), strict=True):
                torch.testing.assert_close(a, b_, rtol=1e-3, atol=1e-6)

    def test_streams_are_independent_when_only_one_resets(self):
        """With learning off, each stream of a batch predicts what it does
        alone, though only stream 1 resets at frame 3."""
        config = replace(
            SMALL,
            time=TimeConfig(learn_interval=math.inf),
            model=replace(SMALL.model, dyn_nodes=2),
        )
        both = Network(replace(config, data=DataConfig(streams=2)))
        alone = [Network(config), Network(config)]
        videos = [Video(seed=s, episode=(100.0, 100.0)) for s in (4, 5)]
        frames = [[v.next() for v in videos] for _ in range(6)]
        resets = [[k == 0, k in (0, 3)] for k in range(6)]
        preds, singles = [], [[], []]
        with (
            mujou.testing.run(both) as run_,
            mujou.testing.run(alone[0]) as run0,
            mujou.testing.run(alone[1]) as run1,
        ):
            for k in range(6):
                images = torch.stack([f.image for f in frames[k]])
                _hold(run_, both, _batch_sample(images, resets[k]))
                preds.append(run_.read(both.compare.next))
                if k == 3:
                    assert both.dynamics.has_prev.tolist() == [True, False]
                    assert both.dynamics.predictions.tolist() == [3, 0]
                    assert both.dynamics2.first.tolist() == [False, True]
                for i, (net, r) in enumerate(zip(alone, (run0, run1), strict=True)):
                    _hold(r, net, _single_sample(frames[k][i], resets[k][i]))
                    singles[i].append(r.read(net.compare.next))
        for k in range(6):
            for i in range(2):
                assert bool(preds[k]["mask"][i]) == bool(singles[i][k]["mask"])
                torch.testing.assert_close(
                    preds[k]["y"][i], singles[i][k]["y"], rtol=1e-3, atol=1e-5
                )
                if bool(singles[i][k]["mask"]):
                    torch.testing.assert_close(
                        preds[k]["y_hat"][i],
                        singles[i][k]["y_hat"],
                        rtol=1e-3,
                        atol=1e-5,
                    )

    @pytest.mark.parametrize("bp", [False, True], ids=["pc", "bp"])
    def test_open_predictions_are_decided_per_stream(self, bp):
        """Streams out of phase (stream 1 resets again at frame 2): in a cycle
        of 4 with half open, at some frame one stream remembers its own
        prediction of the frame before as z_{k-1} and the other the observed
        z; each row is the right one."""
        config = replace(
            SMALL,
            time=TimeConfig(learn_interval=math.inf),
            model=replace(SMALL.model, open_loop=0.5, open_cycle=4),
            data=DataConfig(streams=2),
        )
        net: Any = WorldBPNetwork(config) if bp else Network(config)
        videos = [Video(seed=s, episode=(100.0, 100.0)) for s in (6, 7)]
        node = net.world if bp else net.dynamics
        resets = [[k == 0, k in (0, 2)] for k in range(10)]
        counts = [0, 0]  # the predictions of each video so far
        last: dict[str, torch.Tensor] = {}
        differed = False
        with mujou.testing.run(net) as run_:
            for k in range(10):
                images = torch.stack([v.next().image for v in videos])
                _hold(run_, net, _batch_sample(images, resets[k]))
                is_open_rows = []
                for b in range(2):
                    counts[b] = 0 if resets[k][b] else counts[b] + 1
                    is_open_rows.append(counts[b] > 1 and is_open(counts[b], 0.5, 4))
                differed |= is_open_rows[0] != is_open_rows[1]
                if k:
                    for b in range(2):
                        if resets[k][b]:
                            continue  # no z_{k-1}: the memory is not read
                        want = last["z_hat"][b] if is_open_rows[b] else last["z"][b]
                        torch.testing.assert_close(node.z_prev[b], want)
                z = node.z_last if bp else run_.read(net.z.next)["h"]
                last = {"z": z, "z_hat": run_.read(net.next)["y_hat"]}
        assert differed


def _video_run(tmp_path, name: str, streams: int = 1, **changes) -> list[dict]:
    data = DataConfig(kind="video", seconds=2.0, streams=streams)
    config = replace(SMALL, name=name, data=data, **changes)
    path = tmp_path / f"{name}.json"
    path.write_text(config.to_json())
    out = run.main(["--config", str(path), "--out", str(tmp_path)])
    return [
        json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()
    ]


@pytest.mark.slow
class TestRunVideo:
    def test_evaluating_does_not_change_the_learning(self, tmp_path):
        with_eval = _video_run(tmp_path, "a", eval_interval=1.0)
        without = _video_run(tmp_path, "b", eval_interval=100.0)
        assert any("rank" in r for r in with_eval)
        assert [r.get("error") for r in with_eval] == [r.get("error") for r in without]

    @pytest.mark.parametrize("bp", [False, True], ids=["pc", "bp"])
    def test_a_checkpoint_evaluates_as_the_run_did(self, tmp_path, bp):
        """A checkpoint is written with each evaluation here; loaded, it gives
        the metrics the run logged at that time."""
        name = f"ckpt-{bp}"
        records = _video_run(
            tmp_path, name, bp=bp, eval_interval=1.0, checkpoint_interval=1.0
        )
        (out,) = tmp_path.glob(f"*-{name}")
        paths = sorted((out / "ckpt").glob("*.pt"))
        assert [p.name for p in paths] == ["00001.pt", "00002.pt"]
        config = Config.load(tmp_path / f"{name}.json")
        _, clips = run.source(config)
        d_state = config.model.d_s * config.model.dyn_nodes
        got = evaluate(run.load_checkpoint(paths[-1]), clips, d_state)
        logged = [r for r in records if "rank" in r][-1]
        for key in ("rank", "rollout10", "shrink10", "amp10"):
            assert got[key] == pytest.approx(logged[key], rel=1e-5)

    @pytest.mark.parametrize("bp", [False, True], ids=["pc", "bp"])
    def test_four_streams_run_through_and_log_the_mean(self, tmp_path, bp):
        records = _video_run(
            tmp_path, f"four-{bp}", streams=4, bp=bp, eval_interval=1.0
        )
        assert len(records) == 60
        errors = [r["error"] for r in records if "error" in r]
        assert len(errors) == 59  # every stream resets in the first frame only
        assert all(math.isfinite(e) for e in errors)
        assert all(math.isfinite(r["z_rms"]) for r in records)
        assert sum("rank" in r for r in records) == 2

    @pytest.mark.parametrize("bp", [False, True], ids=["pc", "bp"])
    def test_ten_seconds_of_video_stream_through(self, tmp_path, bp):
        config = replace(
            Config(name=f"ten-{bp}", bp=bp, eval_interval=5.0),
            data=DataConfig(kind="video", seconds=10.0),
        )
        path = tmp_path / "c.json"
        path.write_text(config.to_json())
        out = run.main(["--config", str(path), "--out", str(tmp_path)])
        records = [
            json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()
        ]
        assert len(records) == 300
        assert all(math.isfinite(r["error"]) for r in records if "error" in r)
        assert sum("rank" in r for r in records) == 2
