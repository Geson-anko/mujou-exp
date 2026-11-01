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
from exp.video.evaluate import Snapshot
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


def _video_run(tmp_path, name: str, **changes) -> list[dict]:
    config = replace(
        SMALL, name=name, data=DataConfig(kind="video", seconds=2.0), **changes
    )
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
