import json
import math
from dataclasses import replace

import mujou.pc as pc
import mujou.testing
import pytest
import torch

from experiments.continuous_stream import run
from experiments.continuous_stream.bp import WorldBPNetwork
from experiments.continuous_stream.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TimeConfig,
)
from experiments.continuous_stream.model import Network
from experiments.continuous_stream.nodes import compare_energy, sigreg
from experiments.continuous_stream.video import Video

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
    @pytest.mark.parametrize("stop", [False, True], ids=["", "stop_target"])
    def test_lam_is_the_credit_of_bp_at_the_fixed_point(self, device, stop):
        """Hold a frame with learning off: each node's lam relaxes to the
        credit -dL/dh of L_k (the prediction error and SIGReg), computed by
        autograd through the same modules, memory, buffer and directions;
        with ``stop_target``, through a detached target of the prediction."""
        config = replace(
            SMALL,
            time=TimeConfig(learn_interval=math.inf),
            model=replace(SMALL.model, stop_target=stop),
            device=device,
        )
        net = Network(config)
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
        target = z_leaf.detach() if stop else z_leaf
        loss = compare_energy(net.compare.module, s, target, 0.05) + 0.05 * sigreg(
            z_leaf, sr.buffer[: sr.count], sr.directions
        )
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
