"""The program of M16 (the vision and the dynamics learned together): the
dynamics sends the encoder no credit (its weights are the vision program's,
bit for bit), the dynamics is M14's, the files of the run contract, resume,
the frozen control, and the posthoc decoding of the rollouts."""

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, override

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import nn

import exp.video.programs.joint as joint_program
from exp.lib import ckpt
from exp.lib.host import Host
from exp.lib.optim import adam
from exp.lib.rundir import EnvError, Exit, verify_manifest
from exp.video import loop
from exp.video.compose import compose, from_dict
from exp.video.data.frames import BallFrames
from exp.video.data.physics import render
from exp.video.dyn.modules import build_dynamics
from exp.video.dyn.nets import StateWorldBP, StateWorldPC, make_optimizer, modules
from exp.video.dyn.nodes import Source
from exp.video.dyn.offline import OfflineFit, stream_windows
from exp.video.jepa.nets import build_vision_net
from exp.video.jepa.vit import build_vision
from exp.video.main import local_job
from exp.video.measure.joint import (
    Windows,
    ball_windows,
    image_rows,
    joint_dynamics,
    latent_errors,
)
from exp.video.measure.posthoc import Parent
from exp.video.programs.joint import (
    PROGRAM,
    JointBP,
    JointPC,
    _z_encoder,  # pyright: ignore[reportPrivateUsage]
    build_joint,
    image_objective,
    joint_objective,
    parent_ckpt,
    rebuild,
)
from exp.video.programs.posthoc import decoding
from exp.video.programs.vision import PROGRAM as VISION, STRIDE
from exp.video.schema.run import RunConf
from mujou import TensorSpec
from tests.video.programs.test_vision import TINY as VISION_TINY

TINY = [*VISION_TINY, "dynamics.d_s=16", "optim.warmup=0", "optim.lr=0.1"]
EXPERIMENT = "m16/balls_joint"
FRAMES = 12


@pytest.fixture(autouse=True)
def _no_gil_check(monkeypatch):
    """The GIL of this test process is pytest's."""
    monkeypatch.setattr(loop, "assert_free_threaded", lambda where: None)


def _conf(overrides: list[str] = (), experiment: str = EXPERIMENT) -> RunConf:  # type: ignore[assignment]
    return compose([f"+experiment={experiment}", *TINY, *overrides])


def _images(n: int, b: int, side: int = 16, seed: int = 0) -> torch.Tensor:
    return torch.rand(
        n, b, 1, side, side, generator=torch.Generator().manual_seed(seed)
    )


def _drive(net: Any, conf: RunConf, frames: int = FRAMES) -> None:
    """Show ``frames`` frames of random images to ``net``, learning."""
    b = conf.data.streams
    x = _images(frames, b)
    yes = torch.ones(b, dtype=torch.bool)
    with mujou.testing.run(net) as run:
        host = Host(run, frame=conf.time.frame, period=net.tau.period)
        for k in range(frames):
            reset = torch.full((b,), k == 0)
            sample = pc.msg.Sample(x=x[k], y=torch.zeros(b, 1), mask=yes, reset=reset)
            host.show({net.prev: sample}, net.stats)


def _weights(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class TestShowAll:
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_each_port_gets_its_own_message_in_the_order_asked(self, learner):
        conf = _conf([f"learner={learner}"])
        net = build_joint(conf)
        b = conf.data.streams
        sample = pc.msg.Sample(
            x=_images(1, b)[0],
            y=torch.zeros(b, 1),
            mask=torch.ones(b, dtype=torch.bool),
            reset=torch.ones(b, dtype=torch.bool),
        )
        with mujou.testing.run(net) as run:
            host = Host(run, frame=conf.time.frame, period=net.tau.period)
            (p, q), written = host.show_all({net.prev: sample}, [net.stats, net.next])
            (q2, p2), _ = host.show_all({net.prev: sample}, [net.next, net.stats])
        assert "loss" in p and "y_hat" not in p
        assert "y_hat" in q and "loss" not in q
        assert "loss" in p2 and "y_hat" in q2
        assert p["time"] == q["time"] == written


RECON = ["vision.recon=10.0", "vision.recon_decoder=broadcast"]
JEPA = ["vision.recon=0.0"]  # the JEPA loss alone (the preset reconstructs)
# The learning on windows (BP), small enough for FRAMES: windows at the pushes
# 4, 7 and 10.
WINDOW = [
    "+window@dynamics.window=m16",
    "dynamics.window.window=4",
    "dynamics.window.observed=2",
    "dynamics.window.stride=3",
    "dynamics.window.relax=0.01",
]


class TestNoCreditToTheEncoder:
    @pytest.mark.parametrize(
        ("frames", "recon", "learner", "window"),
        [
            *[
                (frames, recon, learner, [])
                for frames, recon in [(1, JEPA), (2, JEPA), (1, RECON)]
                for learner in ("pc", "bp")
            ],
            (1, JEPA, "bp", WINDOW),
        ],
        ids=[
            *[f"{n}-{lr}" for n in ("n1", "n2", "recon") for lr in ("pc", "bp")],
            "n1-bp-window",
        ],
    )
    def test_the_encoder_is_the_vision_programs_bit_for_bit(
        self, learner, frames, recon, window
    ):
        """The dynamics' error is not a term of the encoder's loss: the same
        seed and data give the vision of M15 and the joint network the same
        encoder, predictor (and decoder of the reconstruction) after K frames,
        however much the dynamics learned."""
        conf = _conf(
            [
                f"learner={learner}",
                f"vision.frames={frames}",
                "optim_vision.warmup=0",  # so that the encoder does move
                *recon,
                *window,
            ]
        )
        alone = build_vision_net(conf)
        init = _weights(alone.mods.encoder)
        _drive(alone, conf)
        assert not _same(init, _weights(alone.mods.encoder))  # the encoder learned
        joint = build_joint(conf)
        assert isinstance(joint, JointPC if learner == "pc" else JointBP)
        before = _weights(joint.dyn.step)
        _drive(joint, conf)
        assert not _same(before, _weights(joint.dyn.step))  # the dynamics learned
        if window:
            assert isinstance(joint, JointBP) and joint.world.win_steps == 3
        # nothing flows back from the dynamics: no .grad, and the z it reads
        # has no graph
        for module in (joint.mods.encoder, joint.mods.predictor, joint.dyn.step):
            assert all(p.grad is None for p in module.parameters())
        assert _same(_weights(alone.mods.encoder), _weights(joint.mods.encoder))
        assert _same(_weights(alone.mods.predictor), _weights(joint.mods.predictor))
        if recon is RECON:
            assert alone.mods.decoder is not None and joint.mods.decoder is not None
            assert _same(_weights(alone.mods.decoder), _weights(joint.mods.decoder))

    def test_the_initial_encoder_is_that_of_the_vision_of_the_same_seed(self):
        conf = _conf(["seed=3"])
        net = build_joint(conf)
        mods = build_vision(conf, seed=3)
        assert _same(_weights(net.mods.encoder), _weights(mods.encoder))
        # and the dynamics' are drawn from seed + 1, not from the seed
        assert conf.vision is not None and conf.dynamics is not None
        d = conf.vision.dim
        other = build_dynamics(conf.dynamics, d, d, seed=3)
        again = build_dynamics(conf.dynamics, d, d, seed=4)
        assert not _same(_weights(net.dyn.step), _weights(other.step))
        assert _same(_weights(net.dyn.step), _weights(again.step))


class TestDynamicsIsM14s:
    def test_the_upstream_is_what_the_state_source_sends(self):
        """The encoder's ``z_out`` has the spec of M14's ``Source.out``, so the
        block is connected to it as it is to the source."""
        conf = _conf(["learner=pc"])
        net = build_joint(conf)
        assert isinstance(net, JointPC)
        b, d = conf.data.streams, conf.vision.dim  # type: ignore[union-attr]
        source = Source(x=TensorSpec((b, d)), period=net.tau.period, batch=b)
        assert net.z_out.send_spec == source.out.send_spec
        assert net.z_out.send_spec == net.block.dynamics.z.recv_spec

    @pytest.mark.parametrize("trace", [False, True])
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_a_fixed_z_sequence_gives_m14s_predictions_and_weights(
        self, learner, trace, tmp_path
    ):
        """With the encoder frozen, the z it sends in each frame is a
        constant; fed to M14's ``Source -> dynamics`` as observations (the same
        weights), the dynamics predicts the same and learns the same, to the
        bit."""
        flag = str(trace).lower()
        conf = _conf(
            [f"learner={learner}", "vision.frozen=true", f"dynamics.trace={flag}"]
        )
        net = build_joint(conf, frozen_from=_frozen_ckpt(tmp_path))
        init = _weights(net.dyn.step)
        b = conf.data.streams
        x, yes = _images(FRAMES, b), torch.ones(b, dtype=torch.bool)
        zs, preds = [], []
        with mujou.testing.run(net) as run:
            host = Host(run, frame=conf.time.frame, period=net.tau.period)
            for k in range(FRAMES):
                reset = torch.full((b,), k == 0)
                sample = pc.msg.Sample(
                    x=x[k], y=torch.zeros(b, 1), mask=yes, reset=reset
                )
                p, _ = host.show({net.prev: sample}, net.next)
                preds.append(p)
                zs.append(run.read(net.z_out)["h"].clone())
        assert not _same(init, _weights(net.dyn.step))  # it learned
        m14 = compose(
            ["+experiment=m14/s2", "dynamics.d_s=16", f"dynamics.trace={flag}"]
        )
        m14.data.streams = b
        m14.learner.kind = learner
        m14.dynamics = conf.dynamics  # the preset's residual and open loop
        kind = StateWorldPC if learner == "pc" else StateWorldBP
        world = kind(
            joint_dynamics(conf),
            conf=m14,
            optimizer=make_optimizer(conf.optim, conf.time.frame),
        )
        ref = []
        with mujou.testing.run(world) as run:
            host = Host(run, frame=conf.time.frame, period=world.tau.period)
            for k in range(FRAMES):
                reset = torch.full((b,), k == 0)
                sample = pc.msg.Sample(
                    x=zs[k], y=torch.zeros(b, 1), mask=~reset, reset=reset
                )
                p, _ = host.show({world.prev: sample}, world.next)
                ref.append(p)
        for a, r in zip(preds, ref, strict=True):
            assert torch.equal(a["y_hat"], r["y_hat"])
            assert torch.equal(a["mask"], r["mask"])
        for name in ("step", "readout"):
            ma, mr = getattr(net.dyn, name), getattr(modules(world), name)
            assert _same(_weights(ma), _weights(mr)), name


def _frozen_ckpt(root: Path, seed: int = 7) -> Path:
    """An evaluation checkpoint of an M15 run of these settings and another
    seed, ``<root>/parent/ckpt/00001.0.pt``."""
    mods = build_vision(_conf([f"seed={seed}"]), seed=seed)
    return ckpt.save_eval(
        root / "parent",
        program="vision",
        group="g",
        t=1.0,
        frame=30,
        modules=mods.named(),
    )


def _finished_parent(out: Path) -> Path:
    """A finished M15 run of these settings (seed 0) with the checkpoint
    ``ckpt/00001.0.pt``."""
    out.mkdir(parents=True)
    child = _conf()
    parent = compose(["+experiment=m15/balls_n1", *VISION_TINY])
    same = [
        f"{group}.{f.name}={json.dumps(getattr(c, f.name))}"
        for group in ("vision", "optim_vision")
        if (c := getattr(child, group)) is not None
        and (p := getattr(parent, group)) is not None
        for f in dataclasses.fields(c)
        if getattr(c, f.name) != getattr(p, f.name)
    ]
    job = local_job(
        ["+experiment=m15/balls_n1", *VISION_TINY, *same, "ctl.ckpt_every=1"], out
    )
    assert loop.run(VISION, job, attempt=1) == Exit.OK
    return out


def _child(out: Path, parent: str) -> Path:
    """The directory of a frozen joint run whose job names ``parent``."""
    out.mkdir(parents=True)
    job = local_job([f"+experiment={EXPERIMENT}", *TINY, "vision.frozen=true"], out)
    data = json.loads(job.read_text())
    data["parent"] = parent
    job.write_text(json.dumps(data))
    return out


class TestFrozen:
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_the_encoder_is_the_checkpoints_and_never_learns(self, learner, tmp_path):
        conf = _conf([f"learner={learner}", "vision.frozen=true"])
        path = _frozen_ckpt(tmp_path)
        net = build_joint(conf, frozen_from=path)
        saved = torch.load(path, weights_only=True)["modules"]
        assert _same(_weights(net.mods.encoder), saved["encoder"])
        init = _weights(build_vision(conf, seed=conf.seed).encoder)
        assert not _same(init, saved["encoder"])
        dyn = _weights(net.dyn.step)
        _drive(net, conf)
        assert _same(_weights(net.mods.encoder), saved["encoder"])
        assert _same(_weights(net.mods.predictor), saved["predictor"])
        assert not _same(dyn, _weights(net.dyn.step))  # the dynamics learns

    def test_it_needs_a_checkpoint(self):
        with pytest.raises(ValueError, match="checkpoint"):
            build_joint(_conf(["vision.frozen=true"]))

    def test_the_parent_names_the_checkpoint_of_the_job(self, tmp_path):
        parent = _finished_parent(tmp_path / "runs" / "b" / "parent")
        conf = _conf(["vision.frozen=true"])
        run_dir = _child(tmp_path / "runs" / "b" / "child", "b/parent")
        assert parent_ckpt(run_dir, conf) == parent / "ckpt" / "00001.0.pt"
        (parent / "ckpt" / "00000.5.pt").touch()  # an earlier one: not the last
        assert parent_ckpt(run_dir, conf).name == "00001.0.pt"

    def test_the_parent_must_exist_and_have_a_checkpoint(self, tmp_path):
        parent = _finished_parent(tmp_path / "runs" / "b" / "parent")
        conf = _conf(["vision.frozen=true"])
        run_dir = _child(tmp_path / "runs" / "b" / "child", "b/nowhere")
        with pytest.raises(EnvError, match="not here"):
            parent_ckpt(run_dir, conf)
        run_dir = _child(tmp_path / "runs" / "b" / "child2", "b/parent")
        (parent / "ckpt" / "00001.0.pt").unlink()
        with pytest.raises(EnvError, match="no evaluation checkpoint"):
            parent_ckpt(run_dir, conf)
        (run_dir / "job.json").write_text(json.dumps({}))
        with pytest.raises(ValueError, match="parent"):
            parent_ckpt(run_dir, conf)

    def test_the_parent_must_have_finished(self, tmp_path):
        parent = _finished_parent(tmp_path / "runs" / "b" / "parent")
        (parent / "result.json").unlink()
        run_dir = _child(tmp_path / "runs" / "b" / "child", "b/parent")
        with pytest.raises(ValueError, match="not finished"):
            parent_ckpt(run_dir, _conf(["vision.frozen=true"]))

    @pytest.mark.parametrize(
        ("override", "differs"),
        [
            ("seed=1", "seed"),
            ("vision.pixel_std=2.0", "vision"),
            ("learner=pc", "learner"),
            ("optim_vision.lr=1e-3", "optim_vision"),
            ("data.balls.size=20", "data.balls"),
        ],
    )
    def test_the_parent_must_be_of_the_run_it_controls(
        self, override, differs, tmp_path
    ):
        """A parent of another seed, normalization, learner or optimizer has
        the same shapes and loads without a word, so ``parent_ckpt`` compares."""
        _finished_parent(tmp_path / "runs" / "b" / "parent")
        run_dir = _child(tmp_path / "runs" / "b" / "child", "b/parent")
        conf = _conf(["vision.frozen=true", override])
        with pytest.raises(ValueError, match=f"differs from this run in {differs}"):
            parent_ckpt(run_dir, conf)

    def test_the_parent_must_be_a_vision_run(self, tmp_path):
        parent = tmp_path / "runs" / "b" / "parent"
        parent.mkdir(parents=True)
        assert _run(parent, ["ctl.ckpt_every=1"]) == Exit.OK  # a joint run
        run_dir = _child(tmp_path / "runs" / "b" / "child", "b/parent")
        with pytest.raises(ValueError, match="not 'vision'"):
            parent_ckpt(run_dir, _conf(["vision.frozen=true"]))

    def test_a_joint_checkpoint_is_not_a_parent(self, tmp_path):
        mods = build_vision(_conf(), seed=0)
        path = ckpt.save_eval(
            tmp_path,
            program="joint",
            group="g",
            t=1.0,
            frame=30,
            modules=mods.named(),
        )
        with pytest.raises(ValueError, match="not 'vision'"):
            build_joint(_conf(["vision.frozen=true"]), frozen_from=path)

    def test_the_vision_program_refuses_a_frozen_encoder(self, tmp_path):
        job = local_job(
            ["+experiment=m15/balls_n1", *VISION_TINY, "vision.frozen=true"],
            tmp_path,
        )
        with pytest.raises(ValueError, match="joint program"):
            VISION.build(
                from_dict(json.loads(job.read_text())["config"]), tmp_path, None
            )

    def test_a_frozen_run_ends_with_the_parents_encoder(self, tmp_path):
        parent = _finished_parent(tmp_path / "runs" / "b" / "parent")
        out = _child(tmp_path / "runs" / "b" / "child", "b/parent")
        job = out / "job.json"
        data = json.loads(job.read_text())
        data["config"]["ctl"]["ckpt_every"] = 1
        job.write_text(json.dumps(data))
        assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
        a = torch.load(parent / "ckpt" / "00001.0.pt", weights_only=True)["modules"]
        b = torch.load(out / "ckpt" / "00001.0.pt", weights_only=True)["modules"]
        assert _same(a["encoder"], b["encoder"])
        assert _same(a["predictor"], b["predictor"])
        assert set(b) == {"encoder", "predictor", "decoder", "step", "readout"}
        assert _same(a["decoder"], b["decoder"])
        assert json.loads((out / "frozen_from.json").read_text()) == {
            "parent": "parent",
            "ckpt": "00001.0.pt",
        }


def _run(
    out: Path, overrides: list[str], program: Any = PROGRAM, attempt: int = 1
) -> int:
    job = local_job([f"+experiment={EXPERIMENT}", *TINY, *overrides], out)
    return loop.run(program, job, attempt=attempt)


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class _Stopping:
    """The program, with ``stop.request`` made after frame ``at``."""

    def __init__(self, at: int) -> None:
        self.at = at

    def build(self, conf, run_dir: Path, resume):
        built = PROGRAM.build(conf, run_dir, resume)
        self.built: Any = built  # to look at its network after the stop
        advance, count = built.advance, [0]

        def stopping() -> None:
            advance()
            count[0] += 1
            if count[0] == self.at:
                (run_dir / "stop.request").touch()

        built.advance = stopping
        return built


class TestContract:
    def test_a_small_job_writes_every_file_of_the_contract(self, tmp_path):
        out = tmp_path / "run"
        confirm = ["ctl.ckpt_every=0.5", "ctl.viz_every=1", "ctl.eval_test=true"]
        assert _run(out, ["learner=pc", *confirm]) == Exit.OK
        for name in (
            "job.json",
            "config.yaml",
            "CODE.json",
            "code.diff",
            "provenance/attempt-1.json",
            "status.json",
            "timing.jsonl",
            "stream.jsonl",
            "evals.jsonl",
            "baselines.json",
            "result.json",
            "manifest.json",
            "ckpt/00000.5.pt",
            "ckpt/00001.0.pt",
            "viz/00001.0/views.png",
        ):
            assert (out / name).is_file(), name
        verify_manifest(out)
        stream = _lines(out / "stream.jsonl")
        assert set(stream[0]) == {
            "t",
            "frames",
            "loss",
            "pred",
            "sigreg",
            "recon",
            "preq",
            "preq_open",
        }
        val = _lines(out / "evals.jsonl")[0]
        for key in (
            "loss",
            "rank",
            "z_spread",
            "probe_pos_min",
            "roll1",
            "copy1",
            "mean1",
            "roll20",
            "copy20",
            "mean20",
            "roll10/mean10",
            "objective",
        ):
            assert key in val, key
        result = json.loads((out / "result.json").read_text())
        assert result["status"] == "completed"
        conf = from_dict(json.loads((out / "job.json").read_text())["config"])
        header = ckpt.load_eval(out / "ckpt" / "00001.0.pt", rebuild(conf))
        assert header["program"] == "joint"

    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_the_same_job_twice_writes_the_same_bytes(self, tmp_path, learner):
        for d in ("a", "b"):
            assert _run(tmp_path / d, [f"learner={learner}"]) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl", "baselines.json"):
            assert (tmp_path / "a" / name).read_bytes() == (
                tmp_path / "b" / name
            ).read_bytes()

    def test_the_bench_task_evaluates_nothing(self, tmp_path):
        out = tmp_path / "run"
        assert _run(out, ["task=bench", "seconds=0.5"]) == Exit.OK
        result = json.loads((out / "result.json").read_text())
        assert set(result["final"]) == {"timing"}
        assert not (out / "evals.jsonl").read_text()


class TestResume:
    @pytest.mark.parametrize("recon", [JEPA, RECON], ids=["jepa", "recon"])
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_resume_equals_uninterrupted(self, tmp_path, learner, recon):
        """Stopped at frame 23 (past the first evaluation) and resumed, the
        run writes the bytes of one never stopped: both learners' states, the
        dynamics' memory and traces among them."""
        extra = [
            *recon,
            f"learner={learner}",
            "vision.frames=2",
            "optim_vision.warmup=0.5",
            "optim_vision.lr=1e-3",
            "ctl.resume_every=0.3",
            "ctl.log_every=0.2",
        ]
        assert _run(tmp_path / "whole", extra) == Exit.OK
        cut = tmp_path / "cut"
        assert _run(cut, extra, _Stopping(23)) == Exit.STOPPED
        assert (cut / "resume" / "00000023.pt").is_file()
        (cut / "stop.request").unlink()
        assert _run(cut, extra, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (cut / name).read_bytes() == (tmp_path / "whole" / name).read_bytes()
        logged = _lines(cut / "stream.jsonl")[-1]
        assert ("recon" in logged) == (recon is RECON)

    def test_a_run_learning_on_windows_resumes_as_one_never_stopped(self, tmp_path):
        """BP on windows, stopped at frame 23 with a window pending (due at
        push 22, learned 0.08 s later) and resumed: the bytes of one never
        stopped, the window records among them."""
        extra = [
            "learner=bp",
            "+window@dynamics.window=m16",
            "dynamics.window.window=6",
            "dynamics.window.observed=3",
            "dynamics.window.stride=4",
            "dynamics.window.relax=0.08",
            "optim=adamw",
            "optim.lr=1e-3",
            "optim_vision.lr=1e-3",
            "ctl.resume_every=0.3",
            "ctl.log_every=0.2",
        ]
        assert _run(tmp_path / "whole", extra) == Exit.OK
        cut = tmp_path / "cut"
        stopping = _Stopping(23)
        assert _run(cut, extra, stopping) == Exit.STOPPED
        assert stopping.built.net.world.pending is not None
        (cut / "stop.request").unlink()
        assert _run(cut, extra, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (cut / name).read_bytes() == (tmp_path / "whole" / name).read_bytes()
        lines = _lines(cut / "stream.jsonl")
        assert lines[0]["win_steps"] == 0 and lines[0]["win_loss"] is None
        # 30 frames: windows at the pushes 6, 10, ..., 30, the last one not
        # learned before the end
        last = lines[-1]
        assert last["win_steps"] == 6 and last["win_skipped"] == 0
        assert last["win_masked"] == 0
        assert last["win_loss"] > 0 and last["win_gnorm"] > 0

    def test_the_pc_learner_has_no_windows_yet(self):
        conf = _conf(["learner=pc", *WINDOW])
        with pytest.raises(ValueError, match="the PC-ALM window learning is not built"):
            build_joint(conf)


def test_the_objective():
    """``roll / mean`` at the horizon, at most PENALTY, PENALTY on a collapse or
    a value that is not finite."""
    from exp.video.measure.objective import PENALTY

    ok = {"z_spread": 0.1, "roll10/mean10": 0.5}
    assert joint_objective(ok, 10) == pytest.approx(0.5)
    assert joint_objective(ok | {"roll10/mean10": 99.0}, 10) == PENALTY
    assert joint_objective(ok | {"roll10/mean10": float("nan")}, 10) == PENALTY
    assert joint_objective(ok | {"z_spread": 0.0}, 10) == PENALTY
    assert joint_objective(ok, 20) == PENALTY


def test_the_image_objective():
    """The max of the logs of ``rec_roll/pix`` at 5 and 10 (the conjunction of
    the image criterion): ratios 0.5 and 2 give log 2 (a gain at 5 does not
    make up for a loss at 10), ratios 0.65 and 0.9 are below 0; PENALTY on a collapse, a missing or a ratio that is not finite
    or not positive."""
    from exp.video.measure.objective import PENALTY

    ok = {"z_spread": 0.1, "rec_roll5/pix5": 0.5, "rec_roll10/pix10": 2.0}
    assert image_objective(ok, [5, 10]) == pytest.approx(math.log(2.0))
    both = ok | {"rec_roll5/pix5": 0.65, "rec_roll10/pix10": 0.9}
    assert image_objective(both, [5, 10]) == pytest.approx(math.log(0.9))
    assert image_objective(both | {"rec_roll10/pix10": 1.1}, [5, 10]) > 0
    assert image_objective(ok, [5]) == pytest.approx(math.log(0.5))
    assert image_objective(ok | {"rec_roll10/pix10": 1e9}, [5, 10]) == PENALTY
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        assert image_objective(ok | {"rec_roll5/pix5": bad}, [5, 10]) == PENALTY
    assert image_objective(ok | {"z_spread": 0.0}, [5, 10]) == PENALTY
    assert image_objective(ok, [5, 20]) == PENALTY


HORIZONS = ["measure.horizons=[1,5,10]"]


def _built(tmp_path: Path, overrides: list[str]):
    conf = _conf([*HORIZONS, *overrides])
    tmp_path.mkdir(parents=True, exist_ok=True)
    built = PROGRAM.build(conf, tmp_path, None)
    return conf, built


class _Frames(nn.Module):
    """A decoder whose standardized output is the standardized last observed
    frame of the windows, whatever z."""

    def __init__(self, frame: torch.Tensor, mean: float, std: float) -> None:
        super().__init__()
        self.target = (frame - mean) / std

    @override
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.target[: len(z)]


class TestDecodedRows:
    def test_the_row_has_the_decoded_keys_only_with_a_decoder(self, tmp_path):
        conf, built = _built(tmp_path / "rec", RECON)
        try:
            row = built.evaluate("val")  # type: ignore[attr-defined]
            assert "rec_recon" in row
            for h in (1, 5, 10):
                for k in (f"rec_roll{h}", f"pix_copy{h}", f"pix_mean{h}"):
                    assert k in row, k
                assert row[f"rec_roll{h}/pix{h}"] == pytest.approx(
                    row[f"rec_roll{h}"] / min(row[f"pix_copy{h}"], row[f"pix_mean{h}"])
                )
            assert row["objective"] == pytest.approx(
                max(math.log(row["rec_roll5/pix5"]), math.log(row["rec_roll10/pix10"]))
            )
            assert not any(k.startswith("img_") for k in row)
        finally:
            built.close()  # type: ignore[attr-defined]
        _, built = _built(tmp_path / "jepa", JEPA)
        try:
            row = built.evaluate("val")  # type: ignore[attr-defined]
            assert not any(k.startswith(("rec_", "pix_")) for k in row)
            assert row["objective"] == joint_objective(row, 10)
        finally:
            built.close()  # type: ignore[attr-defined]

    def test_the_rows_are_image_rows_of_the_unstandardized_decoder(self, tmp_path):
        conf, built = _built(
            tmp_path, [*RECON, "vision.pixel_mean=0.3", "vision.pixel_std=2.0"]
        )
        try:
            row = built.evaluate("val")  # type: ignore[attr-defined]
            mods, dyn = built.mods, built.dyn  # type: ignore[attr-defined]
            w = ball_windows(conf, "val", stride=STRIDE)
            z = w.z(mods)
            _, pred = latent_errors(
                dyn,
                z,
                observed=w.observed,
                horizons=conf.measure.horizons,
                residual=conf.dynamics.residual,  # type: ignore[union-attr]
            )

            def raw(x: torch.Tensor) -> torch.Tensor:
                return mods.decoder(x) * 2.0 + 0.3

            with torch.no_grad():
                by_hand, _ = image_rows(
                    raw,  # type: ignore[arg-type]
                    w,
                    z,
                    pred,
                    conf.measure.horizons,
                    prefix="rec_",
                )
            for k, v in by_hand.items():
                assert row[k] == pytest.approx(v, rel=1e-5), k
        finally:
            built.close()  # type: ignore[attr-defined]

    def test_a_decoder_that_gives_the_standardized_frame_has_no_error(self, tmp_path):
        """Decoding the loss's own target (``(x - mean) / std``) gives back the
        raw frame: ``rec_recon`` is 0 and the decoded rollout is the copy."""
        conf, built = _built(
            tmp_path, [*RECON, "vision.pixel_mean=0.3", "vision.pixel_std=2.0"]
        )
        try:
            w = ball_windows(conf, "val", stride=STRIDE)
            assert len(w.images) < 256  # one decoder batch
            built.mods.decoder = _Frames(w.frame(0), 0.3, 2.0)  # type: ignore[attr-defined]
            row = built.evaluate("val")  # type: ignore[attr-defined]
            assert row["rec_recon"] == pytest.approx(0, abs=1e-10)
            for h in (1, 5, 10):
                assert row[f"rec_roll{h}"] == pytest.approx(
                    row[f"pix_copy{h}"], rel=1e-5
                )
        finally:
            built.close()  # type: ignore[attr-defined]

    def test_evaluating_changes_nothing_it_trains(self, tmp_path):
        _, built = _built(tmp_path, RECON)
        try:
            names = {**built.mods.named(), "step": built.dyn.step}  # type: ignore[attr-defined]
            before = {k: _weights(m) for k, m in names.items()}
            rng = torch.get_rng_state()
            a = built.evaluate("val")  # type: ignore[attr-defined]
            b = built.evaluate("val")  # type: ignore[attr-defined]
            assert a == b
            assert torch.equal(rng, torch.get_rng_state())
            assert all(_same(before[k], _weights(m)) for k, m in names.items())
        finally:
            built.close()  # type: ignore[attr-defined]


class TestPosthoc:
    def test_the_decoder_decodes_the_rollouts_into_images(self, tmp_path):
        """For a joint run the lines of the decoding also have the image keys
        at each horizon, and the PNG is the true frames over the decoded
        rollouts."""
        out = tmp_path / "run"
        assert _run(out, ["ctl.ckpt_every=1", "measure.horizons=[1,5]"]) == Exit.OK
        parent = Parent.read(out, device="cpu")
        d = decoding(parent, tmp_path / "post", frames=16, steps=2)
        lines = [line for _ in range(len(d)) for line in d.step()]
        assert [x["split"] for x in lines] == ["val", "test"]
        for key in (
            "recon",
            "pix_mean",
            "img_recon",
            "img_roll1",
            "pix_copy1",
            "pix_mean1",
            "img_roll5",
            "pix_copy5",
            "pix_mean5",
        ):
            assert key in lines[0], key
        assert (tmp_path / "post" / "00001.0.png").is_file()


OFFLINE = [
    "offline.frames=72",
    "offline.window=36",
    "offline.steps=6",
    "offline.batch=4",
    "offline.every=2",
]


def _offline(
    out: Path,
    parent: str,
    extra: list[str] = (),  # type: ignore[assignment]
    experiment: str = "m16/offline",
) -> Path:
    """The job of an offline bound in ``out`` whose job names ``parent``."""
    out.mkdir(parents=True)
    job = local_job([f"+experiment={experiment}", *TINY, *OFFLINE, *extra], out)
    data = json.loads(job.read_text())
    data["parent"] = parent
    job.write_text(json.dumps(data))
    return job


class TestOffline:
    def test_the_fit_sees_the_z_the_evaluation_sees(self):
        """The windows of states of the fit, rendered and encoded per batch
        (720 frames: three batches), are the $z$ ``Windows.z`` has of the same
        frames, and those frames are the ones the online run streams."""
        conf = _conf()
        balls = conf.data.balls
        assert balls is not None
        mods = build_vision(conf, seed=0)
        states = stream_windows(conf, 360, 36, states=True)
        assert states.shape[:2] == (20, 36)
        images = render(states.flatten(0, 1), balls.size, balls.radii).float()
        images = images.unflatten(0, states.shape[:2])
        w = Windows(images, 1, 16, 20)
        assert torch.equal(_z_encoder(conf, mods)(states), w.z(mods))
        frames = BallFrames.of(
            balls, conf.data.streams, frame=conf.time.frame, seed=conf.seed
        )
        online = torch.stack([frames.next()[0] for _ in range(36)], 1)
        assert torch.equal(images[::10], online)  # window 0 of each stream

    def test_only_the_dynamics_learns(self, tmp_path):
        """The bound ends with its parent's encoder, predictor and decoder and
        with dynamics fit away from their initial weights; its val rows are
        scored by the image objective."""
        parent = _finished_parent(tmp_path / "runs" / "b" / "parent")
        job = _offline(
            tmp_path / "runs" / "b" / "child", "b/parent", ["ctl.ckpt_every=1"]
        )
        out = job.parent
        assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
        rows = _lines(out / "evals.jsonl")
        assert [x["step"] for x in rows if x["split"] == "val"] == [2, 4, 6]
        val = [x for x in rows if x["split"] == "val"]
        assert all(x["objective"] == image_objective(x, [5, 10]) for x in val)
        a = torch.load(parent / "ckpt" / "00001.0.pt", weights_only=True)["modules"]
        last = sorted((out / "ckpt").glob("*.pt"))[-1]
        b = torch.load(last, weights_only=True)["modules"]
        for name in ("encoder", "predictor", "decoder"):
            assert _same(a[name], b[name]), name
        init = joint_dynamics(_conf())
        assert not _same(_weights(init.step), b["step"])
        assert json.loads((out / "frozen_from.json").read_text()) == {
            "parent": "parent",
            "ckpt": "00001.0.pt",
        }

    def test_it_needs_a_frozen_encoder(self, tmp_path):
        conf = compose(
            ["+experiment=m16/offline", *TINY, *OFFLINE, "vision.frozen=false"]
        )
        with pytest.raises(ValueError, match="vision.frozen"):
            PROGRAM.build(conf, tmp_path, None)

    def test_a_window_shorter_than_the_horizons_is_an_error(self, tmp_path):
        conf = compose(
            ["+experiment=m16/offline", *TINY, *OFFLINE, "offline.window=35"]
        )
        with pytest.raises(ValueError, match="offline.window=35"):
            PROGRAM.build(conf, tmp_path, None)

    def test_resume_equals_uninterrupted(self, tmp_path):
        _finished_parent(tmp_path / "runs" / "b" / "parent")
        whole = _offline(tmp_path / "runs" / "b" / "whole", "b/parent")
        assert loop.run(PROGRAM, whole, attempt=1) == Exit.OK
        cut = _offline(tmp_path / "runs" / "b" / "cut", "b/parent")
        assert loop.run(_Stopping(3), cut, attempt=1) == Exit.STOPPED
        (cut.parent / "stop.request").unlink()
        assert loop.run(PROGRAM, cut, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (cut.parent / name).read_bytes() == (
                whole.parent / name
            ).read_bytes()


STREAM = [
    "offline.steps=3",
    "offline.every=1",
    "optim.lr=5.5e-05",
]  # (72 - 36) // 18 + 1 steps


class _SpyFit(OfflineFit):
    """An ``OfflineFit`` that keeps what the program built it from."""

    seen: dict[str, Any] = {}

    def __init__(self, mods, windows, **kw):
        type(self).seen = {"windows": windows, **kw}
        super().__init__(mods, windows, **kw)


class TestOfflineStream:
    def test_the_preset_steps_the_optimizer_of_the_online_run_by_s_frames(self):
        conf = compose(["+experiment=m16/offline_stream", *TINY, *OFFLINE, *STREAM])
        off = conf.offline
        assert off is not None and conf.dynamics is not None
        assert (off.order, off.stride, off.window, off.feedback) == (
            "stream",
            18,
            36,
            "grad",
        )
        assert conf.dynamics.nodes == 3  # big
        full = compose(["+experiment=m16/offline_stream"])
        assert full.offline is not None and full.optim is not None
        assert full.offline.steps == (full.offline.frames - 36) // 18 + 1

    def test_the_program_wires_the_stream_order_and_the_time_of_the_optimizer(
        self, tmp_path, monkeypatch
    ):
        """What the program hands ``OfflineFit``: each stream whole (the
        windows of ``stream_windows`` end to end), the optimizer of ``optim``
        and ``elapsed = stride frame``; those step as ``adam(1e-3, 0.9, 0.999)``
        once per window, on the gradients of the program's own fit."""
        monkeypatch.setattr(joint_program, "OfflineFit", _SpyFit)
        _finished_parent(tmp_path / "runs" / "b" / "parent")
        job = _offline(
            tmp_path / "runs" / "b" / "child",
            "b/parent",
            [*STREAM, "optim.lr=5.5555555555555556e-05"],
            "m16/offline_stream",
        )
        assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
        kw = _SpyFit.seen
        conf = compose(
            ["+experiment=m16/offline_stream", *TINY, *OFFLINE, *STREAM]
            + ["optim.lr=5.5555555555555556e-05"]
        )
        assert (kw["order"], kw["window"], kw["stride"]) == ("stream", 36, 18)
        assert (kw["feedback"], kw["credit"]) == ("grad", "bptt")
        data = kw["windows"]
        assert data.shape[1] == 72 and data.shape[0] == conf.data.streams
        cut = stream_windows(conf, 72, 36, states=True)  # stream after stream
        for b in range(data.shape[0]):
            assert torch.equal(data[b, :36], cut[2 * b])
            assert torch.equal(data[b, 36:], cut[2 * b + 1])
        frame = conf.time.frame
        assert kw["elapsed"] == pytest.approx(18 * frame)
        online, ref = torch.zeros(7), torch.zeros(7)
        a = kw["optimizer"].bind([online])
        b = adam(1e-3, 0.9, 0.999).bind([ref])
        g = torch.Generator().manual_seed(0)
        for _ in range(6):
            grad = torch.randn(7, generator=g)
            a.step_([grad], kw["elapsed"])
            b.step_([grad], 1.0)
            torch.testing.assert_close(online, ref, rtol=1e-6, atol=0)

    def test_the_fit_runs_the_steps_of_the_streams(self, tmp_path):
        _finished_parent(tmp_path / "runs" / "b" / "parent")
        job = _offline(
            tmp_path / "runs" / "b" / "child",
            "b/parent",
            STREAM,
            "m16/offline_stream",
        )
        assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
        rows = _lines(job.parent / "evals.jsonl")
        assert [x["step"] for x in rows if x["split"] == "val"] == [1, 2, 3]
        logs = _lines(job.parent / "stream.jsonl")
        assert all(x["grad_norm"] > 0 for x in logs if "grad_norm" in x)
        assert any("grad_norm" in x for x in logs)

    def test_more_steps_than_windows_is_an_error_before_any_step(self, tmp_path):
        """``(72 - 36) // 18 + 1 = 3`` steps: the configuration is refused when
        composed, not after the steps."""
        with pytest.raises(ValueError, match="offline.steps=4"):
            _offline(
                tmp_path / "runs" / "b" / "child",
                "b/parent",
                ["offline.steps=4"],
                "m16/offline_stream",
            )

    def test_resume_equals_uninterrupted(self, tmp_path):
        _finished_parent(tmp_path / "runs" / "b" / "parent")
        whole = _offline(
            tmp_path / "runs" / "b" / "whole", "b/parent", STREAM, "m16/offline_stream"
        )
        assert loop.run(PROGRAM, whole, attempt=1) == Exit.OK
        cut = _offline(
            tmp_path / "runs" / "b" / "cut", "b/parent", STREAM, "m16/offline_stream"
        )
        assert loop.run(_Stopping(2), cut, attempt=1) == Exit.STOPPED
        (cut.parent / "stop.request").unlink()
        assert loop.run(PROGRAM, cut, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (cut.parent / name).read_bytes() == (
                whole.parent / name
            ).read_bytes()


@pytest.mark.cuda
@pytest.mark.parametrize("learner", ["pc", "bp"])
def test_a_run_on_cuda_completes(tmp_path, learner):
    """The messages the host reads back (into host memory) add to the sums of
    stream.jsonl."""
    assert _run(tmp_path / "run", ["device=cuda", f"learner={learner}"]) == Exit.OK


class TestWindowPresets:
    @pytest.mark.parametrize(
        ("preset", "frozen"),
        [("m16/balls_joint_win", False), ("m16/balls_frozen_win", True)],
    )
    def test_the_online_rule_is_that_of_the_offline_bound(self, preset, frozen):
        """The windows and the optimizer of the online presets are those of
        ``m16/offline_stream``, on the big dynamics by BP, with its feedback
        (grad, the one C1 chose)."""
        conf = compose([f"+experiment={preset}"])
        bound = compose(["+experiment=m16/offline_stream"])
        dyn, off = conf.dynamics, bound.offline
        assert dyn is not None and dyn.window is not None and off is not None
        assert conf.vision is not None and conf.vision.frozen is frozen
        w = dyn.window
        assert (w.window, w.stride, w.credit) == (off.window, off.stride, off.credit)
        assert w.feedback == off.feedback == "grad"
        assert (w.observed, w.relax) == (16, 0.3)
        assert conf.measure.observed == w.observed
        assert (dyn.nodes, dyn.d_s, dyn.trace, dyn.residual) == (3, 512, False, True)
        assert dyn.open_loop == 0.0
        assert conf.optim == bound.optim
        assert conf.learner.kind == "bp"
