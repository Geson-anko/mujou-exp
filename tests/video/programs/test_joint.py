"""The program of M16 (the vision and the dynamics learned together): the
dynamics sends the encoder no credit (its weights are the vision program's,
bit for bit), the dynamics is M14's, the files of the run contract, resume,
the frozen control, and the posthoc decoding of the rollouts."""

import json
from pathlib import Path
from typing import Any

import mujou.pc as pc
import mujou.testing
import pytest
import torch
from torch import nn

from exp.lib import ckpt
from exp.lib.host import Host
from exp.lib.rundir import EnvError, Exit, verify_manifest
from exp.video import loop
from exp.video.compose import compose, from_dict
from exp.video.dyn.modules import build_dynamics
from exp.video.dyn.nets import StateWorldBP, StateWorldPC, make_optimizer, modules
from exp.video.dyn.nodes import Source
from exp.video.jepa.nets import build_vision_net
from exp.video.jepa.vit import build_vision
from exp.video.main import local_job
from exp.video.measure.joint import joint_dynamics
from exp.video.measure.posthoc import Parent
from exp.video.programs.joint import (
    PROGRAM,
    JointBP,
    JointPC,
    build_joint,
    joint_objective,
    parent_ckpt,
    rebuild,
)
from exp.video.programs.posthoc import decoding
from exp.video.programs.vision import PROGRAM as VISION
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


class TestNoCreditToTheEncoder:
    @pytest.mark.parametrize("frames", [1, 2])
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_the_encoder_is_the_vision_programs_bit_for_bit(self, learner, frames):
        """The dynamics' error is not a term of the encoder's loss: the same
        seed and data give the vision of M15 and the joint network the same
        encoder and predictor after K frames, however much the dynamics
        learned."""
        conf = _conf(
            [
                f"learner={learner}",
                f"vision.frames={frames}",
                "optim_vision.warmup=0",  # so that the encoder does move
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
        # nothing flows back from the dynamics: no .grad, and the z it reads
        # has no graph
        for module in (joint.mods.encoder, joint.mods.predictor, joint.dyn.step):
            assert all(p.grad is None for p in module.parameters())
        assert _same(_weights(alone.mods.encoder), _weights(joint.mods.encoder))
        assert _same(_weights(alone.mods.predictor), _weights(joint.mods.predictor))

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
        modules={"encoder": mods.encoder, "predictor": mods.predictor},
    )


def _finished_parent(out: Path) -> Path:
    """A finished M15 run of these settings (seed 0) with the checkpoint
    ``ckpt/00001.0.pt``."""
    out.mkdir(parents=True)
    job = local_job(["+experiment=m15/balls_n1", *VISION_TINY, "ctl.ckpt_every=1"], out)
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
            ("data.balls.size=20", "data.balls"),
        ],
    )
    def test_the_parent_must_be_of_the_run_it_controls(
        self, override, differs, tmp_path
    ):
        """A parent of another seed, normalization or learner has the same
        shapes and loads without a word, so ``parent_ckpt`` compares."""
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
            modules={"encoder": mods.encoder, "predictor": mods.predictor},
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
        assert set(b) == {"encoder", "predictor", "step", "readout"}
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
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_resume_equals_uninterrupted(self, tmp_path, learner):
        """Stopped at frame 23 (past the first evaluation) and resumed, the
        run writes the bytes of one never stopped: both learners' states, the
        dynamics' memory and traces among them."""
        extra = [
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
