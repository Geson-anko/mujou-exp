"""The program of M14 run by the loop on small jobs: the files of the run
contract, determinism, resume, the prequential record, the exit codes and the
records' schedule (``ctl``) leaving the metrics as they are."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from exp.lib import ckpt
from exp.lib.rundir import Exit, verify_manifest
from exp.video import loop
from exp.video.compose import compose
from exp.video.dyn.nets import live_states
from exp.video.main import local_job
from exp.video.measure.dynamics import open_loop
from exp.video.programs.dynamics import PROGRAM

SMALL = [
    "data.streams=2",
    "measure.clips=8",
    "measure.every=1",
    "seconds=3",
]


@pytest.fixture(autouse=True)
def _no_gil_check(monkeypatch):
    """The GIL of this test process is pytest's (tests/video/test_main.py
    checks it in a run of its own)."""
    monkeypatch.setattr(loop, "assert_free_threaded", lambda where: None)


def _run(
    out: Path,
    overrides: list[str],
    program: Any = PROGRAM,
    attempt: int = 1,
    experiment: str = "m14/s0",
) -> int:
    job = local_job([f"+experiment={experiment}", *SMALL, *overrides], out)
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
        confirm = ["ctl.ckpt_every=1.5", "ctl.viz_every=1.5", "ctl.eval_test=true"]
        assert _run(out, ["seconds=3", *confirm]) == Exit.OK
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
            "ckpt/00001.5.pt",
            "ckpt/00003.0.pt",
            "viz/00001.5/frames.png",
            "viz/00003.0/traj.png",
        ):
            assert (out / name).is_file(), name
        assert not (out / "resume").exists()
        verify_manifest(out)
        stream = _lines(out / "stream.jsonl")
        assert [x["frames"] for x in stream] == [30, 60, 90]
        assert set(stream[0]) == {"t", "frames", "preq", "preq_open"}
        evals = _lines(out / "evals.jsonl")
        assert [(x["t"], x["split"]) for x in evals] == [
            (1.0, "val"),
            (1.5, "test"),
            (2.0, "val"),
            (3.0, "val"),
            (3.0, "test"),
        ]
        val = evals[0]
        for key in (
            "roll10",
            "roll10/cv10",
            "fork10",
            "fork10/forkcv10",
            "bounce10",
            "bounceball10",
            "gravity10",
            "roll10@contactball",
            "mean10@contactball",
            "objective",
        ):
            assert key in val
        assert "fork10" not in evals[1] and "objective" not in evals[1]
        result = json.loads((out / "result.json").read_text())
        assert result["status"] == "completed"
        assert result["final"]["test"] == evals[-1]
        assert result["objective"] == evals[-2]["objective"]
        base = json.loads((out / "baselines.json").read_text())
        assert base["val"]["cv10"] == val["cv10"]
        header = ckpt.load_eval(
            out / "ckpt" / "00003.0.pt",
            _modules(out),
            group=json.loads((out / "job.json").read_text())["group"],
        )
        assert header["frame"] == 90

    def test_the_same_job_twice_writes_the_same_bytes(self, tmp_path):
        for d in ("a", "b"):
            assert _run(tmp_path / d, ["learner=pc"]) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (tmp_path / "a" / name).read_bytes() == (
                tmp_path / "b" / name
            ).read_bytes()


def _modules(out: Path) -> dict[str, Any]:
    """Modules of the run's settings, to load an evaluation checkpoint into."""
    from exp.video.compose import from_dict
    from exp.video.dyn.modules import build_dynamics

    conf = from_dict(json.loads((out / "job.json").read_text())["config"])
    assert conf.dynamics is not None
    mods = build_dynamics(conf.dynamics, 4, 4, seed=99)
    return {"step": mods.step, "readout": mods.readout}


class TestResume:
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    @pytest.mark.parametrize("trace", [True], ids=["trace"])
    def test_resume_equals_uninterrupted(self, tmp_path, learner, trace):
        """Stopped at frame 47 (past the first evaluation, between two resume
        checkpoints) and resumed, the run writes the bytes of one never
        stopped: the nodes' states, the traces, the optimizers (AdamW in a
        warm-up), the streams and the sums since the last line all resume."""
        extra = [
            f"learner={learner}",
            f"dynamics.trace={str(trace).lower()}",
            "dynamics.nodes=3",
            "dynamics.d_s=16",
            "optim=adamw",
            "optim.warmup=1.0",
            "ctl.resume_every=0.5",
        ]
        assert _run(tmp_path / "whole", extra) == Exit.OK
        cut = tmp_path / "cut"
        assert _run(cut, extra, _Stopping(47)) == Exit.STOPPED
        assert json.loads((cut / "status.json").read_text())["state"] == "stopped"
        assert (cut / "resume" / "00000047.pt").is_file()
        (cut / "stop.request").unlink()
        assert _run(cut, extra, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (cut / name).read_bytes() == (tmp_path / "whole" / name).read_bytes()

    def test_the_offline_fit_resumes_too(self, tmp_path):
        offline = "[m14/s0,m14/offline]"
        extra = [
            "offline.frames=72",
            "offline.window=36",
            "offline.steps=6",
            "offline.batch=4",
            "offline.every=2",
            "dynamics.d_s=8",
        ]
        assert _run(tmp_path / "whole", extra, experiment=offline) == Exit.OK
        cut = tmp_path / "cut"
        assert _run(cut, extra, _Stopping(3), experiment=offline) == Exit.STOPPED
        (cut / "stop.request").unlink()
        assert _run(cut, extra, attempt=2, experiment=offline) == Exit.OK
        rows = _lines(tmp_path / "whole" / "evals.jsonl")
        assert [x["step"] for x in rows] == [2, 4, 6]
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (tmp_path / "cut" / name).read_bytes() == (
                tmp_path / "whole" / name
            ).read_bytes()


class TestRecords:
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_the_prediction_recorded_is_the_one_before_learning(
        self, tmp_path, learner
    ):
        """The first prediction (frame 1) is made before the frame is learned
        from: its record is the same at any learning rate; the next frames'
        are not (they were learned from)."""
        lines = []
        for lr in ("0.001", "0.3"):
            out = tmp_path / lr
            ctl = [
                "ctl.log_every=0.03",
                f"optim.lr={lr}",
                "optim.warmup=0",
                f"learner={learner}",
            ]
            assert _run(out, ["seconds=0.2", "measure.every=0", *ctl]) == Exit.OK
            lines.append(_lines(out / "stream.jsonl"))
        a, b = lines
        assert a[0]["preq"] is None  # frame 0 starts the video: no prediction
        assert a[1]["frames"] == 2 and a[1]["preq"] == b[1]["preq"]
        assert a[3]["preq"] != b[3]["preq"]

    def test_the_schedule_of_the_records_leaves_the_metrics_as_they_are(self, tmp_path):
        """``ctl`` (lines, checkpoints, pictures, resume checkpoints, the test
        clips) is out of the group's hash: it does not change a val row."""
        plain = ["learner=pc", "ctl.resume_every=300"]
        busy = [
            "learner=pc",
            "ctl.log_every=0.1",
            "ctl.ckpt_every=0.5",
            "ctl.viz_every=1",
            "ctl.resume_every=0.2",
            "ctl.eval_test=true",
        ]
        assert _run(tmp_path / "plain", plain) == Exit.OK
        assert _run(tmp_path / "busy", busy) == Exit.OK
        val = [
            [x for x in _lines(tmp_path / d / "evals.jsonl") if x["split"] == "val"]
            for d in ("plain", "busy")
        ]
        assert len(val[0]) == 3 and val[0] == val[1]


class TestLiveState:
    @pytest.mark.parametrize("nodes", [1, 3])
    @pytest.mark.parametrize("learner", ["pc", "bp"])
    def test_one_step_from_the_live_state_is_the_next_prediction(
        self, tmp_path, learner, nodes
    ):
        """After frame k - 1, ``live_states`` is s_{k-1} at the weights that
        predict frame k, and the history's last row is x_{k-1}: one step of
        ``open_loop`` from them is the prequential prediction of frame k that
        the host reads back, exactly, at every k (the weights change at each
        frame, lr 0.3). This is what ``fork_eval`` rolls out from."""
        conf = compose(
            [
                "+experiment=m14/s0",
                "data.streams=3",
                "measure.clips=8",
                f"learner={learner}",
                f"dynamics.nodes={nodes}",
                "dynamics.d_s=16",
                "dynamics.open_loop=0",
                "dynamics.residual=true",
                "optim=sgd",
                "optim.lr=0.3",
                "optim.warmup=0",
                "seconds=1",
            ]
        )
        built: Any = PROGRAM.build(conf, tmp_path, None)
        try:
            shown: dict[str, Any] = {}
            show = built.host.show

            def capture(writes, watch):
                p, t = show(writes, watch)
                shown["p"] = p
                return p, t

            built.host.show = capture
            built.advance()
            for _ in range(12):
                x = built.measures.std.apply(built._history[:, -1]).float()
                s = live_states(built.net)
                residual = built.conf.dynamics.residual
                pred = open_loop(built.mods, x, s, residual=residual, horizon=1)[:, 0]
                built.advance()
                assert torch.equal(pred, shown["p"]["y_hat"])
        finally:
            built.close()


class TestEnds:
    def test_an_offline_window_shorter_than_the_horizons_is_an_error(self, tmp_path):
        """observed 16 + horizon 20 = 36 frames: a window of 35 never trains
        the prediction 20 frames ahead that the evaluation reads."""
        conf = compose(["+experiment=m14/offline", "offline.window=35"])
        with pytest.raises(ValueError, match="offline.window=35"):
            PROGRAM.build(conf, tmp_path, None)

    def test_a_rate_too_large_diverges(self, tmp_path):
        out = tmp_path / "run"
        assert _run(out, ["optim.lr=100", "optim.warmup=0"]) == Exit.DIVERGED
        result = json.loads((out / "result.json").read_text())
        assert result["status"] == "diverged" and result["reason"]
        assert result["objective"] == pytest.approx(2.302585092994046)
        verify_manifest(out)

    def test_the_baselines_task_writes_the_baselines_only(self, tmp_path):
        out = tmp_path / "run"
        assert _run(out, [], experiment="[m14/s2,m14/baselines]") == Exit.OK
        result = json.loads((out / "result.json").read_text())
        base = json.loads((out / "baselines.json").read_text())
        assert result["final"]["val"] == base["val"]
        assert result["final"]["test"] == base["test"]
        assert "ca10@ball2" in base["test"]
        assert (
            not (out / "evals.jsonl").exists() or not (out / "evals.jsonl").read_text()
        )

    def test_the_bench_task_evaluates_nothing(self, tmp_path):
        out = tmp_path / "run"
        assert _run(out, ["seconds=1"], experiment="m14/bench") == Exit.OK
        result = json.loads((out / "result.json").read_text())
        assert set(result["final"]) == {"timing"}
        assert result["final"]["timing"]["fps"] > 0
        assert len(_lines(out / "stream.jsonl")) == 1
