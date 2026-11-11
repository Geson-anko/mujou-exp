"""The loop on a toy program whose numbers are known in advance: unit ``u``
adds ``u`` to a sum, a line of ``stream.jsonl`` is the mean of the units since
the last line, and an evaluation is ``{"v": sum}``."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from exp.lib.rundir import EnvError, Exit, verify_manifest
from exp.video import loop
from exp.video.compose import to_dict
from exp.video.loop import Schedule
from exp.video.schema.dyn import DynamicsConf
from exp.video.schema.measure import MeasureConf
from exp.video.schema.run import RunConf


@pytest.fixture(autouse=True)
def _no_gil_check(monkeypatch):
    """The GIL of this test process is pytest's (tests/video/test_main.py
    checks it in a run of its own)."""
    monkeypatch.setattr(loop, "assert_free_threaded", lambda where: None)


class Toy:
    """A ``loop.Built`` of plain numbers (module docstring)."""

    def __init__(
        self, schedule: Schedule, resume: Mapping[str, Any] | None, fail: dict
    ) -> None:
        self._schedule = schedule
        self.fail = fail
        self.u, self.total, self.since, self.n = 0, 0.0, 0.0, 0
        if resume is not None:
            h = resume["host"]
            self.u, self.total, self.since, self.n = (
                h["u"],
                h["total"],
                h["since"],
                h["n"],
            )
        self.lin = nn.Linear(1, 1)

    @property
    def schedule(self) -> Schedule:
        return self._schedule

    def advance(self) -> None:
        self.u += 1
        if self.fail.get("raise_at") == self.u:
            raise self.fail.get("error", RuntimeError("the toy failed"))
        self.total += self.u
        self.since += self.u
        self.n += 1

    def log(self) -> dict[str, float | None]:
        x = self.since / self.n
        if self.fail.get("nan_at") == self.u:
            x = math.nan
        self.since, self.n = 0.0, 0
        return {"x": x}

    def evaluate(self, split: str) -> dict[str, float]:
        row = {"v": self.total + (0.5 if split == "test" else 0.0), "gap": math.nan}
        if split == "val":
            row["objective"] = -self.total
        return row

    def diverged(self, row: Mapping[str, float]) -> str | None:
        limit = self.fail.get("diverge_above")
        return None if limit is None or row["v"] <= limit else f"v {row['v']} > {limit}"

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        return 99.0 if diverged else (None if row is None else row["objective"])

    def modules(self) -> dict[str, nn.Module]:
        return {"lin": self.lin}

    def visualize(self, out: Path) -> None:
        out.mkdir(parents=True)
        (out / "seen.txt").write_text(f"{self.u}\n")

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        return {}, {"u": self.u, "total": self.total, "since": self.since, "n": self.n}

    def final(self) -> dict[str, dict[str, Any]]:
        return {"extra": {"u": self.u}}

    def close(self) -> None:
        pass


class ToyProgram:
    def __init__(self, stop_at: int | None = None, **fail) -> None:
        self.stop_at = stop_at
        self.fail = fail
        self.schedule = Schedule(
            units=12,
            unit="frame",
            seconds=0.5,
            log=2,
            evaluate=4,
            ckpt=6,
            viz=6,
            resume=4,
            test=True,
        )

    def build(self, conf, run_dir: Path, resume) -> Toy:
        toy = Toy(self.schedule, resume, self.fail)
        if self.stop_at is not None:
            advance, stop_at = toy.advance, self.stop_at

            def stopping() -> None:
                advance()
                if toy.u == stop_at:
                    (run_dir / "stop.request").touch()

            toy.advance = stopping  # type: ignore[method-assign]
        return toy


def _job(run_dir: Path, **measure) -> Path:
    conf = RunConf(name="toy", dynamics=DynamicsConf(), measure=MeasureConf(**measure))
    run_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "run_id": "toy-s0",
        "group": "g",
        "program": "dynamics",
        "config": to_dict(conf),
        "code": {"snap": "x"},
    }
    path = run_dir / "job.json"
    path.write_text(json.dumps(job))
    return path


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestRun:
    def test_a_complete_run_writes_the_files_of_the_contract(self, tmp_path):
        job = _job(tmp_path / "r", criteria={"v": 100.0, "objective": -1000.0})
        assert loop.run(ToyProgram(), job, attempt=1) == Exit.OK
        r = job.parent
        # a line every 2 units: the mean of the two units, 1.5, 3.5, ...
        stream = _lines(r / "stream.jsonl")
        assert [(x["t"], x["frames"], x["x"]) for x in stream] == [
            (u / 2, u, u - 0.5) for u in range(2, 13, 2)
        ]
        evals = _lines(r / "evals.jsonl")
        val = [(x["frame"], x["v"]) for x in evals if x["split"] == "val"]
        test = [(x["frame"], x["v"]) for x in evals if x["split"] == "test"]
        total = {u: u * (u + 1) / 2 for u in range(13)}
        assert val == [(u, total[u]) for u in (4, 8, 12)]
        assert test == [(u, total[u] + 0.5) for u in (6, 12)]
        assert all(x["gap"] is None for x in evals)  # NaN as null
        assert sorted(p.name for p in (r / "ckpt").iterdir()) == [
            "00003.0.pt",
            "00006.0.pt",
        ]
        assert (r / "viz" / "00003.0" / "seen.txt").read_text() == "6\n"
        result = json.loads((r / "result.json").read_text())
        assert result["status"] == "completed" and result["frames"] == 12
        assert result["final"]["val"]["v"] == total[12]
        assert result["final"]["test"]["v"] == total[12] + 0.5
        assert result["final"]["extra"] == {"u": 12}
        assert result["objective"] == -total[12]
        assert result["criteria"]["v"]["pass"] is True  # 78.5 < 100
        assert result["criteria"]["objective"]["pass"] is False  # judged on test: none
        status = json.loads((r / "status.json").read_text())
        assert status["state"] == "completed" and status["frame"] == 12
        for name in (
            "CODE.json",
            "code.diff",
            "config.yaml",
            "provenance/attempt-1.json",
        ):
            assert (r / name).exists()
        assert not (r / "resume").exists()
        verify_manifest(r)
        listed = json.loads((r / "manifest.json").read_text())["files"]
        assert "evals.jsonl" in listed and "status.json" not in listed

    @pytest.mark.parametrize("stop_at", [3, 4, 12])
    def test_a_run_stopped_and_resumed_writes_the_bytes_of_one_never_stopped(
        self, tmp_path, stop_at
    ):
        """Stopped after unit ``stop_at`` (between two resume checkpoints, on
        one, at the end), it writes a resume checkpoint and exits with
        ``STOPPED``; the next attempt goes on to the same metrics files."""
        whole = _job(tmp_path / "whole")
        assert loop.run(ToyProgram(), whole, attempt=1) == Exit.OK
        cut = _job(tmp_path / "cut")
        assert loop.run(ToyProgram(stop_at=stop_at), cut, attempt=1) == Exit.STOPPED
        assert (
            json.loads((cut.parent / "status.json").read_text())["state"] == "stopped"
        )
        assert (cut.parent / "resume" / "latest").exists()
        (cut.parent / "stop.request").unlink()
        assert loop.run(ToyProgram(), cut, attempt=2) == Exit.OK
        for name in ("evals.jsonl", "stream.jsonl"):
            assert (cut.parent / name).read_bytes() == (
                whole.parent / name
            ).read_bytes()
        assert (cut.parent / "provenance" / "attempt-2.json").exists()

    def test_lines_after_the_checkpoint_are_cut_back_on_resume(self, tmp_path):
        """A run killed after its checkpoint at unit 4 (it wrote lines up to
        unit 6) resumes from 4 and writes those lines once."""
        whole = _job(tmp_path / "whole")
        loop.run(ToyProgram(), whole, attempt=1)
        cut = _job(tmp_path / "cut")
        loop.run(ToyProgram(stop_at=4), cut, attempt=1)
        with (cut.parent / "stream.jsonl").open("a") as f:  # the lines of a crash
            f.write('{"t": 2.5, "frames": 5, "x": 0}\n{"t": 3.0, "frames": 6')
        (cut.parent / "stop.request").unlink()
        assert loop.run(ToyProgram(), cut, attempt=2) == Exit.OK
        assert (cut.parent / "stream.jsonl").read_bytes() == (
            whole.parent / "stream.jsonl"
        ).read_bytes()

    def test_the_schedule_of_the_records_leaves_the_rows_as_they_are(self, tmp_path):
        rows = []
        for i, (log, ckpt, viz, resume) in enumerate(
            [(2, 6, 6, 4), (1, 0, 0, 0), (3, 4, 2, 1)]
        ):
            program = ToyProgram()
            program.schedule = Schedule(
                units=12,
                seconds=0.5,
                log=log,
                evaluate=4,
                ckpt=ckpt,
                viz=viz,
                resume=resume,
            )
            job = _job(tmp_path / str(i))
            assert loop.run(program, job, attempt=1) == Exit.OK
            rows.append(
                [x for x in _lines(job.parent / "evals.jsonl") if x["split"] == "val"]
            )
        assert rows[0] == rows[1] == rows[2]


class TestEnds:
    def test_a_row_that_diverged_ends_the_run_with_its_reason(self, tmp_path):
        job = _job(tmp_path / "r")
        assert loop.run(ToyProgram(diverge_above=20.0), job, attempt=1) == Exit.DIVERGED
        result = json.loads((job.parent / "result.json").read_text())
        assert result["status"] == "diverged"
        # the test row at unit 6: 21 + 0.5 (val at unit 4 was 10)
        assert result["reason"] == "v 21.5 > 20.0"
        assert result["t_end"] == 3.0 and result["objective"] == 99.0
        assert (
            json.loads((job.parent / "status.json").read_text())["state"] == "diverged"
        )
        verify_manifest(job.parent)

    def test_a_log_that_is_not_finite_ends_the_run(self, tmp_path):
        job = _job(tmp_path / "r")
        assert loop.run(ToyProgram(nan_at=6), job, attempt=1) == Exit.DIVERGED
        result = json.loads((job.parent / "result.json").read_text())
        assert result["reason"] == "x not finite at t=3"
        assert _lines(job.parent / "stream.jsonl")[-1]["x"] is None

    def test_an_exception_fails_the_run(self, tmp_path, capsys):
        job = _job(tmp_path / "r")
        assert loop.run(ToyProgram(raise_at=5), job, attempt=1) == Exit.ERROR
        assert json.loads((job.parent / "status.json").read_text())["state"] == "failed"
        assert not (job.parent / "result.json").exists()
        assert "the toy failed" in capsys.readouterr().err

    def test_an_environment_error_says_so(self, tmp_path):
        job = _job(tmp_path / "r")
        error = EnvError("no device")
        assert loop.run(ToyProgram(raise_at=1, error=error), job, attempt=1) == Exit.ENV
        assert json.loads((job.parent / "status.json").read_text())["state"] == "env"

    def test_omp_threads_other_than_one_are_an_environment_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        job = _job(tmp_path / "r")
        assert loop.run(ToyProgram(), job, attempt=1) == Exit.ENV
        assert torch.get_num_threads() >= 1  # nothing was changed


class TestEvery:
    @pytest.mark.parametrize(
        ("seconds", "want"), [(0.0, 0), (1.0, 30), (0.01, 1), (60.0, 1800)]
    )
    def test_seconds_in_frames(self, seconds, want):
        assert loop.every(seconds, 1 / 30) == want
