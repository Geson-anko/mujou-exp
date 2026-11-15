"""The task ``posthoc`` of the vision program as a job of the fleet: started by
``exp.video.main --job`` as the worker starts it, it decodes the parent's
checkpoints into its own run directory and ends by the run contract
(``result.json``, ``manifest.json``); a stop resumes at the next checkpoint
and writes the bytes of a job never stopped. And the entry by hand."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from exp.lib.fleet.jobs import make_job
from exp.lib.rundir import Exit, verify_manifest
from exp.video import loop
from exp.video.compose import compose, to_dict
from exp.video.main import local_job, main as video_main
from exp.video.programs import posthoc
from exp.video.programs.posthoc import main, parent_dir
from exp.video.programs.vision import DATA_ENV, PROGRAM
from tests.video.measure.test_posthoc import TINY
from tests.video.packed import smooth_tours, tiny

CODE = {
    "snap": "unknown",
    "base_head": "unknown",
    "dirty": False,
    "mujou_snap": "unknown",
    "mujou_pushed": False,
    "envkey": "local",
}
RES = {"class": "cpu", "ram_gb": 2.0, "gpu_mem_gb": 0.0}
DECODE = {"frames": 20, "steps": 2}


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    """The runs' root of a worker, ``runs/``, with a tiny vision run (BP, two
    checkpoints) at ``runs/b/parent``."""
    root = tmp_path_factory.mktemp("worker") / "runs"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loop, "assert_free_threaded", lambda where: None)
        job = local_job(TINY, root / "b" / "parent")
        assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
    return root


def _submit(runs: Path, name: str, **extra: Any) -> Path:
    """The posthoc job of the parent in ``runs/b/<name>``, as ``fleet submit``
    writes it (``extra``: its top-level keys)."""
    overrides = [*TINY, "task=posthoc"]
    job = make_job(
        batch="b",
        config=to_dict(compose(overrides)),
        overrides=overrides,
        code=CODE,
        res=RES,
        extra=extra,
    )
    run_dir = runs / "b" / name
    run_dir.mkdir(parents=True)
    (run_dir / "job.json").write_text(json.dumps(job))
    return run_dir


def _run(run_dir: Path, attempt: int, monkeypatch) -> int:
    monkeypatch.setattr(loop, "assert_free_threaded", lambda where: None)
    return video_main(["--job", str(run_dir / "job.json"), "--attempt", str(attempt)])


def _post(run_dir: Path) -> dict[str, bytes]:
    out = run_dir / "post" / "decode"
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


class TestJob:
    def test_it_decodes_into_its_own_run_by_the_contract(self, runs, monkeypatch):
        run_dir = _submit(runs, "whole", parent="b/parent", decode=DECODE)
        assert _run(run_dir, 1, monkeypatch) == Exit.OK
        result = json.loads((run_dir / "result.json").read_text())
        assert result["status"] == "completed" and result["ckpts"] == 2
        assert result["objective"] is None
        lines = [
            json.loads(x)
            for x in (run_dir / "post/decode/metrics.jsonl").read_text().splitlines()
        ]
        assert len(lines) == 4
        assert result["final"]["decode"] == {"val": lines[2], "test": lines[3]}
        manifest = json.loads((run_dir / "manifest.json").read_text())["files"]
        for name in ("00000.5.png", "00001.0.png", "metrics.jsonl", "config.json"):
            assert f"post/decode/{name}" in manifest, name
        verify_manifest(run_dir)
        assert (run_dir / "provenance" / "attempt-1.json").is_file()
        config = json.loads((run_dir / "post/decode/config.json").read_text())
        parent = json.loads((runs / "b/parent/job.json").read_text())
        assert config["parent"]["run_id"] == parent["run_id"]
        assert config["frames"] == 20 and config["steps"] == 2
        assert not (runs / "b" / "parent" / "post").exists()

    def test_a_stopped_job_resumes_at_the_next_checkpoint(self, runs, monkeypatch):
        """A stop asked before the first checkpoint is decoded stops after it
        (``Exit.STOPPED``, one PNG); attempt 2 decodes the second only, into
        the bytes of a job never stopped."""
        whole = _submit(runs, "ref", parent="b/parent", decode=DECODE)
        assert _run(whole, 1, monkeypatch) == Exit.OK
        run_dir = _submit(runs, "stopped", parent="b/parent", decode=DECODE)
        (run_dir / "stop.request").write_text("{}")
        assert _run(run_dir, 1, monkeypatch) == Exit.STOPPED
        assert sorted(_post(run_dir)) == ["00000.5.png", "config.json", "metrics.jsonl"]
        (run_dir / "stop.request").unlink()
        assert _run(run_dir, 2, monkeypatch) == Exit.OK
        assert _post(run_dir) == _post(whole)

    def test_a_job_without_a_parent_fails(self, runs, monkeypatch):
        run_dir = _submit(runs, "orphan", decode=DECODE)
        assert _run(run_dir, 1, monkeypatch) == Exit.ERROR
        assert json.loads((run_dir / "status.json").read_text())["state"] == "failed"

    def test_the_parent_is_relative_to_the_runs_root_or_absolute(self, tmp_path):
        run_dir = tmp_path / "runs" / "b" / "x"
        assert parent_dir({"parent": "a/p"}, run_dir) == tmp_path / "runs" / "a" / "p"
        assert parent_dir({"parent": str(tmp_path / "q")}, run_dir) == tmp_path / "q"
        with pytest.raises(ValueError, match="parent"):
            parent_dir({}, run_dir)


class TestEntry:
    def test_it_decodes_into_the_run_itself(self, runs, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(posthoc, "assert_free_threaded", lambda where: None)
        run = tmp_path / "run"
        shutil.copytree(runs / "b" / "parent", run)
        args = ["--run", str(run), "--frames", "20", "--steps", "1"]
        assert main(args) == Exit.OK
        assert len(capsys.readouterr().out.splitlines()) == 4
        assert (run / "post" / "decode" / "00001.0.png").is_file()

    def test_a_run_without_checkpoints_is_an_error(self, runs, tmp_path, monkeypatch):
        monkeypatch.setattr(posthoc, "assert_free_threaded", lambda where: None)
        (tmp_path / "job.json").write_bytes((runs / "b/parent/job.json").read_bytes())
        assert main(["--run", str(tmp_path)]) == Exit.ERROR

    def test_the_wrong_threads_are_an_environment_error(self, runs, monkeypatch):
        monkeypatch.setattr(posthoc, "assert_free_threaded", lambda where: None)
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        assert main(["--run", str(runs / "b" / "parent")]) == Exit.ENV


class TestPackedJob:
    """M17: the posthoc job of a run of the packed layout evaluates the
    encoder (measure.encoder_eval) by the same contract; task=baselines the
    yardsticks alone."""

    @pytest.fixture(scope="class")
    def m17(self, tmp_path_factory) -> tuple[Path, list[str]]:
        tmp = tmp_path_factory.mktemp("m17")
        root = smooth_tours(tmp / "data" / "d", {"A": 120, "B": 80, "H": 200})
        overrides = tiny(root.name)
        runs = tmp / "runs"
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(DATA_ENV, str(root.parent))
            mp.setattr(loop, "assert_free_threaded", lambda where: None)
            job = local_job(overrides, runs / "b" / "parent")
            assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
        return runs, overrides

    def _job(self, runs: Path, overrides: list[str], name: str, task: str, **extra):
        o = [*overrides, f"task={task}"]
        job = make_job(
            batch="b",
            config=to_dict(compose(o)),
            overrides=o,
            code=CODE,
            res=RES,
            extra=extra,
        )
        run_dir = runs / "b" / name
        run_dir.mkdir(parents=True)
        (run_dir / "job.json").write_text(json.dumps(job))
        return run_dir

    def test_the_last_checkpoint_with_the_objective(self, m17, monkeypatch):
        runs, overrides = m17
        monkeypatch.setenv(DATA_ENV, str(runs.parent / "data"))
        decode = {"windows": 16, "steps": 3, "ckpts": "last"}
        run_dir = self._job(
            runs, overrides, "post", "posthoc", parent="b/parent", decode=decode
        )
        assert _run(run_dir, 1, monkeypatch) == Exit.OK
        result = json.loads((run_dir / "result.json").read_text())
        final = result["final"]
        assert (
            final["decode"]["val"]["t"] == 1.0 and "objective" in final["decode"]["val"]
        )
        assert set(final["baselines"]["val"]) == {"pca", "init"}
        manifest = json.loads((run_dir / "manifest.json").read_text())["files"]
        for name in ("baselines.jsonl", "metrics.jsonl", "00001.0-restore.png"):
            assert f"post/decode/{name}" in manifest, name
        verify_manifest(run_dir)

    def test_the_baselines_task_needs_no_parent(self, m17, monkeypatch):
        runs, overrides = m17
        monkeypatch.setenv(DATA_ENV, str(runs.parent / "data"))
        run_dir = self._job(
            runs, overrides, "base", "baselines", decode={"windows": 16, "steps": 3}
        )
        assert _run(run_dir, 1, monkeypatch) == Exit.OK
        final = json.loads((run_dir / "result.json").read_text())["final"]
        assert final["decode"] == {} and set(final["baselines"]["test"]) == {
            "pca",
            "init",
        }
        assert (run_dir / "post/decode/pca-dyn.png").is_file()
