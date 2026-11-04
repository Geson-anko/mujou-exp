import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from exp.lib.fleet.jobs import group_of, run_id_of
from exp.lib.rundir import Exit, verify_manifest
from exp.video import loop, main

ROOT = Path(__file__).resolve().parents[2]
SMALL = [
    "+experiment=m14/s0",
    "seconds=2",
    "data.streams=2",
    "measure.clips=8",
    "measure.every=1",
]


def _env() -> dict[str, str]:
    """The environment of a run without the settings of the contract: no
    ``PYTHON_GIL``, no ``OMP_NUM_THREADS`` (a local run sets it)."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHON_GIL", "OMP_NUM_THREADS")
    }


class TestLocal:
    def test_a_local_run_composes_a_fleet_job_and_runs_free_threaded(self, tmp_path):
        """``python -m exp.video.main <overrides> --out DIR`` writes the job
        as the fleet makes it and runs it; the GIL stays disabled through the
        import, the build and the evaluations (the loop checks each)."""
        out = tmp_path / "run"
        proc = subprocess.run(
            [sys.executable, "-m", "exp.video.main", *SMALL, "--out", str(out)],
            cwd=ROOT,
            env=_env(),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == Exit.OK, proc.stderr
        job = json.loads((out / "job.json").read_text())
        group = group_of(
            program="dynamics",
            config=job["config"],
            snap=job["code"]["snap"],
            res_class="cpu",
        )
        assert job["group"] == group
        assert job["run_id"] == run_id_of("m14/s0", group, 0)
        assert job["overrides"] == SMALL and job["batch"] == "local"
        assert json.loads((out / "CODE.json").read_text()) == job["code"]
        prov = json.loads((out / "provenance" / "attempt-1.json").read_text())
        assert prov["env"]["OMP_NUM_THREADS"] == "1"
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            assert prov["gil_enabled"] is False
        assert json.loads((out / "status.json").read_text())["state"] == "completed"
        verify_manifest(out)

    def test_it_imports_no_program_until_it_runs_one(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, exp.video.main; print('torch' in sys.modules)",
            ],
            cwd=ROOT,
            env=_env(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.stdout.strip() == "False", proc.stderr


class TestInProcess:
    @pytest.fixture(autouse=True)
    def _no_gil_check(self, monkeypatch):
        """The GIL of this test process is pytest's (checked above in a run of
        its own)."""
        monkeypatch.setattr(loop, "assert_free_threaded", lambda where: None)

    def test_the_same_directory_resumes_the_same_job(self, tmp_path):
        out = tmp_path / "run"
        main.local_job(SMALL, out)
        (out / "stop.request").touch()  # stops after the first frame
        assert main.main([*SMALL, "--out", str(out)]) == Exit.STOPPED
        (out / "stop.request").unlink()
        assert main.main([*SMALL, "--out", str(out)]) == Exit.OK
        assert (out / "provenance" / "attempt-2.json").exists()
        result = json.loads((out / "result.json").read_text())
        assert result["frames"] == 60

    def test_a_finished_run_is_left_as_it_is(self, tmp_path):
        """The command again on a run that completed returns its exit code and
        writes nothing: no attempt, the same records and result."""
        out = tmp_path / "run"
        assert main.main([*SMALL, "--out", str(out)]) == Exit.OK
        names = ("evals.jsonl", "stream.jsonl", "result.json", "status.json")
        before = {n: (out / n).read_bytes() for n in names}
        assert main.main([*SMALL, "--out", str(out)]) == Exit.OK
        assert {n: (out / n).read_bytes() for n in names} == before
        assert not (out / "provenance" / "attempt-2.json").exists()

    def test_the_code_diff_has_the_changed_and_the_untracked_files(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()

        def git(*args: str) -> None:
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
                + list(args),
                check=True,
                capture_output=True,
            )

        git("init", "-q")
        (repo / ".gitignore").write_text("ignored.txt\n")
        (repo / "a.txt").write_text("one\n")
        git("add", "a.txt", ".gitignore")
        git("commit", "-qm", "a")
        (repo / "a.txt").write_text("two\n")
        (repo / "new").mkdir()
        (repo / "new" / "b.txt").write_text("fresh\n")
        (repo / "ignored.txt").write_text("secret\n")
        code, diff = main._local_code(repo)
        assert code["dirty"] and code["envkey"] == "local"
        assert "-one\n+two\n" in diff
        assert "+++ b/new/b.txt\n" in diff and "+fresh\n" in diff
        assert "secret" not in diff
        assert "--- a/a.txt" in diff  # the paths of git diff HEAD, as patch reads them

    def test_a_directory_of_another_job_is_refused(self, tmp_path, capsys):
        out = tmp_path / "run"
        assert main.main([*SMALL, "--out", str(out)]) == Exit.OK
        with pytest.raises(SystemExit) as e:
            main.main([*SMALL, "seed=1", "--out", str(out)])
        assert e.value.code == 2
        assert "holds the run" in capsys.readouterr().err

    def test_a_worker_job_runs_by_its_path_and_attempt(self, tmp_path):
        """``--job``/``--attempt`` as the worker starts it, on a job.json the
        way ``make_job`` writes it (from another directory's local job)."""
        src = tmp_path / "src"
        job = json.loads(main.local_job(SMALL, src).read_text())
        run_dir = tmp_path / "runs" / "b" / job["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "job.json").write_text(json.dumps({**job, "batch": "b"}))
        assert (
            main.main(["--job", str(run_dir / "job.json"), "--attempt", "3"]) == Exit.OK
        )
        assert (run_dir / "provenance" / "attempt-3.json").exists()
        assert json.loads((run_dir / "CODE.json").read_text()) == job["code"]

    def test_a_job_of_an_unknown_program_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "job.json").write_text(json.dumps({"program": "nope"}))
        assert (
            main.main(["--job", str(run_dir / "job.json"), "--attempt", "1"])
            == Exit.ERROR
        )
        assert json.loads((run_dir / "status.json").read_text())["state"] == "failed"

    @pytest.mark.parametrize(
        "argv",
        [
            ["--job", "j.json"],
            ["+experiment=m14/s0"],
            ["--job", "j.json", "--attempt", "1", "x=1"],
        ],
    )
    def test_wrong_arguments_exit_with_2(self, argv):
        with pytest.raises(SystemExit) as e:
            main.main(argv)
        assert e.value.code == 2
