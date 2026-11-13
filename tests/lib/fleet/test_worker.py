import errno
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from exp.lib.fleet import worker as worker_mod
from exp.lib.fleet.contract import read_json, write_json, write_manifest
from exp.lib.fleet.queue import QueueName
from exp.lib.fleet.worker import outcome
from tests.lib.fleet.fleetkit import FAST, REPO, Fleet, run_until, wait_for


def _dead(pid: int) -> bool:
    ps = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True
    )
    return not ps.stdout.strip() or ps.stdout.strip().startswith("Z")


class TestOutcome:
    """The table of design 4.10, from an exit code, a stop reason and the files."""

    @pytest.mark.parametrize(
        ("rc", "stop", "files", "want"),
        [
            (0, None, ("result", "manifest"), ("completed", None)),
            (3, None, ("result", "manifest"), ("diverged", None)),
            (0, None, ("result",), ("failed", "contract")),
            (3, None, (), ("failed", "contract")),
            (4, None, (), ("env", None)),
            (75, "pruned", (), ("stopped", "pruned")),
            (75, "drain", (), ("requeue", "drain")),
            (75, "move", (), ("requeue", "move")),
            (75, "stall", (), ("failed", "stall")),
            (-signal.SIGTERM, "user", (), ("stopped", "user")),
            (-signal.SIGKILL, "cancel", (), ("stopped", "cancel")),
            (-signal.SIGKILL, None, (), ("oom", None)),
            (-signal.SIGSEGV, None, (), ("failed", "signal SIGSEGV")),
            (1, None, (), ("failed", "exit 1")),
            (None, None, ("result", "manifest"), ("completed", None)),
            (None, None, ("result",), ("lost", None)),
            (None, "drain", (), ("requeue", "drain")),
            # adopted: the last state of status.json stands in for the exit code
            (None, None, ("state:failed",), ("failed", "adopted")),
            (None, None, ("state:env",), ("env", None)),
            (None, None, ("state:stopped",), ("stopped", "exit 75")),
            (None, None, ("state:running",), ("lost", None)),
            # a result.json that is no object is no result
            (0, None, ("bad_result", "manifest"), ("failed", "contract")),
            (None, None, ("bad_result", "manifest"), ("lost", None)),
        ],
    )
    def test_table(self, tmp_path: Path, rc, stop, files, want):
        if "result" in files:
            status = "diverged" if rc == 3 else "completed"
            write_json(tmp_path / "result.json", {"status": status})
        if "bad_result" in files:
            write_json(tmp_path / "result.json", ["completed"])
        for f in files:
            if f.startswith("state:"):
                write_json(tmp_path / "status.json", {"state": f.split(":")[1]})
        if "manifest" in files:
            write_manifest(tmp_path)
        status, reason = outcome(rc, stop, tmp_path)
        assert status == want[0]
        if want[1] is None:
            assert reason is None
        else:
            assert reason is not None and want[1] in reason


class TestEnds:
    def test_exit_codes_land_in_done_as_the_table_says(self, fleet: Fleet):
        jobs = {
            "completed": fleet.submit(name="ok"),
            "diverged": fleet.submit(name="div", exit=3, reason="roll10 > 10 mean10"),
            "failed": fleet.submit(name="exc", exit=1),
            "contract": fleet.submit(name="silent", no_manifest=True),
        }
        w = fleet.worker()
        run_until([w], lambda: all(fleet.finished(j["run_id"]) for j in jobs.values()))
        rec = {k: fleet.done(j["run_id"]) for k, j in jobs.items()}
        assert rec["completed"]["status"] == "completed"
        assert rec["diverged"]["status"] == "diverged"
        assert rec["diverged"]["reason"] == "roll10 > 10 mean10"
        assert rec["failed"]["status"] == "failed"
        assert rec["failed"]["exit"] == 1
        assert rec["contract"]["status"] == "failed"
        assert "contract" in rec["contract"]["reason"]
        for k, j in jobs.items():
            nas_dir = fleet.nas.runs / "b" / j["run_id"]
            exit_rec = read_json(nas_dir / "exit" / "attempt-1.json")
            assert exit_rec["returncode"] == rec[k]["exit"]
            assert not (nas_dir / "resume").exists()
        assert (
            fleet.nas.runs / "b" / jobs["failed"]["run_id"] / "logs" / "attempt-1.err"
        ).read_text()
        assert not list(fleet.nas.claimed("h0").iterdir())

    def test_a_job_json_with_the_slots_device(self, fleet: Fleet):
        gpu = fleet.submit(name="gpu", res="4090", gpu_mem_gb=4.0)
        cpu = fleet.submit(name="cpu")
        w = fleet.worker()
        run_until(
            [w], lambda: fleet.finished(gpu["run_id"]) and fleet.finished(cpu["run_id"])
        )
        assert fleet.done(gpu["run_id"])["slot"]["gpu"] == 0
        assert fleet.done(cpu["run_id"])["slot"]["gpu"] is None
        spec = read_json(fleet.nas.runs / "b" / gpu["run_id"] / "job.json")
        assert spec["config"]["device"] == "cuda"

    def test_env_error_cordons_the_class_and_requeues(self, fleet: Fleet):
        job = fleet.submit(exit=[4, 0])
        h0 = fleet.worker("h0")
        run_until([h0], lambda: bool(fleet.nas.entries()) and not h0.running)
        assert "cpu" in fleet.nas.cordoned("h0")
        (back,) = fleet.nas.entries()
        assert read_json(back)["attempt"] == 2
        for _ in range(10):  # the cordoned host does not take it again
            h0.tick()
        assert fleet.nas.entries() == [back]
        h1 = fleet.worker("h1")
        run_until([h0, h1], lambda: fleet.finished(job["run_id"]))
        rec = fleet.done(job["run_id"])
        assert (rec["status"], rec["host"], rec["attempt"]) == ("completed", "h1", 2)

    def test_sigkill_without_a_request_retries_once_with_more_ram(self, fleet: Fleet):
        job = fleet.submit(kill=True, ram_gb=2.0)
        w = fleet.worker()
        run_until([w], lambda: fleet.finished(job["run_id"]))
        rec = fleet.done(job["run_id"])
        assert (rec["status"], rec["reason"], rec["attempt"]) == ("failed", "oom", 2)
        assert rec["res"]["ram_gb"] == pytest.approx(3.0)
        assert rec["retried"]["oom"] == 1


class TestPerHostSlots:
    """h0 has the default slots (4090, 4060), h1 its own (one 5070ti): a worker
    claims only what the GPUs of its own host can take."""

    MORE = (
        "\n[slots.h1]\ncpu = 1\nram_gb = 4\n"
        'gpus = [{index = 0, model = "5070ti", mem_gb = 13.9, jobs = 4}]\n'
    )

    @pytest.fixture
    def fleet(self, tmp_path: Path) -> Fleet:
        return Fleet(tmp_path, more=self.MORE)

    @pytest.mark.parametrize(
        ("res", "never", "lands"), [("5070ti", "h0", "h1"), ("4090", "h1", "h0")]
    )
    def test_a_model_runs_only_on_the_host_that_has_it(self, fleet, res, never, lands):
        job = fleet.submit(res=res, gpu_mem_gb=2.0)
        idle = fleet.worker(never)
        for _ in range(10):
            idle.tick()
        assert len(fleet.nas.entries()) == 1 and not idle.running
        run_until([idle, fleet.worker(lands)], lambda: fleet.finished(job["run_id"]))
        rec = fleet.done(job["run_id"])
        assert (rec["status"], rec["host"]) == ("completed", lands)
        assert (rec["slot"]["model"], rec["slot"]["gpu"]) == (res, 0)

    def test_any_gpu_takes_the_5070ti(self, fleet: Fleet):
        job = fleet.submit(res="any_gpu", gpu_mem_gb=2.0)
        run_until([fleet.worker("h1")], lambda: fleet.finished(job["run_id"]))
        rec = fleet.done(job["run_id"])
        assert (rec["host"], rec["slot"]["model"]) == ("h1", "5070ti")

    def test_h1_takes_no_more_runs_than_its_own_cores(self, fleet: Fleet):
        jobs = [fleet.submit(seed=i, seconds=1.0) for i in range(2)]
        h1 = fleet.worker("h1")
        for _ in range(5):
            h1.tick()
        assert len(h1.running) == 1  # cpu = 1 on h1 (2 on h0)
        assert len(fleet.nas.entries()) == len(jobs) - 1
        run_until([h1], lambda: all(fleet.finished(j["run_id"]) for j in jobs))


class TestStop:
    def test_pruned_ends_stopped_and_drain_resumes(self, fleet: Fleet):
        pruned = fleet.submit(name="pr", seconds=60, frames=600)
        drained = fleet.submit(name="dr", seconds=3, frames=300, resume_every=10)
        w = fleet.worker()
        claims = {}

        def started():
            for p in fleet.nas.claims("h0"):
                claims[QueueName.parse(p.name).run_id] = p
            return len(claims) == 2 and all(
                (fleet.conf.root / "runs" / "b" / r / "resume" / "latest").exists()
                for r in claims
            )

        run_until([w], started)
        fleet.nas.request_stop(
            claims[pruned["run_id"]], pruned["run_id"], reason="pruned", now=False
        )
        fleet.nas.request_stop(
            claims[drained["run_id"]], drained["run_id"], reason="drain", now=False
        )
        run_until([w], lambda: fleet.finished(pruned["run_id"]))
        assert fleet.done(pruned["run_id"])["status"] == "stopped"
        assert fleet.done(pruned["run_id"])["exit"] == 75
        run_until([w], lambda: fleet.finished(drained["run_id"]))
        rec = fleet.done(drained["run_id"])
        assert (rec["status"], rec["attempt"]) == ("completed", 2)
        result = read_json(fleet.nas.runs / "b" / drained["run_id"] / "result.json")
        assert result["resumed_from"] > 0

    def test_a_run_that_ignores_the_request_is_killed_with_its_children(
        self, fleet: Fleet
    ):
        job = fleet.submit(seconds=60, frames=600, ignore_stop=True, child=True)
        w = fleet.worker()
        run_dir = fleet.conf.root / "runs" / "b" / job["run_id"]
        run_until([w], lambda: (run_dir / "child.pid").exists())
        child = int((run_dir / "child.pid").read_text())
        (claim,) = fleet.nas.claims("h0")
        fleet.nas.request_stop(claim, job["run_id"], reason="user", now=False)
        run_until([w], lambda: fleet.finished(job["run_id"]))
        rec = fleet.done(job["run_id"])
        assert (rec["status"], rec["reason"], rec["exit"]) == (
            "stopped",
            "user",
            -signal.SIGTERM,
        )
        wait_for(lambda: _dead(child), timeout=5)

    def test_a_stalled_run_is_stopped_and_failed(self, fleet: Fleet):
        job = fleet.submit(seconds=60, frames=600, hang_at=3)
        w = fleet.worker(timing=replace(FAST, stall=0.6))
        run_until([w], lambda: fleet.finished(job["run_id"]))
        rec = fleet.done(job["run_id"])
        assert (rec["status"], rec["reason"], rec["attempt"]) == ("failed", "stall", 1)

    def test_a_run_whose_claim_is_gone_is_killed_and_not_archived(self, fleet: Fleet):
        job = fleet.submit(seconds=60, frames=600)
        w = fleet.worker()
        run_until([w], lambda: bool(w.running))
        pid = read_json(fleet.conf.root / "state" / f"{job['run_id']}.json")["pid"]
        (claim,) = fleet.nas.claims("h0")
        claim.unlink()
        run_until([w], lambda: not w.running)
        wait_for(lambda: _dead(pid), timeout=5)
        assert not fleet.finished(job["run_id"])
        assert not (fleet.nas.runs / "b" / job["run_id"]).exists()

    def test_a_job_requeued_while_its_run_lives_here_is_not_started_twice(
        self, fleet: Fleet
    ):
        job = fleet.submit(seconds=60, frames=600)
        w = fleet.worker()
        run_until([w], lambda: bool(w.running))
        pid = read_json(fleet.conf.root / "state" / f"{job['run_id']}.json")["pid"]
        (claim,) = fleet.nas.claims("h0")
        back = fleet.nas.requeue(claim, {**read_json(claim), "fresh": True})
        w.tick()  # the claim is missing once (< absent): the job stays queued
        assert fleet.nas.entries() == [back]
        assert w.running == [job["run_id"]]
        run_until([w], lambda: not fleet.nas.entries())  # fenced, then claimed
        wait_for(lambda: _dead(pid), timeout=5)
        new = read_json(fleet.conf.root / "state" / f"{job['run_id']}.json")["pid"]
        assert new != pid


class TestNasOutage:
    """Design 4.10: while the NAS fails, the runs go on."""

    def test_eio_on_the_claims_for_longer_than_absent_ticks(
        self, fleet: Fleet, monkeypatch
    ):
        job = fleet.submit(seconds=1.5, frames=30)
        w = fleet.worker()
        run_until([w], lambda: bool(w.running))
        pid = read_json(fleet.conf.root / "state" / f"{job['run_id']}.json")["pid"]
        real = os.stat
        claimed = str(fleet.nas.root / "claimed")

        def stat(p, *a, **k):
            if str(p).startswith(claimed):
                raise OSError(errno.EIO, "Input/output error")
            return real(p, *a, **k)

        monkeypatch.setattr(os, "stat", stat)
        monkeypatch.setattr(worker_mod, "nas_retry", lambda fn, **_: fn())
        monkeypatch.setattr("exp.lib.fleet.queue.nas_retry", lambda fn, **_: fn())
        for _ in range(3 * FAST.absent):
            w.tick()
        assert w.running == [job["run_id"]]
        assert not _dead(pid)
        monkeypatch.undo()
        run_until([w], lambda: fleet.finished(job["run_id"]))
        rec = fleet.done(job["run_id"])
        assert (rec["status"], rec["attempt"]) == ("completed", 1)

    def test_an_end_that_cannot_be_written_is_written_next_tick(
        self, fleet: Fleet, monkeypatch
    ):
        job = fleet.submit()
        w = fleet.worker()
        real = worker_mod.Worker._finish
        fails = [2]

        def finish(self, *a, **k):
            if fails[0]:
                fails[0] -= 1
                raise OSError(errno.EIO, "Input/output error")
            return real(self, *a, **k)

        monkeypatch.setattr(worker_mod.Worker, "_finish", finish)
        run_until([w], lambda: fleet.finished(job["run_id"]))
        assert fails == [0]
        assert fleet.done(job["run_id"])["status"] == "completed"
        assert w.running == []


class TestWorkerErrors:
    """Design 3.6: whatever the program leaves, the run ends as failed and the
    worker goes on."""

    def test_an_error_in_ending_a_run_fails_it_and_the_worker_goes_on(
        self, fleet: Fleet, monkeypatch
    ):
        def broken(*_):
            raise TypeError("string indices must be integers")

        bad = fleet.submit(name="bad")
        w = fleet.worker()
        monkeypatch.setattr(worker_mod, "outcome", broken)
        run_until([w], lambda: fleet.finished(bad["run_id"]))
        rec = fleet.done(bad["run_id"])
        assert rec["status"] == "failed"
        assert rec["reason"].startswith("worker: TypeError")
        assert fleet.nas.claims("h0") == []
        assert not (fleet.conf.root / "state" / f"{bad['run_id']}.json").exists()
        monkeypatch.undo()
        good = fleet.submit(name="good")
        run_until([w], lambda: fleet.finished(good["run_id"]))
        assert fleet.done(good["run_id"])["status"] == "completed"

    def test_recovering_a_dead_run_that_cannot_be_ended_does_not_stop_the_worker(
        self, fleet: Fleet, monkeypatch
    ):
        job = fleet.submit(seconds=60, frames=600)
        w = fleet.worker()
        run_until([w], lambda: bool(w.running))
        pid = read_json(fleet.conf.root / "state" / f"{job['run_id']}.json")["pid"]
        os.killpg(pid, signal.SIGKILL)
        wait_for(lambda: _dead(pid), timeout=5)

        def broken(*_):
            raise KeyError("path")

        monkeypatch.setattr(worker_mod, "outcome", broken)
        fleet.worker()  # a new worker: recover() must not raise
        rec = fleet.done(job["run_id"])
        assert (rec["status"], rec["reason"]) == ("failed", "worker: KeyError: 'path'")

    def test_an_adopted_run_that_fails_is_failed_not_restarted(self, fleet: Fleet):
        job = fleet.submit(seconds=1.0, frames=20, exit=1, error="boom")
        first = fleet.worker()
        run_until([first], lambda: bool(first.running))
        second = fleet.worker()  # adopts it: its exit code is not ours to see
        assert second.running == [job["run_id"]]
        run_until([second], lambda: fleet.finished(job["run_id"]))
        rec = fleet.done(job["run_id"])
        assert (rec["status"], rec["attempt"], rec["exit"]) == ("failed", 1, None)
        assert "adopted" in rec["reason"]
        assert list((fleet.nas.failures / "b").iterdir())  # marked as a failure


class TestArchive:
    def test_a_new_attempt_replaces_the_files_of_the_last_but_its_logs(
        self, fleet: Fleet
    ):
        from exp.lib.fleet.cli import main

        job = fleet.submit(exit=[1, 0])
        w = fleet.worker()
        run_until([w], lambda: fleet.finished(job["run_id"]))
        nas_dir = fleet.nas.runs / "b" / job["run_id"]
        (nas_dir / "stale.txt").write_text("from attempt 1\n")
        argv = ["--config", str(fleet.toml), "--local", "retry", job["run_id"]]
        assert main(argv) == 0
        run_until([w], lambda: fleet.finished(job["run_id"]))
        assert fleet.done(job["run_id"])["status"] == "completed"
        assert not (nas_dir / "stale.txt").exists()
        assert (nas_dir / "logs" / "attempt-1.err").exists()
        assert (nas_dir / "exit" / "attempt-1.json").exists()
        assert (nas_dir / "exit" / "attempt-2.json").exists()


class TestHolds:
    def test_three_alike_failures_hold_the_batch(self, fleet: Fleet):
        failing = [fleet.submit(seed=s, exit=1, error="boom") for s in range(3)]
        w = fleet.worker()
        run_until([w], lambda: all(fleet.finished(j["run_id"]) for j in failing))
        assert (fleet.nas.holds / "b").exists()
        assert "RuntimeError: boom" in read_json(fleet.nas.holds / "b")["reason"]
        later = fleet.submit(seed=9)
        other = fleet.submit(batch="c")
        run_until([w], lambda: fleet.finished(other["run_id"]))
        assert [QueueName.parse(p.name).run_id for p in fleet.nas.entries()] == [
            later["run_id"]
        ]


class TestRestart:
    """A worker process dies; a new one adopts the live run and restarts the
    dead one from its resume checkpoint."""

    def _start(self, fleet: Fleet) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            [sys.executable, "-m", "exp.lib.fleet", "--local", "--config", str(fleet.toml),
             "worker", "--host", "h0", "--tick", "0.05", "--live", "0"],
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )  # fmt: skip

    def test_adopt_the_live_and_resume_the_dead(self, fleet: Fleet):
        live = fleet.submit(name="live", seconds=4, frames=40, resume_every=5)
        dead = fleet.submit(name="dead", seconds=4, frames=40, resume_every=5)
        state = fleet.conf.root / "state"
        runs = fleet.conf.root / "runs" / "b"
        first = self._start(fleet)
        try:
            wait_for(
                lambda: all(
                    (state / f"{j['run_id']}.json").exists()
                    and (runs / j["run_id"] / "resume" / "latest").exists()
                    for j in (live, dead)
                )
            )
        finally:
            first.kill()
            first.wait()
        live_pid = read_json(state / f"{live['run_id']}.json")["pid"]
        dead_pid = read_json(state / f"{dead['run_id']}.json")["pid"]
        os.killpg(dead_pid, signal.SIGKILL)
        wait_for(lambda: _dead(dead_pid), timeout=5)
        second = self._start(fleet)
        try:
            wait_for(
                lambda: (
                    fleet.finished(live["run_id"]) and fleet.finished(dead["run_id"])
                )
            )
        finally:
            second.terminate()
            second.wait()
        rec_live, rec_dead = fleet.done(live["run_id"]), fleet.done(dead["run_id"])
        assert (rec_live["status"], rec_live["attempt"]) == ("completed", 1)
        assert read_json(
            fleet.nas.runs / "b" / live["run_id"] / "exit" / "attempt-1.json"
        )["adopted"]
        assert (rec_dead["status"], rec_dead["attempt"]) == ("completed", 2)
        assert rec_dead["retried"]["lost"] == 1
        result = read_json(fleet.nas.runs / "b" / dead["run_id"] / "result.json")
        assert result["resumed_from"] > 0
        assert _dead(live_pid)

    def test_a_second_worker_on_a_host_does_not_start(self, fleet: Fleet):
        first = self._start(fleet)
        try:
            wait_for(lambda: (fleet.conf.root / "worker.pid").exists())
            time.sleep(0.3)
            second = self._start(fleet)
            assert second.wait(timeout=20) == 0
            assert first.poll() is None
            assert (
                json.loads((fleet.nas.hosts / "h0.json").read_text())["pid"]
                == first.pid
            )
        finally:
            first.terminate()
            first.wait()
