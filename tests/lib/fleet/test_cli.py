import json
import os
import signal
from pathlib import Path

import pytest

from exp.lib.fleet.cli import main
from exp.lib.fleet.contract import read_json, write_json
from exp.lib.fleet.queue import Nas, QueueName
from exp.lib.fleet.transport import LocalTransport
from tests.lib.fleet.fleetkit import Fleet, run_until, wait_for


def _fleet(fleet: Fleet, *args: str) -> int:
    return main(["--config", str(fleet.toml), "--local", *args])


def _queued(fleet: Fleet) -> list[dict]:
    return [read_json(p) for p in fleet.nas.entries()]


class TestSubmit:
    def test_a_sweep_with_seeds_then_the_same_again_is_duplicate(
        self, fleet: Fleet, capsys
    ):
        args = ["submit", "--batch", "smoke", "--raw", "--entry", "exp.lib.fleet.fake",
                "--sweep", "fake.exit=0,3", "--seeds", "0,1", "--set", "fake.seconds=0.2"]  # fmt: skip
        assert _fleet(fleet, *args) == 0
        jobs = _queued(fleet)
        assert len(jobs) == 4
        assert {j["fake"]["seconds"] for j in jobs} == {0.2}
        assert len({j["group"] for j in jobs}) == 2  # the seeds share a group
        assert {j["seed"] for j in jobs} == {0, 1}
        assert all(j["code"]["envkey"] == "e" * 12 for j in jobs)
        capsys.readouterr()
        assert _fleet(fleet, *args) == 0
        out = capsys.readouterr().out
        assert out.count("duplicate") == 4 and "0 queued" in out
        assert len(fleet.nas.entries()) == 4

    def test_a_failed_run_may_be_submitted_again_a_completed_may_not(
        self, fleet: Fleet, capsys
    ):
        args = ["submit", "--batch", "b", "--raw", "--entry", "exp.lib.fleet.fake"]
        _fleet(fleet, *args)
        (job,) = _queued(fleet)
        fleet.nas.entries()[0].unlink()
        write_json(
            fleet.nas.done / f"{job['run_id']}.json", {**job, "status": "failed"}
        )
        _fleet(fleet, *args)
        assert len(fleet.nas.entries()) == 1
        fleet.nas.entries()[0].unlink()
        write_json(
            fleet.nas.done / f"{job['run_id']}.json", {**job, "status": "completed"}
        )
        _fleet(fleet, *args)
        assert fleet.nas.entries() == []

    def test_a_resubmitted_failed_run_drops_its_old_record(self, fleet: Fleet):
        args = ["submit", "--batch", "b", "--raw", "--entry", "exp.lib.fleet.fake"]
        _fleet(fleet, *args)
        (job,) = _queued(fleet)
        fleet.nas.entries()[0].unlink()
        write_json(
            fleet.nas.done / f"{job['run_id']}.json", {**job, "status": "failed"}
        )
        _fleet(fleet, *args)
        assert not fleet.finished(job["run_id"])
        assert [j["run_id"] for j in _queued(fleet)] == [job["run_id"]]

    def test_a_compose_that_crashes_is_a_message(self, fleet: Fleet, monkeypatch):
        def crash(*_):
            raise RuntimeError("compose failed:\nImportError: hydra")

        monkeypatch.setattr("exp.lib.fleet.cli.compose_configs", crash)
        with pytest.raises(SystemExit, match="ImportError: hydra"):
            _fleet(fleet, "submit", "--batch", "b", "+experiment=m14/s0")

    def test_an_unknown_class_is_refused(self, fleet: Fleet):
        with pytest.raises(SystemExit, match="any_gpu"):
            _fleet(fleet, "submit", "--batch", "b", "--raw", "--res", "a100")

    def test_confirm_needs_a_clean_pushed_snapshot(self, fleet: Fleet):
        code = fleet.conf.root / "code" / "abababababab" / "CODE.json"
        write_json(code, {**read_json(code), "dirty": True})
        with pytest.raises(SystemExit, match="confirm"):
            _fleet(
                fleet, "submit", "--batch", "b", "+experiment=m14/s0", "phase=confirm"
            )


class TestControl:
    def test_stop_and_cancel_of_queued_and_running_runs(self, fleet: Fleet):
        queued = fleet.submit(name="q")
        running = fleet.submit(name="r", seconds=60, frames=600)
        claim = fleet.nas.claim(fleet.nas.entries()[1], "h0")
        assert (
            claim is not None
            and QueueName.parse(claim.name).run_id == running["run_id"]
        )
        assert (
            _fleet(
                fleet, "stop", queued["run_id"], running["run_id"], "--reason", "pruned"
            )
            == 0
        )
        assert fleet.done(queued["run_id"])["status"] == "stopped"
        assert (
            read_json(claim.with_name(f"{running['run_id']}.stop"))["reason"]
            == "pruned"
        )
        assert _fleet(fleet, "cancel", "--batch", "b") == 0
        stop = read_json(claim.with_name(f"{running['run_id']}.stop"))
        assert stop["reason"] == "cancel" and stop["now"]

    def test_stop_of_a_queued_run_a_worker_claims_meanwhile_reaches_the_claim(
        self, fleet: Fleet, monkeypatch
    ):
        job = fleet.submit(seconds=60, frames=600)
        real = Nas.take_back
        claims = []

        def claimed_first(self, path):  # a worker renames it just before us
            claims.append(self.claim(path, "h1"))
            return real(self, path)

        monkeypatch.setattr(Nas, "take_back", claimed_first)
        assert _fleet(fleet, "stop", job["run_id"]) == 0
        (claim,) = claims
        assert claim is not None and claim.exists()
        assert not fleet.finished(job["run_id"])
        assert read_json(claim.with_name(f"{job['run_id']}.stop"))["reason"] == "user"

    def test_retry_puts_a_done_run_back_with_the_next_attempt(self, fleet: Fleet):
        job = fleet.submit(exit=1)
        w = fleet.worker()
        run_until([w], lambda: fleet.finished(job["run_id"]))
        assert _fleet(fleet, "retry", job["run_id"], "--fresh", "--ram", "4") == 0
        assert not fleet.finished(job["run_id"])
        (back,) = _queued(fleet)
        assert (back["attempt"], back["fresh"], back["res"]["ram_gb"]) == (2, True, 4.0)
        assert "status" not in back

    def test_requeue_a_hosts_claims_from_scratch(self, fleet: Fleet):
        job = fleet.submit()
        fleet.nas.claim(fleet.nas.entries()[0], "h1")
        assert _fleet(fleet, "requeue", "--host", "h1") == 0
        (back,) = _queued(fleet)
        assert (
            back["run_id"] == job["run_id"] and back["fresh"] and back["attempt"] == 2
        )
        assert fleet.nas.claims("h1") == []

    def test_release_forgets_the_failures(self, fleet: Fleet):
        (fleet.nas.failures / "b" / "sig").mkdir(parents=True)
        _fleet(fleet, "hold", "b")
        assert (fleet.nas.holds / "b").exists()
        _fleet(fleet, "release", "b")
        assert not (fleet.nas.holds / "b").exists()
        assert not (fleet.nas.failures / "b").exists()

    def test_cordon_and_uncordon(self, fleet: Fleet):
        _fleet(fleet, "cordon", "h0", "--res", "4090")
        _fleet(fleet, "cordon", "h0")
        assert set(fleet.nas.cordoned("h0")) == {"4090", "all"}
        _fleet(fleet, "uncordon", "h0", "--res", "all")
        assert set(fleet.nas.cordoned("h0")) == {"4090"}
        _fleet(fleet, "uncordon", "h0")
        assert fleet.nas.cordoned("h0") == {}


class TestState:
    def test_a_claim_wins_over_an_old_done_record(self, fleet: Fleet, capsys):
        job = fleet.submit(seconds=60, frames=600)
        claim = fleet.nas.claim(fleet.nas.entries()[0], "h0")
        assert claim is not None
        write_json(
            fleet.nas.done / f"{job['run_id']}.json", {**job, "status": "failed"}
        )
        capsys.readouterr()
        _fleet(fleet, "ls", "--json")
        (row,) = json.loads(capsys.readouterr().out)
        assert (row["state"], row["host"]) == ("running", "h0")
        assert _fleet(fleet, "stop", job["run_id"]) == 0
        assert claim.with_name(f"{job['run_id']}.stop").exists()

    def test_status_ls_and_locate(self, fleet: Fleet, capsys):
        done = fleet.submit(name="done")
        running = fleet.submit(name="run", seconds=60, frames=600)
        w = fleet.worker()
        run_until([w], lambda: fleet.finished(done["run_id"]) and bool(w.running))
        run_until(
            [w], lambda: (fleet.nas.live / running["run_id"] / "status.json").exists()
        )
        waiting = fleet.submit(name="wait", host="h1")
        capsys.readouterr()
        assert _fleet(fleet, "status", "--json") == 0
        st = json.loads(capsys.readouterr().out)
        assert st["counts"] == {"completed": 1, "running": 1, "queued": 1}
        assert st["hosts"]["h0"]["running"] == [running["run_id"]]
        assert st["hosts"]["h1"] is None
        assert st["running"][0]["status"]["of"] > 0
        assert _fleet(fleet, "status") == 0
        out = capsys.readouterr().out
        assert (
            "h0  beat" in out and "h1  no heartbeat" in out and running["run_id"] in out
        )
        assert _fleet(fleet, "ls", "--state", "queued") == 0
        out = capsys.readouterr().out
        assert waiting["run_id"] in out and running["run_id"] not in out
        assert _fleet(fleet, "locate", running["run_id"]) == 0
        loc = json.loads(capsys.readouterr().out)
        assert (loc["state"], loc["host"]) == ("running", "h0")
        assert Path(loc["run_dir"], "status.json").exists()
        _fleet(fleet, "stop", "--now", running["run_id"])
        run_until([w], lambda: not w.running)

    def test_table_reads_the_runs_on_the_nas(self, fleet: Fleet, capsys):
        jobs = [fleet.submit(seed=s) for s in (0, 1)]
        w = fleet.worker()
        run_until([w], lambda: all(fleet.finished(j["run_id"]) for j in jobs))
        capsys.readouterr()
        argv = ["--batch", "b", "--rows", "name", "--cols", "program"]
        argv += ["--value", "roll10/cv10", "--split", "val"]
        assert _fleet(fleet, "table", *argv) == 0
        assert "| t | 0.50 / 0.50 |" in capsys.readouterr().out

    def test_a_host_without_a_recent_heartbeat_shows_its_runs_as_lost(
        self, fleet: Fleet, capsys
    ):
        job = fleet.submit()
        fleet.nas.claim(fleet.nas.entries()[0], "h1")
        write_json(fleet.nas.hosts / "h1.json", {"time": 0})
        _fleet(fleet, "status", "--json")
        assert json.loads(capsys.readouterr().out)["counts"] == {"lost?": 1}
        assert job["run_id"]


class TestMac:
    def test_a_command_is_forwarded_to_the_heads_latest_venv(self, fleet: Fleet, capfd):
        fleet.submit()
        code = main(["--config", str(fleet.toml), "ls"], transport=LocalTransport())
        assert code == 0
        assert "queued" in capfd.readouterr().out

    def test_pull_a_batch_without_checkpoints_and_the_logs_of_a_run(
        self, fleet: Fleet, capfd, monkeypatch
    ):
        monkeypatch.setenv("COPYFILE_DISABLE", "1")  # no AppleDouble files from bsdtar
        job = fleet.submit(exit=1, error="boom in the run")
        w = fleet.worker()
        run_until([w], lambda: fleet.finished(job["run_id"]))
        nas_dir = fleet.nas.runs / "b" / job["run_id"]
        (nas_dir / "ckpt").mkdir()
        (nas_dir / "ckpt" / "0001.0.pt").write_bytes(b"weights")
        dest = fleet.tmp / "pulled"
        argv = [
            "--config",
            str(fleet.toml),
            "pull",
            "--batch",
            "b",
            "--dest",
            str(dest),
        ]
        assert main(argv, transport=LocalTransport()) == 0
        got = dest / "b" / job["run_id"]
        assert (got / "job.json").read_bytes() == (nas_dir / "job.json").read_bytes()
        assert (got / "exit" / "attempt-1.json").exists()
        assert not (got / "ckpt").exists()
        capfd.readouterr()
        argv = ["--config", str(fleet.toml), "logs", job["run_id"], "--err"]
        assert main(argv, transport=LocalTransport()) == 0
        assert "RuntimeError: boom in the run" in capfd.readouterr().out

    def test_mac_commands_refuse_local_and_host_commands_need_it(self, fleet: Fleet):
        with pytest.raises(SystemExit, match="Mac"):
            _fleet(fleet, "push")
        with pytest.raises(SystemExit, match="--local"):
            main(["--config", str(fleet.toml), "worker", "--host", "h0"])


class TestUpLocal:
    def test_starts_once_and_restarts(self, fleet: Fleet, capsys):
        pid_file = fleet.conf.root / "worker.pid"
        try:
            assert _fleet(fleet, "up-local", "--host", "h0") == 0
            first = int(pid_file.read_text())
            assert _fleet(fleet, "up-local", "--host", "h0") == 0
            assert "running" in capsys.readouterr().out
            assert int(pid_file.read_text()) == first
            assert _fleet(fleet, "up-local", "--host", "h0", "--restart") == 0
            wait_for(lambda: int(pid_file.read_text()) != first)
        finally:
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGTERM)
                except ProcessLookupError:
                    pass
