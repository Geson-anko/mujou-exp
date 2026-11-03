import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from exp.lib.fleet.queue import Nas, QueueName, nas_exists, nas_retry
from tests.lib.fleet.fleetkit import REPO, Fleet, run_until

_GRAB = """
import json, sys
from pathlib import Path
from exp.lib.fleet.queue import Nas
nas, host = Nas(Path(sys.argv[1])), sys.argv[2]
sys.stdin.readline()  # start together
got = []
for _ in range(3):
    for p in nas.entries():
        if nas.claim(p, host) is not None:
            got.append(p.name)
print(json.dumps(got))
"""


def _eio_then(real, n: int, *, after: bool = False):
    """``os.rename`` raising EIO ``n`` times; ``after``: the rename happens on
    the server but the client still sees EIO (once)."""
    calls = {"n": 0}

    def rename(src, dst):
        calls["n"] += 1
        if calls["n"] <= n:
            if after:
                real(src, dst)
            raise OSError(errno.EIO, "Input/output error")
        return real(src, dst)

    return rename


def _stat_eio(under: Path, n: int):
    """``os.stat`` raising EIO the first ``n`` times for the paths ``under``."""
    real, calls = os.stat, {"n": 0}

    def stat(path, *args, **kwargs):
        if str(path).startswith(str(under)) and calls["n"] < n:
            calls["n"] += 1
            raise OSError(errno.EIO, "Input/output error")
        return real(path, *args, **kwargs)

    return stat


class TestClaim:
    def test_two_processes_claim_each_job_once(self, tmp_path: Path):
        nas = Nas(tmp_path)
        nas.ensure("a")
        nas.ensure("b")
        for i in range(200):
            nas.submit({"prio": 50, "run_id": f"r{i:03d}"}, ns=i + 1)
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _GRAB, str(tmp_path), host],
                cwd=REPO,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            for host in ("a", "b")
        ]
        for p in procs:
            assert p.stdin is not None
            p.stdin.write("go\n")
            p.stdin.flush()
        got = [set(json.loads(p.communicate()[0])) for p in procs]
        assert got[0].isdisjoint(got[1])
        assert len(got[0] | got[1]) == 200
        assert {p.name for p in nas.claims("a")} == got[0]
        assert nas.entries() == []

    def test_eio_three_times_then_the_claim_goes_through(self, tmp_path, monkeypatch):
        nas = Nas(tmp_path)
        nas.ensure("a")
        path = nas.submit({"prio": 1, "run_id": "r"})
        monkeypatch.setattr(os, "rename", _eio_then(os.rename, 3))
        assert nas.claim(path, "a") == nas.claimed("a") / path.name
        assert (nas.claimed("a") / path.name).exists()

    def test_eio_after_the_server_renamed_is_still_ours(self, tmp_path, monkeypatch):
        nas = Nas(tmp_path)
        nas.ensure("a")
        path = nas.submit({"prio": 1, "run_id": "r"})
        monkeypatch.setattr(os, "rename", _eio_then(os.rename, 1, after=True))
        assert nas.claim(path, "a") == nas.claimed("a") / path.name

    def test_a_job_another_host_took_is_not_claimed(self, tmp_path: Path):
        nas = Nas(tmp_path)
        nas.ensure("a")
        nas.ensure("b")
        path = nas.submit({"prio": 1, "run_id": "r"})
        assert nas.claim(path, "b") is not None
        assert nas.claim(path, "a") is None


class TestRetry:
    def test_gives_up_after_the_tries(self):
        calls = []

        def fail():
            calls.append(1)
            raise OSError(errno.ESTALE, "stale")

        with pytest.raises(OSError):
            nas_retry(fail, tries=5, base=0.0)
        assert len(calls) == 5

    def test_does_not_retry_a_missing_file(self):
        calls = []

        def missing():
            calls.append(1)
            raise FileNotFoundError

        with pytest.raises(FileNotFoundError):
            nas_retry(missing, base=0.0)
        assert len(calls) == 1


class TestOrder:
    def test_smaller_prio_first_then_older(self, tmp_path: Path):
        nas = Nas(tmp_path)
        nas.ensure()
        nas.submit({"prio": 50, "run_id": "late"}, ns=2)
        nas.submit({"prio": 50, "run_id": "early"}, ns=1)
        nas.submit({"prio": 10, "run_id": "urgent"}, ns=3)
        order = [QueueName.parse(p.name).run_id for p in nas.entries()]
        assert order == ["urgent", "early", "late"]

    def test_ns_zero_is_a_time_not_a_default(self, tmp_path: Path):
        nas = Nas(tmp_path)
        nas.ensure()
        assert nas.submit({"prio": 1, "run_id": "r"}, ns=0).name == "001-0-r.json"


class TestExists:
    """``Path.exists`` says False to an EIO; the queue must not take an
    unreadable NAS for a missing file."""

    def test_only_a_missing_file_is_false(self, tmp_path: Path, monkeypatch):
        nas = Nas(tmp_path)
        nas.ensure()
        (nas.holds / "b").write_text("{}")
        assert nas_exists(nas.holds / "b") and not nas_exists(nas.holds / "c")
        monkeypatch.setattr(os, "stat", _stat_eio(tmp_path, 10**6))
        with pytest.raises(OSError) as e:
            nas.held("b")
        assert e.value.errno == errno.EIO

    def test_a_transient_eio_is_retried(self, tmp_path: Path, monkeypatch):
        (tmp_path / "f").write_text("x")
        monkeypatch.setattr(os, "stat", _stat_eio(tmp_path, 2))
        assert nas_exists(tmp_path / "f")


class TestBlocked:
    def test_pins_holds_and_after(self, fleet: Fleet):
        nas = fleet.nas
        base = fleet.submit(name="base")
        assert nas.blocked(base, "h0") is None
        assert nas.blocked({**base, "host": "h1"}, "h0") is not None
        assert nas.blocked({**base, "host": "h0"}, "h0") is None
        assert nas.blocked({**base, "avoid_host": "h0"}, "h0") is not None
        assert nas.blocked({**base, "avoid_host": "h0"}, "h1") is None
        after = {**base, "after": ["dep"]}
        assert nas.blocked(after, "h0") is not None
        (nas.done / "dep.json").write_text(
            json.dumps({"run_id": "dep", "status": "failed"})
        )
        assert nas.blocked(after, "h0") is not None
        (nas.done / "dep.json").write_text(
            json.dumps({"run_id": "dep", "status": "completed"})
        )
        assert nas.blocked(after, "h0") is None
        (nas.holds / "b").write_text("{}")
        assert nas.blocked(base, "h0") is not None

    def test_a_worker_waits_for_after_and_skips_a_pinned_job(self, fleet: Fleet):
        first = fleet.submit(name="first", seconds=0.5)
        second = fleet.submit(name="second", after=[first["run_id"]])
        pinned = fleet.submit(name="pinned", host="h1")
        w = fleet.worker()
        run_until([w], lambda: fleet.finished(second["run_id"]))
        assert (
            fleet.done(first["run_id"])["ended"]
            <= fleet.done(second["run_id"])["ended"]
        )
        started = json.loads(
            (
                fleet.nas.runs / "b" / second["run_id"] / "exit" / "attempt-1.json"
            ).read_text()
        )["started"]
        assert started >= fleet.done(first["run_id"])["ended"]
        assert not fleet.finished(pinned["run_id"])
        assert [QueueName.parse(p.name).run_id for p in fleet.nas.entries()] == [
            pinned["run_id"]
        ]
