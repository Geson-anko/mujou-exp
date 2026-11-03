"""The queue on the NAS (design 4.1, 4.5): small files, claimed by ``os.rename``.

::

    queue/<prio:03d>-<submitted_ns>-<run_id>.json   waiting (smaller prio first)
    claimed/<host>/<same name>                      claimed by host (running)
    claimed/<host>/<run_id>.stop                    a request to stop it
    done/<run_id>.json                              the job and how it ended
    holds/<batch>   failures/<batch>/<sig>/<run_id>  hosts/<host>.json  hosts/<host>.cordon
    live/<run_id>/  runs/<batch>/<run_id>/

The state of a run is read off the files: in ``queue/`` it is queued, in
``claimed/<host>/`` and not in ``done/`` it is running, else ``done/`` says.
The rename of SMB is exclusive on the server, so two hosts never claim the same
file. Every operation retries the transient errors of the CIFS mount
(``nas_retry``).
"""

import errno
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exp.lib.fleet.contract import read_json, write_json

__all__ = ["Nas", "QueueName", "nas_exists", "nas_retry"]

_TRANSIENT = frozenset(
    {errno.EIO, errno.ESTALE, errno.EAGAIN, errno.EBUSY, errno.ETIMEDOUT, errno.EINTR}
)
# The statuses of done/ that end a run for good (a duplicate is not resubmitted).
FINAL_OK = frozenset({"completed", "diverged"})


def nas_retry[T](fn: Callable[[], T], *, tries: int = 5, base: float = 0.05) -> T:
    """``fn()``, retried with exponential backoff (``base`` s, doubling) while it
    raises a transient ``OSError`` (EIO, ESTALE, ...), ``tries`` times in all."""
    for i in range(tries):
        try:
            return fn()
        except OSError as e:
            if e.errno not in _TRANSIENT or i == tries - 1:
                raise
            time.sleep(base * 2**i)
    raise AssertionError("unreachable")


def nas_exists(path: Path) -> bool:
    """Whether ``path`` exists, by ``os.stat`` under ``nas_retry``. Only a missing
    file is ``False``; any other ``OSError`` (an EIO that outlived the retries)
    is raised, unlike ``Path.exists``, which says ``False`` to every error."""
    try:
        nas_retry(lambda: os.stat(path))
    except FileNotFoundError:
        return False
    return True


@dataclass(frozen=True)
class QueueName:
    """The parts of a queue file's name."""

    prio: int
    ns: int
    run_id: str

    @classmethod
    def parse(cls, name: str) -> QueueName:
        """``"050-1730000000000000000-m14-s1-3fa29c01-s0.json"`` -> its parts."""
        prio, ns, run_id = name.removesuffix(".json").split("-", 2)
        return cls(int(prio), int(ns), run_id)

    @property
    def name(self) -> str:
        return f"{self.prio:03d}-{self.ns}-{self.run_id}.json"


class Nas:
    """The shared directory of the fleet at ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.queue = root / "queue"
        self.done = root / "done"
        self.holds = root / "holds"
        self.failures = root / "failures"
        self.hosts = root / "hosts"
        self.live = root / "live"
        self.runs = root / "runs"

    def claimed(self, host: str) -> Path:
        return self.root / "claimed" / host

    def ensure(self, host: str | None = None) -> None:
        """Make the directories (and ``claimed/<host>``)."""
        dirs = [self.queue, self.done, self.holds, self.failures, self.hosts]
        dirs += [self.live, self.runs]
        if host is not None:
            dirs.append(self.claimed(host))
        for d in dirs:
            nas_retry(lambda d=d: d.mkdir(parents=True, exist_ok=True))

    # -- the queue -------------------------------------------------------------

    def submit(self, job: dict[str, Any], *, ns: int | None = None) -> Path:
        """Put ``job`` in the queue (written aside, then renamed in)."""
        ns = time.time_ns() if ns is None else ns
        name = QueueName(int(job["prio"]), ns, job["run_id"])
        path = self.queue / name.name
        nas_retry(lambda: write_json(path, job))
        return path

    def entries(self) -> list[Path]:
        """The queue files, first to run first."""
        names = nas_retry(lambda: os.listdir(self.queue))
        return [self.queue / n for n in sorted(names) if _is_entry(n)]

    def claim(self, path: Path, host: str) -> Path | None:
        """Move the queue file ``path`` to ``claimed/<host>/``; ``None`` when
        another host took it first."""
        return self._take(path, self.claimed(host) / path.name)

    def take_back(self, path: Path) -> Path | None:
        """Move the queue file ``path`` aside (a hidden name in ``queue/``, which
        no worker claims), to drop it without racing a claim; ``None`` when a
        host claimed it first."""
        return self._take(path, path.with_name(f".{path.name}.taken"))

    def _take(self, path: Path, dst: Path) -> Path | None:
        for i in range(5):
            try:
                os.rename(path, dst)
                return dst
            except FileNotFoundError:
                # After an EIO the rename may have gone through on the server.
                return dst if i > 0 and nas_exists(dst) else None
            except OSError as e:
                if e.errno not in _TRANSIENT or i == 4:
                    raise
                time.sleep(0.05 * 2**i)
        return None

    def claims(self, host: str) -> list[Path]:
        """The claim files of ``host``."""
        d = self.claimed(host)
        names = nas_retry(lambda: os.listdir(d)) if nas_exists(d) else []
        return [d / n for n in sorted(names) if _is_entry(n)]

    def all_claims(self) -> Iterator[tuple[str, Path]]:
        """``(host, claim file)`` of every host."""
        base = self.root / "claimed"
        if not nas_exists(base):
            return
        for host in sorted(nas_retry(lambda: os.listdir(base))):
            for p in self.claims(host):
                yield host, p

    def finish(self, claim: Path, record: dict[str, Any]) -> None:
        """Write ``done/<run_id>.json`` and drop the claim (and its stop file)."""
        run_id = record["run_id"]
        nas_retry(lambda: write_json(self.done / f"{run_id}.json", record))
        self.unclaim(claim, run_id)

    def requeue(self, claim: Path, job: dict[str, Any]) -> Path:
        """Put the claimed job back in the queue under its old name (its place
        in the order), as ``job``."""
        nas_retry(lambda: write_json(claim, job))
        dst = self.queue / claim.name
        nas_retry(lambda: os.rename(claim, dst))
        self._drop(claim.with_name(f"{job['run_id']}.stop"))
        return dst

    def unclaim(self, claim: Path, run_id: str) -> None:
        self._drop(claim)
        self._drop(claim.with_name(f"{run_id}.stop"))

    def _drop(self, path: Path) -> None:
        try:
            nas_retry(path.unlink)
        except FileNotFoundError:
            pass

    # -- what the workers read -------------------------------------------------

    def done_record(self, run_id: str) -> dict[str, Any] | None:
        try:
            return nas_retry(lambda: read_json(self.done / f"{run_id}.json"))
        except FileNotFoundError:
            return None

    def drop_done(self, run_id: str) -> None:
        """Forget the done record of ``run_id`` (when there is one)."""
        self._drop(self.done / f"{run_id}.json")

    def held(self, batch: str) -> bool:
        return nas_exists(self.holds / batch)

    def cordoned(self, host: str) -> dict[str, str]:
        """The cordoned resource classes of ``host`` (``"all"`` for all) and why."""
        try:
            return nas_retry(lambda: read_json(self.hosts / f"{host}.cordon"))
        except FileNotFoundError:
            return {}

    def cordon(self, host: str, res: str, reason: str) -> None:
        classes = self.cordoned(host)
        classes[res] = reason
        nas_retry(lambda: write_json(self.hosts / f"{host}.cordon", classes))

    def uncordon(self, host: str, res: str | None = None) -> None:
        classes = {} if res is None else self.cordoned(host)
        classes.pop(res or "", None)
        path = self.hosts / f"{host}.cordon"
        if classes:
            nas_retry(lambda: write_json(path, classes))
        else:
            self._drop(path)

    def request_stop(self, claim: Path, run_id: str, *, reason: str, now: bool) -> None:
        stop = {"reason": reason, "now": now, "at": time.time()}
        nas_retry(lambda: write_json(claim.with_name(f"{run_id}.stop"), stop))

    def stop_request(self, claim: Path, run_id: str) -> dict[str, Any] | None:
        try:
            return nas_retry(lambda: read_json(claim.with_name(f"{run_id}.stop")))
        except FileNotFoundError:
            return None

    def blocked(self, job: dict[str, Any], host: str) -> str | None:
        """Why ``host`` may not claim ``job`` now (``None``: it may): pinned to
        another host, kept off this one, waiting for an ``after`` run to
        complete, or its batch on hold."""
        if job.get("host") not in (None, host):
            return f"pinned to {job['host']}"
        if job.get("avoid_host") == host:
            return f"avoids {host}"
        if self.held(job["batch"]):
            return f"batch {job['batch']} on hold"
        for dep in job.get("after", []):
            rec = self.done_record(dep)
            if rec is None or rec.get("status") != "completed":
                return f"after {dep}"
        return None


def _is_entry(name: str) -> bool:
    return name.endswith(".json") and not name.startswith(".")
