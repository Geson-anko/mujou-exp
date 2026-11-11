"""The worker: one per host, in the container (design 4.6, 4.10).

Each tick it watches its runs (how they ended, stop requests, stalls, claims
taken away, the live files), restarts the runs it lost, claims what fits from
the queue, and writes its heartbeat. A run is a process of its own session
(``start_new_session``), so the worker can die and come back: ``recover``
adopts the runs that still live (``<root>/state/<run_id>.json`` keeps their
pid) and restarts the dead ones from their resume checkpoint.

How a run ended (``outcome``), by its exit code and the files it left:

=========================  ==============================================
0 / 3 + result + manifest  completed / diverged: archived, verified, done
0 / 3 without them         failed (contract)
4                          the host's class is cordoned, the job requeued
75, or killed on request   stopped (pruned, user, cancel); drain / move
                           requeue; stall is failed
SIGKILL, no request        oom: requeued once with 1.5x RAM, then failed
other signal, other code   failed; 3 alike in a batch put it on hold
gone while adopted         by the last ``state`` of its ``status.json``
                           (``failed``, ``env``, ``stopped``); else lost:
                           restarted here (``retry.lost`` times)
=========================  ==============================================

An adopted run is not a child of this worker, so its exit code is lost; the
final ``state`` the program writes to ``status.json`` stands in for it.

The NAS may fail for a while: a claim the worker cannot see is not counted as
gone, and a run whose end cannot be written yet stays to be ended again on the
next tick. Any other error in ending a run (a malformed ``result.json``, ...)
ends it as ``failed`` with the reason ``worker: <error>``.
"""

import hashlib
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exp.lib.fleet.config import FleetConf
from exp.lib.fleet.contract import (
    EXIT_DIVERGED,
    EXIT_ENV,
    EXIT_OK,
    EXIT_STOPPED,
    copy_run,
    read_json,
    verify_manifest,
    write_json,
)
from exp.lib.fleet.jobs import DEFAULT_ENTRY
from exp.lib.fleet.provision import venv_python
from exp.lib.fleet.queue import Nas, QueueName, nas_exists, nas_retry
from exp.lib.fleet.slots import Slot, Use, allocate, physical_cores, units

__all__ = ["Timing", "Worker", "outcome"]

log = logging.getLogger("fleet.worker")

_REQUEUE_STOPS = frozenset({"drain", "move"})
_KEEP_ON_FRESH = frozenset({"logs", "exit"})
_ADOPTED_FAILED = "exit ? (adopted; status.json says failed)"
_EXC_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning)\b.*")


@dataclass(frozen=True, kw_only=True)
class Timing:
    """The worker's periods, in seconds: the tick, the copies to ``live/``, the
    grace a stop request gets before SIGTERM and then SIGKILL, how long a run's
    frame may stand still, and how many ticks a claim may be missing before
    the run is killed (fencing)."""

    tick: float = 5.0
    live: float = 60.0
    grace: float = 120.0
    kill: float = 10.0
    stall: float = 1200.0
    absent: int = 2


def outcome(
    rc: int | None, stop_reason: str | None, run_dir: Path
) -> tuple[str, str | None]:
    """``(status, reason)`` of a run that ended with exit code ``rc`` (negative:
    the signal; ``None``: unknown, the run was adopted) after a stop request
    with ``stop_reason`` (``None``: none). ``status`` is one of
    ``completed``, ``diverged``, ``failed``, ``stopped``, ``requeue``, ``env``,
    ``oom``, ``lost``."""
    sealed: dict[str, Any] | None = None  # result.json, when manifest.json is there too
    if (run_dir / "manifest.json").exists():
        try:
            got = read_json(run_dir / "result.json")
        except OSError, ValueError:
            got = None
        sealed = got if isinstance(got, dict) else None
    if rc is None:
        if sealed is not None and sealed.get("status") in ("completed", "diverged"):
            return sealed["status"], sealed.get("reason")
        if stop_reason:
            return _stopped(stop_reason)
        state = _read_status(run_dir).get("state")
        if state == "failed":
            return "failed", _ADOPTED_FAILED
        if state == "env":
            return "env", None
        if state == "stopped":
            return "stopped", "exit 75"
        return "lost", None
    if rc in (EXIT_OK, EXIT_DIVERGED):
        if sealed is None:
            return (
                "failed",
                f"contract: exit {rc} without result.json and manifest.json",
            )
        return ("completed" if rc == EXIT_OK else "diverged"), sealed.get("reason")
    if rc == EXIT_ENV:
        return "env", None
    if rc == EXIT_STOPPED or (rc < 0 and stop_reason):
        return _stopped(stop_reason)
    if rc == -signal.SIGKILL:
        return "oom", None
    if rc < 0:
        return "failed", f"signal {signal.Signals(-rc).name}"
    return "failed", f"exit {rc}"


def _stopped(reason: str | None) -> tuple[str, str | None]:
    if reason == "stall":
        return "failed", "stall"
    if reason in _REQUEUE_STOPS:
        return "requeue", reason
    return "stopped", reason or "exit 75"


@dataclass(kw_only=True)
class _Run:
    job: dict[str, Any]
    claim: Path
    run_dir: Path
    attempt: int
    slot: Slot
    started: float
    pid: int
    proc: subprocess.Popen[bytes] | None  # None: adopted from an earlier worker
    stop: dict[str, Any] | None = None  # the request, with "seen" (our clock)
    reaped: bool = False  # its process group was killed after it ended
    term_at: float | None = None
    missing: int = 0
    frame: Any = None
    frame_at: float = 0.0
    live_at: float = field(default=float("-inf"))

    @property
    def run_id(self) -> str:
        return self.job["run_id"]

    @property
    def use(self) -> Use:
        res = self.job["res"]
        return Use(self.slot, float(res["ram_gb"]), float(res.get("gpu_mem_gb", 0.0)))


class Worker:
    """The worker of ``host``: runs claimed jobs on the slots of ``conf`` over
    the physical ``cores`` (read from ``/sys`` by default)."""

    def __init__(
        self,
        conf: FleetConf,
        host: str,
        *,
        timing: Timing = Timing(),
        cores: list[tuple[int, ...]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._conf = conf
        self._host = host
        self._timing = timing
        self._cores = cores if cores is not None else physical_cores()
        self._clock = clock
        self._nas = Nas(conf.nas)
        self._root = conf.root
        self._state = conf.root / "state"
        self._runs: dict[str, _Run] = {}
        self._pending: list[tuple[dict[str, Any], Path]] = []  # lost, to restart
        self._jobs: dict[str, dict[str, Any]] = {}  # queue file name -> job
        self._snap = Path.cwd().resolve().name
        self._gpu_free: list[float] | None = None
        self._gpu_at = float("-inf")

    @property
    def running(self) -> list[str]:
        """The run_ids running here."""
        return sorted(self._runs)

    # -- life ------------------------------------------------------------------

    def recover(self) -> None:
        """Take up where an earlier worker of this host left off: adopt the runs
        that live, restart the lost ones, finish the ones that ended, and kill
        the ones whose claim is gone."""
        self._runs.clear()
        self._pending.clear()
        self._nas.ensure(self._host)
        self._state.mkdir(parents=True, exist_ok=True)
        claims = {
            QueueName.parse(p.name).run_id: p for p in self._nas.claims(self._host)
        }
        states: dict[str, dict[str, Any]] = {}
        for f in self._state.glob("*.json"):
            try:
                st = read_json(f)
                st["pid"], st["pgid"] = int(st["pid"]), int(st["pgid"])
            except (OSError, ValueError, KeyError, TypeError) as e:
                log.error("%s: unreadable state (%s); taken as lost", f.stem, e)
                continue
            states[f.stem] = st
        for run_id, st in states.items():
            if run_id not in claims:
                if _alive(int(st["pid"]), run_id):
                    log.info("%s: claim gone while away; killing it", run_id)
                    _killpg(int(st["pgid"]), signal.SIGKILL)
                (self._state / f"{run_id}.json").unlink(missing_ok=True)
        for run_id, claim in claims.items():
            try:
                job = nas_retry(lambda claim=claim: read_json(claim))
            except ValueError as e:
                log.error("%s: unreadable claim %s (%s); left alone", run_id, claim, e)
                continue
            st = states.get(run_id)
            if st is None:  # claimed, never started (or its state was lost)
                self._lost(job, claim, attempt=_last_attempt(self._run_dir(job)))
                continue
            run = _Run(
                job=job,
                claim=claim,
                run_dir=Path(st["run_dir"]),
                attempt=int(st["attempt"]),
                slot=Slot(
                    tuple(st["slot"]["cpus"]), st["slot"]["gpu"], st["slot"]["model"]
                ),
                started=float(st["started"]),
                pid=int(st["pid"]),
                proc=None,
                stop=st.get("stop"),
                frame_at=self._clock(),
            )
            self._runs[run_id] = run
            if _alive(run.pid, run_id):
                log.info("%s: adopted (pid %d)", run_id, run.pid)
            else:
                self._settle(run, None)

    def serve(self, stop: threading.Event) -> None:
        """``recover`` (again each tick until it goes through), then tick until
        ``stop`` is set."""
        while not stop.is_set():
            try:
                self.recover()
                break
            except Exception:
                log.exception("recovery failed; trying again")
                stop.wait(self._timing.tick)
        while not stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("tick failed")
            stop.wait(self._timing.tick)

    def tick(self) -> None:
        """One round: watch the runs, restart the lost, claim, heartbeat."""
        now = self._clock()
        for run in list(self._runs.values()):
            try:
                self._watch(run, now)
            except Exception:
                log.exception("%s: watching failed", run.run_id)
        try:
            self._restart_lost()
            self._claim()
        except OSError:
            log.exception("claiming failed (the NAS?)")
        try:
            self._heartbeat(now)
        except OSError:
            log.exception("heartbeat failed (the NAS?)")

    # -- watching a run --------------------------------------------------------

    def _watch(self, run: _Run, now: float) -> None:
        if run.proc is not None:
            rc = run.proc.poll()
            ended = rc is not None
        else:
            rc, ended = None, not _alive(run.pid, run.run_id)
        if ended:
            self._settle(run, rc)
            return
        try:
            if not nas_exists(run.claim):
                run.missing += 1
                if run.missing >= self._timing.absent:
                    self._fence(run)
                return
            run.missing = 0
            if run.stop is None:
                request = self._nas.stop_request(run.claim, run.run_id)
                if request is not None:
                    self._stop(run, request, now)
        except OSError as e:  # not a missing claim: the NAS; the run goes on
            log.warning("%s: cannot read its claim (%s); it goes on", run.run_id, e)
        frame = _read_status(run.run_dir).get("frame")
        if frame != run.frame:
            run.frame, run.frame_at = frame, now
        elif run.stop is None and now - run.frame_at > self._timing.stall:
            log.warning(
                "%s: frame %s for %.0f s; stopping",
                run.run_id,
                frame,
                now - run.frame_at,
            )
            self._stop(run, {"reason": "stall", "now": False}, now)
        if run.stop is not None:
            if run.term_at is None:
                if run.stop.get("now") or now - run.stop["seen"] > self._timing.grace:
                    _killpg(run.pid, signal.SIGTERM)
                    run.term_at = now
            elif now - run.term_at > self._timing.kill:
                _killpg(run.pid, signal.SIGKILL)
        if now - run.live_at >= self._timing.live:
            run.live_at = now
            self._copy_live(run)

    def _stop(self, run: _Run, request: dict[str, Any], now: float) -> None:
        run.stop = {**request, "seen": now}
        write_json(
            run.run_dir / "stop.request", {"reason": request["reason"], "at": now}
        )
        self._save(run)

    def _fence(self, run: _Run) -> None:
        log.warning("%s: the claim is gone; killing it, not archiving", run.run_id)
        _killpg(run.pid, signal.SIGKILL)
        if run.proc is not None:
            run.proc.wait()
        del self._runs[run.run_id]
        (self._state / f"{run.run_id}.json").unlink(missing_ok=True)

    def _copy_live(self, run: _Run) -> None:
        dst = self._nas.live / run.run_id
        nas_retry(lambda: dst.mkdir(parents=True, exist_ok=True))
        for name in ("status.json", "evals.jsonl"):
            src = run.run_dir / name
            if src.exists():
                nas_retry(lambda src=src, name=name: shutil.copyfile(src, dst / name))

    # -- the end of a run ------------------------------------------------------

    def _settle(self, run: _Run, rc: int | None) -> None:
        """``_end`` the run; on an ``OSError`` (the NAS) it stays in ``_runs`` and
        is ended again next tick; any other error ends it as ``failed``."""
        try:
            self._end(run, rc)
        except OSError:
            log.exception("%s: ending failed (the NAS?); again next tick", run.run_id)
        except Exception as e:
            log.exception("%s: ending failed; it is failed", run.run_id)
            job = {**run.job, "attempt": run.attempt}
            reason = f"worker: {type(e).__name__}: {e}"
            try:
                self._finish(run, job, "failed", reason, rc)
            except OSError:
                log.exception("%s: finishing failed (the NAS?)", run.run_id)
                return
            self._forget(run)

    def _forget(self, run: _Run) -> None:
        self._runs.pop(run.run_id, None)
        (self._state / f"{run.run_id}.json").unlink(missing_ok=True)

    def _end(self, run: _Run, rc: int | None) -> None:
        # What is left of its group; not for an adopted run, whose pid may be reused.
        if run.proc is not None and not run.reaped:
            _killpg(run.pid, signal.SIGKILL)
            run.reaped = True
        reason_in = run.stop["reason"] if run.stop else None
        sig = signal.Signals(-rc).name if rc is not None and rc < 0 else None
        exit_dir = run.run_dir / "exit"
        exit_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            exit_dir / f"attempt-{run.attempt}.json",
            {
                "returncode": rc,
                "signal": sig,
                "started": run.started,
                "ended": self._clock(),
                "stop_reason": reason_in,
                "adopted": run.proc is None,
            },
        )
        status, reason = outcome(rc, reason_in, run.run_dir)
        job = dict(run.job)
        job["attempt"] = run.attempt
        retry, retried = job.get("retry", {}), dict(job.get("retried", {}))
        log.info(
            "%s: attempt %d ended (%s): %s %s",
            run.run_id,
            run.attempt,
            rc,
            status,
            reason or "",
        )
        if status == "env":
            self._nas.cordon(self._host, run.slot.model, f"exit 4 of {run.run_id}")
            self._requeue(run, job)
        elif status == "requeue":
            self._requeue(run, job)
        elif status == "oom" and retried.get("oom", 0) < retry.get("oom", 0):
            job["res"] = {**job["res"], "ram_gb": float(job["res"]["ram_gb"]) * 1.5}
            job["retried"] = {**retried, "oom": retried.get("oom", 0) + 1}
            self._requeue(run, job)
        elif status == "lost" and retried.get("lost", 0) < retry.get("lost", 0):
            job["retried"] = {**retried, "lost": retried.get("lost", 0) + 1}
            job.pop("fresh", None)
            self._lost(job, run.claim, attempt=run.attempt)
        else:
            if status in ("oom", "lost"):
                status, reason = "failed", status
            if status == "failed" and (
                (rc is not None and rc > 0 and reason != "stall")
                or reason == _ADOPTED_FAILED
            ):
                self._mark_failure(run, job)
            status = self._archive(run, status)
            self._finish(run, job, status, reason, rc)
        self._forget(run)

    def _requeue(self, run: _Run, job: dict[str, Any]) -> None:
        job["attempt"] = run.attempt + 1
        self._nas.requeue(run.claim, job)
        self._drop_live(run.run_id)

    def _lost(self, job: dict[str, Any], claim: Path, *, attempt: int) -> None:
        job["attempt"] = attempt + 1
        nas_retry(lambda: write_json(claim, job))
        self._pending.append((job, claim))

    def _archive(self, run: _Run, status: str) -> str:
        """Copy the run to ``runs/<batch>/<run_id>/`` of the NAS; for a finished
        run, check it there by its manifest (3 tries, then ``corrupt``)."""
        dst = self._nas.runs / run.job["batch"] / run.run_id
        for _ in range(3):
            try:
                # An earlier attempt's files go, but for its logs and exit records.
                nas_retry(lambda: _clear(dst, _KEEP_ON_FRESH))
                nas_retry(lambda: copy_run(run.run_dir, dst))
            except OSError:
                log.exception("%s: copy to the NAS failed", run.run_id)
                continue
            if status not in ("completed", "diverged"):
                return status
            problem = verify_manifest(dst)
            if problem is None:
                shutil.rmtree(run.run_dir / "resume", ignore_errors=True)
                return status
            log.warning("%s: the copy on the NAS: %s", run.run_id, problem)
        return "corrupt" if status in ("completed", "diverged") else status

    def _finish(
        self,
        run: _Run,
        job: dict[str, Any],
        status: str,
        reason: str | None,
        rc: int | None,
    ) -> None:
        record = {
            **job,
            "status": status,
            "reason": reason,
            "host": self._host,
            "exit": rc,
            "slot": {
                "cpus": list(run.slot.cpus),
                "gpu": run.slot.gpu,
                "model": run.slot.model,
            },
            "ended": self._clock(),
        }
        self._nas.finish(run.claim, record)
        self._drop_live(run.run_id)

    def _mark_failure(self, run: _Run, job: dict[str, Any]) -> None:
        """``failures/<batch>/<sig>/<run_id>``; at 3 alike, ``holds/<batch>``."""
        line = _error_line(run.run_dir / "logs" / f"attempt-{run.attempt}.err")
        sig = hashlib.sha256(f"{job['program']}|{line}".encode()).hexdigest()[:12]
        d = self._nas.failures / job["batch"] / sig
        nas_retry(lambda: d.mkdir(parents=True, exist_ok=True))
        nas_retry(lambda: (d / run.run_id).write_text(line + "\n"))
        n = len(nas_retry(lambda: os.listdir(d)))
        if n >= 3 and not self._nas.held(job["batch"]):
            log.warning("batch %s on hold: %d failures of %s", job["batch"], n, line)
            hold = {
                "reason": f"{n} failures alike: {line}",
                "sig": sig,
                "at": self._clock(),
            }
            nas_retry(lambda: write_json(self._nas.holds / job["batch"], hold))

    def _drop_live(self, run_id: str) -> None:
        shutil.rmtree(self._nas.live / run_id, ignore_errors=True)

    # -- starting runs ---------------------------------------------------------

    def _used(self) -> list[Use]:
        return [r.use for r in self._runs.values()]

    def _slot_for(self, job: dict[str, Any], cordoned: dict[str, str]) -> Slot | None:
        res = job["res"]
        return allocate(
            self._conf.slots,
            self._cores,
            self._used(),
            res_class=res["class"],
            ram_gb=float(res["ram_gb"]),
            gpu_mem_gb=float(res.get("gpu_mem_gb", 0.0)),
            cordoned=cordoned,
        )

    def _restart_lost(self) -> None:
        waiting, self._pending = self._pending, []
        for job, claim in waiting:
            slot = self._slot_for(job, {})
            if slot is None:
                self._pending.append((job, claim))
            else:
                self._launch(job, claim, slot)

    def _ready(self, job: dict[str, Any]) -> bool:
        code = job["code"]
        return (self._root / "code" / code["snap"][:12] / ".ready").exists() and (
            self._root / "venv" / code["envkey"] / ".ready"
        ).exists()

    def _claim(self) -> None:
        entries = self._nas.entries()
        names = {p.name for p in entries}
        self._jobs = {n: j for n, j in self._jobs.items() if n in names}
        cordoned = self._nas.cordoned(self._host)
        n_units = len(units(self._conf.slots, self._cores))
        here = set(self._runs) | {j["run_id"] for j, _ in self._pending}
        for path in entries:
            if len(self._runs) + len(self._pending) >= n_units:
                break
            if QueueName.parse(path.name).run_id in here:
                continue  # requeued while its process is still ours (to be fenced)
            job = self._jobs.get(path.name)
            if job is None:
                try:
                    job = nas_retry(lambda path=path: read_json(path))
                except FileNotFoundError:
                    continue  # another host claimed it
                self._jobs[path.name] = job
            slot = self._slot_for(job, cordoned) if self._ready(job) else None
            if slot is None or self._nas.blocked(job, self._host) is not None:
                continue
            claim = self._nas.claim(path, self._host)
            if claim is not None:
                self._launch(job, claim, slot)

    def _run_dir(self, job: dict[str, Any]) -> Path:
        return self._root / "runs" / job["batch"] / job["run_id"]

    def _launch(self, job: dict[str, Any], claim: Path, slot: Slot) -> None:
        run_dir = self._run_dir(job)
        resume = (run_dir / "resume" / "latest").exists() and not job.get("fresh")
        if not resume:
            _clear(run_dir, _KEEP_ON_FRESH)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "stop.request").unlink(
            missing_ok=True
        )  # a request to an earlier attempt
        attempt = max(int(job.get("attempt", 1)), _last_attempt(run_dir) + 1)
        if not (run_dir / "job.json").exists():
            spec = {**job, "config": {**job["config"], "device": slot.device}}
            spec.pop("fresh", None)
            write_json(run_dir / "job.json", spec)
        job = {k: v for k, v in job.items() if k != "fresh"}
        code = job["code"]
        argv = [
            str(venv_python(self._root, code["envkey"])),
            "-m",
            job.get("entry", DEFAULT_ENTRY),
            "--job",
            str(run_dir / "job.json"),
            "--attempt",
            str(attempt),
        ]
        if shutil.which("taskset"):
            argv = ["taskset", "-c", ",".join(map(str, slot.cpus)), *argv]
        env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
        env.update(
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
            PYTHONFAULTHANDLER="1",
            PYTHONPYCACHEPREFIX="/tmp/pycache",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            CUDA_VISIBLE_DEVICES="" if slot.gpu is None else str(slot.gpu),
            FLEET_HOST=self._host,
        )
        env.setdefault("MUJOU_EXP_DATA", str(self._root / "data"))
        logs = run_dir / "logs"
        with (
            (logs / f"attempt-{attempt}.out").open("wb") as out,
            (logs / f"attempt-{attempt}.err").open("wb") as err,
        ):
            proc = subprocess.Popen(
                argv,
                cwd=self._root / "code" / code["snap"][:12],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        now = self._clock()
        run = _Run(
            job=job,
            claim=claim,
            run_dir=run_dir,
            attempt=attempt,
            slot=slot,
            started=now,
            pid=proc.pid,
            proc=proc,
            frame_at=now,
        )
        self._runs[run.run_id] = run
        self._save(run)
        log.info("%s: attempt %d on %s (pid %d)", run.run_id, attempt, slot, proc.pid)

    def _save(self, run: _Run) -> None:
        self._state.mkdir(parents=True, exist_ok=True)
        write_json(
            self._state / f"{run.run_id}.json",
            {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "attempt": run.attempt,
                "pid": run.pid,
                "pgid": run.pid,
                "started": run.started,
                "slot": {
                    "cpus": list(run.slot.cpus),
                    "gpu": run.slot.gpu,
                    "model": run.slot.model,
                },
                "stop": run.stop,
            },
        )

    # -- heartbeat -------------------------------------------------------------

    def _heartbeat(self, now: float) -> None:
        used = self._used()
        slots = self._conf.slots
        if shutil.which("nvidia-smi") and now - self._gpu_at >= self._timing.live:
            self._gpu_free, self._gpu_at = _gpu_free_mb(), now
        beat = {
            "host": self._host,
            "time": now,
            "pid": os.getpid(),
            "snap": self._snap,
            "cpu": [len(used), len(units(slots, self._cores))],
            "ram_gb": [sum(u.ram_gb for u in used), slots.ram_gb],
            "gpus": [
                {
                    "index": g.index,
                    "model": g.model,
                    "jobs": [sum(u.slot.gpu == g.index for u in used), g.jobs],
                    "mem_gb": [
                        sum(u.gpu_mem_gb for u in used if u.slot.gpu == g.index),
                        g.mem_gb,
                    ],
                }
                for g in slots.gpus
            ],
            "gpu_free_mb": self._gpu_free,
            "running": self.running,
            "pending": [j["run_id"] for j, _ in self._pending],
            "load": list(os.getloadavg()),
            "mem_avail_gb": _mem_available_gb(),
        }
        nas_retry(lambda: write_json(self._nas.hosts / f"{self._host}.json", beat))


def _killpg(pgid: int, sig: int) -> None:
    if pgid <= 0 or pgid == os.getpgid(0):
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError, PermissionError:
        pass


def _alive(pid: int, run_id: str) -> bool:
    """Whether ``pid`` is a live process (not a zombie) of the run ``run_id``."""
    if Path("/proc/self").exists():
        try:
            return run_id.encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False
    ps = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat=,command="],
        capture_output=True,
        text=True,
        check=False,
    )
    out = ps.stdout.strip()
    return bool(out) and not out.startswith("Z") and run_id in out


def _last_attempt(run_dir: Path) -> int:
    """The highest attempt with a log in ``run_dir`` (0 for none)."""
    numbers = [
        int(m.group(1))
        for p in (run_dir / "logs").glob("attempt-*.out")
        if (m := re.fullmatch(r"attempt-(\d+)\.out", p.name))
    ]
    return max(numbers, default=0)


def _read_status(run_dir: Path) -> dict[str, Any]:
    try:
        got = read_json(run_dir / "status.json")
    except OSError, ValueError:
        return {}
    return got if isinstance(got, dict) else {}


def _clear(d: Path, keep: frozenset[str]) -> None:
    """Delete what the directory ``d`` holds but the top-level names ``keep``."""
    if not nas_exists(d):
        return
    for p in d.iterdir():
        if p.name in keep:
            continue
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()


def _error_line(err: Path) -> str:
    """The exception line of the traceback at the end of ``err`` (else its last line)."""
    try:
        lines = [
            x.strip() for x in err.read_text(errors="replace").splitlines() if x.strip()
        ]
    except OSError:
        return "(no stderr)"
    for line in reversed(lines):
        if _EXC_LINE.match(line):
            return line
    return lines[-1] if lines else "(empty stderr)"


def _mem_available_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def _gpu_free_mb() -> list[float] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except OSError, subprocess.SubprocessError:
        return None
    return [float(x) for x in out.split()]
