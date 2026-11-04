"""The host's loop of a run, the same for every program (design 3.6): it reads
the job, checks the environment, resumes, drives the program unit by unit
(frames of the stream, or steps of an offline fit), records, evaluates,
checkpoints, stops on request and ends with the exit code of the contract
(``exp.lib.rundir.Exit``).

A program (``Program``) builds a ``Built``, which advances one unit at a time
and is read between units only: its evaluation reads the live modules
(no copy). The loop decides when, from the ``Schedule`` the ``Built`` gives.
After unit ``u`` (counted from 1) it, in this order:

1. writes a line of ``stream.jsonl`` every ``log`` units (``Built.log``, the
   one point where the program synchronizes with its device): ``{"t",
   "<unit>s", ...}``; a value that is not finite there is a divergence;
2. evaluates ``"val"`` every ``evaluate`` units and at the last unit, and
   ``"test"`` (with ``test``) at each evaluation checkpoint and at the last
   unit: a line ``{"t", "<unit>", "split", ...}`` of ``evals.jsonl`` each, then
   ``fsync``; ``Built.diverged`` of a row is a divergence;
3. writes the evaluation checkpoint (``ckpt/<t>.pt``) every ``ckpt`` units and
   the pictures (``viz/<t>/``) every ``viz`` units;
4. writes a resume checkpoint every ``resume`` units, and when ``stop.request``
   is there (then exits with ``Exit.STOPPED``).

``t`` is ``u`` times the seconds of a unit (the frame), or ``u`` itself for a
program whose units are steps. Nothing but ``Built`` decides the metrics: a
schedule of other ``log``, ``ckpt``, ``viz`` or ``resume`` intervals leaves the
validation rows as they are, and a run resumed from a checkpoint writes the
bytes a run never stopped writes (both are tested).

Non-finite values are written as ``null`` (strict JSON). ``status.json``
(the heartbeat, every ``STATUS_EVERY`` seconds of wall time) and
``timing.jsonl`` hold all that depends on the wall clock; before the process
exits, ``status.json`` gets the final ``state``: ``completed``, ``diverged``,
``stopped``, ``env`` (an ``EnvError``) or ``failed`` (any other exception).
"""

import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import torch
import yaml
from torch import nn

from exp.lib import ckpt
from exp.lib.provenance import assert_free_threaded, collect, configure_determinism
from exp.lib.rundir import METRICS, EnvError, Exit, RunDir
from exp.video.compose import from_dict, to_dict
from exp.video.schema.run import RunConf

__all__ = ["STATUS_EVERY", "Built", "Program", "Schedule", "every", "run"]

STATUS_EVERY = 30.0  # seconds of wall time between two heartbeats
_SPLITS = ("val", "test")


def every(seconds: float, unit: float) -> int:
    """The units in ``seconds`` of virtual time, for units of ``unit``
    seconds: ``round(seconds / unit)``, at least 1; 0 for ``seconds`` 0
    (never)."""
    if seconds <= 0:
        return 0
    return max(1, round(seconds / unit))


@dataclass(frozen=True, kw_only=True)
class Schedule:
    """When the loop does what, in units (module docstring); 0: never.

    ``units`` is the length of the run (0: a run that only computes its
    ``Built.final``); ``unit`` names a unit in the files (``"frame"``,
    ``"step"``); ``seconds`` is a unit's virtual time (``None``: ``t`` counts
    the units).
    """

    units: int
    unit: str = "frame"
    seconds: float | None = None
    log: int = 0
    evaluate: int = 0
    ckpt: int = 0
    viz: int = 0
    resume: int = 0
    test: bool = False

    def t(self, u: int) -> float:
        """The time of unit ``u``."""
        return float(u) if self.seconds is None else u * self.seconds


class Built(Protocol):
    """A program built for one run, which the loop drives (module docstring).

    Every method but ``advance`` is called between units and must not change
    what the next units compute (no random draws from shared generators, no
    writes into the modules).
    """

    @property
    def schedule(self) -> Schedule: ...

    def advance(self) -> None:
        """Run one unit: show a frame, or take a step; accumulate on the
        device, without synchronizing."""
        ...

    def log(self) -> dict[str, float | None]:
        """The values of a line of ``stream.jsonl`` since the last one (the
        prequential error), and reset the sums."""
        ...

    def evaluate(self, split: str) -> dict[str, float]:
        """The metrics of ``split`` (``"val"`` or ``"test"``) now; the rows of
        ``"val"`` carry ``"objective"``."""
        ...

    def diverged(self, row: Mapping[str, float]) -> str | None:
        """Why ``row`` (of ``evaluate``) says the run diverged, or ``None``."""
        ...

    def objective(
        self, row: Mapping[str, Any] | None, *, diverged: bool
    ) -> float | None:
        """The objective of ``result.json`` from the last ``"val"`` row."""
        ...

    def modules(self) -> dict[str, nn.Module]:
        """The live modules, by name, for the evaluation checkpoint."""
        ...

    def visualize(self, out: Path) -> None:
        """Write the pictures to look at into the directory ``out``."""
        ...

    def resume_state(self) -> tuple[Mapping[str, Any], dict[str, Any]]:
        """The tree of mujou (``run.state_dict()``; ``{}`` without a run) and
        the host's state (tensors and plain values) to resume from."""
        ...

    def final(self) -> dict[str, dict[str, Any]]:
        """The entries of ``result.json``'s ``final`` beyond the last rows of
        each split (e.g. the baselines of ``task="baselines"``)."""
        ...

    def close(self) -> None:
        """Release the run (its mujou run, its files)."""
        ...


class Program(Protocol):
    """What ``exp.video.main`` runs: ``program`` of a job names the module, its
    ``PROGRAM`` is an instance."""

    def build(
        self, conf: RunConf, run_dir: Path, resume: Mapping[str, Any] | None
    ) -> Built:
        """The ``Built`` of ``conf`` in ``run_dir``: fresh, or restored from
        the resume checkpoint ``resume`` (the tree of ``ckpt.load_resume``,
        whose ``"mujou"`` and ``"host"`` are of ``Built.resume_state``)."""
        ...


def _finite(value: object) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def _clean(row: Mapping[str, Any]) -> dict[str, Any]:
    """``row`` with the floats that are not finite as ``None``."""
    return {k: (v if _finite(v) else None) for k, v in row.items()}


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak(device: str) -> dict[str, float]:
    """The peak memory of the process: resident (MB) and CUDA allocated (MB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1024  # bytes / KB
    gpu = 0.0
    if device.startswith("cuda"):
        gpu = torch.cuda.max_memory_allocated(torch.device(device)) / 2**20
    return {"rss_mb": round(rss_mb, 1), "gpu_mem_mb": round(gpu, 1)}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


class _Status:
    """``status.json``, the heartbeat the worker reads."""

    def __init__(self, rd: RunDir, attempt: int) -> None:
        self._rd = rd
        self._attempt = attempt
        self.device = "cpu"
        self.unit, self.t, self.of = 0, 0.0, 0.0
        self.fps: float | None = None
        self.eta: float | None = None

    def write(self, state: str) -> None:
        self._rd.write_json(
            "status.json",
            {
                "state": state,
                "attempt": self._attempt,
                "pid": os.getpid(),
                "pgid": os.getpgid(0),
                "frame": self.unit,
                "t": self.t,
                "of": self.of,
                "fps": self.fps,
                "eta": self.eta,
                **_peak(self.device),
                "wall": _now(),
            },
        )


def _write_code(rd: RunDir, job: Mapping[str, Any]) -> None:
    """``CODE.json`` and ``code.diff``: those of the code tree the worker
    started the run in (the cwd), else what the job says (a local run, whose
    ``exp.video.main`` writes them first)."""
    for name in ("CODE.json", "code.diff"):
        if (rd.root / name).exists():
            continue
        if Path(name).is_file():
            shutil.copyfile(name, rd.root / name)
        elif name == "CODE.json":
            rd.write_json(name, dict(job.get("code", {})))
        else:
            (rd.root / name).write_text("")


def _last_rows(rd: RunDir) -> dict[str, dict[str, Any]]:
    """The last row of each split in ``evals.jsonl`` (after a resume cut it
    back)."""

    path = rd.root / "evals.jsonl"
    last: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                last[str(row["split"])] = row
    return last


def run(program: Program, job: Path, *, attempt: int) -> int:
    """Run the job ``job`` (its ``job.json``; the run's directory is its
    parent) as attempt ``attempt`` and return the exit code (module
    docstring). Exceptions do not escape: an ``EnvError`` gives
    ``Exit.ENV``, any other ``Exit.ERROR``, each with its traceback on
    stderr and the final ``state`` in ``status.json``."""
    rd = RunDir(job.parent)
    status = _Status(rd, attempt)
    try:
        return _Loop(program, rd, status, attempt).run()
    except EnvError:
        traceback.print_exc()
        status.write("env")
        return Exit.ENV
    except Exception:
        traceback.print_exc()
        status.write("failed")
        return Exit.ERROR
    finally:
        rd.close()


class _Loop:
    def __init__(
        self, program: Program, rd: RunDir, status: _Status, attempt: int
    ) -> None:
        self._program = program
        self._rd = rd
        self._status = status
        self._attempt = attempt
        self._job = rd.read_job()
        self._evaluated = False  # the GIL was checked after an evaluation

    def run(self) -> int:
        rd, job = self._rd, self._job
        conf = from_dict(job["config"])
        self._conf = conf
        self._status.device = conf.device
        assert_free_threaded("import")
        configure_determinism(conf.device)
        rd.write_json(
            f"provenance/attempt-{self._attempt}.json",
            collect(job, attempt=self._attempt),
        )
        _write_code(rd, job)
        (rd.root / "config.yaml").write_text(
            yaml.safe_dump(to_dict(conf), sort_keys=False)
        )
        tree = ckpt.load_resume(rd.root, group=job["group"])
        if tree is None:  # a fresh start: no lines of an earlier one
            rd.truncate(dict.fromkeys(METRICS, 0))
            u = 0
        else:
            rd.truncate(tree["offsets"])
            u = int(tree["frame"])
        self._status.unit = u
        self._status.write("running")
        built = self._program.build(conf, rd.root, tree)
        try:
            assert_free_threaded("build")
            return self._drive(built, u)
        finally:
            built.close()

    # -- the units --------------------------------------------------------------

    def _drive(self, built: Built, u: int) -> int:
        s = self._s = built.schedule
        rd, status = self._rd, self._status
        self._last = _last_rows(rd)
        self._stream = rd.jsonl("stream.jsonl")
        self._evals = rd.jsonl("evals.jsonl")
        self._timing = rd.jsonl("timing.jsonl")
        status.of = s.t(s.units)
        self._start, self._wall0 = u, time.monotonic()
        beat = self._wall0 + STATUS_EVERY
        while u < s.units:
            built.advance()
            u += 1
            reason = self._after(built, u)
            if reason is not None:
                return self._finish(built, u, diverged=reason)
            saved = bool(s.resume) and u % s.resume == 0
            if saved:
                self._save_resume(built, u)
            if rd.stop_requested():
                if not saved:
                    self._save_resume(built, u)
                self._beat(u)
                status.write("stopped")
                return Exit.STOPPED
            if time.monotonic() >= beat:
                self._beat(u)
                status.write("running")
                beat = time.monotonic() + STATUS_EVERY
        reason = None
        if u == self._start and s.units > 0:  # resumed after the last unit
            reason = self._evaluate_end(built, u)
        return self._finish(built, u, diverged=reason)

    def _after(self, built: Built, u: int) -> str | None:
        """Steps 1-3 of the module docstring after unit ``u``; returns why the
        run diverged, if it did."""
        s = built.schedule
        t = s.t(u)
        if s.log and u % s.log == 0:
            values = built.log()
            self._stream.append({"t": t, f"{s.unit}s": u, **_clean(values)})
            bad = [k for k, v in values.items() if not _finite(v)]
            if bad:
                return f"{', '.join(bad)} not finite at t={t:g}"
        ckpt_due = bool(s.ckpt) and u % s.ckpt == 0
        splits = []
        if s.evaluate and (u % s.evaluate == 0 or u == s.units):
            splits.append("val")
        if s.test and (ckpt_due or u == s.units):
            splits.append("test")
        reason = self._evaluate(built, u, splits)
        if ckpt_due:
            ckpt.save_eval(
                self._rd.root,
                program=self._conf.program,
                group=self._job["group"],
                t=t,
                frame=u,
                modules=built.modules(),
            )
        if s.viz and u % s.viz == 0:
            built.visualize(self._rd.root / "viz" / f"{t:07.1f}")
        return reason

    def _evaluate_end(self, built: Built, u: int) -> str | None:
        """The evaluations of the last unit ``u`` that ``evals.jsonl`` does not
        have yet, for a run resumed after its last unit."""
        s = built.schedule
        due = {"val": bool(s.evaluate), "test": s.test}
        splits = [
            x for x in _SPLITS if due[x] and self._last.get(x, {}).get(s.unit) != u
        ]
        return self._evaluate(built, u, splits)

    def _evaluate(self, built: Built, u: int, splits: list[str]) -> str | None:
        s = built.schedule
        t = s.t(u)
        reason = None
        for split in splits:
            wall = time.monotonic()
            values = built.evaluate(split)
            row = {"t": t, s.unit: u, "split": split, **values}
            self._evals.append(_clean(row))
            self._last[split] = _clean(row)
            self._timing.append(
                {
                    "wall": _now(),
                    "t": t,
                    "split": split,
                    "eval_s": time.monotonic() - wall,
                }
            )
            reason = reason or built.diverged(values)
        if splits:
            self._rd.sync()
            if not self._evaluated:
                assert_free_threaded("evaluate")
                self._evaluated = True
        return reason

    # -- checkpoints, heartbeats, the end ---------------------------------------

    def _save_resume(self, built: Built, u: int) -> None:
        mujou_state, host = built.resume_state()
        ckpt.save_resume(
            self._rd.root,
            group=self._job["group"],
            frame=u,
            mujou_state=mujou_state,
            host=host,
            offsets=self._rd.sync(),
        )

    def _beat(self, u: int) -> None:
        """Bring the heartbeat to unit ``u`` and record it in ``timing.jsonl``
        (``status.json`` is written by the caller, with its state)."""
        status = self._status
        elapsed = max(time.monotonic() - self._wall0, 1e-9)
        status.unit, status.t = u, self._s.t(u)
        status.fps = (u - self._start) / elapsed
        left = self._s.units - u
        status.eta = left / status.fps if status.fps > 0 else None
        self._timing.append({"wall": _now(), "unit": u, "fps": status.fps})

    def _finish(self, built: Built, u: int, *, diverged: str | None) -> int:
        rd, s = self._rd, built.schedule
        self._beat(u)
        rd.sync()
        final = dict(self._last)
        final |= built.final()
        final["timing"] = {"fps": self._status.fps}  # of this attempt
        result = {
            "schema": 1,
            "status": "completed" if diverged is None else "diverged",
            "reason": diverged,
            "t_end": s.t(u),
            f"{s.unit}s": u,
            "final": final,
            "criteria": self._criteria(final),
            "objective": built.objective(
                self._last.get("val"), diverged=diverged is not None
            ),
            "metrics_sha256": {
                name.removesuffix(".jsonl"): _sha256(rd.root / name) for name in METRICS
            },
            "peak": _peak(self._conf.device),
        }
        rd.write_json("result.json", result)
        ckpt.clear_resume(rd.root)
        rd.write_manifest()
        self._status.write(result["status"])
        return Exit.OK if diverged is None else Exit.DIVERGED

    def _criteria(self, final: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Each threshold of ``measure.criteria`` judged on the last row of
        ``"test"`` (``"val"`` without one): passed when the value is below."""
        row = final.get("test") or final.get("val") or {}
        out = {}
        for key, threshold in self._conf.measure.criteria.items():
            value = row.get(key)
            passed = isinstance(value, int | float) and value < threshold
            out[key] = {"value": value, "threshold": threshold, "pass": passed}
        return out
