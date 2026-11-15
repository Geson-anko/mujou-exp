"""A NAS in a temporary directory and a stand-in for the workers of the fleet.

The coordinator sees the fleet only through the NAS: the job files of
``queue/`` and ``claimed/<host>/``, the copies of ``live/<run_id>/``, the
records of ``done/`` and the finished runs of ``runs/<batch>/<run_id>/``.
``FakeFleet`` writes those files as a worker would (``exp.lib.fleet.worker``),
so the tests drive a study by hand: claim a run, show its evaluations, end it.
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

from exp.lib.fleet.contract import read_json, write_json
from exp.lib.fleet.queue import Nas, QueueName

SNAP_A = "a" * 40
SNAP_B = "b" * 40


def code_info(snap: str = SNAP_A, *, dirty: bool = False) -> dict[str, Any]:
    """The ``CODE.json`` of a snapshot, as ``fleet push`` writes it."""
    return {
        "snap": snap,
        "base_head": "c" * 40,
        "dirty": dirty,
        "mujou_snap": "d" * 40,
        "mujou_pushed": True,
        "envkey": "e" * 12,
    }


_GROUPS = ("optim", "learner", "dynamics", "phase")


def fake_compose(lists: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """A stand-in for ``exp.video.compose``: the overrides as a nested dict
    (``+experiment=x`` sets the name, a group choice sets ``<group>.kind``), so
    that different overrides give different configs and run_ids."""
    out = []
    for overrides in lists:
        cfg: dict[str, Any] = {"program": "dynamics", "name": "m14/x", "seed": 0}
        for o in overrides:
            key, _, raw = o.partition("=")
            key = key.lstrip("+~")
            try:
                value: Any = json.loads(raw)
            except ValueError:
                value = raw
            if key == "experiment":
                cfg["name"] = value
            elif key == "bad":
                raise ValueError(f"these do not compose:\n{o}")
            elif key in _GROUPS:
                cfg.setdefault(key, {})["kind"] = value
                if key == "phase":
                    cfg["phase"] = value
            else:
                node = cfg
                *parents, last = key.split(".")
                for p in parents:
                    node = node.setdefault(p, {})
                node[last] = value
        out.append(cfg)
    return out


def row(t: float, objective: float, split: str = "val") -> dict[str, Any]:
    """A line of ``evals.jsonl`` (only ``t``, ``split`` and ``objective`` matter)."""
    return {"t": t, "split": split, "roll10": 0.1, "objective": objective}


class FakeFleet:
    """Writes the NAS files of the workers for the jobs the coordinator queued."""

    def __init__(self, nas: Nas) -> None:
        self.nas = nas

    def queued(self) -> list[dict[str, Any]]:
        """The jobs in the queue, first to run first."""
        return [read_json(p) for p in self.nas.entries()]

    def job(self, run_id: str) -> dict[str, Any]:
        for p in self.nas.entries():
            if QueueName.parse(p.name).run_id == run_id:
                return read_json(p)
        for _, p in self.nas.all_claims():
            if QueueName.parse(p.name).run_id == run_id:
                return read_json(p)
        raise KeyError(run_id)

    def claim(self, run_id: str, host: str = "std-01") -> Path:
        """Move the queue file of ``run_id`` to ``claimed/<host>/``."""
        self.nas.ensure(host)
        for p in self.nas.entries():
            if QueueName.parse(p.name).run_id == run_id:
                got = self.nas.claim(p, host)
                assert got is not None
                return got
        raise KeyError(run_id)

    def claim_all(self, run_ids: Iterable[str], host: str = "std-01") -> None:
        for r in run_ids:
            self.claim(r, host)

    def live(self, run_id: str, rows: Sequence[dict[str, Any]]) -> None:
        """``live/<run_id>/evals.jsonl``, as the worker copies it every minute."""
        d = self.nas.live / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_json(d / "status.json", {"state": "running", "t": rows[-1]["t"]})

    def end(
        self,
        run_id: str,
        status: str,
        rows: Sequence[dict[str, Any]] = (),
        *,
        objective: float | None = None,
        t_end: float | None = None,
        reason: str | None = None,
        final: dict[str, Any] | None = None,
    ) -> None:
        """End the claimed run ``run_id``: for ``completed`` and ``diverged``
        its directory on the NAS (``evals.jsonl``, ``result.json`` with
        ``final``, and the ``job.json``), then the ``done/`` record, the claim
        dropped and ``live/`` removed."""
        claim = next(
            p
            for _, p in self.nas.all_claims()
            if QueueName.parse(p.name).run_id == run_id
        )
        job = read_json(claim)
        if status in ("completed", "diverged"):
            d = self.nas.runs / job["batch"] / run_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            result: dict[str, Any] = {"schema": 1, "status": status}
            if objective is not None:
                result["objective"] = objective
            if t_end is not None:
                result["t_end"] = t_end
            if final is not None:
                result["final"] = final
            write_json(d / "result.json", result)
            write_json(d / "job.json", job)
        self.nas.finish(claim, {**job, "status": status, "reason": reason})
        live = self.nas.live / run_id
        for f in live.glob("*"):
            f.unlink()
        if live.exists():
            live.rmdir()

    def stops(self) -> dict[str, dict[str, Any]]:
        """The stop requests on the claims, by run_id."""
        return {
            p.name.removesuffix(".stop"): read_json(p)
            for p in (self.nas.root / "claimed").glob("*/*.stop")
        }


@pytest.fixture
def nas(tmp_path: Path) -> Nas:
    n = Nas(tmp_path / "nas")
    n.ensure("std-01")
    return n


@pytest.fixture
def fleet(nas: Nas) -> FakeFleet:
    return FakeFleet(nas)
