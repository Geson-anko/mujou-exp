"""The runs of a study as the NAS shows them (``exp/lib/fleet/README.md``).

A run is queued while its file is in ``queue/``, running while it is in
``claimed/<host>/`` (its ``status.json`` and ``evals.jsonl`` copied to
``live/<run_id>/`` every minute), and else ended as ``done/<run_id>.json``
says, a completed or diverged run kept in ``runs/<batch>/<run_id>/``. A queue
or claim file wins over a ``done/`` record (one of an earlier attempt). A run
in none of them is ``absent``: between two renames of a worker for a moment,
or never queued.
"""

import json
from pathlib import Path
from typing import Any

from exp.lib.fleet.contract import read_json
from exp.lib.fleet.queue import FINAL_OK, Nas, QueueName, nas_retry
from sweep.objective import SeedRun

__all__ = ["FleetView"]


class FleetView:
    """The queue and the claims read once per ``refresh``; the ``done/``
    records and the run files read when asked."""

    def __init__(self, nas: Nas) -> None:
        self.nas = nas
        self._queued: dict[str, Path] = {}
        self._claimed: dict[str, Path] = {}

    def refresh(self) -> None:
        """Read the queue, then the claims (a claim made in between shows in
        the second; a run that ends in between has its ``done/`` record)."""
        self._queued = {QueueName.parse(p.name).run_id: p for p in self.nas.entries()}
        self._claimed = {
            QueueName.parse(p.name).run_id: p for _, p in self.nas.all_claims()
        }

    def queued(self, run_id: str, path: Path) -> None:
        """Note the queue file ``path`` of ``run_id`` just submitted."""
        self._queued[run_id] = path

    def is_taken(self, run_id: str) -> bool:
        """Whether ``run_id`` is queued, running, or ended for good (completed
        or diverged): submitting it again would be a duplicate."""
        if run_id in self._queued or run_id in self._claimed:
            return True
        rec = self.nas.done_record(run_id)
        return rec is not None and rec.get("status") in FINAL_OK

    def stopping(self, run_id: str) -> bool:
        """Whether ``run_id`` runs with a request to stop on its claim."""
        claim = self._claimed.get(run_id)
        return claim is not None and self.nas.stop_request(claim, run_id) is not None

    def seed_run(self, run_id: str) -> SeedRun:
        """The status, the evaluations so far and the result of ``run_id``."""
        if run_id in self._queued:
            return SeedRun(status="queued")
        if run_id in self._claimed:
            return SeedRun(
                status="running",
                rows=_jsonl(self.nas.live / run_id / "evals.jsonl"),
            )
        rec = self.nas.done_record(run_id)
        if rec is None:
            return SeedRun(status="absent")
        status = str(rec["status"])
        if status not in FINAL_OK:
            return SeedRun(status=status, reason=rec.get("reason"))
        d = self.nas.runs / rec["batch"] / run_id
        try:
            result = nas_retry(lambda: read_json(d / "result.json"))
        except FileNotFoundError:
            result = None
        return SeedRun(status=status, rows=_jsonl(d / "evals.jsonl"), result=result)

    def stop(self, run_id: str, *, reason: str) -> str:
        """Stop ``run_id`` at once: a queued one is taken out of the queue to
        ``done/`` as ``stopped`` (``"stopped"``); a running one gets a stop
        request on its claim (``"requested"``); else ``"gone"``."""
        path = self._queued.pop(run_id, None)
        if path is not None:
            taken = self.nas.take_back(path)
            if taken is not None:
                job = nas_retry(lambda: read_json(taken))
                record = {**job, "status": "stopped", "reason": reason, "host": None}
                self.nas.finish(taken, record)
                return "stopped"
            self.refresh()  # a worker claimed it meanwhile
        claim = self._claimed.get(run_id)
        if claim is None:
            return "gone"
        self.nas.request_stop(claim, run_id, reason=reason, now=True)
        return "requested"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """The lines of ``path`` (none when it is missing); a partial last line,
    copied while the run wrote it, is left out."""
    try:
        text = nas_retry(path.read_text)
    except FileNotFoundError:
        return []
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
