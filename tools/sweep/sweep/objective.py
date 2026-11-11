"""The value of a trial from the evaluations of its seeds' runs (design 5.2).

A run writes the objective itself: ``objective`` on each ``split == "val"``
line of ``evals.jsonl`` and in ``result.json`` (``exp.video.measure.objective``,
$J = \\frac12 \\sum_{h \\in \\{5, 10\\}} \\log(E^{val}_h / CV^{val}_h)$, minimized;
0 is the constant-velocity extrapolation). This module only combines the
seeds: the trial's value is the mean of $J_s$ over its seeds, a diverged seed
counting as $J_s = \\log 10$ (an error 10 times the extrapolation's) from the
time it diverged on, so that TPE learns where runs diverge. A nonfinite
objective counts as diverged.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

__all__ = [
    "DIVERGED",
    "OBJECTIVE",
    "SeedRun",
    "posthoc_value",
    "reports",
    "seed_value",
    "trial_value",
]

DIVERGED = math.log(10.0)
OBJECTIVE = "objective"  # the key in the val lines of evals.jsonl and result.json


@dataclass(frozen=True)
class SeedRun:
    """What the NAS shows of one seed's run: its ``status`` (``"queued"``,
    ``"running"``, or how it ended), the lines of its ``evals.jsonl`` so far
    (only the val lines count), its ``result.json`` once completed or
    diverged, and the ``reason`` the fleet gave for how it ended."""

    status: str
    rows: Sequence[Mapping[str, Any]] = ()
    result: Mapping[str, Any] | None = None
    reason: str | None = None

    @cached_property
    def val(self) -> list[Mapping[str, Any]]:
        """The lines that carry the objective (``split == "val"``)."""
        return [
            r
            for r in self.rows
            if r.get("split") == "val" and r.get(OBJECTIVE) is not None
        ]

    @property
    def diverged_at(self) -> float | None:
        """The virtual time it diverged at (``t_end`` of the result, else its
        last evaluation, else 0); ``None`` unless it diverged."""
        if self.status != "diverged":
            return None
        t_end = (self.result or {}).get("t_end")
        if t_end is not None:
            return float(t_end)
        return float(self.val[-1]["t"]) if self.val else 0.0

    def value_at(self, t: float) -> float | None:
        """$J_s$ at the evaluation of time ``t``; ``None`` until it got there."""
        div = self.diverged_at
        if div is not None and t >= div:
            return DIVERGED
        for r in self.val:
            if float(r["t"]) == t:
                return _penalized(float(r[OBJECTIVE]))
        return None


def seed_value(run: SeedRun) -> float:
    """$J_s$ of a finished seed: ``log 10`` when it diverged, else the
    objective of its result (or of its last val line). ``ValueError`` when a
    completed run has no objective at all."""
    if run.status == "diverged":
        return DIVERGED
    got = (run.result or {}).get(OBJECTIVE)
    if got is None and run.val:
        got = run.val[-1][OBJECTIVE]
    if got is None:
        raise ValueError("a completed run without an objective")
    return _penalized(float(got))


def posthoc_value(run: SeedRun) -> float:
    """$J_s$ of a completed ``task=posthoc`` job (M17): the objective of its
    val line, ``final.decode.val.objective`` of its ``result.json``.
    ``ValueError`` when it has none."""
    final = (run.result or {}).get("final") or {}
    got = ((final.get("decode") or {}).get("val") or {}).get(OBJECTIVE)
    if got is None:
        raise ValueError("a posthoc job without final.decode.val.objective")
    return _penalized(float(got))


def trial_value(runs: Sequence[SeedRun]) -> float:
    """The mean of $J_s$ over the seeds (all finished)."""
    return sum(seed_value(r) for r in runs) / len(runs)


def reports(runs: Sequence[SeedRun], *, after: int) -> list[tuple[int, float]]:
    """``(step, mean J_s)`` of each evaluation time after the step ``after``
    that every seed has reached, in order; the step is the time rounded to
    the second. Stops at the first time some seed has not reached, so the
    reports of a trial never skip a time. The times are those of the seeds
    still on the schedule (a diverged seed may have stopped between two)."""
    alive = [run for run in runs if run.status != "diverged"] or runs
    times = sorted({float(r["t"]) for run in alive for r in run.val})
    out: list[tuple[int, float]] = []
    for t in times:
        if round(t) <= after:
            continue
        values = [run.value_at(t) for run in runs]
        if any(v is None for v in values):
            break
        out.append((round(t), sum(v for v in values if v is not None) / len(values)))
    return out


def _penalized(j: float) -> float:
    return j if math.isfinite(j) else DIVERGED
