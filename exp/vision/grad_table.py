r"""The gradient statistics of runs (``grad.jsonl``, ``measure.grad_stats``;
``exp.lib.grad_stats``) as a table: per node and window of time, the mean
cosine of the gradient with a past one by its age, and the mean $\rho_s$::

    python -m exp.vision.grad_table runs/<batch>/<run> [...]

prints one markdown row per run, node and window. The windows are 0-300 s
($t \le 300$) and 300-1800 s ($300 < t \le 1800$); a row past 1800 s is
left out. A cosine falls in the bin of the age 0.1, 1, 10 or 100 s nearest
to its age on a log scale (the bin of $10^k$ holds the ages in
$[10^{k - 1/2}, 10^{k + 1/2})$): the previous gradient (``cos_prev``, 0.1 s
at frames of 0.1 s) and the anchors of each span (``cos_<L>`` of
``age_<L>``, 0 to $L$ s old) alike. A mean of no value is ``None``.
"""

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["AGES", "RHOS", "WINDOWS", "main", "table"]

AGES = (0.1, 1.0, 10.0, 100.0)
RHOS = ("rho_0.1", "rho_1", "rho_10")
WINDOWS = (("0-300", 0.0, 300.0), ("300-1800", 300.0, 1800.0))
_BINS = {round(math.log10(a)): a for a in AGES}  # by the power of 10


def _bin(age: float) -> float | None:
    """The age of ``AGES`` whose bin holds ``age``, or ``None``."""
    if not age > 0:
        return None
    return _BINS.get(math.floor(math.log10(age) + 0.5))


def _pairs(row: Mapping[str, Any]) -> Iterable[tuple[float, float]]:
    """``(age, cos)`` of ``row``: the previous gradient and every anchor."""
    for key, value in row.items():
        if key.startswith("cos_") and value is not None:
            age = row.get("age_" + key.removeprefix("cos_"))
            if age is not None:
                yield age, value


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def table(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The rows of the table of the ``grad.jsonl`` rows ``rows``: ``{"node",
    "window", "rows", "cos_0.1", "cos_1", "cos_10", "cos_100", "rho_0.1",
    "rho_1", "rho_10"}``, by node (in order of first appearance) and
    window; a window without rows is left out."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        for name, lo, hi in WINDOWS:
            if lo < row["t"] <= hi:
                groups.setdefault((row["node"], name), []).append(row)
    out = []
    for (node, window), group in groups.items():
        cos: dict[float, list[float]] = {a: [] for a in AGES}
        for row in group:
            for age, value in _pairs(row):
                b = _bin(age)
                if b is not None:
                    cos[b].append(value)
        entry: dict[str, Any] = {"node": node, "window": window, "rows": len(group)}
        entry |= {f"cos_{a:g}": _mean(cos[a]) for a in AGES}
        entry |= {k: _mean([r[k] for r in group if r.get(k) is not None]) for k in RHOS}
        out.append(entry)
    order = {name: i for i, (name, _, _) in enumerate(WINDOWS)}
    nodes = list(dict.fromkeys(node for node, _ in groups))
    return sorted(out, key=lambda e: (nodes.index(e["node"]), order[e["window"]]))


HEADER = (
    "| run | node | window | rows | cos 0.1 s | cos 1 s | cos 10 s | cos 100 s "
    "| ρ 0.1 | ρ 1 | ρ 10 |\n"
    "| -- | -- | -- | --: | --: | --: | --: | --: | --: | --: | --: |"
)


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exp.vision.grad_table")
    ap.add_argument("runs", nargs="+", type=Path, help="run dirs with grad.jsonl")
    args = ap.parse_args(argv)
    print(HEADER)
    for run in args.runs:
        path = run / "grad.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        for e in table(json.loads(x) for x in lines if x.strip()):
            cells = [run.name, e["node"], e["window"], str(e["rows"])]
            cells += [_fmt(e[f"cos_{a:g}"]) for a in AGES]
            cells += [_fmt(e[k]) for k in RHOS]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
