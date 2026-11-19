r"""The gradient statistics of runs (``grad.jsonl``, ``measure.grad_stats``;
``exp.lib.grad_stats``) as a table: per node and window of time, the mean
cosine of the gradient with a past one of about each age, and the mean
$\rho_s$::

    python -m exp.vision.grad_table runs/<batch>/<run> [...]

prints one markdown row per run, node and window. The windows are 0-300 s
($t \le 300$) and 300-1800 s ($300 < t \le 1800$); a row past 1800 s is
left out. The cosines are of the effective ages:

- ``cos_prev``: with the previous gradient, one step old (0.1 s at frames
  of 0.1 s);
- ``cos_<L>`` for $L$ of ``SPANS`` (1, 10, 100 s): with the anchor of span
  $L$, of the rows where it is at least ``NEAR`` $L$ old (an anchor is
  taken again each $L$, so its age runs from 0 to about $L$): the cosine at
  an age of about $L$.

A mean of no value is ``None``.
"""

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["COSINES", "NEAR", "RHOS", "SPANS", "WINDOWS", "main", "table"]

SPANS = (1.0, 10.0, 100.0)  # of the anchors (GradStats.anchors)
NEAR = 0.9  # an anchor's cosine counts at an age of at least NEAR * its span
COSINES = ("cos_prev", *(f"cos_{s:g}" for s in SPANS))
RHOS = ("rho_0.1", "rho_1", "rho_10")
WINDOWS = (("0-300", 0.0, 300.0), ("300-1800", 300.0, 1800.0))


def _cosines(row: Mapping[str, Any]) -> Iterable[tuple[str, float]]:
    """``(column, cos)`` of ``row``: the previous gradient's, and each
    anchor's at an age of at least ``NEAR`` times its span."""
    if row.get("cos_prev") is not None:
        yield "cos_prev", row["cos_prev"]
    for span in SPANS:
        cos, age = row.get(f"cos_{span:g}"), row.get(f"age_{span:g}")
        if cos is not None and age is not None and age >= NEAR * span:
            yield f"cos_{span:g}", cos


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def table(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The rows of the table of the ``grad.jsonl`` rows ``rows``: ``{"node",
    "window", "rows", "cos_prev", "cos_1", "cos_10", "cos_100", "rho_0.1",
    "rho_1", "rho_10"}``, by node (in order of first appearance) and
    window; a window without rows is left out."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        for name, lo, hi in WINDOWS:
            if lo < row["t"] <= hi:
                groups.setdefault((row["node"], name), []).append(row)
    out = []
    for (node, window), group in groups.items():
        cos: dict[str, list[float]] = {c: [] for c in COSINES}
        for row in group:
            for column, value in _cosines(row):
                cos[column].append(value)
        entry: dict[str, Any] = {"node": node, "window": window, "rows": len(group)}
        entry |= {c: _mean(cos[c]) for c in COSINES}
        entry |= {k: _mean([r[k] for r in group if r.get(k) is not None]) for k in RHOS}
        out.append(entry)
    order = {name: i for i, (name, _, _) in enumerate(WINDOWS)}
    nodes = list(dict.fromkeys(node for node, _ in groups))
    return sorted(out, key=lambda e: (nodes.index(e["node"]), order[e["window"]]))


HEADER = (
    "| run | node | window | rows | cos prev (1 step) | cos 1 s (age ≥ 0.9 s) "
    "| cos 10 s (age ≥ 9 s) | cos 100 s (age ≥ 90 s) | ρ 0.1 | ρ 1 | ρ 10 |\n"
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
            cells += [_fmt(e[c]) for c in COSINES]
            cells += [_fmt(e[k]) for k in RHOS]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
