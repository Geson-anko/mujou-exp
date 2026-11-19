"""The judgement tables of V1-V5 (exp/vision/README.md) from posthoc rows.

``python -m exp.vision.tables runs/<batch> [...]`` prints one markdown row per
run: the val metrics at the last checkpoint against t = 0, the diagnostics,
and whether the run is stable and learned under the registered rule:

- stable: the run was not diverged (``result.json``'s status)
- learned: ``stack_mse`` below t = 0 and below 1, ``recon_blur`` below t = 0
  by more than twice the combined standard error of the two means of
  ``posthoc_seeds`` fits, and ``dyn_1/copy_1`` below t = 0
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["learned", "main", "row_of", "stream_learned", "stream_row_of"]

Row = Mapping[str, Any]


def _blur_drop(r0: Row, r1: Row) -> tuple[float, float]:
    """The drop of ``recon_blur`` from ``r0`` to ``r1`` and its threshold
    ``2 sqrt((s0^2 + s1^2) / k)``, ``k`` the seeds of each mean."""
    k = len(r1["recon_blur_k"])
    s0, s1 = r0["recon_blur_sd"], r1["recon_blur_sd"]
    return r0["recon_blur"] - r1["recon_blur"], 2 * math.sqrt((s0**2 + s1**2) / k)


def learned(r0: Row, r1: Row) -> dict[str, bool]:
    """The four conditions of 'learned' for the val rows at t = 0 (``r0``)
    and at the last checkpoint (``r1``), and their conjunction ``all``."""
    drop, thr = _blur_drop(r0, r1)
    c = {
        "stack_mse_down": r1["stack_mse"] < r0["stack_mse"],
        "stack_mse_lt1": r1["stack_mse"] < 1,
        "recon_blur_down": drop > thr,
        "dyn_down": r1["dyn_1/copy_1"] < r0["dyn_1/copy_1"],
    }
    c["all"] = all(c.values())
    return c


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _fmt(x: Any, digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def row_of(run: Path) -> str:
    """One markdown row for the run directory ``run``."""
    status = json.loads((run / "result.json").read_text()).get("status", "?")
    path = run / "posthoc" / "evals.jsonl"
    rows = [r for r in _jsonl(path) if r["split"] == "val"] if path.exists() else []
    rows.sort(key=lambda r: r["t"])
    if len(rows) < 2 or status == "diverged":
        return f"| {run.name} | {status} | " + " | ".join(["—"] * 9) + " | × |"
    r0, r1 = rows[0], rows[-1]
    c = learned(r0, r1)
    drop, thr = _blur_drop(r0, r1)
    cells = [
        run.name,
        status,
        f"{_fmt(r0['stack_mse'], 3)} → {_fmt(r1['stack_mse'], 3)}",
        f"{_fmt(r1['recon_blur'])} ± {_fmt(r1['recon_blur_sd'])}",
        f"{_fmt(drop)} / {_fmt(thr)}",
        f"{_fmt(r0['dyn_1/copy_1'], 3)} → {_fmt(r1['dyn_1/copy_1'], 3)}",
        f"{_fmt(r0.get('dec_spread'), 3)} → {_fmt(r1.get('dec_spread'), 3)}",
        f"{_fmt(r1.get('recent_stack_mse'), 3)} ({_fmt(r1.get('recent_stack_mse_init'), 3)})",
        _fmt(r1.get("recent_dec_spread"), 3),
        _fmt(r1.get("rank"), 1),
        "○"
        if c["all"]
        else "×（" + ", ".join(k for k, v in c.items() if not v and k != "all") + "）",
    ]
    return "| " + " | ".join(cells) + " |"


def stream_learned(s0: Row, s1: Row, p0: Row, p1: Row) -> dict[str, bool]:
    """The four conditions of 'learned' on the run's own stream (V8,
    ``exp.vision.stream_eval``) from the summaries (``s0``, ``s1``) and the
    probes of the group ``span`` (``p0``, ``p1``) at t = 0 and at the last
    checkpoint, and their conjunction ``all``: the span (all past at the
    end) below t = 0 and below 1, the after-span below t = 0, ``recon_blur``
    of the span below t = 0 by more than twice the combined standard error,
    and ``dyn_1/copy_1`` of the span below t = 0."""
    drop, thr = _blur_drop(p0, p1)
    c = {
        "span_down": s1["span"] < s0["span"],
        "span_lt1": s1["span"] < 1,
        "after_down": s1["after"] < s0["after"],
        "recon_blur_down": drop > thr,
        "dyn_down": p1["dyn_1/copy_1"] < p0["dyn_1/copy_1"],
    }
    c["all"] = all(c.values())
    return c


def stream_row_of(run: Path) -> str:
    """One markdown row of the stream evaluation of the run directory
    ``run`` (``STREAM_HEADER``)."""
    status = json.loads((run / "result.json").read_text()).get("status", "?")
    d = run / "posthoc_stream"
    have = (d / "summary.jsonl").exists() and (d / "probes.jsonl").exists()
    summ = sorted(_jsonl(d / "summary.jsonl"), key=lambda r: r["t"]) if have else []
    probes = (
        [r for r in _jsonl(d / "probes.jsonl") if r["group"] == "span"] if have else []
    )
    probes.sort(key=lambda r: r["t"])
    if status == "diverged" or len(summ) < 2 or len(probes) < 2:
        return f"| {run.name} | {status} | " + " | ".join(["—"] * 9) + " | × |"
    s0, s1, p0, p1 = summ[0], summ[-1], probes[0], probes[-1]
    c = stream_learned(s0, s1, p0, p1)
    drop, thr = _blur_drop(p0, p1)
    forgot = [
        _fmt(_minus(s1.get(g), s1.get(f"pre_{g}")), 3)
        for g in ("past_10_60", "past_60_300", "past_300_")
    ]
    cells = [
        run.name,
        status,
        f"{_fmt(s0['span'], 3)} → {_fmt(s1['span'], 3)}",
        f"{_fmt(s0['after'], 3)} → {_fmt(s1['after'], 3)}",
        f"{_fmt(s1.get('now'), 3)} ({_fmt(s1.get('pre_now'), 3)})",
        " / ".join(
            _fmt(s1.get(g), 3) for g in ("past_10_60", "past_60_300", "past_300_")
        ),
        " / ".join(forgot),
        f"{_fmt(p1['recon_blur'])} ± {_fmt(p1['recon_blur_sd'])} ({_fmt(drop)} / {_fmt(thr)})",
        f"{_fmt(p0['dyn_1/copy_1'], 3)} → {_fmt(p1['dyn_1/copy_1'], 3)}",
        _fmt(p1.get("rank"), 1),
        "○"
        if c["all"]
        else "×（" + ", ".join(k for k, v in c.items() if not v and k != "all") + "）",
    ]
    return "| " + " | ".join(cells) + " |"


def _minus(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


STREAM_HEADER = (
    "| run | status | span 0 → T | after 0 → T | now（pre） "
    "| past 10-60 / 60-300 / 300- s | 忘却 10-60 / 60-300 / 300- s "
    "| recon_blur span（差 / 閾値） | dyn_1/copy_1 span 0 → T | rank | 学べた |\n"
    + "|"
    + " -- |" * 11
)


HEADER = (
    "| run | status | stack_mse 0 → 1800 | recon_blur 1800 | 差 / 閾値 "
    "| dyn_1/copy_1 0 → 1800 | dec_spread 0 → 1800 | recent_stack_mse（init） "
    "| recent_dec_spread | rank | 学べた |\n" + "|" + " -- |" * 11
)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exp.vision.tables")
    ap.add_argument("batches", nargs="+", type=Path, help="runs/<batch> dirs")
    ap.add_argument(
        "--stream", action="store_true", help="the stream evaluation (posthoc_stream)"
    )
    args = ap.parse_args(argv)
    sub, head, row = (
        ("posthoc_stream/summary.jsonl", STREAM_HEADER, stream_row_of)
        if args.stream
        else ("posthoc/evals.jsonl", HEADER, row_of)
    )
    print(head)
    for batch in args.batches:
        for run in sorted(p for p in batch.iterdir() if (p / "result.json").exists()):
            if (run / sub).exists() or (
                json.loads((run / "result.json").read_text()).get("status")
                == "diverged"
            ):
                print(row(run))


if __name__ == "__main__":
    main()
