"""The judgement of V17 (exp/vision/README.md) from the posthoc rows.

Reads ``runs/v17-b1/*/posthoc_stream`` and ``runs/v17-chunk/*/posthoc_stream``
(``summary.jsonl``, ``probes.jsonl``, ``z_use.jsonl``, all evaluated with
``--span 6000``; the times are of the single stream, ``eval.stream_view``)
and prints the markdown tables of the check against B = 1, the choice of
tau and the predictions; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v17.judge [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

CHECK_T = [600.0 * i for i in range(1, 11)]  # d(t) at 600 .. 6000 s
SHOWN_T = [
    1800.0,
    6000.0,
    18000.0,
    48000.0,
    60000.0,
    96000.0,
    180000.0,
    300000.0,
    480000.0,
    600000.0,
]
THRESHOLD = 0.07  # not told apart below this (V11-V16)
Z_MIN = 0.2


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def _at(rows: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    return next((r for r in rows if abs(r["t"] - t) < 1e-6), None)


def _fmt(x: float | None, d: int = 3) -> str:
    return (
        "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"
    )


class Run:
    """The rows of one run directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        conf = yaml.safe_load((path / "config.yaml").read_text())
        self.tau = float(conf["optim"]["tau"])
        self.layout = conf["data"].get("layout", "spread")
        post = path / "posthoc_stream"
        self.summary = sorted(_jsonl(post / "summary.jsonl"), key=lambda r: r["t"])
        self.probes = _jsonl(post / "probes.jsonl")
        self.z = sorted(_jsonl(post / "z_use.jsonl"), key=lambda r: r["t"])
        result = path / "result.json"
        self.status = (
            json.loads(result.read_text()).get("status")
            if result.exists()
            else "running"
        )

    def span(self, t: float) -> float | None:
        r = _at(self.summary, t)
        return None if r is None else r["span"]

    def value(self, key: str, t: float) -> float | None:
        r = _at(self.summary, t)
        return None if r is None else r.get(key)

    def z_diff(self, t: float) -> float | None:
        r = _at(self.z, t)
        return None if r is None else r["span_shifted"] - r["span"]

    def probe(self, t: float, group: str, key: str) -> float | None:
        r = next(
            (p for p in self.probes if abs(p["t"] - t) < 1e-6 and p["group"] == group),
            None,
        )
        return None if r is None else r.get(key)


def check(b1: Run, runs: list[Run]) -> tuple[list[dict[str, Any]], Run | None]:
    """The check of each chunk run against B = 1 and the chosen one."""
    rows = []
    for run in runs:
        d = []
        for t in CHECK_T:
            a, b = run.span(t), b1.span(t)
            d.append(None if a is None or b is None else a - b)
        known = [abs(x) for x in d if x is not None]
        z6 = run.z_diff(6000.0)
        ok = (
            d[2] is not None  # 1800 s
            and d[9] is not None  # 6000 s
            and abs(d[2]) < THRESHOLD
            and abs(d[9]) < THRESHOLD
            and z6 is not None
            and z6 > Z_MIN
        )
        rows.append(
            {
                "tau": run.tau,
                "d": d,
                "mean_abs_d": sum(known) / len(known) if known else math.nan,
                "z_6000": z6,
                "passed": ok,
                "run": run,
            }
        )
    passed = [r for r in rows if r["passed"]]
    pool = passed or [r for r in rows if not math.isnan(r["mean_abs_d"])]
    chosen = min(pool, key=lambda r: r["mean_abs_d"])["run"] if pool else None
    return rows, chosen


def predictions(b1: Run, run: Run) -> list[tuple[str, str, str]]:
    """(prediction, verdict, grounds) of predictions 2-5 for the chosen run,
    at its last checkpoint (600000 s if it ran to the end)."""
    out = []
    t_end = run.summary[-1]["t"] if run.summary else 600000.0
    s_end, s_b1 = run.span(t_end), b1.span(6000.0)
    if s_end is None or s_b1 is None:
        out.append(("2 span", "—", f"no {t_end:.0f} s row"))
    else:
        ok = s_end <= 0.40 and s_end <= s_b1 - THRESHOLD
        out.append(
            (
                "2 span <= 0.40, below B=1 6000 s by 0.07",
                "hit" if ok else "miss",
                f"{s_end:.3f} at {t_end:.0f} s vs B=1 {s_b1:.3f}",
            )
        )
    z0, z1 = run.value("zurich", 6000.0), run.value("zurich", t_end)
    if z0 is None or z1 is None:
        out.append(("3 zurich", "—", "no rows"))
    else:
        out.append(
            (
                "3 zurich down by 0.07",
                "hit" if z1 <= z0 - THRESHOLD else "miss",
                f"{z0:.3f} -> {z1:.3f} ({t_end:.0f} s)",
            )
        )
    worst, best, late = 0.0, math.inf, 0
    for r in run.summary:
        if r["t"] < 48000.0 - 1e-6:
            best = min(best, r["span"])
            continue
        worst = max(worst, r["span"] - best)
        best = min(best, r["span"])
        late += 1
    out.append(
        (
            "4 no rise of 0.07 above the running minimum after 48000 s",
            "—" if not late else "hit" if worst < THRESHOLD else "miss",
            f"largest rise {worst:.3f} over {late} checkpoints",
        )
    )
    zs = [
        (r["t"], r["span_shifted"] - r["span"])
        for r in run.z
        if r["t"] >= 1800.0 - 1e-6
    ]
    zmin = min((z for _, z in zs), default=math.nan)
    out.append(
        (
            "5 z diff > 0.2 from 1800 s",
            "hit" if zs and zmin > Z_MIN else "miss",
            f"min {zmin:.3f} over {len(zs)} checkpoints; status {run.status}",
        )
    )
    return out


def stop_rule(run: Run) -> str:
    a, b = run.span(60000.0), run.span(180000.0)
    if a is None or b is None:
        return "—"
    return f"{a:.3f} -> {b:.3f} ({'plateau: stop' if b > a - 0.02 else 'go on'})"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exp.vision.results.v17.judge")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    b1s = [Run(p.parent) for p in sorted((args.runs / "v17-b1").glob("*/config.yaml"))]
    runs = sorted(
        (Run(p.parent) for p in (args.runs / "v17-chunk").glob("*/config.yaml")),
        key=lambda r: r.tau,
    )
    if not b1s or not runs:
        raise SystemExit("no runs under v17-b1 or v17-chunk")
    b1 = b1s[0]
    rows, chosen = check(b1, runs)
    print("### Check against B = 1 (span, d(t) = chunk - B=1)\n")
    print(
        "| tau | "
        + " | ".join(f"{t:.0f}" for t in CHECK_T)
        + " | mean abs d | z diff 6000 | passed |"
    )
    print("| -- " * (len(CHECK_T) + 4) + "|")
    print(
        "| B=1 span | "
        + " | ".join(_fmt(b1.span(t)) for t in CHECK_T)
        + f" | | {_fmt(b1.z_diff(6000.0))} | |"
    )
    for r in rows:
        print(
            f"| {r['tau']:g} | "
            + " | ".join(_fmt(x) for x in r["d"])
            + f" | {_fmt(r['mean_abs_d'])} | {_fmt(r['z_6000'])} | {'yes' if r['passed'] else 'no'} |"
        )
    print(f"\nchosen tau: {chosen.tau:g}" if chosen else "\nchosen tau: none")
    print("\n### The long runs (times of the single stream)\n")
    print("| tau | key | " + " | ".join(f"{t:.0f}" for t in SHOWN_T) + " |")
    print("| -- " * (len(SHOWN_T) + 2) + "|")
    for run in runs:
        for key in ("span", "after", "zurich"):
            print(
                f"| {run.tau:g} | {key} | "
                + " | ".join(_fmt(run.value(key, t)) for t in SHOWN_T)
                + " |"
            )
        print(
            f"| {run.tau:g} | z diff | "
            + " | ".join(_fmt(run.z_diff(t)) for t in SHOWN_T)
            + " |"
        )
        for g in ("span", "zurich"):
            for k in ("recon_blur", "dyn_1/copy_1"):
                print(
                    f"| {run.tau:g} | {g} {k} | "
                    + " | ".join(_fmt(run.probe(t, g, k)) for t in SHOWN_T)
                    + " |"
                )
        print(
            f"| {run.tau:g} | stop rule | {stop_rule(run)} |"
            + " |" * (len(SHOWN_T) - 1)
        )
    if chosen:
        print(f"\n### Predictions (tau {chosen.tau:g})\n")
        print("| prediction | verdict | grounds |\n| -- | -- | -- |")
        for p, v, g in predictions(b1, chosen):
            print(f"| {p} | {v} | {g} |")
    if args.json:
        dump = {
            "b1": {"summary": b1.summary, "z": b1.z, "probes": b1.probes},
            "chunk": {
                f"{r.tau:g}": {
                    "summary": r.summary,
                    "z": r.z,
                    "probes": r.probes,
                    "status": r.status,
                }
                for r in runs
            },
            "chosen": None if chosen is None else f"{chosen.tau:g}",
            "check": [{k: v for k, v in r.items() if k != "run"} for r in rows],
        }
        args.json.write_text(json.dumps(dump))


if __name__ == "__main__":
    main()
