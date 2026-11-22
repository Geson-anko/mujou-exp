"""The extra runs of V17 (exp/vision/README.md, "V17 の追加の調査").

Reads the posthoc rows (``--span 6000``, times of the single stream) of
``v17-b1`` (seed 0), ``v17-b1-long`` (seed 1), ``v17-b1-seeds`` (2, 3),
``v17-chunk`` (or ``--chunk``), ``v17-chunk-x``, ``v17-c10``, ``v17-c30``
and ``v17-c10-long`` under ``--runs`` and prints the markdown tables and the
verdicts of predictions 1-6::

    uv run python -m exp.vision.results.v17.extra [--runs runs] [--chunk DIR]
"""

import argparse
import math
from pathlib import Path

import yaml

from exp.vision.results.v17.judge import CHECK_T, THRESHOLD, Run, _fmt

LONG_T = [6000.0, 12000.0, 18000.0, 30000.0, 48000.0, 60000.0, 96000.0, 120000.0]


class Arm:
    """A run with its seed and the number of consecutive stacks a batch."""

    def __init__(self, path: Path) -> None:
        self.run = Run(path)
        conf = yaml.safe_load((path / "config.yaml").read_text())
        self.seed = int(conf["seed"])
        data = conf["data"]
        self.c = int(data["streams"]) if data.get("layout") == "chunk" else 1
        self.name = path.name

    @property
    def label(self) -> str:
        return f"c={self.c} tau={self.run.tau:g} s{self.seed}"


def _arms(root: Path) -> list[Arm]:
    """The runs under ``root`` that have posthoc rows."""
    return [
        Arm(p.parent)
        for p in sorted(root.glob("*/config.yaml"))
        if (p.parent / "posthoc_stream" / "summary.jsonl").exists()
    ]


def b1_band(b1s: list[Arm]) -> dict[float, tuple[float, float, float]]:
    """(mean, min, max) of the B = 1 seeds' span at each CHECK_T."""
    out = {}
    for t in CHECK_T:
        xs = [x for a in b1s if (x := a.run.span(t)) is not None]
        if xs:
            out[t] = (sum(xs) / len(xs), min(xs), max(xs))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m exp.vision.results.v17.extra")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--chunk", type=Path, help="default: <runs>/v17-chunk")
    args = ap.parse_args(argv)
    r = args.runs
    b1s = sorted(
        _arms(r / "v17-b1") + _arms(r / "v17-b1-long") + _arms(r / "v17-b1-seeds"),
        key=lambda a: a.seed,
    )
    chunks = sorted(
        _arms(args.chunk or r / "v17-chunk")
        + _arms(r / "v17-chunk-x")
        + _arms(r / "v17-c10")
        + _arms(r / "v17-c30"),
        key=lambda a: (-a.c, a.run.tau, a.seed),
    )
    band = b1_band(b1s)
    ref = next(a for a in b1s if a.seed == 0)

    print("### B = 1 seeds (span)\n")
    print("| run | " + " | ".join(f"{t:.0f}" for t in CHECK_T) + " | z diff 6000 |")
    print("| -- " * (len(CHECK_T) + 2) + "|")
    for a in b1s:
        print(
            f"| B=1 s{a.seed} | "
            + " | ".join(_fmt(a.run.span(t)) for t in CHECK_T)
            + f" | {_fmt(a.run.z_diff(6000.0))} |"
        )
    print(
        "| width (max - min) | "
        + " | ".join(
            _fmt(band[t][2] - band[t][1]) if t in band else "—" for t in CHECK_T
        )
        + " | |"
    )
    widths = [band[t][2] - band[t][1] for t in CHECK_T if t in band]
    seeds_n = len(b1s)
    ok1 = bool(widths) and max(widths) < 0.10
    w1800 = band[1800.0][2] - band[1800.0][1] if 1800.0 in band else math.nan

    print("\n### Chunk arms: d(t) = chunk - B=1 seed 0 (and - B=1 seed mean)\n")
    print(
        "| arm | "
        + " | ".join(f"{t:.0f}" for t in CHECK_T)
        + " | mean abs d | mean abs d (seed mean) | z diff 6000 |"
    )
    print("| -- " * (len(CHECK_T) + 4) + "|")
    d1800: dict[str, float | None] = {}
    for a in chunks:
        d = [
            None
            if (x := a.run.span(t)) is None or (y := ref.run.span(t)) is None
            else x - y
            for t in CHECK_T
        ]
        dm = [
            None if (x := a.run.span(t)) is None or t not in band else x - band[t][0]
            for t in CHECK_T
        ]
        k = [abs(x) for x in d if x is not None]
        km = [abs(x) for x in dm if x is not None]
        d1800[a.label] = d[2]
        print(
            f"| {a.label} | "
            + " | ".join(_fmt(x) for x in d)
            + f" | {_fmt(sum(k) / len(k) if k else None)}"
            + f" | {_fmt(sum(km) / len(km) if km else None)}"
            + f" | {_fmt(a.run.z_diff(6000.0))} |"
        )

    long_b1 = next((a for a in b1s if a.run.span(48000.0) is not None), None)
    c10_long = next(iter(_arms(r / "v17-c10-long")), None)
    main_c = next(
        (a for a in chunks if a.c == 100 and a.run.tau == 1e3 and a.seed == 0), None
    )
    print("\n### Long: B=1 seed 1, c=10 tau=1e4, c=100 tau=1e3 (span, zurich)\n")
    print("| run | " + " | ".join(f"{t:.0f}" for t in LONG_T) + " |")
    print("| -- " * (len(LONG_T) + 1) + "|")
    for name, a in (("B=1 s1", long_b1), ("c=10", c10_long), ("c=100", main_c)):
        if a is not None:
            for key in ("span", "zurich"):
                print(
                    f"| {name} {key} | "
                    + " | ".join(_fmt(a.run.value(key, t)) for t in LONG_T)
                    + " |"
                )

    print("\n### Predictions\n")
    print("| prediction | verdict | grounds |\n| -- | -- | -- |")
    c1 = d1800.get("c=100 tau=1000 s0")
    v1 = (
        "—"
        if seeds_n < 2
        else "hit"
        if ok1 and c1 is not None and c1 > w1800
        else "miss"
    )
    print(
        f"| 1 B=1 seed width < 0.10; chunk's +0.186 at 1800 beyond it | {v1} | "
        f"{seeds_n} seeds, largest width {_fmt(max(widths) if widths else None)}, "
        f"width at 1800 {_fmt(w1800)} |"
    )
    t300 = d1800.get("c=100 tau=300 s0")
    t100 = next((a for a in chunks if a.c == 100 and a.run.tau == 100), None)
    finite = t100 is not None and all(
        math.isfinite(x["span"]) for x in t100.run.summary
    )
    print(
        f"| 2 d(1800) at tau=300 below +0.186; tau=100 does not diverge | "
        f"{'hit' if t300 is not None and c1 is not None and t300 < c1 and finite else 'miss'} | "
        f"tau=300 {_fmt(t300)}, tau=1e3 {_fmt(c1)}, tau=100 "
        f"{_fmt(d1800.get('c=100 tau=100 s0'))} |"
    )
    c10 = d1800.get("c=10 tau=10000 s0")
    print(
        f"| 3 c=10 (tau=1e4) d(1800) between 0 and the c=100 one | "
        f"{'hit' if c10 is not None and c1 is not None and 0 < c10 < c1 else 'miss'} | "
        f"c=10 {_fmt(c10)}, c=100 {_fmt(c1)} |"
    )
    if long_b1 is not None and main_c is not None:
        ds = [
            None
            if (x := main_c.run.span(t)) is None or (y := long_b1.run.span(t)) is None
            else x - y
            for t in (18000.0, 30000.0, 48000.0)
        ]
        known = [x for x in ds if x is not None]
        ok4 = len(known) == 3 and all(abs(x) < THRESHOLD for x in known)
        print(
            f"| 4 within 0.07 of B=1 at 18000, 30000, 48000 | "
            f"{'hit' if ok4 else ('miss' if len(known) == 3 else '—')} | "
            + ", ".join(_fmt(x) for x in ds)
            + " |"
        )

    c30 = next((v for k, v in d1800.items() if k.startswith("c=30 ")), None)
    if c30 is None or c10 is None or c1 is None:
        v5 = "—"
    else:
        v5 = "hit" if c10 < c30 < c1 else "miss"
    print(f"| 5 c=30 d(1800) between c=10 and c=100 | {v5} | c=30 {_fmt(c30)} |")
    if c10_long is not None:
        pairs = []
        if long_b1 is not None:
            pairs += [
                (t, c10_long.run.span(t), long_b1.run.span(t))
                for t in (18000.0, 30000.0, 48000.0)
            ]
        a6 = [(t, x - y) for t, x, y in pairs if x is not None and y is not None]
        b6 = [
            (t, x - y)
            for t in LONG_T
            if t >= 18000.0
            and main_c is not None
            and (x := main_c.run.span(t)) is not None
            and (y := c10_long.run.span(t)) is not None
        ]
        ok6 = len(a6) == 3 and b6 and all(abs(d) < THRESHOLD for _, d in a6 + b6)
        print(
            f"| 6 c=10 long within 0.07 of B=1 (18000-48000); c=100 within 0.07 of c=10 from 18000 | "
            f"{'hit' if ok6 else ('—' if not b6 else 'miss')} | "
            f"c10-B1: {', '.join(f'{t:.0f} {d:+.3f}' for t, d in a6)}; "
            f"c100-c10: {', '.join(f'{t:.0f} {d:+.3f}' for t, d in b6)} |"
        )


if __name__ == "__main__":
    main()
