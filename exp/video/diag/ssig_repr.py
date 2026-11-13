r"""DIAGNOSIS ONLY (M17): the check of the mechanism of the SIGReg of the data
points on one stream (results/m17.md, 「ストリームの SIGReg の結果」), shown
and not judged. From the ``stream_stats repr`` outputs (``repr-*.json``) in
``--dir`` of the last checkpoints of the runs, each model labelled
``<arm>-k<k>-l<lambda>-s<seed>``, beside the untrained ViT of the arm
(label ``<arm>``, ``repr-t512.json`` etc. of the grid's derivation):

- the whitened correlation of the flattened tokens at each measured lag, the
  lags of the data points that share frames (1 to 15 steps of the stream:
  a stack is 16 frames 0.1 s apart, so two stacks $\tau$ steps apart share
  $16 - \tau$) apart from the others (16 steps and more)
- the mean correlation of the buffer's pairs $\bar r$ of each curve and span
  (``ssig_window.windows``, $\rho$ = 0.122, $N$ = 256) against the arm's
  ``rbar_max``: the window of the healthy solution is not empty below it
- the whitener's rank and participation ratio (``pr``)

Run from the repository root (CPU, seconds; the directions of seed 0)::

    python -m exp.video.diag.ssig_repr --dir exp/video/results/m17-ssig \
        --init exp/video/results/m17-stream
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from exp.video.diag.ssig_window import N_BUF, RHO, SPANS, load_curves, windows

SHARED = 15  # the largest lag (steps) at which two stacks of 16 frames overlap


def arm_of(label: str) -> str:
    """The arm of a curve's model label (``t512-k1-...`` or ``t512``)."""
    return label.split("/")[0].split("-")[0]


def split_lags(
    lags: Sequence[float], rho: Sequence[float]
) -> tuple[list[float], list[float]]:
    """The correlations of the lags (steps) that share frames (1 to
    ``SHARED``) and of the others (above it); the lag 0 is left out."""
    shared = [r for lag, r in zip(lags, rho, strict=True) if 1 <= lag <= SHARED]
    apart = [r for lag, r in zip(lags, rho, strict=True) if lag > SHARED]
    return shared, apart


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def lag_table(
    curves: dict[str, dict[str, Any]], arm: str, lags: Sequence[float]
) -> str:
    """The whitened correlation of ``arm``'s curves (the untrained ViT
    first) at ``lags`` (steps; each a measured lag), then the plain means of
    the measured lags that share frames and of the others."""
    names = [n for n in curves if arm_of(n) == arm]
    names.sort(key=lambda n: (n.split("/")[0] != arm, n))
    out = [
        "| lag | " + " | ".join(f"`{n.split('/')[0]}`" for n in names) + " |",
        "| -- |" + " -- |" * len(names),
    ]
    for lag in lags:
        vals = []
        for n in names:
            c = curves[n]
            vals.append(
                next(
                    (
                        f"{r:.4f}"
                        for x, r in zip(c["lags"], c["rho"])
                        if abs(x - lag) < 1e-9
                    ),
                    "–",
                )
            )
        out.append(f"| {lag * 0.1:.3g} 秒 | " + " | ".join(vals) + " |")
    for i, name in enumerate(("1〜15 の平均", "16 以上の平均")):
        vals = [
            f"{_mean(split_lags(curves[n]['lags'], curves[n]['rho'])[i]):.4f}"
            for n in names
        ]
        out.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def rbar_table(rows: Sequence[dict[str, Any]]) -> str:
    """$\\bar r$ (band) of each curve and span, of the curve's own arm, with
    the arm's ``rbar_max``."""
    out = [
        "| curve | k | rbar (band) | rbar_max | rbar < rbar_max |",
        "| -- | -- | -- | -- | -- |",
    ]
    for r in rows:
        if arm_of(r["curve"]) != r["arm"]:
            continue
        m, lo, hi = r["mid"], r["low"], r["high"]
        out.append(
            f"| `{r['curve']}` | {r['k']} | {m['rbar']:.4f} ({lo['rbar']:.4f}〜"
            f"{hi['rbar']:.4f}) | {r['rbar_max']:.4f} | "
            f"{'○' if m['rbar'] < r['rbar_max'] else '×'} |"
        )
    return "\n".join(out)


def ranks(dirs: Sequence[Path]) -> str:
    """The whitener's rank and participation ratio of each model's tokens."""
    out = ["| model | rank | pr |", "| -- | -- | -- |"]
    for d in dirs:
        for path in sorted(d.glob("repr-*.json")):
            for label, x in json.loads(path.read_text())["codes"].items():
                w = x.get("whitener", {})
                out.append(f"| `{label}` | {w.get('rank')} | {w.get('pr', 0):.1f} |")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    """The three tables of the module docstring."""
    p = argparse.ArgumentParser(description="The mechanism of m17/ssig_*.")
    p.add_argument("--dir", type=Path, default=Path("exp/video/results/m17-ssig"))
    p.add_argument("--init", type=Path, default=Path("exp/video/results/m17-stream"))
    p.add_argument("--draws", type=int, default=20000)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    learned = sorted(
        f"{label}/white"
        for path in args.dir.glob("repr-*.json")
        for label in json.loads(path.read_text())["codes"]
    )
    init = [f"{a}/tokens/white" for a in ("t512", "t128", "t64")]
    curves = {**load_curves(args.init, init), **load_curves(args.dir, learned)}
    rows = windows(args.dir, learned, [RHO], SPANS, N_BUF, args.draws)
    for arm in ("t512", "t128", "t64"):
        print(f"### {arm}: the whitened correlation\n")
        print(lag_table(curves, arm, curves[f"{arm}/tokens/white"]["lags"]))
        print()
    print("### rbar\n")
    print(rbar_table(rows))
    print("\n### rank\n")
    print(ranks([args.init, args.dir]))
    if args.out is not None:
        args.out.write_text(json.dumps({"rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
