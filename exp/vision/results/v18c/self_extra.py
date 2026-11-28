r"""The descriptive numbers of V18c's identification of the fill in the JEPA's
own space beside ``judge_self`` (exp/vision/README.md, 「V18c: 補完の同定の
結果」). No verdict; nothing here enters ``judge_self``'s stop.

Of every checkpoint of ``judge_self.GROUPS`` whose rows are there:

1. V-s6 at the registered threshold 3 before the user's change to 7.34
   (README, the user's decision 2): the number of $\lvert z \rvert > 3$ of
   its 24, and the largest $\lvert z \rvert$;
2. the hard candidates' $A$ of ``pred`` at each $k$ (4, 8, 12; the
   prediction 8), the run's own $k$ marked;
3. at the run's own $k$: the hard $A$ of ``pred`` by ``l2`` and of
   ``hard_c``, the easy ``top1``, the locality's seam split ($a_U$ of the
   seam and the inside, $a_V$ of the seam and the far cells), $\rho_V$,
   $\eta_U$, the spread over the contexts $j$ of ``fwd``'s hard $A$ of
   ``pred``, and the bands' share of the variance.

Prints markdown; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.self_extra [--runs runs] [--json out.json]
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from exp.vision.results.v18c.judge_self import GROUPS, Row, _one, _rows

KS = (4, 8, 12)
OLD_TOL = 3.0


def extra(rows: Sequence[Row]) -> dict[str, Any]:
    """The numbers of one checkpoint (module docstring)."""
    if not rows:
        return {}
    out: dict[str, Any] = {}
    v_s6 = _one(rows, part="check", name="V-s6")
    if v_s6 is not None:
        z = [abs(x) for x in v_s6["z"]]
        out["v_s6"] = {"over3": sum(x > OLD_TOL for x in z), "n": len(z), "max": max(z)}
    base = {"part": "id", "metric": "cos", "quarter": "mean", "cand": "nat"}
    by_k = {}
    k_run = None
    for k in KS:
        r = _one(rows, **(base | {"set": "hard", "stage": "pred", "k": k}))
        by_k[k] = r["A"] if r else None
        if r and r["in_dist"]:
            k_run = k
    out["hard_pred_by_k"], out["k_run"] = by_k, k_run
    own = base | {"in_dist": True, "stage": "pred"}
    l2 = _one(rows, **(own | {"set": "hard", "metric": "l2"}))
    hard_c = _one(rows, **(own | {"set": "hard_c"}))
    easy = _one(rows, **(own | {"set": "easy"}))
    out["hard_pred_l2"] = l2["A"] if l2 else None
    out["hard_c_pred"] = hard_c["A"] if hard_c else None
    out["top1_easy"] = easy["top1"] if easy else None
    local = _one(rows, part="local", quarter="mean", j="mean")
    keys = ("a_U_seam", "a_U_in", "a_V_seam", "a_V_far", "rho_V", "eta_U")
    out["local"] = {x: local[x] for x in keys} if local else None
    fwd = [
        r["A"]
        for j in range(8)
        if (r := _one(rows, **(own | {"set": "hard", "cand": "fwd", "j": j})))
    ]
    out["fwd_by_j"] = fwd or None
    band = _one(rows, part="band", in_dist=True, quarter="mean")
    out["band_share"] = band["share"] if band else None
    return out


def _f(x: Any, fmt: str = ".3f") -> str:
    return "—" if x is None else format(x, fmt)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    out: dict[str, Any] = {}
    for group, (names, steps) in GROUPS.items():
        out[group] = {
            name: {s: d for s in steps if (d := extra(_rows(a.runs / name, s)))}
            for name in names
        }
    every = [d for g in out.values() for by in g.values() for d in by.values()]
    checked = [d["v_s6"] for d in every if "v_s6" in d]
    over = sum(c["over3"] for c in checked)
    worst = max((c["max"] for c in checked), default=None)
    out["v_s6_total"] = {
        "checkpoints": len(checked),
        "tests": sum(c["n"] for c in checked),
        "over3": over,
        "checkpoints_over3": sum(c["over3"] > 0 for c in checked),
        "max": worst,
    }
    t = out["v_s6_total"]
    print("## V-s6 を元の閾値 3 で数えたとき\n")
    print(
        f"{t['checkpoints']} checkpoint、{t['tests']} 個のうち |z| > 3 は {t['over3']} 個"
        f"（{t['checkpoints_over3']} checkpoint）、最大の |z| は {_f(worst, '.2f')}\n"
    )
    print(
        "## 難しい候補の pred の A（k ごと、* は分布の中の k）と、分布の中の k の副の量\n"
    )
    head = (
        "| run | step | k=4 | k=8 | k=12 | l2 | hard_c | easy top1 | "
        "a_U seam/in | a_V seam/far | ρ_V | η_U | fwd の j の幅 | V-s6 |z|>3 |"
    )
    print(head)
    print("|" + " -- |" * 14)
    for group, got in out.items():
        if group == "v_s6_total":
            continue
        for name, by_step in got.items():
            for s, d in by_step.items():
                ks = [
                    _f(d["hard_pred_by_k"][k]) + ("*" if k == d["k_run"] else "")
                    for k in KS
                ]
                loc = d["local"]
                seam_u = loc and f"{loc['a_U_seam']:.3f}/{loc['a_U_in']:.3f}"
                seam_v = loc and f"{loc['a_V_seam']:.3f}/{loc['a_V_far']:.3f}"
                fwd = d["fwd_by_j"]
                spread = fwd and max(fwd) - min(fwd)
                cells = [
                    *ks,
                    _f(d["hard_pred_l2"]),
                    _f(d["hard_c_pred"]),
                    _f(d["top1_easy"]),
                    seam_u or "—",
                    seam_v or "—",
                    _f(loc and loc["rho_V"]),
                    _f(loc and loc["eta_U"]),
                    _f(spread),
                    str(d["v_s6"]["over3"]) if "v_s6" in d else "—",
                ]
                print(f"| {Path(name).name} | {s} | " + " | ".join(cells) + " |")
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    main()
