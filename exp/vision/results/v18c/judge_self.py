r"""The description of V18c's identification of the fill in the JEPA's own space
(exp/vision/README.md, 「V18c: 自分の表現空間での補完の同定（回す前に決めたこと）」;
``exp.vision.self_eval``). Written before any number.

Reads ``<run>/posthoc_self/rows/<step>.json`` and ``<step>-tr.json`` of the
registered runs and ``runs/v18c-self/checks/rows.json``, and lays out, with
no verdict (V18c judges nothing):

1. the implementation checks V-s0 to V-s6 of every checkpoint (a failure
   stops the reading: the tables after it are printed for the record only);
2. of each run and step, at the run's own $k$ (``in_dist``), the cosine, the
   quarters' per-query mean: the hard candidates' $A$ of ``pred``,
   ``pool``, ``flat``, ``interp`` and ``oracle``, the differences ``pred``
   $-$ ``pool``, $-$ ``flat``, $-$ ``interp``, the easy $A$, ``top1``,
   ``swap``, the transplanted candidates' $A$ (``fwd``, ``rev``, the mean
   over the contexts) and ``fwd``'s ``pred`` $-$ ``flat`` (Fable's
   candidate), and the pooled locality ($a_U$, $a_V$, $L$, $\rho_V$,
   $\eta_U$);
3. the same run's changes between steps of ``pred``'s hard $A$;
4. of V18c's long runs, each setting's two seeds: a difference "上回る" (or
   "下回る") only where both seeds have it of the same sign beyond their
   $2\,\mathrm{SE}^*$, else "区別できない" (the README's wording, not a
   verdict);
5. the companions: the bands' ``pred_ratio``, the complement's $A$ and the
   spectrum's participation; and ``fwd``'s ``pred`` $-$ ``flat`` of each
   context (Codex's condition (2)).

$\mathrm{SE}^*$ is the larger of the block bootstraps of 8 and 16 consecutive
queries (``metrics.block_bootstrap`` with $e$ = 1 − value, $v$ = 1, so its
measure is the values' mean; 2000 resamples, seed 101); a difference of
stages or of steps resamples the same queries together (paired). Prints
markdown; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_self [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from exp.lib.metrics import block_bootstrap, contiguous_blocks

Row = dict[str, Any]

BLOCKS = (8, 16)
BOOT, BOOT_SEED = 2000, 101
CHECKS = ("V-s0", "V-s1", "V-s2", "V-s3", "V-s6")
STAGES = ("pred", "pool", "flat", "interp", "oracle")
DIFFS = (("pred", "pool"), ("pred", "flat"), ("pred", "interp"))
LONG_CKPT = (0, 2000, 6000, 10000, 20000, 30000, 40000, 50000, 60000)
OLD_STEPS = (0, 6000, 20000, 60000)
GROUPS: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "v18": (
        tuple(
            f"v18-long/v18-long-lr1e-4-lam0.5-{r}"
            for r in ("r0.25-s0", "r0.25-s1", "r0.5-s0", "r0.75-s0")
        ),
        OLD_STEPS,
    ),
    "v18b": (
        tuple(
            f"v18b-long/v18b-long-{a}-r0.25-s{s}"
            for a in ("f-lr1e-4-lam0.5", "m-lr3e-5-lam0.8")
            for s in (0, 1)
        ),
        OLD_STEPS,
    ),
    "screen": (
        tuple(
            f"v18c-screen/v18c-lr{lr}-lam{lam}-r0.5-s0"
            for lr in ("1e-4", "3e-5", "1e-5")
            for lam in ("0.5", "0.8", "0.95")
        )
        + tuple(
            f"v18c-screen/v18c-lr1e-4-lam{lam}-r{r}-s0"
            for lam in ("0.5", "0.8", "0.95")
            for r in ("0.25", "0.75")
        ),
        (0, 6000),
    ),
    "long": (
        tuple(
            f"v18c-long/v18c-lr{lr}-lam{lam}-r0.5-s{s}"
            for lr in ("1e-4", "1e-5")
            for lam in ("0.5", "0.8", "0.95")
            for s in (0, 1)
        ),
        LONG_CKPT,
    ),
}
STEP_PAIRS = ((0, 6000), (6000, 60000), (0, 60000))


def se_star(values: Sequence[Sequence[float]], coef: Sequence[float]) -> float:
    r"""$\mathrm{SE}^*$ of $\sum_c a_c \bar x_c$ of the per-query values
    ``values`` ``(C, N)`` (the same queries, in time order) and ``coef``
    $a_c$: the larger of the block bootstraps of ``BLOCKS``."""
    x = torch.tensor(values, dtype=torch.float64)
    n = x.shape[1]
    contrast = torch.tensor([coef], dtype=torch.float64)
    out = 0.0
    for length in BLOCKS:
        blocks = contiguous_blocks(torch.zeros(n, dtype=torch.long), length)
        gen = torch.Generator().manual_seed(BOOT_SEED)
        se = block_bootstrap(1 - x, torch.ones(n), blocks, contrast, BOOT, gen)
        out = max(out, float(se[0]))
    return out


def _load(path: Path) -> list[Row]:
    return json.loads(path.read_text()) if path.exists() else []


def _rows(run: Path, step: int) -> list[Row]:
    rows = run / "posthoc_self" / "rows"
    return _load(rows / f"{step:06d}.json") + _load(rows / f"{step:06d}-tr.json")


def _one(rows: Sequence[Row], **labels: Any) -> Row | None:
    got = [r for r in rows if all(r.get(k) == v for k, v in labels.items())]
    return got[0] if len(got) == 1 else None


def _id(rows: Sequence[Row], **labels: Any) -> Row | None:
    base = {"part": "id", "in_dist": True, "metric": "cos", "quarter": "mean"}
    return _one(rows, **(base | labels))


def _mean(x: Sequence[float]) -> float:
    return sum(x) / len(x)


def _diff(a: Row | None, b: Row | None, key: str = "A_i") -> dict[str, float] | None:
    """The difference of the means of the per-query ``key`` of ``a`` and
    ``b`` and its paired $2\\,\\mathrm{SE}^*$."""
    if a is None or b is None or key not in a or key not in b:
        return None
    d = _mean(a[key]) - _mean(b[key])
    return {"d": d, "se2": 2 * se_star([a[key], b[key]], [1.0, -1.0])}


def _value(a: Row | None, key: str = "A_i") -> dict[str, float] | None:
    if a is None or key not in a:
        return None
    return {"v": _mean(a[key]), "se2": 2 * se_star([a[key]], [1.0])}


def describe(rows: Sequence[Row]) -> dict[str, Any]:
    """The numbers of one checkpoint (module docstring, 1, 2 and 5)."""
    if not rows:
        return {}
    checks = {r["name"]: bool(r["pass"]) for r in rows if r["part"] == "check"}
    hard = {st: _id(rows, set="hard", cand="nat", stage=st) for st in STAGES}
    out: dict[str, Any] = {"checks": checks}
    out["hard"] = {st: (r["A"] if r else None) for st, r in hard.items()}
    for a, b in DIFFS:
        out[f"hard_{a}-{b}"] = _diff(hard[a], hard[b])
    easy = _id(rows, set="easy", cand="nat", stage="pred")
    out["easy_pred"] = _value(easy)
    out["top1_hard_pred"] = hard["pred"]["top1"] if hard["pred"] else None
    out["top1_easy_pred"] = easy["top1"] if easy else None
    out["hard_c_pred"] = _value(_id(rows, set="hard_c", cand="nat", stage="pred"))
    swap = _one(rows, part="swap", stage="pred")
    out["swap"] = _value(swap, "swap_i")
    for kind in ("fwd", "rev"):
        pred = _id(rows, set="hard", cand=kind, j="mean", stage="pred")
        out[f"{kind}_pred"] = _value(pred)
    fwd = {
        st: _id(rows, set="hard", cand="fwd", j="mean", stage=st)
        for st in ("pred", "flat")
    }
    out["fwd_pred-flat"] = _diff(fwd["pred"], fwd["flat"])
    per_j = []
    for j in range(8):
        a = _id(rows, set="hard", cand="fwd", j=j, stage="pred")
        b = _id(rows, set="hard", cand="fwd", j=j, stage="flat")
        if a is None:
            break
        per_j.append(_diff(a, b))
    out["fwd_pred-flat_j"] = per_j
    local = _one(rows, part="local", quarter="mean", j="mean")
    out["local"] = (
        {k: local.get(k) for k in ("a_U", "a_V", "L", "rho_V", "eta_U")}
        if local
        else None
    )
    band = _one(rows, part="band", in_dist=True, quarter="mean")
    out["band"] = band and {"ratio": band["ratio"], "share": band["share"]}
    out["comp"] = {
        f"{st}@{kc}": (
            r["A"]
            if (
                r := _one(
                    rows,
                    part="comp",
                    in_dist=True,
                    quarter="mean",
                    set="hard",
                    stage=st,
                    comp_k=kc,
                    metric="cos",
                )
            )  # fmt: skip
            else None
        )
        for st in ("pred", "pool", "oracle")
        for kc in (4, 16)
    }
    spec = [r["participation"] for r in rows if r["part"] == "spec"]
    out["participation"] = _mean(spec) if spec else None
    return out


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, dict) and "d" in x:
        return f"{x['d']:+.3f} ± {x['se2']:.3f}"
    if isinstance(x, dict) and "v" in x:
        return f"{x['v']:.3f} ± {x['se2']:.3f}"
    if isinstance(x, float):
        return f"{x:.3f}" if math.isfinite(x) else str(x)
    return str(x)


def words(a: dict[str, float] | None, b: dict[str, float] | None) -> str:
    """The README's wording of a difference of two seeds: "上回る" or
    "下回る" where both are of the same sign beyond their own
    $2\\,\\mathrm{SE}^*$, else "区別できない"."""
    if a is None or b is None:
        return "—"
    if a["d"] > a["se2"] and b["d"] > b["se2"]:
        return "上回る"
    if a["d"] < -a["se2"] and b["d"] < -b["se2"]:
        return "下回る"
    return "区別できない"


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    out: dict[str, Any] = {"checks_once": {}, "groups": {}}
    once = _load(a.runs / "v18c-self" / "checks" / "rows.json")
    out["checks_once"] = {r["name"]: r["pass"] for r in once if r["part"] == "check"}
    print("## 実装の確かめ\n")
    print("V-s4・V-s5:", out["checks_once"] or "未実行", "\n")
    stop = False
    for group, (names, steps) in GROUPS.items():
        got: dict[str, dict[int, Any]] = {}
        for name in names:
            run = a.runs / name
            got[name] = {s: describe(_rows(run, s)) for s in steps}
        out["groups"][group] = got
        print(f"### {group}\n")
        print("| run | step | " + " | ".join(CHECKS) + " | V-s3-L |")
        print("|" + " -- |" * (len(CHECKS) + 3))
        for name, by_step in got.items():
            for s, d in by_step.items():
                if not d:
                    continue
                c = d["checks"]
                cells = [_fmt(c.get(x)) for x in (*CHECKS, "V-s3-L")]
                stop |= not all(c.get(x, True) for x in (*CHECKS, "V-s3-L"))
                print(f"| {Path(name).name} | {s} | " + " | ".join(cells) + " |")
        print()
    if stop or not all(out["checks_once"].values()):
        print("**実装の確かめが外れた。以下は記録のためだけに出す。**\n")
    out["stop"] = stop
    cols = (
        ("hard pred", lambda d: d["hard"]["pred"]),
        ("pool", lambda d: d["hard"]["pool"]),
        ("flat", lambda d: d["hard"]["flat"]),
        ("interp", lambda d: d["hard"]["interp"]),
        ("pred−pool", lambda d: d["hard_pred-pool"]),
        ("pred−flat", lambda d: d["hard_pred-flat"]),
        ("pred−interp", lambda d: d["hard_pred-interp"]),
        ("easy pred", lambda d: d["easy_pred"]),
        ("top1 hard", lambda d: d["top1_hard_pred"]),
        ("swap", lambda d: d["swap"]),
        ("fwd pred", lambda d: d["fwd_pred"]),
        ("fwd pred−flat", lambda d: d["fwd_pred-flat"]),
        ("rev pred", lambda d: d["rev_pred"]),
        ("L", lambda d: d["local"] and d["local"]["L"]),
    )
    print("## 自然な候補・対照・移植（分布の中の k、cos、四半分の平均、± は 2SE*）\n")
    print("| run | step | " + " | ".join(c for c, _ in cols) + " |")
    print("|" + " -- |" * (len(cols) + 2))
    for got in out["groups"].values():
        for name, by_step in got.items():
            for s, d in by_step.items():
                if d:
                    cells = [_fmt(f(d)) for _, f in cols]
                    print(f"| {Path(name).name} | {s} | " + " | ".join(cells) + " |")
    print("\n## 同じ run の step の間の変化（難しい候補の pred の A）\n")
    out["steps"] = {}
    for group, (names, steps) in GROUPS.items():
        for name in names:
            run = a.runs / name
            for s0, s1 in STEP_PAIRS:
                if s0 not in steps or s1 not in steps:
                    continue
                r0 = _id(_rows(run, s0), set="hard", cand="nat", stage="pred")
                r1 = _id(_rows(run, s1), set="hard", cand="nat", stage="pred")
                d = _diff(r1, r0)
                if d is not None:
                    out["steps"][f"{name}:{s1}-{s0}"] = d
                    print(f"- {Path(name).name} {s1} − {s0}: {_fmt(d)}")
    print("\n## V18c の長い run: 2 seed の向き（記述の言葉づかい）\n")
    out["seeds"] = {}
    long = out["groups"]["long"]
    keys = ("hard_pred-pool", "hard_pred-flat", "hard_pred-interp", "fwd_pred-flat")
    print("| setting | step | " + " | ".join(keys) + " | 60000 − 6000 |")
    print("|" + " -- |" * (len(keys) + 3))
    for lr in ("1e-4", "1e-5"):
        for lam in ("0.5", "0.8", "0.95"):
            pair = [f"v18c-long/v18c-lr{lr}-lam{lam}-r0.5-s{s}" for s in (0, 1)]
            for s in (6000, 60000):
                d0, d1 = long[pair[0]].get(s), long[pair[1]].get(s)
                if not d0 or not d1:
                    continue
                said = [words(d0.get(k), d1.get(k)) for k in keys]
                st = [out["steps"].get(f"{x}:60000-6000") for x in pair]
                change = words(*st) if s == 60000 else "—"
                out["seeds"][f"lr{lr}-lam{lam}@{s}"] = dict(
                    zip(keys, said, strict=True)
                )
                print(f"| lr{lr} λ{lam} | {s} | " + " | ".join(said) + f" | {change} |")
    print(
        "\n## 相方（帯ごとの pred_ratio・補空間の A・participation）と fwd の j ごと\n"
    )
    for got in out["groups"].values():
        for name, by_step in got.items():
            for s, d in by_step.items():
                if not d:
                    continue
                band = d["band"] and ", ".join(f"{x:.2f}" for x in d["band"]["ratio"])
                comp = ", ".join(f"{k} {_fmt(v)}" for k, v in d["comp"].items())
                per_j = ", ".join(_fmt(x) for x in d["fwd_pred-flat_j"])
                print(
                    f"- {Path(name).name} {s}: band [{band}]; comp {comp}; "
                    f"participation {_fmt(d['participation'])}; fwd pred−flat by j [{per_j}]"
                )
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    main()
