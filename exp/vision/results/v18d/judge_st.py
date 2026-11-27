r"""The judgement of V18d's short runs (spatio-temporal completion) and the
long run's choice (results/v18d/README.md, 「短い run の判定」 and 「長い
run の選び方」), the rules as registered, none of Zurich.

Reads ``<runs>/v18d-st-screen/<run>/`` (``jepa.json``, ``train.jsonl``,
``posthoc_v18d/st_select.jsonl`` of ``st_val``), the thresholds'
yardsticks ``<runs>/v18d-select/yard/select.jsonl`` (``color``,
``color8``, ``init_c`` on the validation video), the content guard v3's rows
of the candidates ``<runs>/v18d-st-select/content3/<run>@6000/content3.json``
and of V18c's short runs ``<runs>/v18d-select/content3/<v18c>@6000/
content3.json`` (``content_eval real3 --no-eval``). Prints markdown tables;
``--json`` writes the numbers::

    uv run python -m exp.vision.results.v18d.judge_st [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from exp.vision.geom_eval import GeomConf
from exp.vision.results.v18d.judge_screen import (
    content_of,
    eligibility,
    lines,
    v18c_name,
)

STEPS = 6000
LRS = (1e-5, 3e-5, 1e-4)
LAMS = (0.5, 0.8, 0.95)
PENDING = 0.05  # sigreg_heldin or L_st over the reference's
PASS_S = 2.0  # sigreg_heldin below this
PASS_SIDE = 0.95  # L_now and L_time, each below this
TIE = 0.01  # L_st within this of the least: the larger E^lin
TRAIN_KEYS = ("loss", "pred", "sigreg", "pred_now", "pred_time")

type Row = dict[str, Any]


def _jsonl(path: Path) -> list[Row]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def _tag(x: float) -> str:
    """``1e-4`` / ``3e-5`` style of a learning rate."""
    return f"{x:.0e}".replace("e-0", "e-")


def run_name(lr: float, lam: float, *, perm: bool = False, seed: int = 0) -> str:
    return f"v18d-st{'perm' if perm else ''}-lr{_tag(lr)}-lam{lam:g}-r0.5-s{seed}"


def _finite(row: Row, keys: Sequence[str]) -> bool:
    return all(
        k not in row or row[k] is None or math.isfinite(float(row[k])) for k in keys
    )


def judge(run_dir: Path, lr: float, lam: float, ln: Mapping[str, float]) -> Row:
    """Collapse (K1 to K3), divergence and the numbers of a short run at
    ``STEPS``."""
    train = _jsonl(run_dir / "train.jsonl")
    sel = {r["step"]: r for r in _jsonl(run_dir / "posthoc_v18d" / "st_select.jsonl")}
    s = sel.get(STEPS, {})
    row: Row = {"name": run_dir.name, "lr": lr, "lam": lam,
                "finished": bool(train) and train[-1]["step"] == STEPS,
                "diverged": not all(_finite(r, TRAIN_KEYS) for r in train)}  # fmt: skip
    for key in ("sigreg_heldin", "rank", "token_r2", "L_now", "L_time", "L_st",
                "sigreg_heldin_stack", "participation", "z_var", "rho_in"):  # fmt: skip
        row[key] = s.get(key)
    row["L_cuts"] = s.get("L_cuts")
    row["sigreg_heldin_times"] = s.get("sigreg_heldin_times")
    last = train[-1] if train else {}
    for key in TRAIN_KEYS:
        row[f"train_{key}"] = last.get(key)
    need = ("sigreg_heldin", "rank", "token_r2", "L_now", "L_time", "L_st")
    complete = row["finished"] and all(
        row[k] is not None and math.isfinite(row[k]) for k in need
    )
    row["complete"] = complete
    if not complete or row["diverged"]:
        return row | {"K1": None, "K2": None, "K3": None, "collapsed": True}
    row["K1"] = row["sigreg_heldin"] >= ln["k1"]
    row["K2"] = row["rank"] < ln["rank_floor"]
    row["K3"] = row["token_r2"] >= ln["theta3"]
    row["collapsed"] = bool(row["K1"] or row["K2"] or row["K3"])
    return row


def apply_pending(rows: list[Row]) -> None:
    """Of each $\\lambda$, the non-collapsed run of the largest learning rate
    is the reference; another is pending when its sigreg_heldin or L_st
    exceeds the reference's by more than ``PENDING``."""
    for row in rows:
        row["pending"], row["reference"] = False, None
    for lam in LAMS:
        same = [x for x in rows if x["lam"] == lam and not x["collapsed"]]
        if not same:
            continue
        ref = max(same, key=lambda x: x["lr"])
        for x in same:
            x["reference"] = ref["name"]
            if x is not ref:
                x["pending"] = any(
                    x[k] > ref[k] + PENDING for k in ("sigreg_heldin", "L_st")
                )


def apply_pass(rows: list[Row]) -> None:
    """The pass: neither collapsed, diverged nor pending, sigreg_heldin <
    ``PASS_S`` and each side's $L^{\\mathrm{rel}}$ < ``PASS_SIDE``."""
    for x in rows:
        ok = not x["collapsed"] and not x["diverged"] and not x["pending"]
        if ok:
            ok = (
                x["sigreg_heldin"] < PASS_S
                and x["L_now"] < PASS_SIDE
                and x["L_time"] < PASS_SIDE
            )
        x["passed"] = bool(ok)


def choose(rows: Sequence[Row]) -> Row:
    """The long run's setting (README's steps 1 to 6)."""
    passed = [x for x in rows if x["passed"] and x.get("content") is not None]
    dropped = [x["name"] for x in rows if x["passed"] and x.get("content") is None]
    if not passed:
        return {"chosen": None, "why": "no pass", "unmeasured": dropped}
    elig = [x for x in passed if x["content"]["eligible"]]
    if elig:
        least = min(x["L_st"] for x in elig)
        near = [x for x in elig if x["L_st"] <= least + TIE]
        pick = max(near, key=lambda x: (x["content"]["E"], x["lr"]))
        why = "eligible, least L_st" + (" (tie: E^lin, lr)" if len(near) > 1 else "")
    else:
        pick = max(passed, key=lambda x: min(x["content"]["D_E"], x["content"]["D_Ed"]))
        why = "内容の適格なし: the largest min(D_E, D_Ed)"
    return {"chosen": pick["name"], "lr": pick["lr"], "lam": pick["lam"], "why": why,
            "eligible": [x["name"] for x in elig], "unmeasured": dropped}  # fmt: skip


def _fmt(x: Any, d: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "×" if x else "·"
    if isinstance(x, float) and not math.isfinite(x):
        return "nan"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def evaluate(runs: Path, g: GeomConf = GeomConf()) -> dict[str, Any]:
    yard = {
        r["code"]: r for r in _jsonl(runs / "v18d-select" / "yard" / "select.jsonl")
    }
    ln = lines(yard)
    rows = [
        judge(runs / "v18d-st-screen" / run_name(lr, lam), lr, lam, ln)
        for lr in LRS
        for lam in LAMS
    ]
    apply_pending(rows)
    apply_pass(rows)
    mine = runs / "v18d-st-select" / "content3"
    theirs = runs / "v18d-select" / "content3"
    for x in rows:
        a = content_of(mine / f"{x['name']}@{STEPS}" / "content3.json")
        b = content_of(
            theirs / f"{v18c_name(x['lr'], x['lam'])}@{STEPS}" / "content3.json"
        )
        if a is not None and b is not None:
            x["content"] = {"E": a["E"], "Ed": a["Ed"], "E_v18c": b["E"],
                            "Ed_v18c": b["Ed"]} | eligibility(a, b, g)  # fmt: skip
        else:
            x["content"] = None
    return {"lines": ln, "yard": yard, "rows": rows, "choice": choose(rows)}


def _print(out: Mapping[str, Any]) -> None:
    ln = out["lines"]
    print(f"K1 ≥ {ln['k1']:.3f}、K2 の rank の床 {ln['rank_floor']:.2f}、"
          f"K3 の θ3 = {ln['theta3']:.3f}\n")  # fmt: skip
    print(
        "| run | sigreg_heldin | rank | token_r2 | L_now | L_time | L_ST "
        "| K1 | K2 | K3 | 発散 | 保留 | 合格 | E^lin | E^d,lin | D_E | D_Ed | 適格 |"
    )
    print("| -- " * 18 + "|")
    for x in out["rows"]:
        c = x.get("content") or {}
        print(f"| `{x['name']}` | {_fmt(x['sigreg_heldin'])} | {_fmt(x['rank'], 1)} "
              f"| {_fmt(x['token_r2'])} | {_fmt(x['L_now'])} | {_fmt(x['L_time'])} "
              f"| {_fmt(x['L_st'])} | {_fmt(x['K1'])} | {_fmt(x['K2'])} "
              f"| {_fmt(x['K3'])} | {_fmt(x['diverged'])} | {_fmt(x['pending'])} "
              f"| {'○' if x['passed'] else '—'} | {_fmt(c.get('E'))} | {_fmt(c.get('Ed'))} "
              f"| {_fmt(c.get('D_E'), 4)} | {_fmt(c.get('D_Ed'), 4)} "
              f"| {'—' if not c else '○' if c['eligible'] else '×'} |")  # fmt: skip
    print()
    print("| run | " + " | ".join(f"L_{k}" for k in ("past", "interp_past", "outward",
          "middle", "interp_future", "future")) + " |")  # fmt: skip
    print("| -- " * 7 + "|")
    for x in out["rows"]:
        cuts = x.get("L_cuts") or {}
        print(f"| `{x['name']}` | " + " | ".join(_fmt(v) for v in cuts.values()) + " |")
    print()
    print(f"- choice: {out['choice']}")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    out = evaluate(a.runs)
    _print(out)
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    main()
