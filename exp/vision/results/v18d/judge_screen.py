r"""The judgement of V18d's short runs and the long run's choice
(results/v18d/README.md, 「短い run の判定」 and 「長い run の選び方」), the
rules as registered, none of Zurich.

Reads ``<runs>/v18d-screen/<run>/`` (``jepa.json``, ``train.jsonl``,
``posthoc_v18d/select.jsonl`` of ``time_val``), the thresholds' yardsticks
``<runs>/v18d-select/yard/select.jsonl`` (``color``, ``color8``,
``init_c`` on the validation video) and the content guard v3's rows on the
training videos' runs alone ``<runs>/v18d-select/content3/<run>@6000/
content3.json`` (``content_eval real3 --no-eval``; V18d's candidates and
V18c's short runs of $r$ = 0.5). Prints markdown tables; ``--json`` writes
the numbers::

    uv run python -m exp.vision.results.v18d.judge_screen [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from exp.lib.fidelity import ContentSums, comparable
from exp.vision.content_eval import sums3
from exp.vision.geom_eval import GeomConf
from exp.vision.width_eval import counts_of

STEPS = 6000
LRS = (1e-5, 3e-5, 1e-4)
LAMS = (0.5, 0.8, 0.95)
ARMS = (("ST", 1.0), ("ST", 0.25), ("T", None))  # (arm, w)
PENDING = (
    0.05  # sigreg_heldin, time_val (and S+T's pred_ratio_calib) over the reference
)
PASS_S = 2.0  # sigreg_heldin below this
PASS_TIME = 0.95  # time_val below this
PASS_PRED = 0.95  # S+T: pred_ratio_calib below this
PSI_STAR = 0.1  # D = E_a - 2^-psi* E_b
TIE = 0.01  # time_val within this of the least: the larger E^lin

type Row = dict[str, Any]


def _jsonl(path: Path) -> list[Row]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def _tag(x: float) -> str:
    """``1e-4`` / ``3e-5`` style of a learning rate."""
    return f"{x:.0e}".replace("e-0", "e-")


def run_name(arm: str, w: float | None, lr: float, lam: float, seed: int = 0) -> str:
    mid = f"-w{w:g}" if w is not None else ""
    return f"v18d-{arm}{mid}-lr{_tag(lr)}-lam{lam:g}-r0.5-s{seed}"


def v18c_name(lr: float, lam: float, seed: int = 0) -> str:
    return f"v18c-lr{_tag(lr)}-lam{lam:g}-r0.5-s{seed}"


def lines(yard: Mapping[str, Row]) -> dict[str, float]:
    """K1's line, K2's rank floor and K3's theta_3 (V18c's), of the
    validation video's yardsticks."""
    init = yard["init_c"]
    return {
        "k1": (init["sigreg_heldin_gauss"] + init["sigreg_heldin_const"]) / 2.0,
        "rank_floor": min(yard["color"]["rank"], init["rank"]),
        "theta3": (1.0 + max(yard["color8"]["token_r2"], init["token_r2"])) / 2.0,
    }


def _finite(row: Row, keys: Sequence[str]) -> bool:
    return all(k not in row or math.isfinite(float(row[k])) for k in keys)


def judge(
    run_dir: Path,
    arm: str,
    w: float | None,
    lr: float,
    lam: float,
    ln: Mapping[str, float],
) -> Row:
    """Collapse, divergence and the pass's value tests of a short run at
    ``STEPS``."""
    train = _jsonl(run_dir / "train.jsonl")
    sel = {r["step"]: r for r in _jsonl(run_dir / "posthoc_v18d" / "select.jsonl")}
    s = sel.get(STEPS, {})
    row: Row = {"name": run_dir.name, "arm": arm, "w": w, "lr": lr, "lam": lam,
                "finished": bool(train) and train[-1]["step"] == STEPS,
                "diverged": not all(_finite(r, ("loss", "pred", "sigreg", "time"))
                                    for r in train)}  # fmt: skip
    for key in ("sigreg_heldin", "rank", "token_r2", "pred_ratio_calib", "time_val",
                "participation", "z_var", "rho_in"):  # fmt: skip
        row[key] = s.get(key)
    row["time_val_lags"] = s.get("time_val_lags")
    last = train[-1] if train else {}
    for key in ("loss", "pred", "sigreg", "time"):
        row[f"train_{key}"] = last.get(key)
    need = ["sigreg_heldin", "rank", "token_r2", "time_val"]
    if arm == "ST":
        need.append("pred_ratio_calib")
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
    """Of each arm (with its $w$) and $\\lambda$, the non-collapsed run of
    the largest learning rate is the reference; another is pending when its
    sigreg_heldin or time_val (of S+T also pred_ratio_calib) exceeds the
    reference's by more than ``PENDING``."""
    for row in rows:
        row["pending"], row["reference"] = False, None
    for arm, w in ARMS:
        for lam in LAMS:
            same = [x for x in rows if (x["arm"], x["w"], x["lam"]) == (arm, w, lam)
                    and not x["collapsed"]]  # fmt: skip
            if not same:
                continue
            ref = max(same, key=lambda x: x["lr"])
            keys = ["sigreg_heldin", "time_val"] + (
                ["pred_ratio_calib"] if arm == "ST" else []
            )
            for x in same:
                x["reference"] = ref["name"]
                if x is not ref:
                    x["pending"] = any(x[k] > ref[k] + PENDING for k in keys)


def apply_pass(rows: list[Row]) -> None:
    for x in rows:
        ok = not x["collapsed"] and not x["diverged"] and not x["pending"]
        if ok:
            ok = x["sigreg_heldin"] < PASS_S and x["time_val"] < PASS_TIME
            if x["arm"] == "ST":
                ok = ok and x["pred_ratio_calib"] < PASS_PRED
        x["passed"] = bool(ok)


def content_of(path: Path) -> dict[str, Any] | None:
    """The least squares' $E$ and $E^d$ on the validation runs (sums too),
    or None when not measured."""
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    s = sums3(rows, "lin", "val")
    return {"E": s.value(), "Ed": s.value(True), "sums": s}


def eligibility(a: Mapping[str, Any], b: Mapping[str, Any], g: GeomConf) -> Row:
    r"""The content eligibility: $D = E_a - 2^{-\psi_*} E_b \ge 0$ (point) of
    $E$ and $E^d$, a quantity whose $E_b$ is not ``comparable`` (lower end
    $\le 0$ of the validation runs' bootstrap) left out and recorded."""
    sb: ContentSums = b["sums"]
    nb = len(sb.tot)
    counts = (counts_of(nb, g), counts_of(nb // 2, g))
    rho = 2.0**-PSI_STAR
    out: Row = {}
    oks = []
    for key, change in (("E", False), ("Ed", True)):
        comp = comparable(sb, *counts, change=change)
        d = a[key] - rho * b[key]
        out[f"D_{key}"] = d
        out[f"comparable_{key}"] = {k: comp[k] for k in ("point", "se", "lo", "ok")}
        if comp["ok"]:
            oks.append(d >= 0)
    out["eligible"] = all(oks)
    out["checked"] = len(oks)
    return out


def choose(rows: Sequence[Row]) -> dict[str, Any]:
    """The long run's setting of each arm (S+T's two $w$ one pool; T)."""
    out: dict[str, Any] = {}
    for arm in ("ST", "T"):
        passed = [x for x in rows if x["arm"] == arm and x["passed"]
                  and x.get("content") is not None]  # fmt: skip
        dropped = [x["name"] for x in rows if x["arm"] == arm and x["passed"]
                   and x.get("content") is None]  # fmt: skip
        if not passed:
            out[arm] = {"chosen": None, "why": "no pass", "unmeasured": dropped}
            continue
        elig = [x for x in passed if x["content"]["eligible"]]
        if elig:
            least = min(x["time_val"] for x in elig)
            near = [x for x in elig if x["time_val"] <= least + TIE]
            pick = max(near, key=lambda x: (x["content"]["E"], x["lr"]))
            why = "eligible, least time_val" + (
                " (tie: E^lin, lr)" if len(near) > 1 else ""
            )
        else:
            pick = max(
                passed, key=lambda x: min(x["content"]["D_E"], x["content"]["D_Ed"])
            )
            why = "内容の適格なし: the largest min(D_E, D_Ed)"
        out[arm] = {"chosen": pick["name"], "lr": pick["lr"], "lam": pick["lam"],
                    "w": pick["w"], "why": why, "eligible": [x["name"] for x in elig],
                    "unmeasured": dropped}  # fmt: skip
    return out


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
    rows = []
    for arm, w in ARMS:
        for lr in LRS:
            for lam in LAMS:
                name = run_name(arm, w, lr, lam)
                rows.append(judge(runs / "v18d-screen" / name, arm, w, lr, lam, ln))
    apply_pending(rows)
    apply_pass(rows)
    content = runs / "v18d-select" / "content3"
    for x in rows:
        a = content_of(content / f"{x['name']}@{STEPS}" / "content3.json")
        b = content_of(
            content / f"{v18c_name(x['lr'], x['lam'])}@{STEPS}" / "content3.json"
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
        "| run | sigreg_heldin | rank | token_r2 | pred_ratio_calib | time_val "
        "| K1 | K2 | K3 | 発散 | 保留 | 合格 | E^lin | E^d,lin | D_E | D_Ed | 適格 |"
    )
    print("| -- " * 17 + "|")
    for x in out["rows"]:
        c = x.get("content") or {}
        print(f"| `{x['name']}` | {_fmt(x['sigreg_heldin'])} | {_fmt(x['rank'], 1)} "
              f"| {_fmt(x['token_r2'])} | {_fmt(x['pred_ratio_calib'])} "
              f"| {_fmt(x['time_val'])} | {_fmt(x['K1'])} | {_fmt(x['K2'])} "
              f"| {_fmt(x['K3'])} | {_fmt(x['diverged'])} | {_fmt(x['pending'])} "
              f"| {'○' if x['passed'] else '—'} | {_fmt(c.get('E'))} | {_fmt(c.get('Ed'))} "
              f"| {_fmt(c.get('D_E'), 4)} | {_fmt(c.get('D_Ed'), 4)} "
              f"| {'—' if not c else '○' if c['eligible'] else '×'} |")  # fmt: skip
    print()
    for arm, v in out["choice"].items():
        print(f"- {arm}: {v}")


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
