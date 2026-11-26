r"""The validity checks of V18c's remade main measure $m$ and of E1's repair,
on the second try (exp/vision/README.md, "V18c: 作り直した主の指標の有効性
（2026-11-25）"; the rules in "V18c: 主の指標を作り直す（回す前に決めたこと、
2026-11-25）", with V-m0's reference changed by the user's decision written
there).

Applies the registered rules as ``judge_m`` has them, to what exists before
the screen's added seeds and the long runs:

- V-m0 to V-m4 and V-m6 (``judge_m.validity``) from ``runs/v18c-sem/
  targets.json`` and the yardsticks' rows; V-m5 needs $\sigma_{\mathrm{seed}}$
  (choice 2 of the long runs' rules) and is listed as not yet checkable.
  V-m0 also lists the hires checks (a), (b) and (c) of every kept frame
  (``runs/v18c-sem/hires_rgb/<video>.json``, the ``frames.json`` of
  ``hires.prepare``);
- the yardsticks' $m$ (sel, final, the fit set's halves), $\delta_{yy}$ of
  the neighbouring yardsticks, the share of ``pca1024`` held by ``pca16``,
  ``pca64`` and ``color8`` (described), and the scales at the grid's edge;
- E1's repaired checks V-E1'1, V-E1'3 and V-E1'4 on the V18c screen
  checkpoints ``E1_RUNS`` (the README reads them on the long runs, which do
  not exist yet: these are the checks before any long run), V-E1'2's
  residual of their ``e1diag`` (listed), and their ``e1f`` values by $k$
  (listed, no C1' read: C1' is of the long runs);
- the diagnosis of ``shuf`` of the old checkpoints (``judge_m.diagnosis``).

Prints markdown; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_m_checks [--runs runs] [--json out.json]
"""

import argparse
import json
from pathlib import Path
from typing import Any

from exp.vision.fill_eval import OUT, OUT_SEM
from exp.vision.results.v18c.judge_m import (
    K3_TOL,
    RESID,
    Boot,
    _fmt,
    _load,
    _resid,
    diagnosis,
    validity,
)

Row = dict[str, Any]

VIDEOS = ("Amsterdam", "Bangkok", "Zurich")
E1_RUNS = {  # V18c screen checkpoints: (run, steps)
    "c-lr1e-4-lam0.5-r0.5-s0": ("v18c-screen/v18c-lr1e-4-lam0.5-r0.5-s0", (0, 6000)),
    "c-lr1e-5-lam0.95-r0.5-s0": ("v18c-screen/v18c-lr1e-5-lam0.95-r0.5-s0", (6000,)),
}
ORDER = (  # the yardsticks in the README's order, for the neighbours' delta_yy
    "mean",
    "color",
    "color8",
    "pca16",
    "pca64",
    "pca256",
    "pca1024",
    "v16",
    "v17",
    "init_c_k3",
    "init_c",
    "inc64",
)
SHARE = ("pca16", "pca64", "color8")  # of pca1024 (described)
STAGES = ("pred", "pool", "pred_k3", "vis_color", "oracle", "mean")


def hires_checks(runs: Path) -> Row:
    """The hires checks of every kept frame, by video (``CHECK_LOG``)."""
    out: Row = {}
    for v in VIDEOS:
        got = _load(runs / "v18c-sem" / "hires_rgb" / f"{v}.json")
        if got is None:
            out[v] = None
            continue
        c = got["checks"]
        n = len(c["diff"])
        b = [m for m in c["b_margin"] if m is not None]
        cm = [m for m in c["c_margin"] if m is not None]
        out[v] = {
            "frames": n,
            "diff_mean": sum(c["diff"]) / n,
            "diff_max": max(c["diff"]),
            "data_diff_mean": sum(c["data_diff"]) / n,
            "data_diff_max": max(c["data_diff"]),
            "b_frames": len(b),
            "b_margin_min": min(b, default=None),
            "c_frames": len(cm),
            "c_compared_mean": sum(c["c_compared"]) / n,
            "c_margin_min": min(cm, default=None),
            "c_margin_all_min": min(c["c_margin_all"]),
            "c_all_nonpositive": sum(m <= 0 for m in c["c_margin_all"]),
        }
    return out


def yard_table(yard: dict[str, Row], boot: Boot) -> Row:
    """The yardsticks' values, the neighbours' $\\delta_{yy}$ and the shares."""
    names = [n for n in ORDER if n in yard]
    rows = {
        n: {
            "m_sel": yard[n]["m_sel"],
            "m_final": yard[n]["m_final"],
            "half": yard[n].get("m_sel_half"),
            "alpha_scale": yard[n].get("alpha_scale"),
            "alpha_edge": yard[n].get("alpha_edge"),
            "id_easy_sel": yard[n].get("id_easy_sel"),
            "m_scene_sel": yard[n].get("m_scene_sel"),
        }
        for n in names
    }
    pairs = [
        {
            "pair": f"{a}/{b}",
            "diff": yard[b]["m_sel"] - yard[a]["m_sel"],
            "d_yy": boot.delta(yard[b], yard[a]),
        }
        for a, b in zip(names, names[1:], strict=False)
    ]
    top = yard.get("pca1024")
    share = (
        {n: yard[n]["m_sel"] / top["m_sel"] for n in SHARE if n in yard}
        if top is not None and top["m_sel"] > 0
        else None
    )
    return {"rows": rows, "neighbours": pairs, "share_of_pca1024": share}


def e1_checks(runs: Path, boot: Boot) -> Row:
    """V-E1'1, V-E1'3 and V-E1'4 on ``E1_RUNS`` (``judge_m.e1_repair``'s
    rules), the ``e1diag`` residual and the ``e1f`` values."""
    v1, v3, v4 = True, True, True
    read = {"e1": 0, "step0": 0}
    per: Row = {}
    resid: float | None = None
    for label, (path, steps) in E1_RUNS.items():
        run_dir = runs / path
        got_resid = _resid(run_dir / OUT / "rows", steps)
        if got_resid is not None:
            resid = got_resid if resid is None else max(resid, got_resid)
        rows_dir = run_dir / OUT_SEM / "rows"
        for step in steps:
            e1 = _load(rows_dir / f"{step:06d}-e1.json")
            e1f = _load(rows_dir / f"{step:06d}-e1f.json")
            if e1 is None or e1f is None:
                per[f"{label}/{step}"] = None
                continue
            read["e1"] += 1
            k3 = []
            for x in e1:
                if x.get("mask") == "S" and x["stage"] in (
                    "mean",
                    "oracle_k3",
                    "pred_k3",
                ):
                    ok = x["swap"] == 0.5 and x["n_tie"] == x["n_cand"]
                    v1 &= ok
                    k3.append(
                        {
                            "stage": x["stage"],
                            "target": x.get("target"),
                            "swap": x["swap"],
                            "n_tie": x["n_tie"],
                            "n_cand": x["n_cand"],
                            "id_easy": x.get("id_easy"),
                            "ok": ok,
                        }
                    )
            by = {(x["k"], x["stage"]): x for x in e1f}
            k3_diff = {}
            for k in (4, 8, 12):
                mean = by[(k, "mean")]
                v3 &= mean["m_sel"] == 0.0 and mean["m_final"] == 0.0
                a, b = by[(k, "pred_k3")], by[(k, "pool")]
                k3_diff[k] = a["m_final"] - b["m_final"]
                v3 &= abs(k3_diff[k]) <= K3_TOL
            oracle = None
            if step == 0:
                read["step0"] += 1
                o = by[(4, "oracle")]
                w = 2 * boot.se_v([o["e_final"]], o["v_final"], [1.0], "final")
                v4 &= o["m_final"] > w
                oracle = {"m_final": o["m_final"], "delta_w": w}
            gaps = {}
            for k in (4, 8, 12):
                p = by[(k, "pred")]
                for other in ("pool", "vis_color"):
                    q = by[(k, other)]
                    w = 2 * boot.se_v(
                        [p["e_final"], q["e_final"]],
                        p["v_final"],
                        [1.0, -1.0],
                        "final",
                    )
                    gaps[f"{k}/pred-{other}"] = {
                        "diff": p["m_final"] - q["m_final"],
                        "delta_w": w,
                    }
            per[f"{label}/{step}"] = {
                "k3_e1": k3,
                "k3_diff": k3_diff,
                "oracle_k4": oracle,
                "pool_cond": {k: by[(k, "pool")].get("cond") for k in (4, 8, 12)},
                "m": {
                    f"{k}/{s}": [by[(k, s)]["m_sel"], by[(k, s)]["m_final"]]
                    for k in (4, 8, 12)
                    for s in STAGES
                    if (k, s) in by
                },
                "gaps": gaps,
            }
    return {
        "V-E1'1": {"pass": v1 if read["e1"] else None},
        "V-E1'2": {
            "resid": resid,
            "pass": None if resid is None else resid <= RESID,
            "note": "screen checkpoints (the README reads the long runs')",
        },
        "V-E1'3": {"tol": K3_TOL, "pass": v3 if read["e1"] else None},
        "V-E1'4": {"pass": v4 if read["step0"] else None},
        "runs": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    runs = args.runs
    out: Row = {"hires": hires_checks(runs)}
    print("## V-m0: 高解像度のフレームの確かめ（残したすべてのフレーム）\n")
    for v, h in out["hires"].items():
        print(f"- {v}: {json.dumps(h)}")
    meta = _load(runs / "v18c-sem" / "targets.json")
    if meta is None:
        print("\n的が無い（V-m0 で止まったか、まだ作っていない）")
    else:
        boot = Boot(meta)
        ydir = runs / "v18c-sem" / "yardsticks" / "posthoc_sem" / "rows"
        yard = {
            p.stem.removeprefix("yardstick-"): _load(p)
            for p in sorted(ydir.glob("*.json"))
        }
        out["validity"] = validity(meta, yard, boot)
        out["validity"]["V-m5"] = {"pass": None, "note": "needs sigma_seed"}
        out["yardsticks"] = yard_table(yard, boot)
        print("\n## 有効性の確かめ\n")
        for k, v in out["validity"].items():
            if k.startswith("V-"):
                print(f"- {k}: {v['pass']}")
        lv = out["validity"]["L"]
        print(f"- L = {lv['L']}（tau = {_fmt(lv['tau'], 2)}、上限: {lv['at_max']}）")
        print(f"- α の端: {out['validity']['alpha_edge']}")
        print(f"- V-m3: {out['validity']['V-m3']}")
        print(
            "\n| 物差し | m_sel | m_final | 半分 0 | 半分 1 | c |\n|--|--|--|--|--|--|"
        )
        for n, r in out["yardsticks"]["rows"].items():
            h = r["half"] or [None, None]
            print(
                f"| {n} | {_fmt(r['m_sel'])} | {_fmt(r['m_final'])} | "
                f"{_fmt(h[0])} | {_fmt(h[1])} | {r['alpha_scale']} |"
            )
        print("\n隣どうしの差と δ_yy:")
        for p in out["yardsticks"]["neighbours"]:
            print(f"- {p['pair']}: {_fmt(p['diff'])} (δ_yy {_fmt(p['d_yy'])})")
        print(f"- pca1024 に対する割合: {out['yardsticks']['share_of_pca1024']}")
        out["e1"] = e1_checks(runs, boot)
        print("\n## E1 の修理（V18c の掃引の checkpoint）\n")
        for k in ("V-E1'1", "V-E1'2", "V-E1'3", "V-E1'4"):
            print(f"- {k}: {out['e1'][k]['pass']}")
    out["diagnosis"] = diagnosis(runs)
    d = out["diagnosis"]
    print(
        f"\n## shuf の診断\n\n- H-bug {d['H-bug']}・H-time {d['H-time']}・"
        f"H-skew {d['H-skew']}（残差 {d['resid']}）"
    )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
