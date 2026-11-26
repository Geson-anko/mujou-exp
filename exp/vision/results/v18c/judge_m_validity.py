r"""Where the validity checks of V18c's remade main measure $m$ stopped, and
the diagnosis of E1's ``shuf`` (exp/vision/README.md, "V18c: 作り直した主の
指標の有効性（2026-11-25）"; the rules in "V18c: 主の指標を作り直す（回す前に
決めたこと、2026-11-25）").

V-m0 is the first check: the hires frames' position checks (a) and (b) on
every kept frame, before any target is made (``exp.lib.tours.hires``; the
plan: "外れたら書き出さずに止める。閾値は緩めない"). ``hires.prepare`` stops at
the first frame that fails, so this reads the full check of
``hires_check`` (``runs/v18c-sem/hires_check/<video>.json``) instead, which
writes no frame. If V-m0 fails, no target exists, and V-m1 to V-m6, the
visual check, $m$ of any code and E1's repaired parts that need the
targets' split (``oracle_k3``, ``pred_k3``, ``e1f``) are not run: they are
listed as such.

Also reads the diagnosis of ``shuf`` (``fill_eval diag``) of the old
checkpoints of the README (``judge_m.diagnosis``: H-bug, H-time, H-skew,
described, not judged) and, listed only, of two V18c screen checkpoints
(``SCREEN_DIAG``: their identity residual is V-E1'2's quantity, which the
README reads on the long runs). Prints markdown; ``--json`` also writes the
numbers::

    uv run python -m exp.vision.results.v18c.judge_m_validity [--runs runs] [--json out.json]
"""

import argparse
import json
from pathlib import Path
from typing import Any

from exp.vision.results.v18c.judge_m import MAX_DIFF, RESID, diagnosis

Row = dict[str, Any]

VIDEOS = ("Amsterdam", "Bangkok", "Zurich")
SCREEN_DIAG = {  # listed only: (run, steps)
    "c-lr1e-4-lam0.5-r0.5-s0": ("v18c-screen/v18c-lr1e-4-lam0.5-r0.5-s0", (0, 6000)),
    "c-lr1e-5-lam0.95-r0.5-s0": ("v18c-screen/v18c-lr1e-5-lam0.95-r0.5-s0", (6000,)),
}
TAUS = (1, 2, 256)


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def v_m0(runs: Path) -> Row:
    """V-m0's position checks (a) and (b) on every kept frame."""
    per: Row = {}
    for v in VIDEOS:
        got = _load(runs / "v18c-sem" / "hires_check" / f"{v}.json")
        if got is None:
            per[v] = None
            continue
        per[v] = {
            "frames": got["frames"],
            "listed": got["listed"],
            "a_fail": got["a_fail"],
            "b_fail": got["b_fail"],
            "d": got["d"],
            "luma": got["luma"],
            "signed_mean": got["signed_mean"],
            "near_min_min": got["near_min_min"],
            "parts": got["parts"],
        }
    done = all(p is not None for p in per.values())
    all_kept = done and all(p["frames"] == p["listed"] for p in per.values())
    passed = all_kept and all(
        p["a_fail"] == 0 and p["b_fail"] == 0 for p in per.values()
    )
    return {"videos": per, "max_diff": MAX_DIFF, "complete": all_kept, "pass": passed}


def screen_diag(runs: Path) -> Row:
    """The diagnosis rows of the V18c screen checkpoints (listed only)."""
    out: Row = {"rows": {}, "resid": None}
    resid = 0.0
    for label, (path, steps) in SCREEN_DIAG.items():
        for step in steps:
            rows = _load(
                runs / path / "posthoc_fill" / "rows" / f"{step:06d}-e1diag.json"
            )
            if rows is None:
                continue
            for r in rows:
                resid = max(resid, r["resid"])
                out["rows"][f"{label}/{step}/{r['target']}/{r['stage']}"] = {
                    "swap": r["swap"],
                    **{f"C{t}": r["C"][t] for t in TAUS},
                    "s256": r["s"][256],
                    "s_off": r["s_off"],
                }
    if out["rows"]:
        out["resid"] = resid
        out["resid_ok"] = resid <= RESID
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out: Row = {"V-m0": v_m0(args.runs)}
    m0 = out["V-m0"]
    stopped = not m0["pass"]
    out["not_run"] = (
        [
            "V-m1",
            "V-m2",
            "V-m3",
            "V-m4",
            "V-m5",
            "V-m6",
            "visual",
            "m of every yardstick and run",
            "oracle_k3",
            "pred_k3",
            "e1f",
            "V-E1'1",
            "V-E1'3",
            "V-E1'4",
        ]  # fmt: skip
        if stopped
        else []
    )
    out["m_valid"] = False if stopped else None  # None: read judge_m next
    print("## V-m0（高解像度のフレームの位置）\n")
    print(
        "| 動画 | フレーム | (a) 外れ | (b) 外れ | d 平均 | d 中央 | d 95% | d 最大 "
        "| 輝度の d 平均 | 符号つきの差（R, G, B） | 動く隣の d の最小 |"
    )
    print("| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    for v, p in m0["videos"].items():
        if p is None:
            print(f"| {v} | — | | | | | | | | | |")
            continue
        d, s = p["d"], p["signed_mean"]
        print(
            f"| {v} | {p['frames']} / {p['listed']} | {p['a_fail']} | {p['b_fail']} "
            f"| {d['mean']:.3f} | {d['median']:.3f} | {d['q95']:.3f} | {d['max']:.3f} "
            f"| {p['luma']['mean']:.3f} | {s[0]:+.2f}, {s[1]:+.2f}, {s[2]:+.2f} "
            f"| {p['near_min_min']:.2f} |"
        )
    print("\n先頭のフレームでの分解\n")
    print(
        "| 動画 | 枚数 | 確かめの d | 同じフィルタの 64 を再び | RGB で縮めた 320 と 64 | RGB で縮めた 64 とデータ | d が最小のずらし |"
    )
    print("| -- | -- | -- | -- | -- | -- | -- |")
    for v, p in m0["videos"].items():
        if p is None:
            continue
        q = p["parts"]
        print(
            f"| {v} | {q['frames']} | {q['check']:.3f} | {q['fresh64']:.3f} "
            f"| {q['rgb']:.3f} | {q['rgb64_data']:.3f} | {q['shift_best']} |"
        )
    print(f"\n- V-m0: {'通る' if m0['pass'] else '外れる'}")
    if stopped:
        print(f"- 走らせなかったもの: {', '.join(out['not_run'])}")
    out["diagnosis"] = diagnosis(args.runs)
    d = out["diagnosis"]
    print(
        f"\n## E1 の shuf の診断（旧 checkpoint）\n\n- 残差の最大 {d['resid']:.1e}"
        f"、H-bug {d['H-bug']}・H-time {d['H-time']}・H-skew {d['H-skew']}\n"
    )
    print("| 行 | swap | C(1) | C(2) | C(256) | s(256) | s_off | time | skew |")
    print("| -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    for k, r in d["rows"].items():
        print(
            f"| {k} | {r['swap']:.3f} | {r['C1']:+.3f} | {r['C2']:+.3f} "
            f"| {r['C256']:+.3f} | {r['s256']:.3f} | {r['s_off']:.3f} "
            f"| {r['time']} | {r['skew']} |"
        )
    out["screen_diag"] = screen_diag(args.runs)
    sd = out["screen_diag"]
    if sd["rows"]:
        print(f"\n## 参考: V18c の短い run の診断（残差の最大 {sd['resid']:.1e}）\n")
        print("| 行 | swap | C(1) | C(2) | C(256) | s(256) | s_off |")
        print("| -- | -- | -- | -- | -- | -- | -- |")
        for k, r in sd["rows"].items():
            print(
                f"| {k} | {r['swap']:.3f} | {r['C1']:+.3f} | {r['C2']:+.3f} "
                f"| {r['C256']:+.3f} | {r['s256']:.3f} | {r['s_off']:.3f} |"
            )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
