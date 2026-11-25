r"""The description of V18c's time linearity and information width
(exp/vision/README.md, 「V18c: 時間の線形性と情報の幅（回す前に決めたこと）」;
``exp.vision.geom_eval``). Written before any number. No verdict on a code:
the stages lay out the checks and the material for $\Delta_*$.

``--stage checks`` reads ``<geom>/checks/rows.json`` (V-d0a to V-d0f): each
check's passes and failures, and the records (the TwoNN's recovery beyond
the registered two).

``--stage yard`` reads ``<geom>/yard/vd.json`` and ``yard/rows/*.json``:

1. V-d0h, V-d1 and V-d2 (a failure stops the reading; the tables after it
   are printed for the record only), with V-d2's $\hat\delta$, its recovery
   $\hat\delta / \delta$ and whether $\hat\delta \pm 2\,\mathrm{SE}^*$
   holds 0;
2. V-d3's material, of the yardsticks (``YARDSTICKS``, each place apart):
   (a) of each lag, the ``in`` stratum's $G$, $s$ and their
   $\mathrm{SE}^*$; (b) the widths $4\,\mathrm{SE}^*$ of the paired
   differences of $G$ of every two yardsticks of a place (the median and the
   largest of each lag); (c) $G$ of the reference's halves, $k$ = 48, 96
   and $c_\epsilon = 10^{-3}$, and of each the pairs of yardsticks whose
   order of $G$ flips from the main whitening; (d) the TwoNN of 8 s (the
   offsets' mean, smallest and largest, ``pre`` and ``white``, $k$ = 16, 64,
   256), the thinning's curve ($N$, $\rho$, $\hat d$ of each spacing), the
   share of near neighbours and $\hat d$ without them, and the standard
   deviation of the paired half-samples' differences of $\hat d$ (median and
   largest of the pairs; a guide to the SE of the whole sample's difference,
   likely on the large side: a half-sample's TwoNN draws its neighbours
   anew, near an estimate of $N/2$ points); (e) the synthetic controls
   against their stationary values (``theory``): of each the main row of
   its population's whitening (``ref`` ``population``, $k = D$), and as
   records the rows whitened by its reference at $k$ and ``k_side``; (f) of
   the yardsticks' main whitening, ``geom_eval.spread_diagnosis`` ($V/k$,
   the largest variance of a direction and the top 5's share of $V$).

Before them, whether the step 0 of a seed's runs are the same
(``step0_same``; where not, each run's step 0 is a yardstick
``<init>@<run>``).

$\mathrm{SE}^*$ is ``geom_eval.ratio_se`` and ``geom_eval.s_se``: the larger
of the block bootstraps of 128 s and 256 s, 2000 resamples, seed 101. The
``apply`` stage (the models' values) is written after $\Delta_*$ is
decided. Prints markdown; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_geom --stage checks|yard \
        [--geom runs/v18c-geom] [--json out.json]
"""

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from exp.lib.midpoint import permutation_terms, stationary_terms
from exp.lib.synth_series import gauss_kernel, kernel_acf
from exp.vision.geom_eval import GeomConf, cheb_js, ratio_se, s_se

Row = dict[str, Any]

YARDSTICKS = {
    "own": ("pca", "color", "mean8", "init", "init_f", "init_m", "init_c", "init_c1",
            "init_v16"),
    "h_mean": ("init", "init_f", "init_m", "init_c", "init_c1", "init_v16"),
}  # fmt: skip
SENS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("k48", {"k": 48}),
    ("k96", {"k": 96}),
    ("eps1e-3", {"eps": 1e-3}),
    ("front", {"ref": "front"}),
    ("back", {"ref": "back"}),
)
CONTROLS = ("pca_perm", "ou_mix_1", "ou_mix_3", "ou_mix_10", "ou_mix_30", "cheb_a",
            "cheb_b", "cheb_c", "k3_z", "k3_hmean")  # fmt: skip


def _yardsticks(by_code: Mapping[str, Sequence[Row]]) -> dict[str, tuple[str, ...]]:
    """``YARDSTICKS`` and the step 0 of each run of a seed whose step 0
    differ (``<init>@<run>``)."""
    extra = tuple(sorted(c for c in by_code if "@" in c))
    return {place: codes + extra for place, codes in YARDSTICKS.items()}


def _load(path: Path) -> list[Row]:
    return json.loads(path.read_text()) if path.exists() else []


def _of(rows: Sequence[Row], **labels: Any) -> list[Row]:
    return [r for r in rows if all(r.get(k) == v for k, v in labels.items())]


def _fmt(x: float | None, digits: int = 4) -> str:
    return (
        "—"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{x:.{digits}g}"
    )


# -- the checks ----------------------------------------------------------------------


def checks_stage(rows: Sequence[Row]) -> dict[str, Any]:
    """Each check's passes and failures, and the records."""
    out: dict[str, Any] = {"checks": {}, "failures": [], "records": []}
    for r in rows:
        if r["part"] == "check":
            c = out["checks"].setdefault(r["name"], {"pass": 0, "fail": 0})
            c["pass" if r["pass"] else "fail"] += 1
            if not r["pass"]:
                out["failures"].append(r)
        elif r["part"] == "record":
            out["records"].append(r)
    print("## V-d0a〜V-d0f\n")
    print("| 確かめ | 通った | 外れた |\n| -- | -- | -- |")
    for name, c in out["checks"].items():
        print(f"| {name} | {c['pass']} | {c['fail']} |")
    if out["failures"]:
        print("\n**外れた行（ここで止める）**\n")
        for r in out["failures"]:
            print("-", json.dumps({k: v for k, v in r.items() if k != "snap"}))
    print("\n## TwoNN の記録\n")
    for r in out["records"]:
        label = ", ".join(f"{k}={r[k]}" for k in ("dim", "noise") if k in r)
        each = r.get("each") or []
        print(f"- {r['name']} {label}: d = {_fmt(r.get('d'))}, each = "
              + ", ".join(_fmt(x, 3) for x in each))  # fmt: skip
    return out


# -- the yard ------------------------------------------------------------------------


def _main(rows: Sequence[Row], g: GeomConf, **labels: Any) -> list[Row]:
    return _of(rows, k=g.k, eps=g.c_eps, ref="all", **labels)


def _mid(rows: Sequence[Row], g: GeomConf, place: str, lag: int, stratum: str = "in",
         **variant: Any) -> Row | None:  # fmt: skip
    base = {"k": g.k, "eps": g.c_eps, "ref": "all"} | variant
    got = _of(rows, part="mid", place=place, lag=lag, stratum=stratum, **base)
    return got[0] if len(got) == 1 and "G" in got[0] else None


def _g_se(rows: Sequence[Row], g: GeomConf) -> float:
    num = torch.tensor([r["num_b"] for r in rows], dtype=torch.float64)
    den = torch.tensor([r["den_b"] for r in rows], dtype=torch.float64)
    coef = torch.tensor([1.0] if len(rows) == 1 else [1.0, -1.0], dtype=torch.float64)
    return ratio_se(num, den, coef, g)


def curves(
    by_code: Mapping[str, Sequence[Row]],
    g: GeomConf,
    yardsticks: Mapping[str, Sequence[str]] = YARDSTICKS,
) -> dict[str, Any]:
    """(a): the ``in`` stratum's $G$, $s$ and their $\\mathrm{SE}^*$."""
    out: dict[str, Any] = {}
    for place, codes in yardsticks.items():
        for code in codes:
            rows = by_code.get(code, [])
            spread = _main(rows, g, part="spread", place=place)
            for lag in g.lags:
                m = _mid(rows, g, place, lag)
                if m is None or len(spread) != 1:
                    continue
                out.setdefault(place, {}).setdefault(code, {})[lag] = {
                    "G": m["G"],
                    "s": m["s"],
                    "se_G": _g_se([m], g),
                    "se_s": s_se(spread[0], m, g),
                    "n": m["n_triples"],
                }
    return out


def pair_widths(
    by_code: Mapping[str, Sequence[Row]],
    g: GeomConf,
    yardsticks: Mapping[str, Sequence[str]] = YARDSTICKS,
) -> dict[str, Any]:
    """(b): the widths $4\\,\\mathrm{SE}^*$ of the paired differences of
    $G$ of every two yardsticks of a place, by lag."""
    out: dict[str, Any] = {}
    for place, codes in yardsticks.items():
        for lag in g.lags:
            widths = []
            for a, b in combinations(codes, 2):
                ma = _mid(by_code.get(a, []), g, place, lag)
                mb = _mid(by_code.get(b, []), g, place, lag)
                if ma is not None and mb is not None:
                    widths.append(4 * _g_se([ma, mb], g))
            if widths:
                out.setdefault(place, {})[lag] = {
                    "median": statistics.median(widths), "max": max(widths), "pairs": len(widths),
                }  # fmt: skip
    return out


def _flips(main: Mapping[str, float], other: Mapping[str, float]) -> int:
    both = [c for c in main if c in other]
    return sum(
        (main[a] - main[b]) * (other[a] - other[b]) < 0
        for a, b in combinations(both, 2)
    )


def sensitivity(
    by_code: Mapping[str, Sequence[Row]],
    g: GeomConf,
    yardsticks: Mapping[str, Sequence[str]] = YARDSTICKS,
) -> dict[str, Any]:
    """(c): $G$ of each variant of the whitening and the pairs whose order
    flips from the main."""
    out: dict[str, Any] = {}
    for place, codes in yardsticks.items():
        for lag in g.lags:
            main = {
                c: m["G"]
                for c in codes
                if (m := _mid(by_code.get(c, []), g, place, lag))
            }
            entry: dict[str, Any] = {"main": main}
            for name, variant in SENS:
                got = {c: m["G"] for c in codes
                       if (m := _mid(by_code.get(c, []), g, place, lag, **variant))}  # fmt: skip
                entry[name] = {"G": got, "flips": _flips(main, got)}
            out.setdefault(place, {})[lag] = entry
    return out


def twonn_table(
    by_code: Mapping[str, Sequence[Row]],
    g: GeomConf,
    yardsticks: Mapping[str, Sequence[str]] = YARDSTICKS,
) -> dict[str, Any]:
    """(d): the TwoNN's offsets, thinning, near neighbours and the paired
    half-samples."""
    out: dict[str, Any] = {"codes": {}, "pairs": {}}
    first = g.spacings[0]
    for place, codes in yardsticks.items():
        for code in codes:
            rows = by_code.get(code, [])
            for k in (g.k, *g.k_side):
                for metric in ("pre", "white"):
                    tw = _of(rows, part="twonn", place=place, k=k, metric=metric)
                    if not tw:
                        continue
                    entry: dict[Any, Any] = {}
                    for spacing in g.spacings:
                        ds = [r["d"] for r in tw if r["spacing"] == spacing]
                        ok = [d for d in ds if d is not None]
                        rhos = [
                            r["rho"]
                            for r in tw
                            if r["spacing"] == spacing and r["rho"] is not None
                        ]
                        entry[spacing] = {
                            "mean": sum(ok) / len(ok) if ok and len(ok) == len(ds) else None,
                            "min": min(ok) if ok else None, "max": max(ok) if ok else None,
                            "N": statistics.mean(r["N"] for r in tw if r["spacing"] == spacing),
                            "rho": statistics.mean(rhos) if rhos else None,
                        }  # fmt: skip
                    s8 = [r for r in tw if r["spacing"] == first and "near_share" in r]
                    if s8:
                        entry["near_share"] = statistics.mean(
                            r["near_share"] for r in s8
                        )
                        far = [r["d_far"] for r in s8 if r.get("d_far") is not None]
                        entry["d_far"] = (
                            statistics.mean(far) if len(far) == len(s8) else None
                        )
                    out["codes"].setdefault(place, {}).setdefault(code, {})[
                        f"{metric}-k{k}"
                    ] = entry
        for metric in ("pre", "white"):
            sds = []
            for a, b in combinations(codes, 2):
                sa = _of(
                    by_code.get(a, []),
                    part="twonn_sub",
                    place=place,
                    k=g.k,
                    metric=metric,
                )
                sb = _of(
                    by_code.get(b, []),
                    part="twonn_sub",
                    place=place,
                    k=g.k,
                    metric=metric,
                )
                if len(sa) == 1 and len(sb) == 1:
                    diffs = [x - y for x, y in zip(sa[0]["d"], sb[0]["d"], strict=True)
                             if x is not None and y is not None]  # fmt: skip
                    if len(diffs) > 1:
                        sds.append(statistics.stdev(diffs))
            if sds:
                out["pairs"].setdefault(place, {})[metric] = {
                    "median": statistics.median(sds), "max": max(sds), "pairs": len(sds),
                }  # fmt: skip
    return out


def theory(code: str, lag: int, g: GeomConf, n: int) -> tuple[float, float] | None:
    r"""The stationary $s$ and $G_{\mathrm{mid}}$ of a synthetic control at
    ``lag`` (``pca_perm``: $E[s] = N/(N-1)$ and the ratio of the expected
    terms, 3/4), None for the K3's."""
    if code == "pca_perm":
        num, den = permutation_terms(1.0, n)
        return n / (n - 1), num / den
    if code.startswith("ou_mix_"):
        c = int(code.removeprefix("ou_mix_"))
        r = math.exp(-1 / g.ou_tau)
        k = gauss_kernel(c * g.ou_tau)

        def sbar(m: int) -> float:
            return 1 - (r**m + float(kernel_acf(k, torch.tensor([m]))[0])) / 2

    elif code.startswith("cheb_"):
        js = cheb_js(code, g).double()
        omega = 2 * math.pi / (g.cheb_period * float(js.mean()))

        def sbar(m: int) -> float:
            return 1 - float(torch.cos(js * omega * m).mean())

    else:
        return None
    s1, s2 = sbar(lag), sbar(2 * lag)
    num, den = stationary_terms(torch.tensor(1 - s1, dtype=torch.float64),
                                torch.tensor(1 - s2, dtype=torch.float64))  # fmt: skip
    return s1, float(num / den)


def _control_variants(
    rows: Sequence[Row], g: GeomConf, place: str
) -> list[tuple[str, dict[str, Any]]]:
    """The whitenings of a control in order, the main first: its
    population's (``k`` its $D$) where it has one, then its reference's at
    $k$ and ``k_side``."""
    out: list[tuple[str, dict[str, Any]]] = []
    pop = _of(rows, part="spec", place=place, ref="population")
    if pop:
        out.append(("population", {"ref": "population", "k": pop[0]["k"]}))
    return out + [(f"ref-k{k}", {"k": k}) for k in (g.k, *g.k_side)]


def _diag(row: Row) -> dict[str, Any]:
    return {k: row.get(k) for k in ("V_over_k", "var_max", "top_share")}


def controls(
    by_code: Mapping[str, Sequence[Row]], g: GeomConf, n: int
) -> dict[str, Any]:
    """(e): the synthetic controls' $G$, $s$ (``all`` stratum; None where
    not scoreable) of each whitening (``_control_variants``, ``main`` the
    first) with its ``spread_diagnosis``, against ``theory``."""
    out: dict[str, Any] = {}
    for code in CONTROLS:
        rows = by_code.get(code, [])
        place = "h_mean" if code == "k3_hmean" else "own"
        entry: dict[str, Any] = {"main": None, "theory": {}, "variants": {}}
        for lag in g.lags:
            t = theory(code, lag, g, n)
            entry["theory"][lag] = {"s": t[0] if t else None, "G": t[1] if t else None}
        for name, variant in _control_variants(rows, g, place):
            base = {"k": g.k, "eps": g.c_eps, "ref": "all"} | variant
            spec = _of(rows, part="spec", place=place, **base)
            spread = _of(rows, part="spread", place=place, **base)
            v: dict[str, Any] = {
                "scoreable": spec[0]["scoreable"] if spec else None,
                "why": spec[0]["why"] if spec else None,
                "diag": _diag(spread[0]) if len(spread) == 1 else None,
                "lags": {},
            }
            for lag in g.lags:
                m = _mid(rows, g, place, lag, stratum="all", **variant)
                v["lags"][lag] = {
                    "G": m["G"] if m else None,
                    "s": m["s"] if m else None,
                }
            entry["variants"][name] = v
            entry["main"] = entry["main"] or name
        out[code] = entry
    return out


def spread_table(
    by_code: Mapping[str, Sequence[Row]],
    g: GeomConf,
    yardsticks: Mapping[str, Sequence[str]] = YARDSTICKS,
) -> dict[str, Any]:
    """(f): ``spread_diagnosis`` of the yardsticks' main whitening."""
    out: dict[str, Any] = {}
    for place, codes in yardsticks.items():
        for code in codes:
            got = _main(by_code.get(code, []), g, part="spread", place=place)
            if len(got) == 1:
                out.setdefault(place, {})[code] = _diag(got[0])
    return out


def yard_stage(geom: Path, g: GeomConf) -> dict[str, Any]:
    vd = _load(geom / "yard" / "vd.json")
    by_code = {
        p.stem: _load(p) for p in sorted((geom / "yard" / "rows").glob("*.json"))
    }
    sets = (
        json.loads((geom / "sets.json").read_text())
        if (geom / "sets.json").exists()
        else {}
    )
    checks = [r for r in vd if r["part"] == "check"]
    stop = not checks or not all(r["pass"] for r in checks)
    same = _of(vd, part="record", name="step0_same")
    out: dict[str, Any] = {"checks": checks, "stop": stop, "step0_same": same}
    ys = _yardsticks(by_code)
    print("## step 0 の一致（`self_eval.same_weights`）\n")
    for r in same:
        what = "同じ" if r["same"] else "違う（各 run の step 0 も物差しとして回した）"
        print(f"- {r['init']}: {what}（{len(r['runs'])} 本）")
    print("\n## V-d0h・V-d1・V-d2\n")
    print("| 確かめ | ラベル | 合否 |\n| -- | -- | -- |")
    for r in checks:
        label = ", ".join(
            f"{k}={r[k]}" for k in ("set", "code", "lag", "delta") if k in r
        )
        print(f"| {r['name']} | {label} | {'通った' if r['pass'] else '外れた'} |")
    if stop:
        print("\n**外れたか未実行（ここで止める。以下は記録だけ）**\n")
    print(
        "\n### V-d2 の回収\n\n| 刻み | δ | δ̂ | SE* | 回収率 | 0 から分かれる |\n| -- | -- | -- | -- | -- | -- |"
    )
    for r in _of(checks, name="V-d2"):
        print(f"| {r['lag'] / 10:g} 秒 | {r['delta']} | {_fmt(r['hat'])} | {_fmt(r['se'])} | "
              f"{_fmt(r['recovery'], 3)} | {'はい' if r['apart_from_0'] else 'いいえ'} |")  # fmt: skip
    ident = sets.get("ident", {})
    n = ident["frames"] - (ident["stack"] - 1) * ident["stride"] if ident else 0
    out["curves"] = curves(by_code, g, ys)
    out["pair_widths"] = pair_widths(by_code, g, ys)
    out["sensitivity"] = sensitivity(by_code, g, ys)
    out["twonn"] = twonn_table(by_code, g, ys)
    out["controls"] = controls(by_code, g, n) if n > 1 else {}
    out["spread"] = spread_table(by_code, g, ys)
    print("\n## V-d3 の材料\n\n### (a) `in` の G と s（2 SE*）\n")
    for place, codes in out["curves"].items():
        print(
            f"\n{place}\n\n| 符号 | "
            + " | ".join(f"{lag / 10:g} 秒" for lag in g.lags)
            + " |"
        )
        print("|" + " -- |" * (len(g.lags) + 1))
        for code, by_lag in codes.items():
            cells = [f"{_fmt(v['G'], 3)}±{_fmt(2 * v['se_G'], 2)} (s {_fmt(v['s'], 3)})"
                     if (v := by_lag.get(lag)) else "—" for lag in g.lags]  # fmt: skip
            print(f"| {code} | " + " | ".join(cells) + " |")
    print("\n### (b) 物差しの対の差の区間の幅 4 SE*（中央値・最大）\n")
    for place, by_lag in out["pair_widths"].items():
        print(f"- {place}: " + "、".join(f"{lag / 10:g} 秒 {_fmt(v['median'], 3)}・{_fmt(v['max'], 3)}"
                                        for lag, v in by_lag.items()))  # fmt: skip
    print("\n### (c) 白色化の感度（順位の入れ替わりの対の数）\n")
    for place, by_lag in out["sensitivity"].items():
        print(f"- {place}: " + "、".join(
            f"{lag / 10:g} 秒 " + "/".join(f"{name} {e[name]['flips']}" for name, _ in SENS)
            for lag, e in by_lag.items()))  # fmt: skip
    print("\n### (d) TwoNN（8 秒、開始の秒の平均 [最小, 最大]）\n")
    for place, codes in out["twonn"]["codes"].items():
        for code, by_metric in codes.items():
            cells = []
            for key, e in by_metric.items():
                first = e.get(g.spacings[0], {})
                cells.append(f"{key} {_fmt(first.get('mean'), 3)} [{_fmt(first.get('min'), 3)}, "
                             f"{_fmt(first.get('max'), 3)}]")  # fmt: skip
            print(f"- {place} {code}: " + "; ".join(cells))
    for place, by_metric in out["twonn"]["pairs"].items():
        print(f"- 半分の部分標本の対の差の標準偏差（全体の差の SE の目安。おそらく保守側） {place}: " + "、".join(
            f"{m} {_fmt(v['median'], 3)}・{_fmt(v['max'], 3)}" for m, v in by_metric.items()))  # fmt: skip
    print("\n### (e) 合成の対照（all の層。主は母集団の白色化、参照の白色化は記録）\n")
    for code, e in out["controls"].items():
        for name, v in e["variants"].items():
            tag = f"{code} {name}" + ("（主）" if name == e["main"] else "（記録）")
            if not v["scoreable"]:
                print(f"- {tag}: 採点できない（{v['why']}）")
                continue
            d = v["diag"] or {}
            print(f"- {tag} [V/k {_fmt(d.get('V_over_k'), 3)}, 最大 {_fmt(d.get('var_max'), 3)}, "
                  f"上位 5 {_fmt(d.get('top_share'), 3)}]: " + "、".join(
                f"{lag / 10:g} 秒 G {_fmt(x['G'], 3)}（理論 {_fmt(e['theory'][lag]['G'], 3)}）"
                f" s {_fmt(x['s'], 3)}（理論 {_fmt(e['theory'][lag]['s'], 3)}）"
                for lag, x in v["lags"].items()))  # fmt: skip
    print("\n### (f) 白色化した評価の分散の診断（主の白色化）\n")
    for place, codes in out["spread"].items():
        for code, d in codes.items():
            print(f"- {place} {code}: V/k {_fmt(d['V_over_k'], 3)}、方向の分散の最大 "
                  f"{_fmt(d['var_max'], 3)}、上位 5 方向の割合 {_fmt(d['top_share'], 3)}")  # fmt: skip
    return out


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--stage", choices=("checks", "yard"), required=True)
    p.add_argument("--geom", type=Path, default=Path("runs/v18c-geom"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    g = GeomConf()
    if a.stage == "checks":
        out = checks_stage(_load(a.geom / "checks" / "rows.json"))
    else:
        out = yard_stage(a.geom, g)
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    main()
