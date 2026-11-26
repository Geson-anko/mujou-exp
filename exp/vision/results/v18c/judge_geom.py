r"""The description of V18c's time linearity and information width
(exp/vision/README.md, 「V18c: 時間の線形性と情報の幅（回す前に決めたこと）」;
``exp.vision.geom_eval``). Written before any number. No verdict on a code:
the stages lay out the checks, the material for $\Delta_*$ and the
models' values with the registered wording.

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

``--stage apply`` (written after the user decided $\Delta_*$, before any
model's value) reads the models' rows (``MODELS``: each run's
``posthoc_geom/rows/<step>.json`` and V18c's ``k3/<step>.json`` under
``--runs``; the VAEs' ``<geom>/vae/rows/<name>.json``) with the yard's, and
lays out their curves, their paired differences (against the yardsticks,
of the models at 60000, of V18c's 6000 and 60000 and of its seeds at 6000)
with the wording of README's 「言葉づかいの規則」 ($\Delta_*$ 0.03 of $G$ at
the lags V-d2 resolves, ``judged_lags``; 4 of $\hat d$, no order of two
codes over 20), the TwoNN, the spread's diagnosis against the provisional
bounds and the K3's rows.

$\mathrm{SE}^*$ is ``geom_eval.ratio_se`` and ``geom_eval.s_se``: the larger
of the block bootstraps of 128 s and 256 s, 2000 resamples, seed 101.
Prints markdown; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_geom --stage checks|yard|apply \
        [--geom runs/v18c-geom] [--runs runs] [--json out.json]
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

from exp.lib.metrics import block_counts, max_t_critical, ratio_resamples
from exp.lib.midpoint import permutation_terms, stationary_terms
from exp.lib.synth_series import gauss_kernel, kernel_acf
from exp.vision.geom_eval import OUT, GeomConf, cheb_js, ratio_se, s_se

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


# -- the apply stage -----------------------------------------------------------------

DELTA_G = 0.03  # Delta* of G_mid, every lag (README, the user's decision 2026-11-26)
DELTA_D = 4.0  # Delta* of d-hat (pre, k = 64, 8 s)
D_TRUST = 20.0  # no order of two codes whose d-hat both exceed it
DIAG_V = (0.5, 2.0)  # the provisional bounds of the spread's diagnosis
DIAG_MAX = 10.0
DIAG_TOP_SHARE = 0.4

V18 = "v18-long/v18-long-lr1e-4-lam0.5-"
V18B = "v18b-long/v18b-long-"
MODELS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    *((f"{V18}{r}", "v18", (60000,)) for r in ("r0.25-s0", "r0.5-s0", "r0.75-s0", "r0.25-s1")),
    *((f"{V18B}{r}", "v18b_f", (60000,)) for r in ("f-lr1e-4-lam0.5-r0.25-s0", "f-lr1e-4-lam0.5-r0.25-s1")),
    *((f"{V18B}{r}", "v18b_m", (60000,)) for r in ("m-lr3e-5-lam0.8-r0.25-s0", "m-lr3e-5-lam0.8-r0.25-s1")),
    *((f"v18c-long/v18c-lr{lr}-lam{lam}-r0.5-s{seed}", "v18c", (6000, 60000))
      for lr in ("1e-4", "1e-5") for lam in ("0.5", "0.8", "0.95") for seed in (0, 1)),
)  # fmt: skip
VAES = ("v16", "v17")
FINAL = 60000


def classify(lo: float, hi: float, delta: float) -> str:
    r"""The wording of an interval $[\mathrm{lo}, \mathrm{hi}]$ of a paired
    difference (README, 「言葉づかいの規則」): ``distinct`` (「実用上の差を区別
    できた」) where it clears $\Delta_*$ wholly on one side, ``small``
    (「実用上の差は小さい」) where it lies in $[-\Delta_*, \Delta_*]$,
    ``unresolved`` (「解像度が足りない」) otherwise."""
    if lo > delta or hi < -delta:
        return "distinct"
    if -delta <= lo and hi <= delta:
        return "small"
    return "unresolved"


def judged_lags(vd_rows: Sequence[Row], delta: float, g: GeomConf) -> tuple[int, ...]:
    r"""The lags whose $G$ is judged (README, 「ユーザーの決定（2026-11-26）」):
    where V-d2's smallest registered $\delta \ge \Delta_*$ was made and its
    interval $\hat\delta \pm 2\,\mathrm{SE}^*$ holds no 0. A lag that cannot
    make it (no row; $\delta \ge \tfrac34 - G$) is not judged, as one whose
    interval holds 0."""
    d0 = min(d for d in g.deltas if d >= delta - 1e-12)
    return tuple(
        lag
        for lag in g.lags
        if any(
            r["lag"] == lag and math.isclose(r["delta"], d0) and r["apart_from_0"]
            for r in _of(vd_rows, part="check", name="V-d2")
        )
    )


def paired_g(
    rows_a: Sequence[Row],
    rows_b: Sequence[Row],
    place: str,
    g: GeomConf,
    judged: Sequence[int],
    delta: float = DELTA_G,
    stratum: str = "in",
) -> dict[str, Any]:
    r"""The paired differences $G_a - G_b$ of each lag of ``g.lags`` (the
    main whitening, ``stratum``): $\mathrm{SE}^*$ (``ratio_se``), the
    interval $\pm 2\,\mathrm{SE}^*$ and the simultaneous one $\pm
    c\,\mathrm{SE}^*$ over all the lags ($c$ of ``metrics.max_t_critical``
    of the 128 s resamples, each lag's $t$ by its own resamples' standard
    deviation), and of a ``judged`` lag the wording of each (``classify``;
    None of another lag: described only). A lag of zero spread (a
    difference the same in every resample) adds no $t$."""
    lags, est, res, out = [], [], [], {}
    coef = torch.tensor([1.0, -1.0], dtype=torch.float64)
    for lag in g.lags:
        ma = _mid(rows_a, g, place, lag, stratum)
        mb = _mid(rows_b, g, place, lag, stratum)
        if ma is None or mb is None:
            continue
        num = torch.tensor([ma["num_b"], mb["num_b"]], dtype=torch.float64)
        den = torch.tensor([ma["den_b"], mb["den_b"]], dtype=torch.float64)
        gen = torch.Generator().manual_seed(g.boot_seed)
        res.append(
            ratio_resamples(num, den, block_counts(num.shape[1], g.boot, gen)) @ coef
        )
        lags.append(lag)
        est.append(ma["G"] - mb["G"])
        out[lag] = {"diff": est[-1], "se": ratio_se(num, den, coef, g)}
    c = 0.0
    if lags:
        r = torch.stack(res, 1)
        r = r[torch.isfinite(r).all(1)]
        sd = r.std(0)
        keep = sd > 0
        if bool(keep.any()):
            c = max_t_critical(r[:, keep], torch.tensor(est, dtype=torch.float64)[keep],
                               sd[keep])  # fmt: skip
    for lag, e in out.items():
        e["lo"], e["hi"] = e["diff"] - 2 * e["se"], e["diff"] + 2 * e["se"]
        e["lo_sim"], e["hi_sim"] = e["diff"] - c * e["se"], e["diff"] + c * e["se"]
        on = lag in judged
        e["cls"] = classify(e["lo"], e["hi"], delta) if on else None
        e["cls_sim"] = classify(e["lo_sim"], e["hi_sim"], delta) if on else None
    return {"c": c, "lags": out}


def _d_mean(rows: Sequence[Row], place: str, g: GeomConf) -> float | None:
    tw = _of(rows, part="twonn", place=place, k=g.k, metric="pre",
             spacing=g.spacings[0])  # fmt: skip
    ds = [r["d"] for r in tw]
    return sum(ds) / len(ds) if ds and None not in ds else None


def paired_d(
    rows_a: Sequence[Row], rows_b: Sequence[Row], place: str, g: GeomConf
) -> dict[str, Any] | None:
    r"""The paired difference of $\hat d$ (``pre``, $k$ = 64, 8 s, the
    offsets' mean) $a - b$, the standard deviation of the paired
    half-samples' differences ``sd`` (a guide to its SE, likely on the large
    side), the interval $\pm 2\,\mathrm{sd}$ and its wording with
    $\Delta_* = 4$; ``above_20`` where both exceed 20 (no order of such
    codes; README, 「ユーザーの決定（2026-11-26）」)."""
    da, db = _d_mean(rows_a, place, g), _d_mean(rows_b, place, g)
    sa = _of(rows_a, part="twonn_sub", place=place, k=g.k, metric="pre")
    sb = _of(rows_b, part="twonn_sub", place=place, k=g.k, metric="pre")
    if da is None or db is None or len(sa) != 1 or len(sb) != 1:
        return None
    diffs = [x - y for x, y in zip(sa[0]["d"], sb[0]["d"], strict=True)
             if x is not None and y is not None]  # fmt: skip
    sd = statistics.stdev(diffs)
    diff = da - db
    lo, hi = diff - 2 * sd, diff + 2 * sd
    cls = "above_20" if da > D_TRUST and db > D_TRUST else classify(lo, hi, DELTA_D)
    return {"a": da, "b": db, "diff": diff, "sd": sd, "lo": lo, "hi": hi, "cls": cls}


def diag_flags(diag: Mapping[str, float]) -> list[str]:
    """The measures of ``spread_diagnosis`` beyond the provisional bounds
    ($V/k \\notin [0.5, 2]$, the largest variance of a direction over 10,
    the top 5's share over 0.4): 「計量が参照と評価のずれに依存する」."""
    out = []
    if not DIAG_V[0] <= diag["V_over_k"] <= DIAG_V[1]:
        out.append("V_over_k")
    if diag["var_max"] > DIAG_MAX:
        out.append("var_max")
    if diag["top_share"] > DIAG_TOP_SHARE:
        out.append("top_share")
    return out


def _label(run: str, step: int | None) -> str:
    name = run.rsplit("/", 1)[-1]
    return name if step is None else f"{name}@{step}"


def load_models(
    geom: Path, runs: Path
) -> tuple[dict[str, list[Row]], dict[str, list[Row]], dict[str, dict[str, Any]]]:
    """The models' rows by label (``<run>@<step>``, the VAEs by name), the
    V18c's K3 rows by the same label, and each label's family and step."""
    rows: dict[str, list[Row]] = {}
    k3: dict[str, list[Row]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for name in VAES:
        rows[name] = _load(geom / "vae" / "rows" / f"{name}.json")
        meta[name] = {"family": "vae", "step": None, "run": name}
    for run, family, steps in MODELS:
        for step in steps:
            label = _label(run, step)
            rows[label] = _load(runs / run / OUT / "rows" / f"{step:06d}.json")
            meta[label] = {
                "family": family,
                "step": step,
                "run": run.rsplit("/", 1)[-1],
            }
            if family == "v18c":
                k3[label] = _load(runs / run / OUT / "k3" / f"{step:06d}.json")
    return rows, k3, meta


def _init_of(meta: Mapping[str, Any]) -> str | None:
    """The yardstick of a model's own step 0: the yard's checkpoint of its
    run, or for V18c its seed's (``step0_same`` holds); None where the yard
    has none (V17, V18 other than ``r0.25-s0``, V18b ``s1``: whether their
    step 0 equal the yard's is not checked)."""
    run, fam = meta["run"], meta["family"]
    if fam == "v18c":
        return "init_c" if run.endswith("-s0") else "init_c1"
    own = {"v16": "init_v16", "v18-long-lr1e-4-lam0.5-r0.25-s0": "init",
           "v18b-long-f-lr1e-4-lam0.5-r0.25-s0": "init_f",
           "v18b-long-m-lr3e-5-lam0.8-r0.25-s0": "init_m"}  # fmt: skip
    return own.get(run)


def apply_stage(geom: Path, runs: Path, g: GeomConf) -> dict[str, Any]:
    """The models' values (README, 「V18c: 時間の線形性と情報の幅の結果」):
    the curves, the paired differences against the yardsticks, of the
    models' pairs at step 60000, of V18c's 6000 and 60000 and of its seeds
    at 6000 (their wording at the judged lags), the TwoNN and its paired
    differences, the spread's diagnosis with the provisional bounds, and
    the K3's rows. Stops where a row's ``key`` is not the sets'."""
    sets = json.loads((geom / "sets.json").read_text())
    vd = _load(geom / "yard" / "vd.json")
    yard = {p.stem: _load(p) for p in sorted((geom / "yard" / "rows").glob("*.json"))}
    rows, k3, meta = load_models(geom, runs)
    for label, rs in [*rows.items(), *k3.items()]:
        keys = {r["key"] for r in rs}
        if not rs or keys != {sets["key"]}:
            raise SystemExit(
                f"{label}: key {sorted(keys)} is not the sets' {sets['key']}"
            )
    judged = judged_lags(vd, DELTA_G, g)
    labels = tuple(rows)
    final = tuple(x for x in labels if meta[x]["step"] in (None, FINAL))
    every = {**yard, **rows}
    out: dict[str, Any] = {
        "delta_g": DELTA_G, "delta_d": DELTA_D, "judged": judged, "meta": meta,
        "curves": curves(every, g, {p: (*YARDSTICKS[p], *labels) for p in YARDSTICKS}),
    }  # fmt: skip

    def pairs(specs: Sequence[tuple[str, str]], place: str) -> list[dict[str, Any]]:
        got = []
        for a, b in specs:
            pg = paired_g(every[a], every[b], place, g, judged)
            if pg["lags"]:
                got.append(
                    {
                        "a": a,
                        "b": b,
                        "G": pg,
                        "d": paired_d(every[a], every[b], place, g),
                    }
                )
        return got

    out["vs_yard"] = {
        p: pairs([(m, y) for m in labels for y in YARDSTICKS[p]], p) for p in YARDSTICKS
    }
    out["models"] = {p: pairs(list(combinations(final, 2)), p) for p in YARDSTICKS}
    c_runs = [
        m for m in labels if meta[m]["family"] == "v18c" and meta[m]["step"] == FINAL
    ]
    out["steps"] = {p: pairs([(m, m.replace(f"@{FINAL}", "@6000")) for m in c_runs], p)
                    for p in YARDSTICKS}  # fmt: skip
    early = [
        m.replace(f"@{FINAL}", "@6000")
        for m in c_runs
        if m.split("@")[0].endswith("-s0")
    ]
    out["seeds_6000"] = {p: pairs([(m, m.replace("-s0@", "-s1@")) for m in early], p)
                         for p in YARDSTICKS}  # fmt: skip
    out["init_of"] = {m: _init_of(meta[m]) for m in labels}
    out["twonn"] = twonn_table(
        every, g, {p: (*YARDSTICKS[p], *labels) for p in YARDSTICKS}
    )
    spread = spread_table(every, g, {p: (*YARDSTICKS[p], *labels) for p in YARDSTICKS})
    out["spread"] = {p: {c: d | {"flags": diag_flags(d)} for c, d in codes.items()}
                     for p, codes in spread.items()}  # fmt: skip
    out["k3"] = {}
    for m, rs in k3.items():
        e: dict[str, Any] = {"k3_z": {}, "own_k16": {}}
        for lag in g.lags:
            kz = _mid(rs, g, "own", lag, "in", k=16)
            oz = _mid(rows[m], g, "own", lag, "in", k=16)
            e["k3_z"][lag] = {"G": kz["G"], "s": kz["s"]} if kz else None
            e["own_k16"][lag] = {"G": oz["G"], "s": oz["s"]} if oz else None
        spec = _of(rs, part="spec", code="k3_z", place="own", k=g.k)
        e["k3_z_k64_scoreable"] = spec[0]["scoreable"] if spec else None
        same = all(
            {k: v for k, v in a.items() if k not in _LABELS}
            == {k: v for k, v in b.items() if k not in _LABELS}
            for a, b in zip(_of(rs, code="k3_hmean"), _of(rows[m], place="h_mean"), strict=True)
        )  # fmt: skip
        e["k3_hmean_same"] = same
        out["k3"][m] = e
    _print_apply(out, g)
    return _lean(out)


def _lean(x: Any) -> Any:
    """``x`` for the JSON: floats to 6 significant digits, and no interval
    bounds of a paired $G$ (``diff`` $\\pm 2\\,$``se`` and $\\pm c\\,$``se``)."""
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, Mapping):
        return {k: _lean(v) for k, v in x.items()
                if not ("cls_sim" in x and k in ("lo", "hi", "lo_sim", "hi_sim"))}  # fmt: skip
    if isinstance(x, list | tuple):
        return [_lean(v) for v in x]
    return x


_LABELS = ("code", "run", "step", "t", "snap", "place", "source")


def _print_apply(out: Mapping[str, Any], g: GeomConf) -> None:
    print(f"## 適用（Δ* = {out['delta_g']}・{out['delta_d']}、判定する刻み "
          + "・".join(f"{lag / 10:g} 秒" for lag in out["judged"]) + "）\n")  # fmt: skip
    for place, codes in out["curves"].items():
        print(f"\n### `in` の G（2 SE*）と s: {place}\n\n| 符号 | "
              + " | ".join(f"{lag / 10:g} 秒" for lag in g.lags) + " |")  # fmt: skip
        print("|" + " -- |" * (len(g.lags) + 1))
        for code, by_lag in codes.items():
            cells = [f"{_fmt(v['G'], 3)}±{_fmt(2 * v['se_G'], 2)} (s {_fmt(v['s'], 3)})"
                     if (v := by_lag.get(lag)) else "—" for lag in g.lags]  # fmt: skip
            print(f"| {code} | " + " | ".join(cells) + " |")
    for key in ("vs_yard", "models", "steps", "seeds_6000"):
        for place, got in out[key].items():
            counts: dict[str, int] = {}
            for p in got:
                for lag in out["judged"]:
                    e = p["G"]["lags"].get(lag)
                    if e:
                        counts[e["cls"]] = counts.get(e["cls"], 0) + 1
            print(
                f"- {key} {place}: "
                + "、".join(f"{k} {v}" for k, v in sorted(counts.items()))
            )


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--stage", choices=("checks", "yard", "apply"), required=True)
    p.add_argument("--geom", type=Path, default=Path("runs/v18c-geom"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    g = GeomConf()
    if a.stage == "checks":
        out = checks_stage(_load(a.geom / "checks" / "rows.json"))
    elif a.stage == "yard":
        out = yard_stage(a.geom, g)
    else:
        out = apply_stage(a.geom, a.runs, g)
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    main()
