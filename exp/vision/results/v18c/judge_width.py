r"""The description of the linear horizon (``exp/vision/README.md``, 「線形の
幅（回す前に決めたこと、2026-11-26）」; ``exp.vision.width_eval`` and
``exp.vision.width_guard``). Written before any number; the wording is the
README's 「言葉づかいの規則」 with $\Delta_* = 0.5$ octave (``DELTA``).

``--stage checks`` reads ``<width>/checks/rows.json`` (V-h0): each check's
passes and failures (a failure stops) and the records.

``--stage yard`` (V-h1, no verdict) reads ``<width>/yard/rows/*.json`` and
``vh.json``: of each yardstick and place, the widths (``summary``: the
first windows of $p_q \ge 1/2$ and of the pooled $G < q$, $\Delta_{\mathrm{
typ}}$, $\bar\Delta$, the survival's crossing, the intervals of 128 s and
256 s blocks and their union, the share of draws of no window, the
windows of the whole grid below 0.8 s beside ``mean8``, $N_{\mathrm{lin}}$),
and the material of $\Delta_*$: the paired ratios of every two yardsticks
of a place (``pair_ratio``: the interval of $\log_2$ of $\bar\Delta$ and of
$\Delta_{\mathrm{typ}}$, their widths), $\bar\Delta$ at $k$ = 48, 96 against
64 and ``init_c`` against ``init_c1``. $\Delta_*$ is not changed by it.

``--stage guard`` (V-h3) reads ``<width>/guard``: of each decoder and lag the
ceiling, floor, extrapolation and midpoint (``guard_values``), their paired
differences' $\mathrm{SE}^*$ (``width_guard.diff_se``), the controls and
``uncond``; $\sigma_{\mathrm{dec}}$ of ``pca``'s three decoders
(``sigma_dec``); the lags where the ceiling clears the floor (``resolves``);
the stop where ``pca``'s first decoder clears it at none.

``--stage cheat`` (V-h2) reads ``<width>/guard/cheat``: of each cheat its
$\bar\Delta$ against strength 0 (``pair_ratio``) and its guard's verdict
(``guard_verdict``, $\sigma_{\mathrm{dec}}$ of the guard stage), and the
families' pass (``cheat_pass``, C21).

``--stage apply`` (the exploration, a description: 「探索の記述」) reads the
models' ``posthoc_width/rows/<step>.json`` (``judge_geom.MODELS``) and the
VAEs' ``<width>/vae/rows``, their widths, their paired ratios against
``pca``, ``mean8`` and their own step 0, and their guard where there is
one. Prints markdown; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_width --stage checks|yard|guard|cheat|apply \
        [--width runs/v18c-width] [--geom runs/v18c-tlin] [--runs runs] [--json out.json]
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

from exp.lib.linwidth import Interval, paired_ratio, union
from exp.lib.metrics import ratio_resamples
from exp.vision.geom_eval import GeomConf
from exp.vision.results.v18c.judge_geom import (
    MODELS,
    VAES,
    _init_of,
    _label,
    classify,
    step0_inits,
)
from exp.vision.width_eval import (
    OUT,
    LinSums,
    WidthConf,
    counts_of,
    draws,
    lin_sums,
    paired_blocks,
    point,
)
from exp.vision.width_guard import OUT as GUARD_OUT, GuardConf, diff_se

Row = dict[str, Any]

DELTA = 0.5  # Delta*, octaves of a width (README, registered 2026-11-26)
YARDSTICKS = {
    "own": ("pca", "color", "mean8", "init", "init_f", "init_m", "init_c", "init_c1",
            "init_v16"),
    "h_mean": ("init", "init_f", "init_m", "init_c", "init_c1", "init_v16"),
}  # fmt: skip
CHEAT_BASE = "base"


def _load(path: Path) -> list[Row]:
    return json.loads(path.read_text()) if path.exists() else []


def _fmt(x: float | None, digits: int = 3) -> str:
    return (
        "—"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{x:.{digits}g}"
    )


def _iv(iv: Interval | None) -> str:
    if iv is None:
        return "—"
    lo, hi = iv
    return f"[{'無い' if lo is None else _fmt(lo)}, {_fmt(hi)}]"


# -- the checks -----------------------------------------------------------------------


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
    print("## V-h0\n\n| 確かめ | 通った | 外れた |\n| -- | -- | -- |")
    for name, c in out["checks"].items():
        print(f"| {name} | {c['pass']} | {c['fail']} |")
    if out["failures"]:
        print("\n**外れた行（ここで止める）**\n")
        for r in out["failures"]:
            print("-", json.dumps({k: v for k, v in r.items() if k != "snap"}))
    return out


# -- the widths of a code -------------------------------------------------------------


def summary(ls: LinSums, rows: Sequence[Row], wconf: WidthConf, gconf: GeomConf,
            **labels: Any) -> dict[str, Any]:  # fmt: skip
    """The point widths (``width_eval.point``), their intervals
    (``width_eval.draws``) and the directions' ``n_lin`` (of the main
    whitening's ``dir`` row of ``labels``, if any)."""
    out = {"point": point(ls, wconf), "draws": draws(ls, wconf, gconf)}
    d = [r for r in rows if r.get("part") == "dir"
         and all(r.get(k) == v for k, v in labels.items())]  # fmt: skip
    if d:
        out["n_lin"] = d[0]["n_lin"]
    return out


def pair_ratio(
    a: LinSums, b: LinSums, wconf: WidthConf, gconf: GeomConf
) -> dict[str, Any]:
    r"""The paired ratio $\log_2(W_a / W_b)$ of the main range (the same
    draws, 128 s and 256 s blocks, their union): of $\bar\Delta$ its
    interval and wording (``classify`` with ``DELTA``), of
    $\Delta_{\mathrm{typ}}$ the same where both have a window in 97.5% of
    the draws of both lengths (else the shares of draws with a window), and
    whether a draw is cut at 25.6 s."""
    ma, mb = a.restrict(wconf.main_lo), b.restrict(wconf.main_lo)
    x = wconf.seconds(ma.lags)
    got = []
    for pa, pb in ((ma, mb), (paired_blocks(ma), paired_blocks(mb))):
        counts = counts_of(pa.n.shape[1], gconf)
        ra = ratio_resamples(pa.win["25"], pa.n, counts)
        rb = ratio_resamples(pb.win["25"], pb.n, counts)
        got.append(paired_ratio(x, ra, rb, 0.5))
    mean = union([g.mean for g in got])
    typs = [g.typ for g in got]
    typ = (
        None
        if any(t is None for t in typs)
        else union([t for t in typs if t is not None])
    )
    return {
        "mean": mean,
        "mean_cls": _classify(mean),
        "mean_width": _width(mean),
        "typ": typ,
        "typ_cls": None if typ is None else _classify(typ),
        "windowed": [g.windowed for g in got],
        "cut": any(g.cut for g in got),
    }


def _width(iv: Interval) -> float | None:
    lo, hi = iv
    return None if lo is None or hi is None else hi - lo


def _classify(iv: Interval) -> str:
    lo, hi = iv
    return classify(
        -math.inf if lo is None else lo, math.inf if hi is None else hi, DELTA
    )


def _sums(
    rows: Sequence[Row], place: str, g: GeomConf, **variant: Any
) -> LinSums | None:
    lab = {"place": place, "k": g.k, "ref": "all"} | variant
    try:
        return lin_sums(rows, **lab)
    except ValueError:
        return None


def yard_stage(width: Path, wconf: WidthConf, g: GeomConf) -> dict[str, Any]:
    """V-h1's material (module docstring; no verdict, $\\Delta_*$ unchanged)."""
    by_code = {
        p.stem: _load(p) for p in sorted((width / "yard" / "rows").glob("*.json"))
    }
    out: dict[str, Any] = {"vh": _load(width / "yard" / "vh.json"), "codes": {},
                           "pairs": {}, "k_moves": {}, "seeds": {}}  # fmt: skip
    for place, names in YARDSTICKS.items():
        ls = {n: _sums(by_code.get(n, []), place, g) for n in names}
        ls = {n: v for n, v in ls.items() if v is not None}
        for n, v in ls.items():
            out["codes"][f"{n}/{place}"] = summary(v, by_code[n], wconf, g, place=place,
                                                   k=g.k, ref="all")  # fmt: skip
            moves = {}
            for k in g.k_sens:
                alt = _sums(by_code[n], place, g, k=k)
                if alt is not None:
                    moves[k] = math.log2(point(alt, wconf)["25"]["mean"]
                                         / point(v, wconf)["25"]["mean"])  # fmt: skip
            out["k_moves"][f"{n}/{place}"] = moves
        for a, b in combinations(ls, 2):
            out["pairs"][f"{a}-{b}/{place}"] = pair_ratio(ls[a], ls[b], wconf, g)
        if "init_c" in ls and "init_c1" in ls:
            out["seeds"][place] = out["pairs"][f"init_c-init_c1/{place}"]
    widths = [
        p["mean_width"] for p in out["pairs"].values() if p["mean_width"] is not None
    ]
    out["pair_width_median"] = statistics.median(widths) if widths else None
    _print_codes(out["codes"])
    print(f"\n物差しの対の $\\bar\\Delta$ の比の区間の幅（中央値、オクターブ）: "
          f"{_fmt(out['pair_width_median'])}")  # fmt: skip
    return out


def _print_codes(codes: Mapping[str, Any]) -> None:
    print(
        "| 符号 | Δ_typ | 区間 | Δ̄ | 区間 | 窓が無い抽出 | プールの窓（1/4） | 0.8 秒未満の窓 | N_lin |"
    )
    print("| -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    for name, s in codes.items():
        p, d = s["point"]["25"], s["draws"]["25"]
        pool = p["G"]["first"]
        whole = p["whole_p"]["first"]
        print(f"| {name} | {_fmt(p['typ'])} | {_iv(d['typ'])} | {_fmt(p['mean'])} | "
              f"{_iv(d['mean'])} | {_fmt(d['none'])} | "
              f"{'無い' if pool is None else _iv((pool['lo'], pool['hi']))} | "
              f"{'無い' if whole is None else _iv((whole['lo'], whole['hi']))} | "
              f"{s.get('n_lin', '—')} |")  # fmt: skip


# -- the guard -------------------------------------------------------------------------


def guard_values(row: Row, gconf: GeomConf) -> dict[str, Any]:
    r"""Of a guard row: the ceiling $c$ (``future``'s share on the
    extrapolation's target), the floor $f$ (``now``'s), the extrapolation
    $e$, the midpoint's ceiling (``now``), floor (the mean of ``past`` and
    ``future``) and $m$, the scene's (``shuffle``) and the positive control's
    (``pos``) shares, and the $\mathrm{SE}^*$ of the paired differences $c -
    f$, $e - f$, $m - f_{\mathrm{mid}}$."""
    ext, mid = row["read"]["ext"], row["read"]["mid"]
    blocks = torch.tensor(row["blocks"])

    def h(part: Mapping[str, Any], name: str) -> torch.Tensor:
        return torch.tensor(part[name]["hits"], dtype=torch.float64)

    f_mid = (h(mid, "past") + h(mid, "future")) / 2
    out = {
        "c": ext["future"]["share"], "f": ext["now"]["share"], "e": ext["ext"]["share"],
        "c_mid": mid["now"]["share"], "f_mid": float(f_mid.mean()), "m": mid["mid"]["share"],
        "shuffle": ext.get("shuffle", {}).get("share"), "pos": ext.get("pos", {}).get("share"),
        "pick_max": {k: v["pick_max"] for k, v in ext.items()},
        "se_cf": diff_se(h(ext, "future"), h(ext, "now"), blocks, gconf),
        "se_ef": diff_se(h(ext, "ext"), h(ext, "now"), blocks, gconf),
        "se_mf": diff_se(h(mid, "mid"), f_mid, blocks, gconf),
        "n_anchors": row["n_anchors"],
    }  # fmt: skip
    return out


def sigma_dec(by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]]) -> float:
    r"""$\sigma_{\mathrm{dec}}$ (C12): of each lag the sample variance (over
    the decoder seeds, $n - 1$) of $c - f$, pooled as their mean over the
    lags (the degrees of freedom add up), its square root. ``by_seed`` maps
    a seed to each lag's ``guard_values``."""
    seeds = sorted(by_seed)
    lags = sorted(set.intersection(*(set(by_seed[s]) for s in seeds)))
    var = [statistics.variance([by_seed[s][lag]["c"] - by_seed[s][lag]["f"] for s in seeds])
           for lag in lags]  # fmt: skip
    return math.sqrt(sum(var) / len(var))


def _tot(se: float, sigma: float) -> float:
    return math.sqrt(se * se + sigma * sigma)


def resolves(v: Mapping[str, Any], sigma: float) -> bool:
    r"""The ceiling clears the floor: $c - f - 2\,\mathrm{SE}_{\mathrm{tot}} >
    0$, $\mathrm{SE}_{\mathrm{tot}} = \sqrt{\mathrm{SE}^{*2} +
    \sigma_{\mathrm{dec}}^2}$."""
    return v["c"] - v["f"] - 2 * _tot(v["se_cf"], sigma) > 0


def guard_verdict(
    values: Mapping[int, Mapping[str, Any]], won: Sequence[int], sigma: float
) -> dict[str, Any]:
    r"""The guard's verdict (C13) of the lags ``won`` (the guard's lags where
    the point $p_{1/4} \ge 1/2$, $L$) and each lag's ``guard_values``:
    ``no_lag`` ($L$ empty: no width is called wide), ``unverifiable`` (the
    ceiling does not clear the floor at a lag of $L$), ``broken`` (it does,
    and $e - f + 2\,\mathrm{SE}_{\mathrm{tot}} < 0$ at a lag), ``passed``
    ($e - f - 2\,\mathrm{SE}_{\mathrm{tot}} > 0$ at every lag of $L$),
    ``pending`` otherwise (an interval holding 0)."""
    lags = [lag for lag in won if lag in values]
    if not lags:
        return {"verdict": "no_lag", "L": []}
    each = {}
    for lag in lags:
        v = values[lag]
        se = _tot(v["se_ef"], sigma)
        each[lag] = {"resolves": resolves(v, sigma), "beats": v["e"] - v["f"] - 2 * se > 0,
                     "loses": v["e"] - v["f"] + 2 * se < 0}  # fmt: skip
    if not all(e["resolves"] for e in each.values()):
        verdict = "unverifiable"
    elif any(e["loses"] for e in each.values()):
        verdict = "broken"
    elif all(e["beats"] for e in each.values()):
        verdict = "passed"
    else:
        verdict = "pending"
    return {"verdict": verdict, "L": lags, "each": each}


def _guard_rows(path: Path, g: GeomConf) -> dict[int, dict[str, Any]]:
    return {
        r["lag"]: guard_values(r, g) for r in _load(path) if r.get("part") == "guard"
    }


def guard_stage(width: Path, g: GeomConf) -> dict[str, Any]:
    """V-h3 (module docstring)."""
    root = width / "guard"
    out: dict[str, Any] = {"codes": {}, "controls": {}}
    for r in _load(root / "controls.json"):
        out["controls"][r["lag"]] = {t: {k: v["share"] for k, v in part.items()}
                                     for t, part in r["read"].items()}  # fmt: skip
    for path in sorted(root.glob("*/rows-d*.json")):
        if path.parent.name == "cheat":
            continue
        seed = int(path.stem.removeprefix("rows-d"))
        out["codes"].setdefault(path.parent.name, {})[seed] = _guard_rows(path, g)
    pca = out["codes"].get("pca", {})
    out["sigma_dec"] = sigma_dec(pca) if len(pca) >= 2 else None
    sig = out["sigma_dec"] or 0.0
    out["resolves"] = {
        f"{code}/d{seed}": [lag for lag, v in lags.items() if resolves(v, sig)]
        for code, seeds in out["codes"].items()
        for seed, lags in seeds.items()
    }
    out["stop"] = not out["resolves"].get("pca/d0")
    print(f"σ_dec = {_fmt(out['sigma_dec'])}")
    print(
        "\n| 符号 | 刻み | 天井 c | 床 f | 外挿 e | 中点 m | SE*(c−f) | SE*(e−f) | 天井が床から離れる |"
    )
    print("| -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    for code, seeds in out["codes"].items():
        for seed, lags in seeds.items():
            for lag, v in lags.items():
                print(f"| {code}/d{seed} | {lag / 10} | {_fmt(v['c'])} | {_fmt(v['f'])} | "
                      f"{_fmt(v['e'])} | {_fmt(v['m'])} | {_fmt(v['se_cf'])} | "
                      f"{_fmt(v['se_ef'])} | {resolves(v, sig)} |")  # fmt: skip
    if out["stop"]:
        print("\n**pca の d0 で、どの刻みでも天井が床から離れない（ここで止める）**")
    return out


# -- the cheats -------------------------------------------------------------------------


def won_lags(ls: LinSums, lags: Sequence[int]) -> list[int]:
    r"""The lags of ``lags`` where the point $p_{1/4} \ge 1/2$."""
    p = ls.p("25")
    return [lag for lag in lags if lag in ls.lags.tolist()
            and float(p[ls.lags.tolist().index(lag)]) >= 0.5]  # fmt: skip


def _clears(iv: Interval) -> bool:
    lo = iv[0]
    return lo is not None and lo > DELTA


def cheat_pass(cheats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    r"""V-h2 (C21) of each family (``mean``, ``drop``): ``claimable`` the
    strengths whose $\bar\Delta$ against strength 0 clears $+\Delta_*$
    wholly; the family passes where it has one and every one's guard is
    ``broken``; it fails where one's is ``passed`` (the guard is not
    enough), ``unverifiable`` or ``pending`` (no catch shown); with none,
    ``none_claimable`` (「この族では確かめられない」). Records where
    $\bar\Delta$ does not rise with the strength."""
    out = {}
    for family in ("mean", "drop"):
        names = sorted((n for n in cheats if n.startswith(f"{family}-")),
                       key=lambda n: int(n.split("-")[1]))  # fmt: skip
        claim = [n for n in names if _clears(cheats[n]["ratio"]["mean"])]
        verdicts = {n: cheats[n]["guard"]["verdict"] for n in claim}
        if not claim:
            result = "none_claimable"
        elif all(v == "broken" for v in verdicts.values()):
            result = "pass"
        else:
            result = "fail"
        means = [cheats[n]["mean"] for n in names]
        out[family] = {"claimable": claim, "verdicts": verdicts, "result": result,
                       "monotone": all(b >= a for a, b in zip(means, means[1:], strict=False))}  # fmt: skip
    return out


def cheat_stage(
    width: Path, wconf: WidthConf, g: GeomConf, sigma: float
) -> dict[str, Any]:
    """V-h2 (module docstring) with $\\sigma_{\\mathrm{dec}}$ ``sigma``."""
    root = width / "guard" / "cheat"
    guard = GuardConf()
    ls = {p.parent.name: lin_sums(_load(p)) for p in sorted(root.glob("*/width.json"))}
    base = ls[CHEAT_BASE]
    cheats: dict[str, Any] = {}
    for name, v in ls.items():
        values = _guard_rows(root / name / "rows-d0.json", g)
        cheats[name] = {
            "mean": point(v, wconf)["25"]["mean"],
            "ratio": pair_ratio(v, base, wconf, g),
            "guard": guard_verdict(values, won_lags(v, guard.lags), sigma),
            "beyond": bool(won_lags(v, range(guard.lags[-1] + 1, wconf.max_lag + 1))),
        }  # fmt: skip
    out = {"cheats": cheats, "families": cheat_pass(cheats), "sigma_dec": sigma}
    print("| ずる | Δ̄ | 強さ 0 との比の区間 | ガード |\n| -- | -- | -- | -- |")
    for name, c in cheats.items():
        print(f"| {name} | {_fmt(c['mean'])} | {_iv(c['ratio']['mean'])} | "
              f"{c['guard']['verdict']} |")  # fmt: skip
    for family, f in out["families"].items():
        print(f"\n{family}: {f['result']}（主張できる強さ {f['claimable']}）")
    return out


# -- the exploration ----------------------------------------------------------------------


def apply_stage(width: Path, geom: Path, runs: Path, wconf: WidthConf, g: GeomConf,
                sigma: float | None) -> dict[str, Any]:  # fmt: skip
    """The exploration (「探索の記述」; module docstring): each model's
    widths of each place, its paired ratios against ``pca``, ``mean8`` and
    its own step 0 (of the same place), and its guard's verdict where its
    guard rows are (``sigma`` the guard stage's; None: not read)."""
    yard = {p.stem: _load(p) for p in sorted((width / "yard" / "rows").glob("*.json"))}
    step0 = step0_inits(_load(geom / "yard" / "vd.json"))
    rows: dict[str, list[Row]] = {name: _load(width / "vae" / "rows" / f"{name}.json")
                                  for name in VAES}  # fmt: skip
    meta: dict[str, dict[str, Any]] = {n: {"family": "vae", "step": None, "run": n}
                                       for n in VAES}  # fmt: skip
    for run, family, steps in MODELS:
        for step in steps:
            label = _label(run, step)
            rows[label] = _load(runs / run / OUT / "rows" / f"{step:06d}.json")
            meta[label] = {"family": family, "step": step, "run": run.rsplit("/", 1)[-1],
                           "path": str(runs / run)}  # fmt: skip
    out: dict[str, Any] = {"note": "探索の記述", "models": {}}
    guard = GuardConf()
    for label, rs in rows.items():
        if not rs:
            continue
        init = None
        try:
            init = _init_of(meta[label], step0)
        except SystemExit:
            init = None
        m: dict[str, Any] = {}
        for place in ("own", "h_mean"):
            ls = _sums(rs, place, g)
            if ls is None:
                continue
            e: dict[str, Any] = {"summary": summary(ls, rs, wconf, g, place=place, k=g.k,
                                                    ref="all"), "ratios": {}}  # fmt: skip
            for other in ("pca", "mean8", init):
                if other is None or other not in yard:
                    continue
                lo = _sums(yard[other], place, g)
                if lo is not None:
                    e["ratios"][other] = pair_ratio(ls, lo, wconf, g)
            m[place] = e
        if (
            "path" in meta[label]
            and sigma is not None
            and meta[label]["step"] is not None
        ):
            gpath = (
                Path(meta[label]["path"])
                / GUARD_OUT
                / "rows"
                / f"{meta[label]['step']:06d}.json"
            )
            grows = [r for r in _load(gpath) if r.get("cond") == "pca64"]
            if grows:
                values = {r["lag"]: guard_values(r, g) for r in grows}
                own = _sums(rs, "own", g)
                assert own is not None
                m["guard"] = guard_verdict(values, won_lags(own, guard.lags), sigma)
        out["models"][label] = m
    print("（探索の記述。V19 の確証ではない）\n")
    for label, m in out["models"].items():
        for place in ("own", "h_mean"):
            if place not in m:
                continue
            for other, r in m[place]["ratios"].items():
                print(
                    f"- {label}/{place} 対 {other}: Δ̄ の比 {_iv(r['mean'])} → {r['mean_cls']}"
                )
    return out


def _lean(x: Any) -> Any:
    """``x`` without the per-anchor lists (for the JSON)."""
    if isinstance(x, Mapping):
        return {k: _lean(v) for k, v in x.items() if k not in ("hits", "local_hits")}
    if isinstance(x, list | tuple):
        return [_lean(v) for v in x]
    return x


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--stage", choices=("checks", "yard", "guard", "cheat", "apply"),
                   required=True)  # fmt: skip
    p.add_argument("--width", type=Path, default=Path("runs/v18c-width"))
    p.add_argument("--geom", type=Path, default=Path("runs/v18c-tlin"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    g, w = GeomConf(), WidthConf()
    if a.stage == "checks":
        out = checks_stage(_load(a.width / "checks" / "rows.json"))
    elif a.stage == "yard":
        out = yard_stage(a.width, w, g)
    elif a.stage == "guard":
        out = guard_stage(a.width, g)
    else:
        sig = guard_stage(a.width, g)["sigma_dec"]
        if a.stage == "cheat":
            if sig is None:
                raise SystemExit("no sigma_dec: run the guard's yard first")
            out = cheat_stage(a.width, w, g, sig)
        else:
            out = apply_stage(a.width, a.geom, a.runs, w, g, sig)
    if a.json is not None:
        a.json.write_text(json.dumps(_lean(out), indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    main()
