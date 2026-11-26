r"""The verdicts of the main evaluation dyn/copy (``exp/vision/plans/dyn.md``;
``results/dyn/README.md``, 「回す前に決めたこと」; ``exp.vision.dyn_eval``).
Written before any number.

**The interval** (``psi_interval``) of a pair $(a, b)$ of one denominator:
$\hat\psi = \overline{\log R_a} - \overline{\log R_b}$ (``dynpred.mean_log``
of the main range), $\hat\psi \pm z\,\mathrm{SE}_{\mathrm{tot}}$,
$\mathrm{SE}_{\mathrm{tot}} = \sqrt{\mathrm{SE}^{*2} + \sigma_{ab}^2}$,
$\mathrm{SE}^*$ the larger standard deviation of the draws of 128 s and of
256 s blocks (``metrics.block_counts`` of ``boot``, ``boot_seed``; the same
draws for every code, so a pair is paired); none where more than 2.5% of a
blocking's draws are invalid (a zero denominator) or the point is not
finite (an exact prediction). $z = 2$ for the main comparison, $z_{1 -
0.025/m}$ (``bonferroni_z``) for a side family of $m$.

**The words** (``words``): ``a_better`` (「a の予測が b より良いと区別でき
た」) where both $\psi_{\mathrm{copy}}$'s and $\psi_{\mathrm{clim}}$'s
intervals lie below 0, ``b_better`` above, ``split`` (「分母で向きが割れた」)
where both clear 0 on opposite sides, ``none`` (「区別できない」) else. 「良い」
is never said here: it needs the collapse measures and the content guard v3
besides (``GOOD_NOTE``).

**$\sigma_{ab}$** (``sigma_ab``): $\max(\sigma_{\mathrm{rot}}, s_{ab})$, the
seeds' sample standard deviations of $\psi_s$ of ``rot`` against ``pca``
and of the pair itself; 0 of the linear family.

``--stage checks`` reads ``<dyn>/checks/rows.json`` (V-p1); ``--stage yard``
(V-p2) ``<dyn>/yard/rows/*.json``: of each code and place the main numbers
(``summary``), V-p2a (``rot`` against ``pca``'s linear, every lag and set, to
``REL_ROT``, the same $\alpha$), V-p2b (the ridge's bounds and the MLP's
validation), V-p2d (the MLP's non-inferiority on the record set; the main
family), $\sigma_{\mathrm{rot}}$ and the yardsticks' pairs; ``--stage
apply`` the exploration's runs (a description)::

    uv run python -m exp.vision.results.dyn.judge_dyn --stage checks|yard|apply \
        [--dyn runs/v18c-dyn] [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist, stdev
from typing import Any

import torch
from torch import Tensor

from exp.lib.dynpred import GRID, MAIN, mean_log, mean_log_draws, pair_blocks
from exp.lib.metrics import block_counts

__all__ = [
    "GOOD_NOTE",
    "Interval",
    "bonferroni_z",
    "main",
    "psi_interval",
    "sigma_ab",
    "summary",
    "words",
]

type Row = dict[str, Any]

BOOT, BOOT_SEED = 2000, 101
REC_SEED = 102
INVALID = 0.025  # the share of invalid draws beyond which no interval
REL_ROT = 1e-6  # V-p2a
REL_BOUND = 1e-9  # V-p2b
WORDS = {
    "a_better": "a の予測が b より良いと区別できた",
    "b_better": "b の予測が a より良いと区別できた",
    "split": "分母で向きが割れた",
    "none": "区別できない",
}
GOOD_NOTE = "内容の比べは未確立（内容のガード v3 の有効性が確かめられるまで「良い」とは書かない）"
_F64 = torch.float64


@dataclass(frozen=True)
class Interval:
    """$\\hat\\psi$, its interval (None where none), $\\mathrm{SE}^*$ of each
    blocking, $\\sigma_{ab}$, $z$ and the invalid draws' larger share."""

    psi: float | None
    lo: float | None
    hi: float | None
    se128: float | None
    se256: float | None
    sigma: float
    z: float
    invalid: float


def _draws(lags: Sequence[int], num: Tensor, den: Tensor, boot: int, seed: int,
           lo: int, hi: int) -> Tensor:  # fmt: skip
    counts = block_counts(num.shape[1], boot, torch.Generator().manual_seed(seed))
    return mean_log_draws(lags, num, den, counts, lo, hi)


def psi_interval(
    lags: Sequence[int],
    num_a: Tensor,
    den_a: Tensor,
    num_b: Tensor,
    den_b: Tensor,
    boot: int = BOOT,
    boot_seed: int = BOOT_SEED,
    *,
    sigma: float = 0.0,
    z: float = 2.0,
    main: tuple[int, int] = MAIN,
) -> Interval:
    """The interval of $\\psi$ (module docstring) of the blocks' sums ``(L,
    B)`` of the two codes at ``lags``."""
    lo, hi = main
    ra, rb = num_a.sum(1) / den_a.sum(1), num_b.sum(1) / den_b.sum(1)
    psi = float(mean_log(lags, ra, lo, hi) - mean_log(lags, rb, lo, hi))
    ses, bad = [], 0.0
    for pa in (False, True):
        f = pair_blocks if pa else (lambda x: x)
        d = _draws(lags, f(num_a), f(den_a), boot, boot_seed, lo, hi) - _draws(
            lags, f(num_b), f(den_b), boot, boot_seed, lo, hi
        )
        ok = torch.isfinite(d)
        bad = max(bad, 1 - float(ok.double().mean()))
        ses.append(float(d[ok].std()) if int(ok.sum()) > 1 else None)
    if not math.isfinite(psi) or bad > INVALID or None in ses:
        return Interval(psi if math.isfinite(psi) else None, None, None, ses[0], ses[1],
                        sigma, z, bad)  # fmt: skip
    se = math.sqrt(max(s for s in ses if s is not None) ** 2 + sigma**2)
    return Interval(psi, psi - z * se, psi + z * se, ses[0], ses[1], sigma, z, bad)


def words(copy: Interval, clim: Interval) -> str:
    """The words of a pair's two intervals (module docstring)."""

    def side(iv: Interval) -> int:
        if iv.lo is None or iv.hi is None:
            return 0
        return -1 if iv.hi < 0 else (1 if iv.lo > 0 else 0)

    a, b = side(copy), side(clim)
    if a == b == -1:
        return "a_better"
    if a == b == 1:
        return "b_better"
    if a * b == -1:
        return "split"
    return "none"


def sigma_ab(psi_rot: Sequence[float], psi_ab: Sequence[float]) -> float:
    """$\\max(\\sigma_{\\mathrm{rot}}, s_{ab})$ of the seeds' $\\psi_s$."""
    return max(stdev(psi_rot), stdev(psi_ab))


def bonferroni_z(m: int) -> float:
    """$z_{1 - 0.025/m} = \\Phi^{-1}(1 - 0.025/m)$ of a side family of
    ``m`` comparisons."""
    return NormalDist().inv_cdf(1 - 0.025 / m)


# -- the rows -----------------------------------------------------------------------


def _load(path: Path) -> list[Row]:
    return json.loads(path.read_text())


def _f64(x: Any) -> Tensor:
    return torch.tensor(x, dtype=_F64)


def curve(rows: Sequence[Row], family: str, set_: str = "zurich",
          seed: int | None = None) -> dict[str, Tensor] | None:  # fmt: skip
    """The blocks' sums ``(L, B)`` of a family on a set by lag (``lags``):
    of ``mlp`` the seeds' mean numerator (``seed`` None) or one seed's.
    None where no row."""
    got = [r for r in rows if r.get("part") == "dyn" and r["set"] == set_
           and r["family"] == family and (family != "mlp" or seed is None or r["seed"] == seed)]  # fmt: skip
    if not got:
        return None
    lags = sorted({r["lag"] for r in got})
    out: dict[str, list[Any]] = {"num_b": [], "copy_b": [], "clim_b": []}
    for n in lags:
        at = [r for r in got if r["lag"] == n]
        for k in out:
            out[k].append(_f64([r[k] for r in at]).mean(0).tolist())
    return {"lags": torch.tensor(lags)} | {k: _f64(v) for k, v in out.items()}


def _ratios(c: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {d: c["num_b"].sum(1) / c[f"{d}_b"].sum(1) for d in ("copy", "clim")}


def _ml(c: Mapping[str, Tensor], d: str, rng: tuple[int, int] = MAIN) -> float | None:
    lags = c["lags"].tolist()
    if rng[0] not in lags or rng[1] not in lags:
        return None
    v = float(mean_log(lags, _ratios(c)[d], *rng))
    return v if math.isfinite(v) else None


def summary(rows: Sequence[Row]) -> dict[str, Any]:
    """Of one code's place: the main numbers $\\overline{\\log R}$ of each
    family and denominator on the main range and below it, the curves of
    $R$ on Zurich, $V/D$, the train, val, rec and Zurich values of the
    linear and MLP, the lags where the linear's $R_{\\mathrm{copy}} > 1$ on
    Zurich, the chosen $\\alpha$ and steps, the spectrum."""
    out: dict[str, Any] = {"main": {}, "below": {}, "curves": {}, "sets": {}}
    for fam in ("lin", "mlp", "lin1", "vel", "mean", "scale", "copy"):
        c = curve(rows, fam)
        if c is None:
            continue
        grid = _grid(c)
        out["main"][fam] = {d: _ml(grid, d) for d in ("copy", "clim")}
        out["below"][fam] = {d: _ml(grid, d, (1, 8)) for d in ("copy", "clim")}
        out["curves"][fam] = {"lags": c["lags"].tolist()} | {
            k: v.tolist() for k, v in _ratios(c).items()
        }
    lin = curve(rows, "lin")
    if lin is not None:
        r = _ratios(lin)
        out["over_one"] = [
            int(n) for n, v in zip(lin["lags"], r["copy"], strict=True) if v > 1
        ]
        out["vd"] = (r["copy"] / r["clim"]).tolist()
    for set_ in ("train", "val", "rec", "zurich"):
        for fam in ("lin", "mlp"):
            got = [x for x in rows if x.get("part") == "dyn" and x["set"] == set_
                   and x["family"] == fam and x["lag"] in GRID]  # fmt: skip
            if not got:
                continue
            lags = sorted({x["lag"] for x in got})
            by: dict[str, Any] = {"lags": lags}
            for d in ("copy", "clim"):
                vals = []
                for n in lags:
                    at = [x for x in got if x["lag"] == n]
                    num = sum(_tot(x, "num") for x in at) / len(at)
                    vals.append(num / _tot(at[0], d))
                by[d] = vals
            out["sets"][f"{set_}/{fam}"] = by
    out["alpha"] = {r["lag"]: r["scale"] for r in rows if r.get("part") == "dyn"
                    and r["set"] == "zurich" and r["family"] == "lin" and r["lag"] in GRID}  # fmt: skip
    out["steps"] = {r["lag"]: r["step"] for r in rows if r.get("part") == "mlp_val"}
    spec = [r for r in rows if r.get("part") == "spec"]
    if spec:
        out["spec"] = {k: spec[0].get(k) for k in ("pr", "raised", "kept_k", "share")}
    return out


def _tot(r: Row, k: str) -> float:
    return float(r[k]) if k in r else float(sum(r[f"{k}_b"]))


def _grid(c: Mapping[str, Tensor]) -> dict[str, Tensor]:
    keep = [i for i, n in enumerate(c["lags"].tolist()) if n in GRID]
    return {k: v[keep] for k, v in c.items()}


def _place(rows: Sequence[Row], place: str) -> list[Row]:
    return [r for r in rows if r.get("place") == place]


def pair(a: Sequence[Row], b: Sequence[Row], fam: str, sigma: float = 0.0,
         z: float = 2.0, set_: str = "zurich") -> dict[str, Any] | None:  # fmt: skip
    """The intervals and words of ``a`` against ``b`` (one place each) of the
    family ``fam`` on ``set_``."""
    ca, cb = curve(a, fam, set_), curve(b, fam, set_)
    if ca is None or cb is None:
        return None
    ga, gb = _grid(ca), _grid(cb)
    lags = ga["lags"].tolist()
    ivs = {d: psi_interval(lags, ga["num_b"], ga[f"{d}_b"], gb["num_b"], gb[f"{d}_b"],
                           sigma=sigma, z=z) for d in ("copy", "clim")}  # fmt: skip
    w = words(ivs["copy"], ivs["clim"])
    return {"words": w, "text": WORDS[w], "note": GOOD_NOTE} | {
        d: asdict(iv) for d, iv in ivs.items()
    }


def _seed_psis(a: Sequence[Row], b: Sequence[Row], d: str) -> list[float]:
    seeds = sorted(
        {r["seed"] for r in a if r.get("family") == "mlp" and r.get("set") == "zurich"}
    )
    out = []
    for s in seeds:
        ca, cb = curve(a, "mlp", "zurich", s), curve(b, "mlp", "zurich", s)
        assert ca is not None and cb is not None
        out.append(
            float(_ml(_grid(ca), d) or math.nan) - float(_ml(_grid(cb), d) or math.nan)
        )
    return out


def sigma_rot(rot: Sequence[Row], pca: Sequence[Row]) -> dict[str, float]:
    """$\\sigma_{\\mathrm{rot}}$ of each denominator: the seeds' sample
    standard deviation of the MLP's $\\psi_s$ of ``rot`` against ``pca``."""
    return {d: stdev(_seed_psis(rot, pca, d)) for d in ("copy", "clim")}


def vp2a(rot: Sequence[Row], pca: Sequence[Row]) -> dict[str, Any]:
    """V-p2a: ``rot``'s linear rows against ``pca``'s, every lag and set."""
    key = lambda r: (r["set"], r["lag"])  # noqa: E731
    pa = {key(r): r for r in pca if r.get("part") == "dyn" and r["family"] == "lin"}
    worst, alpha_same, n = 0.0, True, 0
    for r in rot:
        if r.get("part") != "dyn" or r["family"] != "lin":
            continue
        o = pa[key(r)]
        n += 1
        alpha_same &= r["scale"] == o["scale"]
        for d in ("copy", "clim"):
            x, y = _tot(r, "num") / _tot(r, d), _tot(o, "num") / _tot(o, d)
            worst = max(worst, abs(x - y) / abs(y))
    return {"pass": worst <= REL_ROT and alpha_same and n > 0, "worst_rel": worst,
            "alpha_same": alpha_same, "rows": n}  # fmt: skip


def vp2b(rows: Sequence[Row]) -> dict[str, Any]:
    """V-p2b of one place: the ridge's bounds on the training set and the
    MLP's validation numerator not above the linear's."""
    ok, worst = True, []
    for r in rows:
        if r.get("part") == "dyn" and r["set"] == "train" and r["family"] == "lin":
            bound_c = r["copy"] + r.get("k", 64) * r["alpha_min"]  # + alpha |[I, 0]|^2
            ok &= r["train_min"] <= bound_c * (1 + REL_BOUND)
            ok &= r["train_min"] <= r["clim"] * (1 + REL_BOUND)
    val_lin = {r["lag"]: r["num"] for r in rows if r.get("part") == "dyn"
               and r["set"] == "val" and r["family"] == "lin"}  # fmt: skip
    for n in GRID:
        at = [r["num"] for r in rows if r.get("part") == "dyn" and r["set"] == "val"
              and r["family"] == "mlp" and r["lag"] == n]  # fmt: skip
        if at:
            m = sum(at) / len(at)
            ok &= m <= val_lin[n] * (1 + REL_BOUND)
            worst.append(m / val_lin[n])
    return {"pass": bool(ok), "mlp_over_lin_val": worst}


def vp2d(rows: Sequence[Row]) -> dict[str, Any] | None:
    """V-p2d of one place: the MLP's $\\overline{\\log R_{\\mathrm{copy}}}$
    on the record set at most the linear's plus twice the SE of the paired
    difference (``REC_SEED``'s draws of its 128 s blocks)."""
    m, lin = curve(rows, "mlp", "rec"), curve(rows, "lin", "rec")
    if m is None or lin is None:
        return None
    gm, gl = _grid(m), _grid(lin)
    lags = gm["lags"].tolist()
    counts = block_counts(
        gm["num_b"].shape[1], BOOT, torch.Generator().manual_seed(REC_SEED)
    )
    d = mean_log_draws(lags, gm["num_b"], gm["copy_b"], counts) - mean_log_draws(
        lags, gl["num_b"], gl["copy_b"], counts
    )
    se = float(d[torch.isfinite(d)].std())
    a, b = _ml(gm, "copy"), _ml(gl, "copy")
    ok = a is not None and b is not None and a <= b + 2 * se
    return {"pass": bool(ok), "mlp": a, "lin": b, "se": se}


# -- the stages ---------------------------------------------------------------------


def checks_stage(rows: Sequence[Row]) -> dict[str, Any]:
    """V-p1: each check's pass (a failure stops) and the records."""
    checks = [r for r in rows if r.get("part") == "check"]
    failed = [
        f"{r['name']} {r.get('code', '')}".strip() for r in checks if not r["pass"]
    ]
    print(f"V-p1: {len(checks) - len(failed)} / {len(checks)} passed")
    for f in failed:
        print(f"  FAILED {f}")
    return {"pass": not failed, "failed": failed, "rows": list(rows)}


def yard_stage(dyn: Path) -> dict[str, Any]:
    """V-p2 (module docstring)."""
    yard = {p.stem: _load(p) for p in sorted((dyn / "yard" / "rows").glob("*.json"))}
    out: dict[str, Any] = {"codes": {}, "checks": {}}
    for code, rows in yard.items():
        for place in sorted({r["place"] for r in rows if "place" in r}):
            pr = _place(rows, place)
            out["codes"][f"{code}/{place}"] = {"summary": summary(pr), "vp2b": vp2b(pr),
                                               "vp2d": vp2d(pr)}  # fmt: skip
    if "rot" in yard and "pca" in yard:
        rot, pca = _place(yard["rot"], "own"), _place(yard["pca"], "own")
        out["checks"]["V-p2a"] = vp2a(rot, pca)
        out["sigma_rot"] = sigma_rot(rot, pca)
    out["checks"]["V-p2b"] = all(v["vp2b"]["pass"] for v in out["codes"].values())
    d = [
        v["vp2d"]
        for k, v in out["codes"].items()
        if not k.startswith("clock/") and v["vp2d"]
    ]
    out["main_family"] = "mlp" if d and all(x["pass"] for x in d) else "lin"
    sig = out.get("sigma_rot", {"copy": 0.0, "clim": 0.0})
    out["pairs"] = {}
    if "pca" in yard:
        pca = _place(yard["pca"], "own")
        for code, rows in yard.items():
            if code == "pca":
                continue
            own = _place(rows, "own")
            out["pairs"][f"{code}/pca"] = {
                "lin": pair(own, pca, "lin"),
                "mlp": pair(own, pca, "mlp", sigma=max(sig.values())),
            }
    for k, v in out["codes"].items():
        mn = v["summary"]["main"]
        print(f"| {k} | lin {mn.get('lin')} | mlp {mn.get('mlp')} |")
    print(json.dumps(out["checks"], default=str))
    print(f"main family: {out['main_family']}")
    return out


def apply_stage(dyn: Path, runs: Path) -> dict[str, Any]:
    """The exploration (a description): each model's summary of each place
    and its pairs against ``pca`` and ``mean8`` (own place)."""
    from exp.vision.results.v18c.judge_geom import MODELS, VAES, _label

    yard = {p.stem: _load(p) for p in sorted((dyn / "yard" / "rows").glob("*.json"))}
    sig = {"copy": 0.0, "clim": 0.0}
    if "rot" in yard and "pca" in yard:
        sig = sigma_rot(_place(yard["rot"], "own"), _place(yard["pca"], "own"))
    rows: dict[str, list[Row]] = {}
    for name in VAES:
        p = dyn / "vae" / "rows" / f"{name}.json"
        if p.exists():
            rows[name] = _load(p)
    for run, _, steps in MODELS:
        for step in steps:
            p = runs / run / "posthoc_dyn" / "rows" / f"{step:06d}.json"
            if p.exists():
                rows[_label(run, step)] = _load(p)
    out: dict[str, Any] = {"note": "探索の記述", "models": {}}
    for label, rs in rows.items():
        m: dict[str, Any] = {}
        for place in ("own", "h_mean"):
            pr = _place(rs, place)
            if not pr:
                continue
            e: dict[str, Any] = {"summary": summary(pr), "pairs": {}}
            for other in ("pca", "mean8"):
                if other in yard:
                    o = _place(yard[other], "own")
                    e["pairs"][other] = {"lin": pair(pr, o, "lin"),
                                         "mlp": pair(pr, o, "mlp", sigma=max(sig.values()))}  # fmt: skip
            m[place] = e
        out["models"][label] = m
    return out


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--stage", choices=("checks", "yard", "apply"), required=True)
    p.add_argument("--dyn", type=Path, default=Path("runs/v18c-dyn"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    if a.stage == "checks":
        out = checks_stage(_load(a.dyn / "checks" / "rows.json"))
    elif a.stage == "yard":
        out = yard_stage(a.dyn)
    else:
        out = apply_stage(a.dyn, a.runs)
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    main()
