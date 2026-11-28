r"""The verdicts of the foundation's main evaluation (``plans/foundation.md``,
4.1; ``exp.vision.fnd_dyn``), in dyn/copy's words
(``results/dyn/judge_dyn.py``). Written before any number.

**A code's numbers** (``gru_summary``): of each direction and count of
layers, of each seed on Zurich $R_{\mathrm{copy}}(\Delta)$ and
$R_{\mathrm{clim}}(\Delta)$ (the sums over the blocks) and $\overline{\log
R}$ over 0.8 .. 25.6 s (``dynpred.mean_log`` of the lags $\Delta = 1 ..
32$ steps), their mean over the seeds and its sample deviation. The
**main** count of layers is the one of the least validation loss (the
seeds' mean of each seed's best epoch's; fewer layers on a tie), chosen on
the validation video, never on Zurich; both are kept.

**A pair** (``pair``) $(a, b)$ of one direction and count of layers:
``judge_dyn.psi_interval`` of the seeds' mean numerators (each code's own
denominators), $\sigma_{ab} = \max(\sigma_{\mathrm{rot}}, s_{ab})$ of each
denominator apart (as ``judge_dyn.sigma_ab``: the seeds' sample deviations of
$\psi_s$ of ``rot`` against ``pca`` and of the pair, seed by seed in
order), $z = 2$, the main range $(1, 32)$; the words ``judge_dyn.words``
(「a の予測が b より良いと区別できた」 only where both intervals lie below
0). Beside it the shifted blocks' interval (a description). The two-point
linear (``lin``) has $\sigma_{ab} = 0$.

**The checks** (``check_summary``): each unit's ``ok`` and the minimum
over them.

::

    uv run python -m exp.vision.results.foundation.judge_fnd --dir runs/fnd-dyn \
        [--json out.json]
"""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from torch import Tensor

from exp.lib.dynpred import mean_log
from exp.vision.results.dyn.judge_dyn import Interval, psi_interval, words

__all__ = ["check_summary", "gru_summary", "lin_summary", "main", "pair"]

type Row = dict[str, Any]

DENS = ("copy", "clim")
_F64 = torch.float64


def _t(x: Any) -> Tensor:
    return torch.tensor(x, dtype=_F64)


def _main_range(lags: Sequence[int]) -> tuple[int, int]:
    return min(lags), max(lags)


def _ml(lags: Sequence[int], num_b: Tensor, den_b: Tensor) -> float:
    r = num_b.sum(-1) / den_b.sum(-1)
    return float(mean_log(lags, r, *_main_range(lags)))


def _gru(rows: Sequence[Row], direction: str, layers: int) -> list[Row]:
    got = [r for r in rows if r.get("part") == "gru" and r["dir"] == direction
           and r["layers"] == layers]  # fmt: skip
    return sorted(got, key=lambda r: r["seed"])


def _seed_psis(a: Sequence[Row], b: Sequence[Row], d: str) -> list[float]:
    out = []
    for ra, rb in zip(a, b, strict=True):
        za, zb = ra["sets"]["zurich"], rb["sets"]["zurich"]
        lags = ra["lags"]
        out.append(_ml(lags, _t(za["num_b"]), _t(za[f"{d}_b"]))
                   - _ml(lags, _t(zb["num_b"]), _t(zb[f"{d}_b"])))  # fmt: skip
    return out


def gru_summary(rows: Sequence[Row]) -> dict[str, Any]:
    """Of one code's ``gru`` rows (module docstring): by direction, the
    numbers of each count of layers and the ``main`` one."""
    out: dict[str, Any] = {}
    if any(r.get("part") == "unmeasurable" for r in rows):
        return {"unmeasurable": True}
    for direction in sorted({r["dir"] for r in rows if r.get("part") == "gru"}):
        by: dict[str, Any] = {}
        vals: dict[int, float] = {}
        for layers in sorted({r["layers"] for r in rows if r.get("part") == "gru"}):
            got = _gru(rows, direction, layers)
            if not got:
                continue
            lags = got[0]["lags"]
            per: dict[str, list[float]] = {d: [] for d in DENS}
            curves: dict[str, list[list[float]]] = {d: [] for d in DENS}
            for r in got:
                z = r["sets"]["zurich"]
                for d in DENS:
                    num, den = _t(z["num_b"]), _t(z[f"{d}_b"])
                    per[d].append(_ml(lags, num, den))
                    curves[d].append((num.sum(-1) / den.sum(-1)).tolist())
            vals[layers] = mean(r["val_curve"][r["best"]] for r in got)
            by[str(layers)] = {
                "seeds": [r["seed"] for r in got],
                "val": vals[layers],
                "best": [r["best"] for r in got],
                "mean_log": {d: mean(per[d]) for d in DENS},
                "sd": {d: stdev(per[d]) if len(per[d]) > 1 else None for d in DENS},
                "per_seed": per,
                "curves": curves,
                "lags": lags,
            }
        main_layers = min(vals, key=lambda n: (vals[n], n)) if vals else None
        out[direction] = {"layers": by, "main": main_layers}
    return out


def _mean_num(rows: Sequence[Row], key: str) -> Tensor:
    return torch.stack([_t(r["sets"]["zurich"][key]) for r in rows]).mean(0)


def _iv(i: Interval) -> dict[str, Any]:
    return i.__dict__.copy()


def pair(
    a: Sequence[Row], b: Sequence[Row], direction: str, layers: int,
    rot: Sequence[Row] | None = None, pca: Sequence[Row] | None = None, z: float = 2.0,
) -> dict[str, Any] | None:  # fmt: skip
    """The intervals and the words of ``a`` against ``b`` (their ``gru``
    rows) of one direction and count of layers (module docstring);
    $\\sigma_{\\mathrm{rot}}$ of ``rot`` against ``pca`` (0 without). None
    where a side has no rows."""
    ra, rb = _gru(a, direction, layers), _gru(b, direction, layers)
    if not ra or not rb:
        return None
    lags = ra[0]["lags"]
    out: dict[str, Any] = {}
    ivs = {}
    for d in DENS:
        s_ab = stdev(_seed_psis(ra, rb, d)) if len(ra) > 1 else 0.0
        s_rot = 0.0
        if rot is not None and pca is not None:
            rr, rp = _gru(rot, direction, layers), _gru(pca, direction, layers)
            if len(rr) > 1 and len(rr) == len(rp):
                s_rot = stdev(_seed_psis(rr, rp, d))
        sigma = max(s_rot, s_ab)  # judge_dyn.sigma_ab of the two spreads
        args = (_mean_num(ra, "num_b"), _mean_num(ra, f"{d}_b"),
                _mean_num(rb, "num_b"), _mean_num(rb, f"{d}_b"))  # fmt: skip
        ivs[d] = psi_interval(lags, *args, sigma=sigma, z=z, main=_main_range(lags))
        out[d] = _iv(ivs[d]) | {"s_ab": s_ab, "sigma_rot": s_rot}
        if "num_b_s" in ra[0]["sets"]["zurich"]:
            sh = (_mean_num(ra, "num_b_s"), _mean_num(ra, f"{d}_b_s"),
                  _mean_num(rb, "num_b_s"), _mean_num(rb, f"{d}_b_s"))  # fmt: skip
            out[f"{d}_shifted"] = _iv(psi_interval(lags, *sh, sigma=sigma, z=z,
                                                   main=_main_range(lags)))  # fmt: skip
    out["words"] = words(ivs["copy"], ivs["clim"])
    return out


def lin_summary(rows: Sequence[Row]) -> dict[str, Any]:
    """Of one code's ``lin`` rows: by direction $\\overline{\\log R}$ on
    Zurich and the chosen scales."""
    out: dict[str, Any] = {}
    for r in rows:
        if r.get("part") != "lin":
            continue
        z = r["sets"]["zurich"]
        out[r["dir"]] = {"mean_log": {d: _ml(r["lags"], _t(z["num_b"]), _t(z[f"{d}_b"]))
                                      for d in DENS}, "scale": r["scale"]}  # fmt: skip
    return out


def lin_pair(
    a: Sequence[Row], b: Sequence[Row], direction: str
) -> dict[str, Any] | None:
    """The two-point linear's pair ($\\sigma_{ab} = 0$)."""
    ra = [r for r in a if r.get("part") == "lin" and r["dir"] == direction]
    rb = [r for r in b if r.get("part") == "lin" and r["dir"] == direction]
    if not ra or not rb:
        return None
    za, zb = ra[0]["sets"]["zurich"], rb[0]["sets"]["zurich"]
    lags = ra[0]["lags"]
    ivs = {d: psi_interval(lags, _t(za["num_b"]), _t(za[f"{d}_b"]), _t(zb["num_b"]),
                           _t(zb[f"{d}_b"]), main=_main_range(lags)) for d in DENS}  # fmt: skip
    return {d: _iv(v) for d, v in ivs.items()} | {
        "words": words(ivs["copy"], ivs["clim"])
    }


def check_summary(rows: Sequence[Row]) -> dict[str, Any]:
    """Each check unit's ``ok`` (by ``<check>-L<layers>``) and ``all``."""
    oks = {f"{r['check']}-L{r['layers']}": bool(r["ok"]) for r in rows
           if r.get("part") == "check"}  # fmt: skip
    return {"units": oks, "all": bool(oks) and all(oks.values())}


def _rows(path: Path) -> list[Row]:
    return json.loads(path.read_text())


def judge(root: Path, against: str = "pca") -> dict[str, Any]:
    """The summaries of every code under ``root`` (``fnd_dyn``'s outputs)
    and their pairs against ``against``."""
    gru: dict[str, list[Row]] = {}
    for f in sorted((root / "gru").glob("*/*.json")):
        gru.setdefault(f.parent.name, []).extend(_rows(f))
    lin = {f.stem: _rows(f) for f in sorted((root / "lin").glob("*.json"))}
    checks = [r for f in sorted((root / "checks").glob("*.json"))
              if f.name != "config.json" for r in _rows(f)]  # fmt: skip
    out: dict[str, Any] = {"checks": check_summary(checks), "codes": {}}
    for code in sorted(set(gru) | set(lin)):
        entry: dict[str, Any] = {}
        if code in gru:
            entry["gru"] = gru_summary(gru[code])
            if code != against and against in gru:
                pairs: dict[str, Any] = {}
                for direction in ("future", "past"):
                    for layers in (1, 2):
                        got = pair(gru[code], gru[against], direction, layers,
                                   gru.get("rot"), gru.get("pca"))  # fmt: skip
                        if got is not None:
                            pairs[f"{direction}-L{layers}"] = got
                entry["pairs"] = pairs
        if code in lin:
            entry["lin"] = lin_summary(lin[code])
            if code != against and against in lin:
                entry["lin_pairs"] = {d: lin_pair(lin[code], lin[against], d)
                                      for d in ("future", "past")}  # fmt: skip
        out["codes"][code] = entry
    return out


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(
        prog="python -m exp.vision.results.foundation.judge_fnd"
    )
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--against", default="pca")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    out = judge(a.dir, a.against)
    text = json.dumps(out, indent=1, default=lambda x: None if isinstance(x, float)
                      and not math.isfinite(x) else x)  # fmt: skip
    if a.json:
        a.json.write_text(text + "\n")
    else:
        print(text)
    return out


if __name__ == "__main__":
    main()
