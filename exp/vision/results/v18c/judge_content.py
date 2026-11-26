r"""The verdicts of the guard rebuilt on the observation's content
(``exp/vision/README.md``, 「線形の幅のガードの作り直し（回す前に決めたこと、
2026-11-26）」; ``exp.vision.content_eval``). Written before any number.

``--stage checks`` reads ``<width>/content/checks/rows.json`` (V-h4a): each
check's pass (a failure stops) and the records.

``--stage real`` (V-h4b) reads ``<width>/content/<code>/content.json``, the
cheats' widths ``<width>/guard/cheat/<name>/width.json`` and the yard's
``<width>/yard/rows/<code>.json``:

- $\sigma_{\mathrm{seed}}$ (``sigmas``): of ``rot`` against ``base``
  (``pca``), the per-seed $\psi_s$'s and $\psi^d_s$'s sample deviations
- V-h4b1: of each ``mean-<w>`` whose $\bar\Delta_1$ against ``base``
  (``judge_width.pair_ratio``) clears $+\Delta_*$ wholly, the front's
  verdict (``content_eval.front_of``) against ``base``; every one
  ``dropped`` passes, none claimable does not (nothing is caught)
- V-h4b2: the point $E$ of ``drop-16`` .. ``drop-56`` does not rise with
  the strength (``base`` recorded beside)
- V-h4b3: ``rot`` against ``base``: $\psi$'s and $\psi^d$'s intervals in
  $[-\psi_*, +\psi_*]$; every yardstick (``base`` as ``pca``) comparable
  ($E$ and $E^d$; a yardstick missing fails)
- V-h4b4 (records): each code's $E$, $E^d$ of ``mlp`` and ``lin``, of the
  local features where given, and the extrapolation in the observation;
  of ``mlp`` the level error's lag-1 autocorrelation (``rho1``: $E^d$ lies
  outside propositions 1 and 2) and $E$, $E^d$ on the training set and on
  the held-out runs of the training videos (``fit``: the training
  distribution's in-sample and held-out values beside Zurich's); of each
  code against ``base`` the pair's own per-seed spread (``pair_sigma``) and
  those above $\sigma_{\mathrm{seed}}$ (the verdicts keep
  $\sigma_{\mathrm{seed}}$ as registered)

**v3** (``exp/vision/README.md``, 「線形の幅の内容のガード v3（回す前に決めた
こと、2026-11-26）」). ``--stage checks3`` reads
``<width>/content3/checks/rows.json`` (V-h5a, merged by
``content_eval.merge_checks3``). ``--stage real3`` (V-h5b) reads
``<width>/content3/<code>/content3.json`` and the same widths:

- $\sigma_{\mathrm{rot}}$: of ``rot`` against ``base``, the per-seed
  $D_s$'s and $D^d_s$'s sample deviations; each pair's $\sigma$ the larger
  of it and the pair's own (``content_eval.pair3``)
- V-h5b1: of each ``mean-<w>`` whose $\bar\Delta_1$ against ``base``
  clears $+\Delta_*$ wholly, ``content_eval.front3`` against ``base``:
  every one ``dropped`` and at least one
- V-h5b2: the chosen readout's point $E$ of ``base``, ``drop-16`` ..
  ``drop-56`` does not rise with the strength ($E^d$ recorded)
- V-h5b3: ``rot`` against ``base`` and ``base`` against ``rot``: both
  lower ends of $D$ and $D^d$ above 0; every yardstick (``base`` as
  ``pca``) comparable ($E$ and $E^d$; one missing fails)
- V-h5b4: of every code, the chosen readout against the least squares on
  the record's runs (``fidelity.noninferior``, the runs drawn 2000 times of
  seed 102), margin $(1 - 2^{-\psi_*}) \hat E_{\mathrm{lin}}(\text{base})$
  there ($E$ and $E^d$); recorded beside, the same on the record's runs at
  least ``run + gap`` from every training run (``content_eval.far_runs``:
  the record's runs are drawn with no gap, so their scenes leak as much)
- V-h5b5 (records): each code's choices, the candidates' validation
  scores, the gap to the runner-up of another kind and its SE, $E$ and
  $E^d$ of each readout on each set, the extrapolation in the observation

Prints markdown; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_content --stage checks|real \
        [--width runs/v18c-width] [--json out.json]
    uv run python -m exp.vision.results.v18c.judge_content --stage checks3|real3 ..
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from exp.lib.fidelity import (
    ContentSums,
    comparable,
    contrast_seed_sd,
    noninferior,
    ratio_interval,
    seed_sigma,
)
from exp.lib.metrics import block_counts
from exp.vision.content_eval import (
    NI_DRAWS,
    NI_SEED,
    YARD,
    chosen3,
    far_runs,
    front3,
    front_of,
    noninferior_far,
    pair3,
    sums3,
)
from exp.vision.geom_eval import GeomConf
from exp.vision.results.v18c.judge_width import DELTA, _main, pair_ratio
from exp.vision.width_eval import WidthConf, counts_of, lin_sums, point

Row = dict[str, Any]

PSI_STAR = 0.1  # psi*, octaves of the explained variance (plans/jepa.md, registered)
BASE = "base"  # pca's whitened code, the yardstick pca


def _load(path: Path) -> list[Row]:
    return json.loads(path.read_text()) if path.exists() else []


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if math.isinf(x):
        return "−∞" if x < 0 else "+∞"
    return f"{x:.{digits}g}"


def _iv(v: Mapping[str, Any]) -> str:
    return f"[{_fmt(v.get('lo'))}, {_fmt(v.get('hi'))}]"


# -- the synthetic checks -----------------------------------------------------------------


def checks_stage(rows: Sequence[Row], title: str = "V-h4a") -> dict[str, Any]:
    """Each check's pass and the records (V-h4a; V-h5a of ``title``)."""
    out: dict[str, Any] = {"checks": {}, "failures": [], "records": []}
    for r in rows:
        if r["part"] == "check":
            out["checks"][r["name"]] = r["pass"]
            if not r["pass"]:
                out["failures"].append(r["name"])
        elif r["part"] == "record":
            out["records"].append(r)
    print(f"## {title}\n\n| 確かめ | 合否 |\n| -- | -- |")
    for name, ok in out["checks"].items():
        print(f"| {name} | {'通った' if ok else '外れた'} |")
    for r in out["records"]:
        print("-", json.dumps({k: v for k, v in r.items() if k != "snap"}))
    if out["failures"]:
        print(f"\n**外れた確かめ（ここで止める）**: {out['failures']}")
    return out


# -- the real data -------------------------------------------------------------------------


def load_content(root: Path) -> dict[str, dict[str, Any]]:
    """Of each code's ``content.json`` under ``root``: ``sums`` by
    (``obs``, ``readout``), the ``obs_ext`` rows, ``rho1`` (the ``mlp``
    readouts' ``err_rho1`` by ``obs``) and ``fit`` (the sums by ``set``)."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/content.json")):
        rows = _load(path)
        content = [r for r in rows if r["part"] == "content"]
        sums = {(r["obs"], r["readout"]): ContentSums.of_row(r) for r in content}
        rho1 = {r["obs"]: r["err_rho1"] for r in content if "err_rho1" in r}
        fit = {r["set"]: ContentSums.of_row(r)
               for r in rows if r["part"] == "fit" and r["readout"] == "mlp"}  # fmt: skip
        ext = [r for r in rows if r["part"] == "obs_ext"]
        out[path.parent.name] = {"sums": sums, "ext": ext, "rho1": rho1, "fit": fit}
    return out


def sigmas(content: Mapping[str, Mapping[str, Any]]) -> tuple[float, float]:
    r"""$\sigma_{\mathrm{seed}}$ of $E$ and $E^d$: ``rot`` against ``base``
    (``pix``, ``mlp``)."""
    a, b = content["rot"]["sums"][("pix", "mlp")], content[BASE]["sums"][("pix", "mlp")]
    return seed_sigma(a, b), seed_sigma(a, b, change=True)


def _ext_shares(ext: Sequence[Row]) -> dict[str, dict[str, dict[str, float]]]:
    """Of each readout the seeds' mean error shares (copy, ext, ceil) at
    each lag."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for r in ext:
        per = r["lags"]
        lags = per[0].keys()
        out[r["readout"]] = {
            lag: {k: sum(p[lag][k] / p[lag]["tot"] for p in per) / len(per)
                  for k in ("copy", "ext", "ceil")}
            for lag in lags
        }  # fmt: skip
    return out


def real_stage(width: Path, wconf: WidthConf, g: GeomConf) -> dict[str, Any]:
    """V-h4b (module docstring)."""
    content = load_content(width / "content")
    base = content[BASE]["sums"][("pix", "mlp")]
    nb = len(base.tot)
    counts = (counts_of(nb, g), counts_of(nb // 2, g))
    sig = sigmas(content)
    out: dict[str, Any] = {"sigma_seed": list(sig), "codes": {}}
    for name, c in content.items():
        out["codes"][name] = {
            f"{obs}/{ro}": {"E": s.value(), "Ed": s.value(True),
                            "E_seeds": s.per_seed().tolist()}
            for (obs, ro), s in c["sums"].items()
        } | {"ext": _ext_shares(c["ext"]), "rho1": c["rho1"],
             "fit": {k: {"E": s.value(), "Ed": s.value(True)} for k, s in c["fit"].items()}}  # fmt: skip
    # the pairs' own per-seed spread against base (a record)
    pair_sigma = {}
    for name, c in content.items():
        if name != BASE and ("pix", "mlp") in c["sums"]:
            a = c["sums"][("pix", "mlp")]
            pair_sigma[name] = [seed_sigma(a, base), seed_sigma(a, base, change=True)]
    out["pair_sigma"] = pair_sigma
    out["pair_sigma_above"] = [n for n, v in pair_sigma.items()
                               if any(x > s for x, s in zip(v, sig, strict=True))]  # fmt: skip
    # V-h4b1: the past means
    cheat = width / "guard" / "cheat"
    ls_base = lin_sums(_load(cheat / BASE / "width.json"))
    means = sorted((n for n in content if n.startswith("mean-")),
                   key=lambda n: int(n.split("-")[1]))  # fmt: skip
    b1: dict[str, Any] = {}
    for name in means:
        pr = pair_ratio(lin_sums(_load(cheat / name / "width.json")), ls_base, wconf, g)
        b1[name] = front_of(pr["mean1"], content[name]["sums"][("pix", "mlp")], base,
                            *counts, sig, DELTA, PSI_STAR) | {"width_cls": pr["mean1_cls"]}  # fmt: skip
    claim = [n for n, f in b1.items() if f["verdict"] != "width"]
    out["b1"] = {"fronts": b1, "claimable": claim,
                 "pass": bool(claim) and all(b1[n]["verdict"] == "dropped" for n in claim)}  # fmt: skip
    # V-h4b2: the fast directions dropped
    drops = sorted((n for n in content if n.startswith("drop-")),
                   key=lambda n: int(n.split("-")[1]))  # fmt: skip
    es = [content[n]["sums"][("pix", "mlp")].value() for n in drops]
    out["b2"] = {"codes": drops, "E": es, "E_base": base.value(),
                 "pass": bool(es) and all(b <= a for a, b in zip(es, es[1:], strict=False))}  # fmt: skip
    # V-h4b3: no false catch, every yardstick comparable
    rot = content["rot"]["sums"][("pix", "mlp")]
    psi = ratio_interval(rot, base, *counts, sig[0])
    psi_d = ratio_interval(rot, base, *counts, sig[1], change=True)

    def inside(v: Mapping[str, Any]) -> bool:
        return v["lo"] is not None and v["hi"] is not None and \
            -PSI_STAR <= v["lo"] and v["hi"] <= PSI_STAR  # fmt: skip

    yard = {}
    for name in (BASE, *YARD):
        if name not in content:
            yard[name] = None
            continue
        s = content[name]["sums"][("pix", "mlp")]
        yard[name] = [comparable(s, *counts), comparable(s, *counts, change=True)]
    all_comp = all(
        v is not None and all(bool(c["ok"]) for c in v) for v in yard.values()
    )
    out["b3"] = {"psi": psi, "psi_d": psi_d, "comparable": yard,
                 "pass": inside(psi) and inside(psi_d) and all_comp}  # fmt: skip
    out["yard_width"] = _yard_points(width, wconf, g)
    _print(out)
    return out


def _yard_points(width: Path, wconf: WidthConf, g: GeomConf) -> dict[str, float | None]:
    """The yardsticks' point $\\bar\\Delta_1$ (``own``, the main whitening;
    a record beside their content)."""
    out = {}
    for name in ("pca", *YARD):
        rows = _load(width / "yard" / "rows" / f"{name}.json")
        try:
            out[name] = point(lin_sums(rows, **_main("own", g)), wconf)["25"]["mean1"]
        except ValueError:
            out[name] = None
    return out


def _print(out: Mapping[str, Any]) -> None:
    s, sd = out["sigma_seed"]
    print(f"σ_seed(ψ) = {_fmt(s)}, σ_seed(ψ^d) = {_fmt(sd)}\n")
    print("| 符号 | E | E^d | E_lin | E^d_lin | E（局所） | E^d（局所） | E（訓練） | "
          "E（訓練の外の区間） | 誤差の lag 1 の自己相関 | 対の σ（ψ, ψ^d） |\n"
          "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")  # fmt: skip
    for name, c in out["codes"].items():

        def v(key: str, q: str) -> str:
            return _fmt(c[key][q]) if key in c else "—"

        fit = c["fit"]
        rho = ", ".join(_fmt(x) for x in c["rho1"].get("pix", []))
        ps = out["pair_sigma"].get(name)
        print(f"| {name} | {v('pix/mlp', 'E')} | {v('pix/mlp', 'Ed')} | {v('pix/lin', 'E')} | "
              f"{v('pix/lin', 'Ed')} | {v('local/mlp', 'E')} | {v('local/mlp', 'Ed')} | "
              f"{_fmt(fit['train']['E']) if 'train' in fit else '—'} | "
              f"{_fmt(fit['holdout']['E']) if 'holdout' in fit else '—'} | {rho or '—'} | "
              f"{', '.join(_fmt(x) for x in ps) if ps else '—'} |")  # fmt: skip
    print(
        f"\nσ_seed を超える対（記録。判定の σ_seed は登録のまま）: {out['pair_sigma_above']}"
    )
    print("\n### V-h4b1（時間の平均）\n\n| ずる | 幅の比の区間 | ψ | ψ^d | 判定 |\n"
          "| -- | -- | -- | -- | -- |")  # fmt: skip
    for name, f in out["b1"]["fronts"].items():
        w = f["width"]
        print(f"| {name} | [{_fmt(w[0])}, {_fmt(w[1])}] | {_iv(f['psi'])} | "
              f"{_iv(f['psi_d'])} | {f['verdict']} |")  # fmt: skip
    print(f"\nV-h4b1: {'通った' if out['b1']['pass'] else '外れた'}"
          f"（主張できる強さ {out['b1']['claimable']}）")  # fmt: skip
    b2 = out["b2"]
    print(f"V-h4b2: {'通った' if b2['pass'] else '外れた'}（E: "
          f"{', '.join(f'{n} {_fmt(e)}' for n, e in zip(b2['codes'], b2['E'], strict=True))}"
          f"、base {_fmt(b2['E_base'])}）")  # fmt: skip
    b3 = out["b3"]
    print(f"V-h4b3: {'通った' if b3['pass'] else '外れた'}（rot 対 base の ψ {_iv(b3['psi'])}、"
          f"ψ^d {_iv(b3['psi_d'])}、比べられない物差し "
          f"{[n for n, v in b3['comparable'].items() if v is None or not all(c['ok'] for c in v)]}）")  # fmt: skip


# -- v3 (V-h5b) ---------------------------------------------------------------------------


def _drop_strength(name: str) -> int:
    return 0 if name == BASE else int(name.split("-")[1])


def real3_stage(width: Path, wconf: WidthConf, g: GeomConf) -> dict[str, Any]:
    """V-h5b (module docstring)."""
    content = {p.parent.name: _load(p)
               for p in sorted((width / "content3").glob("*/content3.json"))}  # fmt: skip

    def s_(name: str, readout: str = "sel", set_: str = "eval") -> ContentSums:
        return sums3(content[name], readout, set_)

    base = s_(BASE)
    nb = len(base.tot)
    counts = (counts_of(nb, g), counts_of(nb // 2, g))
    sig = (contrast_seed_sd(s_("rot"), base, PSI_STAR),
           contrast_seed_sd(s_("rot"), base, PSI_STAR, change=True))  # fmt: skip
    out: dict[str, Any] = {"sigma_rot": list(sig), "codes": {}}
    for name, rows in content.items():
        sel = {r["target"]: {k: r.get(k) for k in ("selected", "runner", "gap", "gap_se",
                                                     "scores")}
               for r in rows if r["part"] == "select"}  # fmt: skip
        vals = {f"{r['readout']}/{r['set']}": {"E": (v := ContentSums.of_row(r)).value(),
                                               "Ed": v.value(True)}
                for r in rows if r["part"] == "sums"}  # fmt: skip
        ext = [r for r in rows if r["part"] == "obs_ext"]
        out["codes"][name] = {"chosen": chosen3(rows), "select": sel, "values": vals,
                              "ext": _ext_shares(ext) if ext else {}}  # fmt: skip
    # V-h5b1: the past means
    cheat = width / "guard" / "cheat"
    ls_base = lin_sums(_load(cheat / BASE / "width.json"))
    means = sorted((n for n in content if n.startswith("mean-")),
                   key=lambda n: int(n.split("-")[1]))  # fmt: skip
    b1: dict[str, Any] = {}
    for name in means:
        pr = pair_ratio(lin_sums(_load(cheat / name / "width.json")), ls_base, wconf, g)
        b1[name] = front3(pr["mean1"], s_(name), base, *counts, sig, DELTA, PSI_STAR) | {
            "width_cls": pr["mean1_cls"]}  # fmt: skip
    claim = [n for n, f in b1.items() if f["verdict"] != "width"]
    out["b1"] = {"fronts": b1, "claimable": claim,
                 "pass": bool(claim) and all(b1[n]["verdict"] == "dropped" for n in claim)}  # fmt: skip
    # V-h5b2: the fast directions dropped, from the strength 0
    drops = sorted((n for n in content if n == BASE or n.startswith("drop-")),
                   key=_drop_strength)  # fmt: skip
    es = [s_(n).value() for n in drops]
    out["b2"] = {"codes": drops, "E": es, "Ed": [s_(n).value(True) for n in drops],
                 "pass": len(es) > 1 and all(b <= a for a, b in zip(es, es[1:], strict=False))}  # fmt: skip
    # V-h5b3: no false catch either way, every yardstick comparable
    rot = s_("rot")
    both = {
        "rot-base": pair3(rot, base, *counts, sig, PSI_STAR),
        "base-rot": pair3(base, rot, *counts, sig, PSI_STAR),
    }
    kept = all(p["D"]["lo"] > 0 and p["Dd"]["lo"] > 0 for p in both.values())
    yard: dict[str, Any] = {}
    for name in (BASE, *YARD):
        yard[name] = None if name not in content else [
            comparable(s_(name), *counts, change=ch) for ch in (False, True)]  # fmt: skip
    all_comp = all(
        v is not None and all(bool(c["ok"]) for c in v) for v in yard.values()
    )
    out["b3"] = {"pairs": both, "comparable": yard, "pass": kept and all_comp}
    # V-h5b4: the chosen readout not worse than the least squares (the record's runs)
    nr = len(s_(BASE, "sel", "record").tot)
    rec = block_counts(nr, NI_DRAWS, torch.Generator().manual_seed(NI_SEED))
    lin_b = s_(BASE, "lin", "record")
    margins = [(1 - 2**-PSI_STAR) * lin_b.value(ch) for ch in (False, True)]
    ni = {n: [noninferior(s_(n, "sel", "record"), s_(n, "lin", "record"), m, rec, change=ch)
              for m, ch in zip(margins, (False, True), strict=True)] for n in content}  # fmt: skip
    out["b4"] = {"margins": margins, "codes": ni,
                 "pass": all(bool(x["ok"]) for v in ni.values() for x in v)}  # fmt: skip
    man_rows = _load(width / "content3" / "intervals.json")
    if (
        man_rows
    ):  # a record: the record's runs as far from the training runs as the val's
        man = {r["set"]: r["starts"] for r in man_rows}
        conf = json.loads((width / "content3" / "config.json").read_text())
        least = man_rows[0]["length"] + conf["content3"]["val_gap"]
        far = far_runs(man["record"], man["train"], least)
        out["b4_far"] = noninferior_far(content, BASE, far, PSI_STAR)
    out["val_distance"] = _val_distance(width / "content3", content)
    out["yard_width"] = _yard_points(width, wconf, g)
    _print3(out)
    return out


def _val_distance(root: Path, content: Mapping[str, Sequence[Row]]) -> dict[str, Any]:
    """Of each validation run (a record of the scenes' correlation left
    across the gap): its distance in stacks to the nearest training run's
    start and each code's $E$ of the chosen readout there (the run's own
    block)."""
    man = {r["set"]: r["starts"] for r in _load(root / "intervals.json")}
    if not man:
        return {}
    dist = [min(abs(v - j) for j in man["train"]) for v in man["val"]]
    es = {}
    for name, rows in content.items():
        s = sums3(rows, "sel", "val")
        es[name] = (1 - s.err.mean(0) / s.tot).tolist()
    return {"distance": dist, "E": es}


def _print3(out: Mapping[str, Any]) -> None:
    s, sd = out["sigma_rot"]
    print(f"σ_rot(D) = {_fmt(s)}, σ_rot(D^d) = {_fmt(sd)}\n")
    print("| 符号 | 水準の読み出し | 変化の読み出し | E | E^d | E_lin | E^d_lin | E（訓練） | "
          "E（検証） | E（記録用） | E（v2） | E^d（v2） |\n"
          "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")  # fmt: skip
    for name, c in out["codes"].items():
        v = c["values"]

        def f(key: str, q: str = "E") -> str:
            return _fmt(v[key][q]) if key in v else "—"

        ch = c["chosen"]
        print(f"| {name} | {ch.get('level')} | {ch.get('change')} | {f('sel/eval')} | "
              f"{f('sel/eval', 'Ed')} | {f('lin/eval')} | {f('lin/eval', 'Ed')} | "
              f"{f('sel/train')} | {f('sel/val')} | {f('sel/record')} | {f('v2/eval')} | "
              f"{f('v2/eval', 'Ed')} |")  # fmt: skip
    print("\n### V-h5b1（時間の平均）\n\n| ずる | 幅の比の区間 | D | D^d | ψ, ψ^d | 判定 |\n"
          "| -- | -- | -- | -- | -- | -- |")  # fmt: skip
    for name, f3 in out["b1"]["fronts"].items():
        w = f3["width"]
        print(f"| {name} | [{_fmt(w[0])}, {_fmt(w[1])}] | {_iv(f3['D'])} | {_iv(f3['Dd'])} | "
              f"{', '.join(_fmt(x) for x in f3['psi'])} | {f3['verdict']} |")  # fmt: skip
    print(f"\nV-h5b1: {'通った' if out['b1']['pass'] else '外れた'}"
          f"（主張できる強さ {out['b1']['claimable']}）")  # fmt: skip
    b2 = out["b2"]
    print(f"V-h5b2: {'通った' if b2['pass'] else '外れた'}（E: "
          f"{', '.join(f'{n} {_fmt(e)}' for n, e in zip(b2['codes'], b2['E'], strict=True))}）")  # fmt: skip
    b3 = out["b3"]
    pairs = "、".join(
        f"{k} の D {_iv(p['D'])}・D^d {_iv(p['Dd'])}" for k, p in b3["pairs"].items()
    )
    bad = [
        n
        for n, v in b3["comparable"].items()
        if v is None or not all(c["ok"] for c in v)
    ]
    print(
        f"V-h5b3: {'通った' if b3['pass'] else '外れた'}（{pairs}、比べられない物差し {bad}）"
    )
    b4 = out["b4"]
    worst = [n for n, v in b4["codes"].items() if not all(x["ok"] for x in v)]
    print(f"V-h5b4: {'通った' if b4['pass'] else '外れた'}（余裕 "
          f"{', '.join(_fmt(m) for m in b4['margins'])}、外れた符号 {worst}）")  # fmt: skip
    if "b4_far" in out:
        far = out["b4_far"]
        bad = [n for n, v in far.items() if n != "runs" and not all(x["ok"] for x in v)]
        print(f"V-h5b4 の記録（訓練の区間から run + gap 以上離れた記録用の {far['runs']} 区間だけ）: "
              f"非劣でない符号 {bad}")  # fmt: skip


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--stage", choices=("checks", "real", "checks3", "real3"), required=True
    )
    p.add_argument("--width", type=Path, default=Path("runs/v18c-width"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)
    if a.stage == "checks":
        out = checks_stage(_load(a.width / "content" / "checks" / "rows.json"))
    elif a.stage == "checks3":
        out = checks_stage(
            _load(a.width / "content3" / "checks" / "rows.json"), "V-h5a"
        )
    elif a.stage == "real3":
        out = real3_stage(a.width, WidthConf(), GeomConf())
    else:
        out = real_stage(a.width, WidthConf(), GeomConf())
    if a.json is not None:
        a.json.write_text(json.dumps(out, default=str))
    return out


if __name__ == "__main__":
    main()
