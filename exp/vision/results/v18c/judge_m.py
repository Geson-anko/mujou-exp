r"""The judgement of V18c's remade main measure $m$ and of E1's repair
(exp/vision/README.md, "V18c: 主の指標を作り直す（回す前に決めたこと、
2026-11-25）"; the design in exp/vision/plans/jepa.md, "V18c: 主の指標の作り直し
と E1 の修理").

Reads ``runs/v18c-sem/targets.json`` (the cache's ``targets.json``: the
split, the items' $v_i$ and blocks, $L$, $b_{16}$, the sensitivity, V-m0's
shift identification and the hires data's checks), the yardsticks' rows
``runs/v18c-sem/yardsticks/posthoc_sem/rows``, the screen's runs
``runs/v18c-screen/<run>/posthoc_sem/rows`` (with ``posthoc_jepa`` for the
added seeds' pass, by ``judge_screen``'s rules), the long runs
``runs/v18c-long/<run>/posthoc_sem``, ``posthoc_fill_sem`` (``e1``, ``e1f``)
and ``posthoc_fill`` (``e1diag``, V-E1'2), and the diagnosis of ``shuf`` of
the old checkpoints (``<run>/posthoc_fill/rows/<step>-e1diag.json``), and
applies the rules registered before the runs, as they are:

- the validity checks V-m0 to V-m6 (the visual check is listed, not read);
- the choice of the long runs' setting per learning rate (seed 0 at step
  6000 on sel, the top 2 and those within $\delta_1$ with seeds 1 and 2,
  $\sigma_{\mathrm{seed}}$ of the top 2's only, the settings an added seed
  failed taken out (their seed 0 kept for the maximum and $\delta_1$), the
  ties (i) to (iii) and their breaking);
- the long runs' judgement L1 to L3 on final, and the rules of V19;
- E1's repair: the diagnosis of ``shuf`` (H-bug, H-time, H-skew, described),
  V-E1'1 to V-E1'4, C1' and C4'.

$\mathrm{SE}_{\mathrm{blk}}$ is the block bootstrap's (``metrics.block_bootstrap``,
``BOOT`` resamples from ``BOOT_SEED``, the same draws for every contrast) of
the contrast at hand from the rows' items' $e_i$ and the targets' $v_i$ and
blocks. V-m5's $\delta_1$ takes the largest $\mathrm{SE}_{\mathrm{blk}}$ of
the differences of the pairs of the seed-0 candidates of one learning rate
(the README fixes $\delta_1$'s form, not the pair). C4' takes
$\mathrm{SE}_{\mathrm{blk}}$ of the contrast $(p_{60} - q_{60}) - (p_6 -
q_6)$ of ``pred`` $p$ and ``pool`` $q$ at 60000 and 6000. V-E1'3's
``pred_k3`` is fit at $P$ times ``pool``'s scale (``fill_eval.e1f_fits``),
the same fit up to rounding: $\lvert \Delta m \rvert \le$ ``K3_TOL``.
Prints markdown;
``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_m [--runs runs] [--json out.json]
"""

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from exp.lib.metrics import block_bootstrap
from exp.vision.fill_eval import OUT, OUT_SEM
from exp.vision.results.v18c.judge_screen import Run, judge, lines, yardsticks
from exp.vision.sem_eval import BOOT, BOOT_SEED

Row = dict[str, Any]

LAMS = (0.5, 0.8, 0.95)
CANDIDATES = {
    1e-4: [(lam, r) for lam in LAMS for r in (0.25, 0.5, 0.75)],
    1e-5: [(lam, 0.5) for lam in LAMS],
}
LISTED = {3e-5: [(lam, 0.5) for lam in LAMS]}  # listed, not chosen from
SHORT, LONG = 6000, 60000
LONG_SEEDS = (0, 1)
NEST = (
    ("color", "color8"),
    ("pca16", "pca64"),
    ("pca64", "pca256"),
    ("pca256", "pca1024"),
    ("init_c_k3", "init_c"),
)
V_M5 = 10  # pca1024 >= V_M5 delta_1 (a design choice fixed in advance)
MAX_DIFF = 1.0  # V-m0 (a): exp.lib.tours.hires's
RESID = 1e-9  # H-bug and V-E1'2
# V-E1'3: pred_k3 and pool are one fit in exact arithmetic; a backward-stable
# float64 solve errs by about eps kappa, kappa = cond(X^T X + alpha I) <= 1 +
# D / c (alpha = c tr / D >= c lambda_max / D), <= 1.6e9 at D = 16, c = 1e-8:
# eps kappa ~ 3.6e-7, so |m(pred_k3) - m(pool)| stays under 1e-6
K3_TOL = 1e-6
STEP2 = 2  # choice 2: the top settings of seed 0 per learning rate
DIAG_RUNS = {  # the diagnosis of shuf: (run, step)
    "v18-r0.25-s0": ("v18-long/v18-long-lr1e-4-lam0.5-r0.25-s0", (0, 6000, 60000)),
    "f-s0": ("v18b-long/v18b-long-f-lr1e-4-lam0.5-r0.25-s0", (60000,)),
    "m": ("v18b-long/v18b-long-m-lr3e-5-lam0.8-r0.25-s0", (60000,)),
    "v18-r0.5-s0": ("v18-long/v18-long-lr1e-4-lam0.5-r0.5-s0", (60000,)),
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _tag(x: float) -> str:
    return f"{x:.0e}".replace("e-0", "e-")


def screen_name(lr: float, lam: float, r: float, seed: int) -> str:
    return f"v18c-lr{_tag(lr)}-lam{lam:g}-r{r:g}-s{seed}"


def long_name(lr: float, lam: float, r: float, seed: int) -> str:
    return f"v18c-long-lr{_tag(lr)}-lam{lam:g}-r{r:g}-s{seed}"


def _fmt(x: float | None, d: int = 4) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:.{d}f}"


class Boot:
    """The block bootstrap of the targets' split (module docstring)."""

    def __init__(self, meta: Row) -> None:
        self._split = meta["sets_split"]

    def se(
        self, es: Sequence[Sequence[float]], contrast: Sequence[float], s: str
    ) -> float:
        r"""$\mathrm{SE}_{\mathrm{blk}}$ of $\sum_c a_c m_c$ on the set ``s``
        of the codes' $e$ ``es`` and the coefficients ``contrast``."""
        part = self._split[s]
        return self.se_v(es, part["v"], contrast, s)

    def se_v(
        self,
        es: Sequence[Sequence[float]],
        v: Sequence[float],
        contrast: Sequence[float],
        s: str,
    ) -> float:
        """The same with the items' $v_i$ ``v`` (E1f's own)."""
        part = self._split[s]
        got = block_bootstrap(
            torch.tensor(es, dtype=torch.float64),
            torch.tensor(v, dtype=torch.float64),
            torch.tensor(part["blocks"]),
            torch.tensor([list(contrast)], dtype=torch.float64),
            BOOT,
            torch.Generator().manual_seed(BOOT_SEED),
        )
        return float(got[0])

    def delta(self, a: Row, b: Row, s: str = "sel") -> float:
        """$2 \\mathrm{SE}_{\\mathrm{blk}}$ of $m_a - m_b$ (``delta_w``,
        ``delta_yy``)."""
        return 2 * self.se([a[f"e_{s}"], b[f"e_{s}"]], [1.0, -1.0], s)


# -- validity -------------------------------------------------------------------------


def validity(meta: Row, yard: dict[str, Row], boot: Boot) -> Row:
    """V-m0 to V-m4 and V-m6 (V-m5 needs sigma_seed: ``choose``)."""
    out: Row = {}
    videos = meta["hires"]["index"]["videos"]
    diffs = [v["max_diff"] for v in videos.values()]
    # (b) and (c) held on every kept frame, or hires.prepare wrote no index;
    # the places to the stacks were checked, or build_targets made no targets
    # (the user's decision of 2026-11-25: (a) against the RGB reference)
    places = meta.get("places_checked") is True
    out["V-m0"] = {
        "max_diff": max(diffs),
        "checks": {k: v.get("checks") for k, v in videos.items()},
        "places_checked": places,
        "places_max_diff": meta["places_max_diff"],
        "places_b_margin": meta.get("places_b_margin"),
        "shift_top1": meta["shift_top1"],
        "pass": max(diffs) <= MAX_DIFF and places and meta["shift_top1"] >= 0.5,
    }
    mean, k3 = yard.get("mean"), yard.get("init_c_k3")
    out["V-m1"] = {
        "mean_m": None if mean is None else [mean["m_sel"], mean["m_final"]],
        "k3": None if k3 is None else [k3["m_sel"], k3["k3_bound_sel"]],
        "pass": mean is not None
        and k3 is not None
        and mean["m_sel"] == 0.0
        and mean["m_final"] == 0.0
        and k3["m_sel"] <= k3["k3_bound_sel"]
        and k3["m_final"] <= k3["k3_bound_final"],
    }
    nest = []
    for small, big in NEST:
        if small in yard and big in yard:
            a, b = yard[big], yard[small]
            d, dyy = a["m_sel"] - b["m_sel"], boot.delta(a, b)
            nest.append(
                {"pair": f"{small}<{big}", "diff": d, "d_yy": dyy, "ok": d >= -dyy}
            )
    out["V-m2"] = {
        "pairs": nest,
        "pass": len(nest) == len(NEST) and all(x["ok"] for x in nest),
    }
    sens = meta["sens"]
    rho = sens["rho"]
    out["V-m3"] = {
        "rho": rho,
        "rho_pix": sens["rho_pix"],
        "pass": all(rho[j] < 1 and rho[j] < sens["rho_pix"][j] for j in rho),
    }
    if "init_c" in yard and k3 is not None:
        d, dyy = yard["init_c"]["m_sel"] - k3["m_sel"], boot.delta(yard["init_c"], k3)
        out["V-m4"] = {"diff": d, "d_yy": dyy, "pass": d > dyy}
    else:
        out["V-m4"] = {"pass": False}
    flips = []
    names = sorted(yard)  # mean too: its halves are exactly 0
    for a, b in itertools.combinations(names, 2):
        ya, yb = yard[a], yard[b]
        d = ya["m_sel"] - yb["m_sel"]
        if abs(d) <= boot.delta(ya, yb):
            continue
        for h in (0, 1):
            dh = ya["m_sel_half"][h] - yb["m_sel_half"][h]
            if (dh > 0) != (d > 0):
                flips.append(
                    {"pair": f"{a}/{b}", "half": h, "full": d, "half_diff": dh}
                )
    out["V-m6"] = {"flips": flips, "pass": not flips}
    out["L"] = {"L": meta["L"], "tau": meta["tau"], "at_max": meta["L_at_max"]}
    out["alpha_edge"] = sorted(c for c, r in yard.items() if r.get("alpha_edge"))
    out["explained"] = meta["explained"]
    out["k3_bound"] = {s: meta["sets_split"][s]["k3_bound"] for s in ("sel", "final")}
    return out


# -- the choice of the long runs' settings --------------------------------------------------


def _row(runs: Path, group: str, name: str, step: int) -> Row | None:
    return _load(runs / group / name / "posthoc_sem" / "rows" / f"{step:06d}-z.json")


def _added_ok(runs: Path, name: str, ln: dict[str, float]) -> bool | None:
    """An added seed's pass (README, choice 2): no collapse nor divergence,
    pred_ratio_calib < 0.95, sigreg_heldin < 2.0 (the pending rule not
    applied); None where the run is not there."""
    path = runs / "v18c-screen" / name
    if not (path / "jepa.json").exists():
        return None
    j = judge(Run(path), ln)
    return bool(
        not j["collapsed"]
        and not j["diverged"]
        and j.get("beats_pred")
        and j.get("healthy_s")
    )


Setting = tuple[float, float]


def _settings(runs: Path) -> dict[float, dict[Setting, Row]]:
    """Each candidate's rows by seed (``seeds``) and the added seeds that
    failed their pass (``failed``; README, choice 2)."""
    ln = lines(yardsticks(runs))
    out: dict[float, dict[Setting, Row]] = {}
    for lr, cands in (CANDIDATES | LISTED).items():
        out[lr] = {}
        for lam, r in cands:
            seeds: dict[int, Row] = {}
            failed = []
            for s in (0, 1, 2):
                name = screen_name(lr, lam, r, s)
                row = _row(runs, "v18c-screen", name, SHORT)
                if row is None:
                    continue
                if s > 0 and _added_ok(runs, name, ln) is not True:
                    failed.append(s)
                    continue
                path = runs / "v18c-screen" / name
                q = (
                    Run(path).stat("pred_ratio_calib")
                    if (path / "jepa.json").exists()
                    else None
                )
                seeds[s] = row | {"name": name, "pred_ratio_calib": q}
            out[lr][(lam, r)] = {"seeds": seeds, "failed": failed}
    return out


def _label(k: Setting) -> str:
    return f"lam{k[0]:g}-r{k[1]:g}"


def _full(v: Row) -> bool:
    """3 seeds, none failed: a candidate of choice 4."""
    return len(v["seeds"]) == 3 and not v["failed"]


def choose(runs: Path, boot: Boot, yard: dict[str, Row]) -> Row:
    r"""The choice per learning rate (README 「長い run の選び方」).

    $\sigma_{\mathrm{seed}}$ pools only the settings of choice 2 (each
    learning rate's top ``STEP2`` of seed 0) that have 3 seeds: the near
    settings of choice 3 are chosen by $\delta_1$, itself of
    $\sigma_{\mathrm{seed}}$, so they never enter it. A setting with a failed
    added seed is out of choices 2 and 4 (``removed``); its seed 0 stays in
    the maximum and the near set of choice 3."""
    settings = _settings(runs)
    out: Row = {"lr": {}}
    step2: dict[float, list[Setting]] = {}
    for lr in CANDIDATES:
        seed0 = {k: v["seeds"][0] for k, v in settings[lr].items() if 0 in v["seeds"]}
        step2[lr] = sorted(seed0, key=lambda k: -seed0[k]["m_sel"])[:STEP2]
    ss, df = 0.0, 0
    for lr in CANDIDATES:
        for k in step2[lr]:
            v = settings[lr][k]
            if not _full(v):
                continue
            ms = [v["seeds"][s]["m_sel"] for s in (0, 1, 2)]
            mu = sum(ms) / 3
            ss += sum((m - mu) ** 2 for m in ms)
            df += 2
    sigma = math.sqrt(ss / df) if df else None
    out["sigma_seed"], out["df"] = sigma, df
    out["step2"] = {_tag(lr): [_label(k) for k in ks] for lr, ks in step2.items()}
    out["removed"] = {
        _tag(lr): {
            _label(k): v["failed"] for k, v in settings[lr].items() if v["failed"]
        }
        for lr in CANDIDATES
    }
    out["listed"] = {
        _tag(lr): {_label(k): v["seeds"].get(0, {}).get("m_sel") for k, v in by.items()}
        for lr, by in settings.items()
    }
    se1 = []
    for lr in CANDIDATES:
        rows0 = [v["seeds"][0] for v in settings[lr].values() if 0 in v["seeds"]]
        for a, b in itertools.combinations(rows0, 2):
            se1.append(boot.se([a["e_sel"], b["e_sel"]], [1.0, -1.0], "sel"))
    if sigma is not None and se1 and "pca1024" in yard:
        d1 = 2 * math.sqrt(max(se1) ** 2 + 2 * sigma**2)
        out["V-m5"] = {
            "pca1024": yard["pca1024"]["m_sel"],
            "delta_1": d1,
            "pass": yard["pca1024"]["m_sel"] >= V_M5 * d1,
        }
    short = {
        _tag(lr): [_label(k) for k in ks if _full(settings[lr][k])]
        for lr, ks in step2.items()
    }
    if sigma is None or any(len(v) < STEP2 for v in short.values()):
        out["stopped"] = (
            f"a learning rate has fewer than {STEP2} settings of 3 seeds in choice "
            f"2 (of 3 seeds: {short}; removed: {out['removed']}): stop and report"
        )
        return out
    for lr in CANDIDATES:
        out["lr"][_tag(lr)] = _choose_lr(settings[lr], step2[lr], boot, sigma)
    return out


def _choose_lr(
    settings: dict[Setting, Row],
    step2: list[Setting],
    boot: Boot,
    sigma: float,
) -> Row:
    """Choices 3 to 5 of one learning rate."""
    seed0 = {k: v["seeds"][0] for k, v in settings.items() if 0 in v["seeds"]}
    if not seed0:
        return {"stopped": "no seed-0 row"}
    top = max(seed0, key=lambda k: seed0[k]["m_sel"])
    near = []
    for k, row in seed0.items():
        se = boot.se([seed0[top]["e_sel"], row["e_sel"]], [1.0, -1.0], "sel")
        d1 = 2 * math.sqrt(se**2 + 2 * sigma**2)
        if k != top and seed0[top]["m_sel"] - row["m_sel"] <= d1:
            near.append(k)
    asked = {top, *step2, *near}
    full = {
        k: v["seeds"] for k, v in settings.items() if k in asked and _full(settings[k])
    }
    to_add = [
        k
        for k in near
        if k not in step2 and not _full(settings[k]) and not settings[k]["failed"]
    ]
    head = {
        "seed0_top": _label(top),
        "near": [_label(k) for k in near],
        "to_add": [_label(k) for k in to_add],
    }
    if to_add:
        return head | {"stopped": "choice 3: add seeds 1 and 2 to to_add first"}
    if not full:
        return head | {"stopped": "no setting of 3 seeds"}

    def mean(k: Setting, h: int | None = None) -> float:
        vals = [
            full[k][s]["m_sel"] if h is None else full[k][s]["m_sel_half"][h]
            for s in (0, 1, 2)
        ]
        return sum(vals) / 3

    a = max(full, key=mean)
    ties, table = [], []
    for b in full:
        row: Row = {"setting": _label(b), "m_mean": mean(b)}
        row["m_seeds"] = [full[b][s]["m_sel"] for s in (0, 1, 2)]
        if b != a:
            es = [full[a][s]["e_sel"] for s in (0, 1, 2)] + [
                full[b][s]["e_sel"] for s in (0, 1, 2)
            ]
            se = boot.se(es, [1 / 3] * 3 + [-1 / 3] * 3, "sel")
            d3 = 2 * math.sqrt(se**2 + 2 * sigma**2 / 3)
            i = mean(a) - mean(b) <= d3
            ii = any(full[a][s]["m_sel"] <= full[b][s]["m_sel"] for s in (0, 1, 2))
            iii = any(mean(a, h) < mean(b, h) for h in (0, 1))
            row |= {"delta_3": d3, "tie_i": i, "tie_ii": ii, "tie_iii": iii}
            if i or ii or iii:
                ties.append(b)
        table.append(row)
    group = [a, *ties]
    pick = sorted(
        group,
        key=lambda k: (
            k[1] != 0.5,
            -k[0],
            sum(full[k][s]["pred_ratio_calib"] or math.inf for s in (0, 1, 2)),
        ),
    )[0]
    return head | {
        "best": _label(a),
        "tied": [_label(k) for k in ties],
        "chosen": {"lam": pick[0], "r": pick[1]},
        "table": table,
    }


# -- the long runs ------------------------------------------------------------------------


def _collapse_at(run: Run, ln: dict[str, float], step: int) -> bool | None:
    s, rank, r2 = (
        run.stat("sigreg_heldin", step),
        run.probe("rank", step),
        run.stat("token_r2", step),
    )
    if s is None or rank is None or r2 is None:
        return None
    return s >= ln["k1"] or rank < ln["rank_floor"] or r2 >= ln["theta3"]


def long_runs(runs: Path, chosen: Row, boot: Boot) -> Row:
    """L1 to L3 per learning rate and seed on final (README 「長い run の判定」)."""
    ln = lines(yardsticks(runs))
    out: Row = {}
    for tag, c in chosen.items():
        if "chosen" not in c:
            continue
        lr = float(tag)
        lam, r = c["chosen"]["lam"], c["chosen"]["r"]
        per = {}
        for s in LONG_SEEDS:
            name = long_name(lr, lam, r, s)
            rows = {st: _row(runs, "v18c-long", name, st) for st in (0, SHORT, LONG)}
            if any(v is None for v in rows.values()):
                per[s] = {"name": name, "missing": True}
                continue
            run = Run(runs / "v18c-long" / name)
            q6, q60 = (
                run.stat("pred_ratio_calib", SHORT),
                run.stat("pred_ratio_calib", LONG),
            )
            premise = q6 is not None and q60 is not None and q60 < q6
            r0, r6, r60 = rows[0], rows[SHORT], rows[LONG]
            assert r0 is not None and r6 is not None and r60 is not None
            d2, w2 = r60["m_final"] - r6["m_final"], boot.delta(r60, r6, "final")
            d3, w3 = r60["m_final"] - r0["m_final"], boot.delta(r60, r0, "final")
            collapsed = _collapse_at(run, ln, LONG)
            per[s] = {
                "name": name,
                "premise": premise,
                "m": [r0["m_final"], r6["m_final"], r60["m_final"]],
                "L1": collapsed is False,
                "L2": d2 > w2,
                "L3": d3 > w3,
                "L2_diff": d2,
                "L3_diff": d3,
                "delta_w": [w2, w3],
            }
        judged = [p for p in per.values() if p.get("premise")]
        both = len(judged) == len(LONG_SEEDS) and all(
            p["L1"] and p["L2"] and p["L3"] for p in judged
        )
        refuted = len(judged) == len(LONG_SEEDS) and not any(p["L2"] for p in judged)
        out[tag] = {"seeds": per, "both": both, "refuted_L2": refuted}
    passed = [t for t, v in out.items() if v["both"]]
    missing = any(p.get("missing") for v in out.values() for p in v["seeds"].values())
    out["V19"] = (
        "not yet: a long run's rows are missing"
        if missing
        else f"go with {passed[0]}"
        if len(passed) == 1
        else "ask the user (both passed)"
        if len(passed) == 2
        else "do not go: find the cause"
    )
    return out


# -- E1's repair -----------------------------------------------------------------------------


def diagnosis(runs: Path) -> Row:
    """The diagnosis of shuf (H-bug, H-time, H-skew; described, not judged)."""
    out: Row = {"rows": {}}
    resid = 0.0
    time_ok, skew_ok = [], []
    step0_aligned = None
    for label, (path, steps) in DIAG_RUNS.items():
        for step in steps:
            rows = _load(
                runs / path / "posthoc_fill" / "rows" / f"{step:06d}-e1diag.json"
            )
            if rows is None:
                continue
            for r in rows:
                resid = max(resid, r["resid"])
                key = f"{label}/{step}/{r['target']}/{r['stage']}"
                c, s = r["C"], r["s"]
                fits_time = c[1] > 0 and c[2] > 0 and c[256] < 0 and s[256] < 0.5
                nw = list(zip(r["C_nw"], r["s_nw"], strict=True))
                below_pos = [sv < 0.5 for cv, sv in nw if cv > 0]
                below_neg = [sv < 0.5 for cv, sv in nw if cv < 0]

                def most(x: list[bool]) -> bool:
                    return bool(x) and sum(x) > len(x) / 2

                skew = r["s_off"] < 0.5 and most(below_pos) and most(below_neg)
                out["rows"][key] = {
                    "C1": c[1], "C2": c[2], "C256": c[256], "s256": s[256],
                    "s_off": r["s_off"], "swap": r["swap"], "time": fits_time,
                    "skew": skew,
                }  # fmt: skip
                if step == 0 and r["stage"] == "pred":
                    step0_aligned = (
                        fits_time
                        if step0_aligned is None
                        else step0_aligned and fits_time
                    )
                elif step > 0:
                    time_ok.append(fits_time)
                skew_ok.append(skew)
    out["resid"] = resid
    if not out["rows"]:
        return out | {"H-bug": None, "H-time": None, "H-skew": None}
    out["H-bug"] = resid > RESID
    out["H-time"] = bool(time_ok) and all(time_ok) and step0_aligned is False
    out["H-skew"] = bool(skew_ok) and sum(skew_ok) > len(skew_ok) / 2
    return out


def _resid(rows_dir: Path, steps: Sequence[int]) -> float | None:
    """The largest identity residual of the ``e1diag`` rows of ``steps`` in
    ``rows_dir`` (None if none is there)."""
    got = [_load(rows_dir / f"{step:06d}-e1diag.json") for step in steps]
    vals = [r["resid"] for rows in got if rows is not None for r in rows]
    return max(vals) if vals else None


def e1_repair(runs: Path, chosen: Row, boot: Boot, diag: Row) -> Row:
    """V-E1'1 to V-E1'4, C1' and C4' of the long runs (README 「E1 の修理」)."""
    v1, v3, v4 = True, True, True
    read = {"e1": 0, "step0": 0}
    per: Row = {}
    resid: float | None = None
    for tag, c in chosen.items():
        if "chosen" not in c:
            continue
        lr = float(tag)
        lam, r = c["chosen"]["lam"], c["chosen"]["r"]
        for s in LONG_SEEDS:
            name = long_name(lr, lam, r, s)
            run_dir = runs / "v18c-long" / name
            got_resid = _resid(run_dir / OUT / "rows", (0, SHORT, LONG))
            if got_resid is not None:
                resid = got_resid if resid is None else max(resid, got_resid)
            rows_dir = run_dir / OUT_SEM / "rows"
            k_run = (_load(run_dir / "jepa.json") or {}).get("k")
            got: dict[int, Row] = {}
            for step in (0, SHORT, LONG):
                e1 = _load(rows_dir / f"{step:06d}-e1.json")
                e1f = _load(rows_dir / f"{step:06d}-e1f.json")
                if e1 is None or e1f is None:
                    continue
                read["e1"] += 1
                for x in e1:
                    if x["mask"] == "S" and x["stage"] in (
                        "mean",
                        "oracle_k3",
                        "pred_k3",
                    ):
                        v1 &= x["swap"] == 0.5 and x["n_tie"] == x["n_cand"]
                by = {(x["k"], x["stage"]): x for x in e1f}
                for k in (4, 8, 12):
                    mean = by[(k, "mean")]
                    v3 &= mean["m_sel"] == 0.0 and mean["m_final"] == 0.0
                    a, b = by[(k, "pred_k3")], by[(k, "pool")]
                    v3 &= abs(a["m_final"] - b["m_final"]) <= K3_TOL
                if step == 0:
                    read["step0"] += 1
                    o = by[(4, "oracle")]
                    w = 2 * boot.se_v([o["e_final"]], o["v_final"], [1.0], "final")
                    v4 &= o["m_final"] > w
                if k_run in (4, 8, 12):
                    p, q, vc = (
                        by[(k_run, "pred")],
                        by[(k_run, "pool")],
                        by[(k_run, "vis_color")],
                    )
                    dq = p["m_final"] - q["m_final"]
                    wq = 2 * boot.se_v(
                        [p["e_final"], q["e_final"]], p["v_final"], [1.0, -1.0], "final"
                    )
                    dv = p["m_final"] - vc["m_final"]
                    wv = 2 * boot.se_v(
                        [p["e_final"], vc["e_final"]],
                        p["v_final"],
                        [1.0, -1.0],
                        "final",
                    )
                    got[step] = {
                        "pred_pool": dq, "w_pool": wq, "pred_vis": dv, "w_vis": wv,
                        "pool_cond": q.get("cond"),
                        "e_pred": p["e_final"], "e_pool": q["e_final"],
                        "v": p["v_final"],
                        "all_k": {f"{k}/{st}": by[(k, st)]["m_final"] for (k, st) in by},
                    }  # fmt: skip
            per[name] = got
    out: Row = {
        "V-E1'2": {
            "resid": resid,
            "resid_old": diag["resid"],
            "pass": None if resid is None else resid <= RESID,
        },
        "V-E1'1": {"pass": v1 if read["e1"] else None},
        "V-E1'3": {"tol": K3_TOL, "pass": v3 if read["e1"] else None},
        "V-E1'4": {"pass": v4 if read["step0"] else None},
    }
    c1, c4 = {}, {}
    for name, got in per.items():
        if LONG in got:
            g = got[LONG]
            pool = g["pred_pool"] > g["w_pool"]
            vis = g["pred_vis"] > g["w_vis"]
            c1[name] = {
                "read": "both" if pool and vis else "pool only" if pool else "neither",
                "pool_cond": g["pool_cond"],  # a singular mean A: see the README
            }
        if SHORT in got and LONG in got:
            a, b = got[LONG], got[SHORT]
            drop = b["pred_pool"] - a["pred_pool"]
            se = boot.se_v(
                [a["e_pred"], a["e_pool"], b["e_pred"], b["e_pool"]],
                a["v"],
                [1.0, -1.0, -1.0, 1.0],
                "final",
            )
            c4[name] = {"drop": drop, "w": 2 * se, "pass": drop <= 2 * se}
    out["runs"] = {
        name: {
            st: {k: v for k, v in g.items() if k not in ("e_pred", "e_pool", "v")}
            for st, g in got.items()
        }
        for name, got in per.items()
    }
    out["C1'"], out["C4'"] = c1, c4
    return out


# -- the output ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    runs = args.runs
    meta = _load(runs / "v18c-sem" / "targets.json")
    if meta is None:
        raise SystemExit(f"no {runs / 'v18c-sem' / 'targets.json'}: the targets first")
    boot = Boot(meta)
    ydir = runs / "v18c-sem" / "yardsticks" / "posthoc_sem" / "rows"
    yard = {
        p.stem.removeprefix("yardstick-"): _load(p) for p in sorted(ydir.glob("*.json"))
    }
    out: Row = {"validity": validity(meta, yard, boot)}
    print("## 有効性の確かめ\n")
    for k, v in out["validity"].items():
        if k.startswith("V-"):
            print(f"- {k}: {'通る' if v['pass'] else '外れる'}")
    lv = out["validity"]["L"]
    print(
        f"- L = {lv['L']}（tau = {_fmt(lv['tau'], 2)}、上限に当たった: {lv['at_max']}）"
    )
    print(f"- α の格子の端に当たった物差し: {out['validity']['alpha_edge']}")
    print("- 目視: 記録を README で確かめる（このスクリプトは読まない）\n")
    print("| 物差し | m_sel | m_final | 半分 0 | 半分 1 |\n| -- | -- | -- | -- | -- |")
    for name, r in yard.items():
        half = r.get("m_sel_half", [None, None])
        print(
            f"| {name} | {_fmt(r['m_sel'])} | {_fmt(r['m_final'])} | {_fmt(half[0])} | {_fmt(half[1])} |"
        )
    out["choice"] = choose(runs, boot, yard)
    ch = out["choice"]
    print(
        f"\n## 長い run の選び方\n\nsigma_seed = {_fmt(ch['sigma_seed'])}"
        f"（自由度 {ch['df']}、選び方の 2 の設定 {ch['step2']}）"
    )
    print(f"- 候補から外した設定（落ちた seed）: {ch['removed']}")
    if "stopped" in ch:
        print(f"- 止まった: {ch['stopped']}")
    if "V-m5" in out["choice"]:
        v = out["choice"]["V-m5"]
        print(
            f"- V-m5: pca1024 {_fmt(v['pca1024'])} ≥ 10 δ_1 = {_fmt(10 * v['delta_1'])}: {v['pass']}"
        )
    for tag, c in out["choice"]["lr"].items():
        keys = ("seed0_top", "near", "to_add", "stopped", "best", "tied", "chosen")
        print(f"- {tag}: {json.dumps({k: c.get(k) for k in keys})}")
    out["long"] = long_runs(runs, out["choice"]["lr"], boot)
    print(f"\n## 長い run\n\n```\n{json.dumps(out['long'], indent=1)}\n```")
    out["diagnosis"] = diagnosis(runs)
    out["e1"] = e1_repair(runs, out["choice"]["lr"], boot, out["diagnosis"])
    d = out["diagnosis"]
    print(
        f"\n## E1 の修理\n\n- H-bug {d['H-bug']}・H-time {d['H-time']}・H-skew {d['H-skew']}"
    )
    for k in ("V-E1'1", "V-E1'2", "V-E1'3", "V-E1'4"):
        print(f"- {k}: {out['e1'][k]['pass']}")
    print(f"- C1': {out['e1']["C1'"]}\n- C4': {out['e1']["C4'"]}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
