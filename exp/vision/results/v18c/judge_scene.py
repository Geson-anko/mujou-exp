r"""The judgement of V18c's main measure of the scene categories and of the
long runs (exp/vision/README.md, 「V18c: 場面のカテゴリで主の指標を作る（回す前
に決めたこと、2026-11-25）」; the design in exp/vision/plans/jepa.md, 「V18c:
主の指標（場面のカテゴリ、2026-11-25）」), the rules applied as registered.

Reads ``runs/v18c-scene/``: ``labels.json`` (the labels' cache's: the kept
labels ``t`` of ``full`` and the quarters, the videos, $L$, $L_{\max}$, V-c1
and V-E1'5), ``places_check.json`` (``python -m exp.lib.places365 check
--out``), ``audit/key.json`` and ``audit/answers.json`` (the judge's: number
-> a list of 1 or 2 group names, or ``"判定不能"``), ``yardsticks/rows`` and
``checks/rows``; and of each long run ``runs/v18c-long/<run>/``
``posthoc_scene/rows`` (``z``), ``posthoc_scene_fill/rows`` (E1f),
``posthoc_fill_sem/rows`` (the frozen E1, V-E1'1) and ``posthoc_fill/rows``
(``e1diag``, V-E1'2); L1 and its premise from ``judge_long``'s JSON
(``--long``).

$\mathrm{SE}^*$ of a contrast is $\max(\mathrm{SE}_{\mathrm{blk}}(L),
\mathrm{SE}_{\mathrm{blk}}(2L))$ (``metrics.scene_bootstrap``, ``BOOT``
resamples from ``BOOT_SEED`` each time, so every contrast has the same
draws; $2L$ cut at $L_{\max}$). The jackknife's values of a code without
video $u$ are ``scene_m`` of its ``jack_pred[u]`` on the other videos
(classes of 10 rows or more). A comparison $d$ against a threshold $\delta$
reads (README 「比べの 3 つの読み」): ``above`` ($d > \delta$ and $d_{(u)} >
0$ for every $u$), ``below`` ($d < -\delta$ and every $d_{(u)} < 0$),
``excl`` ($d + \delta < \Delta_*$, alongside ``below`` possibly), else
``undecided``. Prints markdown; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_scene [--runs runs] \
        [--long exp/vision/results/v18c/judge_long.json] [--json out.json]
"""

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from exp.lib.metrics import contiguous_blocks, scene_bootstrap, scene_resamples
from exp.lib.readout import scene_m
from exp.vision.results.v18c.judge_screen import run_name
from exp.vision.scene_eval import BOOT, BOOT_SEED, DELTA_STAR, N_RAND, YARD_PREFIX

Row = dict[str, Any]

LRS = (1e-4, 1e-5)
LAMS = (0.5, 0.8, 0.95)
SEEDS = (0, 1)
R = 0.5
SHORT, LONG = 6000, 60000
EXACT = 1e-12  # V-c3
K3_TOL = 1e-6  # V-E1'3
RESID = 1e-9  # V-E1'2
AGREE, UNSURE = 0.7, 0.2  # V-c2, V-E1'6
UNSURE_LABEL = "判定不能"
LISTED = ("init_c", "pca1024", "color8", "v16", "inc64")  # set against the runs
LONG_JSON = Path(__file__).with_name("judge_long.json")


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


class Scene:
    """The labels, the blocks and the bootstrap of the judgement."""

    def __init__(self, labels: Row) -> None:
        self.meta = labels
        self.video = torch.tensor(labels["video"])
        self.t = torch.tensor(labels["full"]["t"])
        self.k = labels["full"]["K"]
        self.quads = [
            torch.tensor(labels[f"quad/{q}"]["t"]) for q in "tl tr bl br".split()
        ]
        self.length = labels["full"]["L"]
        self.l2 = min(2 * self.length, labels["full"]["L_max"])
        self.blocks = {
            ell: contiguous_blocks(self.video, ell) for ell in (self.length, self.l2)
        }
        self.held = labels["videos"].index(labels["held_out"])

    def se(
        self, preds: Sequence[Tensor], coef: Sequence[float], *, quad: bool = False
    ) -> float:
        r"""$\mathrm{SE}^*$ of $\sum_c a_c m_c$ of the predictions ``preds``
        (each ``(N,)``, or ``(4, N)`` of the quarters with ``quad``)."""
        p = torch.stack(list(preds))
        lab = torch.stack(self.quads) if quad else self.t
        c = torch.tensor([list(coef)], dtype=torch.float64)
        out = []
        for blocks in self.blocks.values():
            gen = torch.Generator().manual_seed(BOOT_SEED)
            out.append(
                float(scene_bootstrap(p, lab, self.video, blocks, c, BOOT, gen)[0])
            )
        return max(out)

    def resamples(self, preds: Sequence[Tensor], y: Tensor, ell: int) -> Tensor:
        """The resampled $m$ ``(BOOT, C)`` at the block length ``ell``."""
        gen = torch.Generator().manual_seed(BOOT_SEED)
        p = torch.stack(list(preds))
        return scene_resamples(p, y, self.video, self.blocks[ell], BOOT, gen)

    def jack(self, jack_pred: Tensor, *, quad: bool = False) -> list[float]:
        """The values without each video $u$ of a code's ``jack_pred``
        ``(U, N)`` (or ``(4, U, N)`` of the quarters, averaged)."""
        if quad:
            vals = [self._jack(jack_pred[i], y) for i, y in enumerate(self.quads)]
            return [sum(v[u] for v in vals) / len(vals) for u in range(len(vals[0]))]
        return self._jack(jack_pred, self.t)

    def _jack(self, jp: Tensor, y: Tensor) -> list[float]:
        out = []
        for u in range(jp.shape[0]):
            on = (self.video != u) & (y >= 0)
            out.append(scene_m(jp[u][on], y[on], self.video[on], int(y.max()) + 1,
                               min_count=10))  # fmt: skip
        return out


def reading(d: float, delta: float, d_jack: Sequence[float] | None) -> list[str]:
    """The readings of a comparison (module docstring)."""
    out = []
    jack = list(d_jack or [])
    if d > delta and jack and all(x > 0 for x in jack):
        out.append("above")
    if d < -delta and jack and all(x < 0 for x in jack):
        out.append("below")
    if d + delta < DELTA_STAR:
        out.append("excl")
    return out or ["undecided"]


def _pred(row: Row) -> Tensor:
    return torch.tensor(row["pred"])


def _jp(row: Row) -> Tensor | None:
    return torch.tensor(row["jack_pred"]) if "jack_pred" in row else None


def compare(sc: Scene, a: Row, b: Row | None, factor: float = 2.0,
            extra_var: float = 0.0) -> Row:  # fmt: skip
    r"""$d = m_a - m_b$ (``b`` None: 0), its $\mathrm{SE}^*$, $\delta =
    \text{factor} \sqrt{\mathrm{SE}^{*2} + \text{extra\_var}}$, the
    jackknife's differences and the readings."""
    if b is None:
        d, se = a["m"], sc.se([_pred(a)], [1.0])
        ja = _jp(a)
        dj = None if ja is None else sc.jack(ja)
    else:
        d = a["m"] - b["m"]
        se = sc.se([_pred(a), _pred(b)], [1.0, -1.0])
        ja, jb = _jp(a), _jp(b)
        dj = None
        if ja is not None and jb is not None:
            dj = [x - y for x, y in zip(sc.jack(ja), sc.jack(jb))]
    delta = factor * math.sqrt(se**2 + extra_var)
    return {
        "d": d,
        "se": se,
        "delta": delta,
        "d_jack": dj,
        "read": reading(d, delta, dj),
    }


# -- the validity checks -----------------------------------------------------------------


def audit(key: list[Row] | None, answers: Row | None, quarter: bool) -> Row:
    """V-c2 (``quarter`` False) or V-E1'6: of each teacher's group, the share
    of the judge's answers holding it (the unsure left out), and the unsure's
    share."""
    if key is None or answers is None:
        return {"pass": None}
    items = [it for it in key if (it["quarter"] is not None) == quarter]
    by: dict[str, list[int]] = {}
    unsure = 0
    for it in items:
        given = answers.get(str(it["number"]))
        if given is None or given == UNSURE_LABEL:
            unsure += 1
            continue
        hit = it["teacher"] in given
        by.setdefault(it["teacher"], []).append(int(hit))
    agree = {g: sum(v) / len(v) for g, v in by.items()}
    share = unsure / len(items) if items else None
    ok = bool(agree) and min(agree.values()) >= AGREE and share is not None
    return {"agree": agree, "unsure": share, "pass": ok and (share or 0.0) <= UNSURE}


def validity(sc: Scene, yard: dict[str, Row], checks: dict[str, Row],
             places: Row | None, key: Any, answers: Any) -> Row:  # fmt: skip
    """V-c0 to V-c6 (README 「有効性の確かめ」)."""
    out: Row = {}
    out["V-c0"] = {
        "places365": None if places is None else places.get("passed"),
        "places_checked": sc.meta.get("places_checked"),
        "pass": bool(places and places.get("passed") and sc.meta.get("places_checked")),
    }
    out["V-c1"] = {
        "checks": sc.meta["full"].get("checks"),
        "pass": sc.meta["full"]["passed"],
    }
    out["V-c2"] = audit(key, answers, quarter=False)
    mean, label, video = (
        yard.get(f"{YARD_PREFIX}-{c}") for c in ("mean", "label", "video")
    )
    if mean and label and video:
        ok = (mean["m"] == 0.0 and abs(mean["ba"] - 1 / sc.k) <= EXACT
              and label["ba"] == 1.0 and abs(label["m"] - (1 - label["chance"])) <= EXACT
              and abs(video["m"]) <= EXACT)  # fmt: skip
        out["V-c3"] = {"mean": mean["m"], "label_ba": label["ba"], "video": video["m"],
                       "ceiling": label["m"], "pass": ok}  # fmt: skip
    else:
        out["V-c3"] = {"pass": None}
    out["V-c4"] = {}
    v4 = True
    for base in ("color", "inc64"):
        row = yard.get(f"{YARD_PREFIX}-{base}")
        ms = [checks[f"rand-{base}-{i:02d}"]["m"] for i in range(N_RAND)
              if f"rand-{base}-{i:02d}" in checks]  # fmt: skip
        if row is None or len(ms) < N_RAND:
            out["V-c4"][base] = None
            v4 = False
            continue
        se = sc.se([_pred(row)], [1.0])
        mean_m = sum(ms) / len(ms)
        sd = float(torch.tensor(ms).std())
        ok = abs(mean_m) <= 3 * se / math.sqrt(N_RAND) and 0.5 * se <= sd <= 2 * se
        out["V-c4"][base] = {"mean": mean_m, "sd": sd, "se": se, "pass": ok}
        v4 &= ok
    out["V-c4"]["pass"] = v4
    syn = sorted(
        (r for n, r in checks.items() if n.startswith("syn_")), key=lambda r: r["q"]
    )
    if len(syn) == 4:
        near = [abs(r["m"] - r["m_true"]) <= DELTA_STAR / 2 for r in syn]
        lo, hi = syn[1], syn[2]
        c = compare(sc, hi, lo)
        truth = hi["m_true"] - lo["m_true"]
        ok = all(near) and "above" in c["read"] and abs(c["d"] - truth) <= c["delta"]
        out["V-c5"] = {"rows": [{k: r[k] for k in ("q", "m", "m_true")} for r in syn],
                       "step": c, "truth_step": truth, "pass": ok}  # fmt: skip
    else:
        out["V-c5"] = {"pass": None}
    inc = yard.get(f"{YARD_PREFIX}-inc64")
    colors = [yard.get(f"{YARD_PREFIX}-{c}") for c in ("color", "color8")]
    if inc and all(colors):
        best = max((c for c in colors if c), key=lambda r: r["m"])
        d = inc["m"] - best["m"]
        se = sc.se([_pred(inc), _pred(best)], [1.0, -1.0])
        ji = sc.jack(torch.tensor(inc["jack_pred"]))
        jc = [sc.jack(torch.tensor(c["jack_pred"])) for c in colors if c]
        dj = [ji[u] - max(j[u] for j in jc) for u in range(len(ji))]
        ok = d - 2 * se >= 2 * DELTA_STAR and all(x > 0 for x in dj)
        out["V-c6"] = {"d": d, "delta_yy": 2 * se, "d_jack": dj, "pass": ok}
    else:
        out["V-c6"] = {"pass": None}
    out["valid"] = all(out[f"V-c{i}"]["pass"] is True for i in range(7))
    return out


# -- the long runs -------------------------------------------------------------------------


def _rows(run: Path, out: str) -> dict[str, Any]:
    d = run / out / "rows"
    return (
        {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}
        if d.is_dir()
        else {}
    )


type Key = tuple[float, float, int]


def long_runs(
    sc: Scene, runs: Path, l1: dict[str, Row]
) -> tuple[list[Row], dict[Key, dict[int, Row]]]:
    """L2, L3, the gain on Zurich and the other 8, and E1f's C1', C4' and
    V-E1'1 to V-E1'4 of each long run; and its ``z`` rows by step."""
    out = []
    zrows: dict[Key, dict[int, Row]] = {}
    for lr, lam, s in itertools.product(LRS, LAMS, SEEDS):
        name = run_name(lr, lam, R, s)
        rows = _rows(runs / "v18c-long" / name, "posthoc_scene")
        z = {int(k.split("-")[0]): r for k, r in rows.items() if k.endswith("-z")}
        zrows[(lr, lam, s)] = z
        row: Row = {"name": name, "lr": lr, "lam": lam, "seed": s,
                    "m": {str(k): v["m"] for k, v in sorted(z.items())},
                    "ba": {str(k): v["ba"] for k, v in sorted(z.items())},
                    "premise": l1.get(name, {}).get("premise"),
                    "L1": l1.get(name, {}).get("L1")}  # fmt: skip
        if all(k in z for k in (0, SHORT, LONG)):
            row["L2"] = compare(sc, z[LONG], z[SHORT])
            row["L3"] = compare(sc, z[LONG], z[0])
            row["seen"] = _seen(sc, z[LONG], z[SHORT])
        row["e1f"] = e1f(sc, runs / "v18c-long" / name)
        out.append(row)
    return out, zrows


def _seen(sc: Scene, a: Row, b: Row) -> Row:
    """The gain $\\Delta = m(60000) - m(6000)$ on the other 8 less that on
    Zurich, with its paired $2\\,\\mathrm{SE}^*$ (README 「学習で見た動画」)."""
    on_z = sc.video == sc.held
    sub = {
        "zurich": torch.where(on_z, sc.t, -1),
        "seen8": torch.where(~on_z, sc.t, -1),
    }
    d_z = (a["m_zurich"] or 0.0) - (b["m_zurich"] or 0.0)
    d_8 = (a["m_seen8"] or 0.0) - (b["m_seen8"] or 0.0)
    ses = []
    for ell in sc.blocks:
        res = {k: sc.resamples([_pred(a), _pred(b)], y, ell) for k, y in sub.items()}
        diff = (res["seen8"][:, 0] - res["seen8"][:, 1]) - (
            res["zurich"][:, 0] - res["zurich"][:, 1]
        )
        ses.append(float(diff.std()))
    se = max(ses)
    return {"gain_zurich": d_z, "gain_seen8": d_8, "d": d_8 - d_z, "2se": 2 * se,
            "seen_only": d_8 - d_z > 2 * se}  # fmt: skip


def _quarter_preds(row: Row, key: str) -> Tensor:
    return torch.stack(
        [torch.tensor(row["quarters"][q][key]) for q in "tl tr bl br".split()]
    )


def e1f(sc: Scene, run: Path) -> Row:
    """E1f's checks and readings of a run (README 「``e1f``」)."""
    rows = _rows(run, "posthoc_scene_fill")
    by_step = {int(k.split("-")[0]): r["rows"] for k, r in rows.items()}
    out: Row = {"steps": sorted(by_step)}
    v3 = True
    for got in by_step.values():
        by = {(x["k"], x["stage"]): x for x in got}
        for k in (4, 8, 12):
            v3 &= by[(k, "mean")]["m"] == 0.0
            v3 &= abs(by[(k, "pred_k3")]["m"] - by[(k, "pool")]["m"]) <= K3_TOL
    out["V-E1'3"] = v3 if by_step else None
    fill_sem = _rows(run, "posthoc_fill_sem")
    e1 = [r for k, rs in fill_sem.items() if k.endswith("-e1") for r in rs]
    out["V-E1'1"] = all(
        r["swap"] == 0.5 and r["n_tie"] == r["n_cand"]
        for r in e1 if r["mask"] == "S" and r.get("target") == "color"
        and r["stage"] in ("mean", "oracle_k3", "pred_k3")
    ) if e1 else None  # fmt: skip
    diag = [
        r
        for k, rs in _rows(run, "posthoc_fill").items()
        if k.endswith("-e1diag")
        for r in rs
    ]
    out["V-E1'2"] = max(r["resid"] for r in diag) <= RESID if diag else None

    def m_of(step: int, k: int, stage: str) -> Row:
        x = {(r["k"], r["stage"]): r for r in by_step[step]}[(k, stage)]
        return {"m": x["m"], "pred": _quarter_preds(x, "pred"),
                "jack": _quarter_preds(x, "jack_pred") if "jack_pred" in x["quarters"]["tl"] else None}  # fmt: skip

    def cmp(terms: Sequence[tuple[Row, float]]) -> Row:
        d = sum(c * t["m"] for t, c in terms)
        se = sc.se([t["pred"] for t, _ in terms], [c for _, c in terms], quad=True)
        jacks = [sc.jack(t["jack"], quad=True) for t, _ in terms]
        dj = [
            sum(c * j[u] for (_, c), j in zip(terms, jacks))
            for u in range(len(jacks[0]))
        ]
        return {
            "d": d,
            "se": se,
            "delta": 2 * se,
            "d_jack": dj,
            "read": reading(d, 2 * se, dj),
        }

    if 0 in by_step:
        out["V-E1'4"] = cmp([(m_of(0, 4, "oracle"), 1.0)])
    if SHORT in by_step and LONG in by_step:
        p60, q60, c60 = (m_of(LONG, 8, s) for s in ("pred", "pool", "vis_color"))
        p6, q6 = (m_of(SHORT, 8, s) for s in ("pred", "pool"))
        out["C1_pool"] = cmp([(p60, 1.0), (q60, -1.0)])
        out["C1_color"] = cmp([(p60, 1.0), (c60, -1.0)])
        out["C4"] = cmp([(p60, 1.0), (q60, -1.0), (p6, -1.0), (q6, 1.0)])
        conds = {q: x["quarters"][q].get("cond") for x in by_step[LONG]
                 if (x["k"], x["stage"]) == (8, "pool") for q in "tl tr bl br".split()}  # fmt: skip
        out["cond_pool_60000"] = conds
    return out


def settings(
    sc: Scene, rows: list[Row], zrows: dict[Key, dict[int, Row]], yard: dict[str, Row]
) -> Row:
    """The readings of each setting, $\\sigma_{\\mathrm{seed}}$, the
    comparisons of $\\lambda$ and of the learning rates, and the yardsticks
    set against the runs (README 「設定ごとの読み」「学習率ごとの読み」)."""
    by = {(r["lr"], r["lam"], r["seed"]): r for r in rows}
    pairs = [(by.get((lr, lam, 0)), by.get((lr, lam, 1))) for lr in LRS for lam in LAMS]
    var = [(a["m"][str(LONG)] - b["m"][str(LONG)]) ** 2 / 2 for a, b in pairs
           if a and b and str(LONG) in a["m"] and str(LONG) in b["m"]]  # fmt: skip
    s2 = sum(var) / len(var) if var else None
    s2_short = [(a["m"][str(SHORT)] - b["m"][str(SHORT)]) ** 2 / 2 for a, b in pairs
                if a and b and str(SHORT) in a["m"] and str(SHORT) in b["m"]]  # fmt: skip
    out: Row = {"sigma_seed": None if s2 is None else math.sqrt(s2), "dof": len(var),
                "sigma_seed_6000": math.sqrt(sum(s2_short) / len(s2_short)) if s2_short else None}  # fmt: skip
    per: list[Row] = []
    for lr in LRS:
        for lam in LAMS:
            a, b = by.get((lr, lam, 0)), by.get((lr, lam, 1))
            if not (a and b and "L2" in a and "L2" in b):
                per.append({"lr": lr, "lam": lam, "verdict": None})
                continue
            # a run whose premise fails (or is unknown) is not judged
            judged = [x["premise"] is True for x in (a, b)]
            full = [ok and x["L1"] is True and "above" in x["L2"]["read"]
                    and "above" in x["L3"]["read"] for ok, x in zip(judged, (a, b))]  # fmt: skip
            refuted = all(
                ok and "excl" in x["L2"]["read"] for ok, x in zip(judged, (a, b))
            )
            verdict = ("supported" if all(full) else "refuted" if refuted
                       else "one seed" if any(full) else "neither")  # fmt: skip
            per.append({"lr": lr, "lam": lam, "judged": judged, "seeds_full": full,
                        "verdict": verdict})  # fmt: skip
    out["per_setting"] = per
    out["v19"] = any(p.get("verdict") == "supported" for p in per)
    if s2 is None:
        return out
    z = zrows

    def mean_cmp(x: Sequence[tuple[Row, float]], extra: float) -> Row:
        d = sum(c * r["m"] for r, c in x)
        se = sc.se([_pred(r) for r, _ in x], [c for _, c in x])
        jps = [_jp(r) for r, _ in x]
        jacks = [None if j is None else sc.jack(j) for j in jps]
        dj = None
        if all(j is not None for j in jacks):
            dj = [
                sum(c * j[u] for (_, c), j in zip(x, jacks) if j)
                for u in range(len(jacks[0] or []))
            ]
        delta = 2 * math.sqrt(se**2 + extra)
        return {
            "d": d,
            "se": se,
            "delta": delta,
            "d_jack": dj,
            "read": reading(d, delta, dj),
        }

    lam_cmp = []
    for lr in LRS:
        for l1, l2 in itertools.combinations(LAMS, 2):
            keys = [(lr, l1, s) for s in SEEDS] + [(lr, l2, s) for s in SEEDS]
            if not all(k in z and z[k] for k in keys):
                continue
            lvl = [(z[(lr, l2, s)][LONG], 0.5) for s in SEEDS]
            lvl += [(z[(lr, l1, s)][LONG], -0.5) for s in SEEDS]
            gain = [(z[(lr, l2, s)][LONG], 0.5) for s in SEEDS]
            gain += [(z[(lr, l2, s)][SHORT], -0.5) for s in SEEDS]
            gain += [(z[(lr, l1, s)][LONG], -0.5) for s in SEEDS]
            gain += [(z[(lr, l1, s)][SHORT], 0.5) for s in SEEDS]
            signs = [
                (z[(lr, l2, s)][LONG]["m"] - z[(lr, l1, s)][LONG]["m"]) > 0
                for s in SEEDS
            ]
            lam_cmp.append({"lr": lr, "pair": [l2, l1], "level": mean_cmp(lvl, s2),
                            "gain": mean_cmp(gain, s2), "seeds_agree": len(set(signs)) == 1})  # fmt: skip
    out["lambda"] = lam_cmp
    lr_cmp = []
    for lam in LAMS:
        keys = [(lr, lam, s) for lr in LRS for s in SEEDS]
        if not all(k in z and z[k] for k in keys):
            continue
        a, b = LRS
        for ref, tag in ((SHORT, "L2"), (0, "L3")):
            terms = [(z[(a, lam, s)][LONG], 0.5) for s in SEEDS]
            terms += [(z[(a, lam, s)][ref], -0.5) for s in SEEDS]
            terms += [(z[(b, lam, s)][LONG], -0.5) for s in SEEDS]
            terms += [(z[(b, lam, s)][ref], 0.5) for s in SEEDS]
            lr_cmp.append({"lam": lam, "gain": tag, "cmp": mean_cmp(terms, s2)})
    out["lr"] = lr_cmp
    listed = []
    for lr in LRS:
        for lam in LAMS:
            keys = [(lr, lam, s) for s in SEEDS]
            if not all(k in z and z[k] for k in keys):
                continue
            for code in LISTED:
                ys = [r for n, r in yard.items() if n == f"{YARD_PREFIX}-{code}"
                      or n.startswith(f"{YARD_PREFIX}-{code}-")]  # fmt: skip
                if not ys:
                    continue
                terms = [(z[k][LONG], 0.5) for k in keys] + [(ys[0], -1.0)]
                c = mean_cmp(terms, s2 / 2)
                listed.append(
                    {
                        "lr": lr,
                        "lam": lam,
                        "code": code,
                        "d": c["d"],
                        "delta_y": c["delta"],
                    }
                )
    out["yardsticks"] = listed
    return out


def main(argv: Sequence[str] | None = None) -> Row:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--long", type=Path, default=LONG_JSON, help="judge_long's JSON")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    base = args.runs / "v18c-scene"
    sc = Scene(json.loads((base / "labels.json").read_text()))
    yard = {p.stem: json.loads(p.read_text())
            for p in sorted((base / "yardsticks" / "rows").glob("*.json"))}  # fmt: skip
    checks = {p.stem: json.loads(p.read_text())
              for p in sorted((base / "checks" / "rows").glob("*.json"))}  # fmt: skip
    v = validity(sc, yard, checks, _load(base / "places_check.json"),
                 _load(base / "audit" / "key.json"), _load(base / "audit" / "answers.json"))  # fmt: skip
    v["V-E1'5"] = all(sc.meta[f"quad/{q}"]["passed"] for q in "tl tr bl br".split())
    v["V-E1'6"] = audit(_load(base / "audit" / "key.json"),
                        _load(base / "audit" / "answers.json"), quarter=True)  # fmt: skip
    result: Row = {"validity": v, "yardsticks": {n: {k: r.get(k) for k in ("m", "ba", "chance", "m_zurich", "const_folds")} for n, r in yard.items()}}  # fmt: skip
    _print_validity(v)
    if not v["valid"]:
        print("\n有効性の確かめが外れた: m を判定に使わない（README）。")
    else:
        long = _load(args.long) or {"runs": []}
        l1 = {r["name"]: r for r in long["runs"]}
        rows, zrows = long_runs(sc, args.runs, l1)
        result["runs"] = rows
        result["settings"] = settings(sc, rows, zrows, yard)
        _print_runs(rows, result["settings"])
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    return result


def _print_validity(v: Row) -> None:
    print("## 有効性の確かめ\n")
    print("| 名前 | 合否 |")
    print("| -- | -- |")
    for k in (*(f"V-c{i}" for i in range(7)), "V-E1'5", "V-E1'6"):
        x = v.get(k)
        ok = x.get("pass") if isinstance(x, dict) else x
        print(f"| {k} | {ok} |")


def _print_runs(rows: list[Row], st: Row) -> None:
    print("\n## 長い run（L1〜L3）\n")
    print("| run | premise | L1 | m(0) | m(6000) | m(60000) | L2 | L3 |")
    print("| -- | -- | -- | -- | -- | -- | -- | -- |")
    for r in rows:
        m = r["m"]
        cells = [f"{m[s]:.4f}" if s in m else "—" for s in ("0", str(SHORT), str(LONG))]
        l2 = ",".join(r["L2"]["read"]) if "L2" in r else "—"
        l3 = ",".join(r["L3"]["read"]) if "L3" in r else "—"
        print(
            f"| {r['name']} | {r['premise']} | {r['L1']} | {' | '.join(cells)} | {l2} | {l3} |"
        )
    print(
        f"\nσ_seed = {st['sigma_seed']}（自由度 {st['dof']}）、V19 に進める: {st['v19']}\n"
    )
    for p in st["per_setting"]:
        print(f"- lr {p['lr']:g}、λ {p['lam']:g}: {p['verdict']}")


if __name__ == "__main__":
    main()
