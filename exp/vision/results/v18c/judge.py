r"""The judgement of V18c's evaluation (exp/vision/README.md, the sections
"V18c: 生成と Inception で JEPA の符号を測り直す" and "V18c: 未観測の部分の評価").

Reads the rows of G (``<run>/posthoc_gen/rows``, the yardsticks and ``refs.json``
in ``runs/v18c-gen/yardsticks/posthoc_gen``), of E1, E1e and E1g
(``<run>/posthoc_fill/rows``, the yardsticks in ``runs/v18c-fill/yardsticks``)
and of E2 (``<run>/posthoc_side/rows``), and applies the rules registered before
the runs, as they are:

- G: the validity checks V-G1 to V-G3, the decoder's and the seed's spread
  ($\sigma_{\mathrm{dec}}$ from the 3 pairs of decoder seeds, $\sigma_{\mathrm{seed}}$
  from the 16 pairs of run seeds), the thresholds $\delta_w$, $\delta_y$,
  $\delta_a$, questions 1 to 3, P16 against F, M, and the reading of the two
  views (the user's: the distinguishing information stays; Codex's: the
  hard-to-predict part was dropped).
- E1: the validity checks V-E1 to V-E3, the four stages and ``gain``, the
  seed spread $\sigma_{\mathrm{seed},E}$ from 8 pairs, and Fable's conditions
  1 to 4 per arm; E1e and E2 against their yardsticks; E1g listed.

The pair standard error $\mathrm{SE}_{\mathrm{pair}} = \mathrm{sd}_i(h^A_i -
h^B_i)/\sqrt{n}$ takes the per-candidate values of the rows (``hits``,
``hard_hits``, ``feat_i``, ``hits_easy``, ``hits_hard``, ``swap_i``). ``r2``'s
comes from the per-candidate squared errors ``sq_i``: with the coordinates'
total sums of squares taken equal, $r_2 = 1 - \sum_i \mathrm{sq}_i / S$, so
$S = \sum_i \mathrm{sq}_i / (1 - r_2)$ of the reference row B and
$\mathrm{SE}(\Delta r_2) = \sqrt{n}\, \mathrm{sd}_i(\Delta \mathrm{sq}_i) / S$
(an approximation; the rows do not keep the coordinates' sums). FID and
``top5`` have no per-candidate values, so their SE is 0. Prints markdown;
``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

Row = dict[str, Any]

N_CAND = 512
CHANCE = 1 / N_CAND
STEPS = (0, 6000, 20000, 60000)
LONG = 60000
SEEDS = (0, 1)
ARMS = {
    "p16": "v18-long/v18-long-lr1e-4-lam0.5-r0.25-s{s}",
    "f": "v18b-long/v18b-long-f-lr1e-4-lam0.5-r0.25-s{s}",
}
M_RUN = "v18b-long/v18b-long-m-lr3e-5-lam0.8-r0.25-s0"
R_RUNS = {
    0.5: "v18-long/v18-long-lr1e-4-lam0.5-r0.5-s0",
    0.75: "v18-long/v18-long-lr1e-4-lam0.5-r0.75-s0",
}
CODES = ("z", "z_w")
YARDSTICKS = ("color", "pca", "v16", "v17", "init_m", "uncond")
# the decoder-seed pairs of sigma_dec: (run, step, code)
DEC_PAIRS = (("p16", 0, 0, "z"), ("p16", 0, LONG, "z"), ("f", 0, LONG, "z_w"))
INTERVALS = ((6000, LONG), (0, 6000), (6000, 20000), (20000, LONG))


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _mean(x: Sequence[float]) -> float:
    return sum(x) / len(x)


def _sd(x: Sequence[float]) -> float:
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def _avg(vs: Sequence[Sequence[float]]) -> list[float]:
    return [_mean(c) for c in zip(*vs, strict=True)]


def _fmt(x: float | None, d: int = 4) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:+.{d}f}" if x < 0 else f"{x:.{d}f}"


def classify(d: Sequence[float], thr: Sequence[float]) -> str:
    """The rule of question 1 over the seeds: ``up`` if every seed's difference
    passes its threshold upward, ``down`` downward, ``same`` if every one is
    within, else ``split``."""
    if all(x > t for x, t in zip(d, thr, strict=True)):
        return "up"
    if all(x < -t for x, t in zip(d, thr, strict=True)):
        return "down"
    if all(abs(x) <= t for x, t in zip(d, thr, strict=True)):
        return "same"
    return "split"


# -- G -------------------------------------------------------------------------------


class GMetric:
    """A metric of G: its value, its per-candidate values (or none)."""

    def __init__(self, name: str, per: Callable[[Row], list[float]] | None) -> None:
        self.name = name
        self.per = per

    def value(self, row: Row) -> float:
        return float(row[self.name])


def _feat_ratio_i(row: Row) -> list[float]:
    den = row["feat_dist"] / row["feat_ratio"]
    return [v / den for v in row["feat_i"]]


G_METRICS = {
    "top1": GMetric("top1", lambda r: r["hits"]),
    "top5": GMetric("top5", None),
    "hard_top1": GMetric("hard_top1", lambda r: r["hard_hits"]),
    "fid": GMetric("fid", None),
    "feat_ratio": GMetric("feat_ratio", _feat_ratio_i),
}


class G:
    """G's rows: ``rows[(arm, seed, step, code, dec)]`` and the yardsticks."""

    def __init__(self, runs: Path) -> None:
        self.rows: dict[tuple[str, int, int, str, int], Row] = {}
        for arm, pattern in ARMS.items():
            for s in SEEDS:
                self._read(runs / pattern.format(s=s) / "posthoc_gen/rows", arm, s)
        self._read(runs / M_RUN / "posthoc_gen/rows", "m", 0)
        ys = runs / "v18c-gen/yardsticks/posthoc_gen"
        self.yard = {c: _load(ys / f"rows/yardstick-{c}-d0.json") for c in YARDSTICKS}
        self.refs = _load(ys / "refs.json")
        self.sig_dec = {m: self._sigma_dec(m) for m in G_METRICS}
        self.sig_seed = {m: self._sigma_seed(m) for m in G_METRICS}

    def _read(self, d: Path, arm: str, seed: int) -> None:
        for p in sorted(d.glob("*.json")):
            r = _load(p)
            key = (arm, seed, int(r["step"]), r["code"], int(r["dec_seed"]))
            self.rows[key] = r

    def row(self, arm: str, seed: int, step: int, code: str, dec: int = 0) -> Row:
        return self.rows[(arm, seed, step, code, dec)]

    def _sigma_dec(self, m: str) -> float:
        met = G_METRICS[m]
        d = [
            met.value(self.row(a, s, t, c, 1)) - met.value(self.row(a, s, t, c, 0))
            for a, s, t, c in DEC_PAIRS
        ]
        return math.sqrt(_mean([x * x / 2 for x in d]))

    def _sigma_seed(self, m: str) -> float:
        met = G_METRICS[m]
        d = [
            met.value(self.row(a, 1, t, c)) - met.value(self.row(a, 0, t, c))
            for a in ARMS
            for c in CODES
            for t in STEPS
        ]
        return math.sqrt(_mean([x * x / 2 for x in d]))

    @staticmethod
    def se_pair(m: str, a: Sequence[Row], b: Sequence[Row]) -> float:
        """SE of the difference of the means over ``a`` and over ``b`` (their
        per-candidate values averaged first)."""
        per = G_METRICS[m].per
        if per is None:
            return 0.0
        ha, hb = _avg([per(r) for r in a]), _avg([per(r) for r in b])
        diff = [x - y for x, y in zip(ha, hb, strict=True)]
        return _sd(diff) / math.sqrt(len(diff))

    def delta_w(self, m: str, a: Row, b: Row) -> float:
        se = self.se_pair(m, [a], [b])
        return 2 * math.sqrt(se**2 + 2 * self.sig_dec[m] ** 2)

    def delta_y(self, m: str, a: Sequence[Row], b: Row) -> float:
        se = self.se_pair(m, a, [b])
        return 2 * math.sqrt(se**2 + self.sig_seed[m] ** 2 / 2 + self.sig_dec[m] ** 2)

    def delta_a(self, m: str, a: Sequence[Row], b: Sequence[Row]) -> float:
        se = self.se_pair(m, a, b)
        return 2 * math.sqrt(se**2 + self.sig_seed[m] ** 2)

    # rules

    def within(self, m: str, arm: str, code: str, t0: int, t1: int) -> dict[str, Any]:
        """Seed by seed, ``m``(t1) - ``m``(t0) of one run against delta_w."""
        met = G_METRICS[m]
        d, thr = [], []
        for s in SEEDS:
            a, b = self.row(arm, s, t1, code), self.row(arm, s, t0, code)
            d.append(met.value(a) - met.value(b))
            thr.append(self.delta_w(m, a, b))
        return {"d": d, "delta_w": thr, "class": classify(d, thr)}

    def code_gap(self, m: str, arm: str, step: int) -> dict[str, Any]:
        """Question 3: ``z_w`` - ``z`` per seed against delta_w."""
        met = G_METRICS[m]
        d, thr = [], []
        for s in SEEDS:
            a, b = self.row(arm, s, step, "z_w"), self.row(arm, s, step, "z")
            d.append(met.value(a) - met.value(b))
            thr.append(self.delta_w(m, a, b))
        return {"d": d, "delta_w": thr, "class": classify(d, thr)}

    def vs_yardstick(self, m: str, arm: str, code: str, y: str) -> dict[str, Any]:
        met = G_METRICS[m]
        a = [self.row(arm, s, LONG, code) for s in SEEDS]
        b = self.yard[y]
        d = _mean([met.value(r) for r in a]) - met.value(b)
        thr = self.delta_y(m, a, b)
        return {"d": d, "delta_y": thr, "class": classify([d], [thr])}

    def arms_gap(self, m: str, code: str) -> dict[str, Any]:
        met = G_METRICS[m]
        a = [self.row("f", s, LONG, code) for s in SEEDS]
        b = [self.row("p16", s, LONG, code) for s in SEEDS]
        d = _mean([met.value(r) for r in a]) - _mean([met.value(r) for r in b])
        thr = self.delta_a(m, a, b)
        return {"d": d, "delta_a": thr, "class": classify([d], [thr])}

    def above_uncond(self, arm: str, code: str) -> dict[str, Any]:
        u = self.yard["uncond"]
        d, thr = [], []
        for s in SEEDS:
            a = self.row(arm, s, LONG, code)
            d.append(a["top1"] - u["top1"])
            thr.append(self.delta_w("top1", a, u))
        return {"d": d, "delta_w": thr, "class": classify(d, thr)}

    def validity(self) -> dict[str, Any]:
        u = self.yard["uncond"]
        g2 = {
            y: {
                "d": self.yard[y]["top1"] - u["top1"],
                "delta_w": self.delta_w("top1", self.yard[y], u),
            }
            for y in ("pca", "v17")
        }
        rows = list(self.rows.values()) + list(self.yard.values())
        g3 = [
            r.get("code", "?")
            for r in rows
            if not (
                math.isfinite(r["fit_loss"]) and r["fit_loss"] < r["fit_loss_first"]
            )
        ]
        return {
            "V-G1": {
                "uncond_top1": u["top1"],
                "bound": 0.0059,
                "ok": u["top1"] <= 0.0059,
            },
            "V-G2": g2 | {"ok": all(v["d"] > v["delta_w"] for v in g2.values())},
            "V-G3": {"failed": g3, "ok": not g3},
        }

    def judge(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "refs": {k: v for k, v in self.refs.items() if isinstance(v, (int, float))},
            "sigma_dec": self.sig_dec,
            "sigma_seed": self.sig_seed,
            "validity": self.validity(),
        }
        q1: dict[str, Any] = {}
        q2: dict[str, Any] = {}
        q3: dict[str, Any] = {}
        yard: dict[str, Any] = {}
        for m in G_METRICS:
            for arm in ARMS:
                for code in CODES:
                    key = f"{m}/{arm}/{code}"
                    q1[key] = {t: self.within(m, arm, code, 0, t) for t in STEPS[1:]}
                    q2[key] = {
                        f"{a}-{b}": self.within(m, arm, code, a, b)
                        for a, b in INTERVALS
                    }
                    yard[key] = {
                        y: self.vs_yardstick(m, arm, code, y) for y in YARDSTICKS
                    }
                q3[f"{m}/{arm}"] = {t: self.code_gap(m, arm, t) for t in STEPS}
        out |= {"q1": q1, "q2": q2, "q3": q3, "yardsticks": yard}
        out["arms"] = {
            f"{m}/{c}": self.arms_gap(m, c) for m in G_METRICS for c in CODES
        }
        out["above_uncond"] = {
            f"{arm}/{c}": self.above_uncond(arm, c) for arm in ARMS for c in CODES
        }
        user = {
            c: all(out["above_uncond"][f"{a}/{c}"]["class"] == "up" for a in ARMS)
            and all(
                q2[f"top1/{a}/{c}"][f"6000-{LONG}"]["class"] != "down" for a in ARMS
            )
            for c in CODES
        }
        codex = all(
            q2[f"top1/{a}/z"][f"6000-{LONG}"]["class"] == "down"
            and q1[f"top1/{a}/z"][LONG]["class"] == "down"
            for a in ARMS
        )
        out["views"] = {"user": user, "codex": codex}
        u = self.yard["uncond"]
        out["overfit"] = {
            "uncond": u["fm_zurich"] - u["fit_loss"],
            "max_row": max(r["fm_zurich"] - r["fit_loss"] for r in self.rows.values()),
        }
        return out


# -- E1, E1e, E1g, E2 ----------------------------------------------------------------


def _r2_scale(row: Row) -> float:
    return sum(row["sq_i"]) / (1 - row["r2"])


PER_E: dict[str, str] = {
    "id_easy": "hits_easy",
    "id_hard": "hits_hard",
    "swap": "swap_i",
    "r2": "sq_i",
}


def se_e(m: str, a: Sequence[Row], b: Sequence[Row]) -> float:
    """SE of the difference of the means over the rows ``a`` and ``b`` of an
    E-metric (``r2`` by the approximation of the module docstring)."""
    key = PER_E[m]
    ha, hb = _avg([r[key] for r in a]), _avg([r[key] for r in b])
    diff = [x - y for x, y in zip(ha, hb, strict=True)]
    se = _sd(diff) / math.sqrt(len(diff))
    if m == "r2":
        return se * len(diff) / _mean([_r2_scale(r) for r in b])
    return se


class Fill:
    """The rows of E1 (``e1[(run, step)]``), E1e, E1g and E2 (``side``)."""

    def __init__(self, runs: Path) -> None:
        self.runs = runs
        self.names = (
            {
                (arm, s): pattern.format(s=s)
                for arm, pattern in ARMS.items()
                for s in SEEDS
            }
            | {("m", 0): M_RUN}
            | {(f"r{r}", 0): p for r, p in R_RUNS.items()}
        )
        self.part: dict[str, dict[tuple[str, int, int], list[Row]]] = {}
        for part in ("e1", "e1e", "e1g"):
            self.part[part] = {}
            for (arm, s), name in self.names.items():
                for p in sorted(
                    (runs / name / "posthoc_fill/rows").glob(f"*-{part}.json")
                ):
                    step = int(p.name.split("-")[0])
                    self.part[part][(arm, s, step)] = _load(p)
        self.side: dict[tuple[str, int, int, str], list[Row]] = {}
        for (arm, s), name in self.names.items():
            for p in sorted((runs / name / "posthoc_side/rows").glob("*.json")):
                step, code = p.stem.split("-", 1)
                self.side[(arm, s, int(step), code)] = _load(p)
        ys = runs / "v18c-fill/yardsticks"
        self.e1e_yard: list[Row] = _load(ys / "posthoc_fill/rows/yardstick-e1e.json")
        self.side_yard: dict[str, list[Row]] = {
            p.stem.removeprefix("yardstick-"): _load(p)
            for p in sorted((ys / "posthoc_side/rows").glob("*.json"))
        }

    # lookups

    def e1(
        self, arm: str, s: int, step: int, target: str, mask: str, stage: str
    ) -> Row:
        return next(
            r
            for r in self.part["e1"][(arm, s, step)]
            if r["target"] == target and r["mask"] == mask and r["stage"] == stage
        )

    def e1e(self, arm: str, s: int, step: int, target: str, mask: str) -> Row:
        return next(
            r
            for r in self.part["e1e"][(arm, s, step)]
            if r["target"] == target and r["mask"] == mask
        )

    def e1e_y(self, code: str, target: str, mask: str) -> Row:
        return next(
            r
            for r in self.e1e_yard
            if r["code"] == code and r["target"] == target and r["mask"] == mask
        )

    def e2(self, arm: str, s: int, step: int, code: str, target: str, band: str) -> Row:
        return next(
            r
            for r in self.side[(arm, s, step, code)]
            if r["target"] == target and r["band"] == band
        )

    def e2_y(self, code: str, target: str, band: str) -> Row:
        return next(
            r
            for r in self.side_yard[code]
            if r["target"] == target and r["band"] == band
        )

    # thresholds

    def sigma_seed(self, get: Callable[[str, int, int], Row], m: str) -> float:
        d = [get(arm, 1, t)[m] - get(arm, 0, t)[m] for arm in ARMS for t in STEPS]
        return math.sqrt(_mean([x * x / 2 for x in d]))

    def vs(
        self,
        m: str,
        a: Sequence[Row],
        b: Sequence[Row],
        sigma: float,
        mean_a: bool = True,
    ) -> dict[str, Any]:
        """``m`` of the mean of ``a`` against the mean of ``b`` with
        delta_Ey (``b`` a yardstick, sigma^2/2) or delta_Ea (both means)."""
        se = se_e(m, a, b)
        var = sigma**2 / 2 if mean_a and len(b) == 1 else sigma**2
        thr = 2 * math.sqrt(se**2 + var)
        d = _mean([r[m] for r in a]) - _mean([r[m] for r in b])
        return {"d": d, "delta": thr, "class": classify([d], [thr])}

    # E1

    def stage_table(self, arm: str, s: int, step: int, target: str, mask: str) -> Row:
        out: Row = {}
        for stage in ("oracle", "pred", "mean", "shuf", "interp"):
            r = self.e1(arm, s, step, target, mask, stage)
            out[stage] = {
                m: r[m] for m in ("r2", "id_easy", "id_hard", "pick_max_easy")
            }
        for stage in ("oracle", "pred", "mean", "shuf"):
            out[stage]["swap"] = self.e1(arm, s, step, target, "S", stage)["swap"]
        for m in ("r2", "id_easy", "id_hard", "swap"):
            o, p, mn = (out[x][m] for x in ("oracle", "pred", "mean"))
            out.setdefault("gain", {})[m] = (p - mn) / (o - mn) if o != mn else math.nan
        return out

    def stage_diff(
        self, arm: str, s: int, step: int, target: str, m: str, a: str, b: str
    ) -> dict[str, float]:
        mask = "S" if m == "swap" else "Q"
        ra = self.e1(arm, s, step, target, mask, a)
        rb = self.e1(arm, s, step, target, mask, b)
        d = ra[m] - rb[m]
        thr = 2 * se_e(m, [ra], [rb])
        return {"d": d, "delta_E": thr}

    def step_diff(
        self, arm: str, s: int, t0: int, t1: int, target: str, stage: str, m: str
    ) -> dict[str, float]:
        mask = "S" if m == "swap" else "Q"
        ra = self.e1(arm, s, t1, target, mask, stage)
        rb = self.e1(arm, s, t0, target, mask, stage)
        return {"d": ra[m] - rb[m], "delta_E": 2 * se_e(m, [ra], [rb])}

    def validity(self) -> dict[str, Any]:
        bad1 = []
        for (arm, s, step), _rows in self.part["e1"].items():
            for target in ("incep", "color"):
                for mask, m_n in (("Q", 4), ("H", 4), ("R", 1)):
                    r = self.e1(arm, s, step, target, mask, "shuf")
                    bound = CHANCE + 2 * math.sqrt(
                        CHANCE * (1 - CHANCE) / (N_CAND * m_n)
                    )
                    if r["id_easy"] > bound:
                        bad1.append(
                            (arm, s, step, target, mask, "id_easy", r["id_easy"])
                        )
                sh = self.e1(arm, s, step, target, "S", "shuf")
                se = _sd(sh["swap_i"]) / math.sqrt(N_CAND)
                if abs(sh["swap"] - 0.5) > 2 * se:
                    bad1.append((arm, s, step, target, "S", "swap", sh["swap"]))
                mn = self.e1(arm, s, step, target, "S", "mean")
                if mn["swap"] != 0.5:
                    bad1.append((arm, s, step, target, "S", "mean_swap", mn["swap"]))
        v2 = self.stage_diff("p16", 0, 0, "incep", "id_easy", "oracle", "mean")
        v3 = self.stage_diff("m", 0, LONG, "incep", "swap", "pred", "mean")
        v3c = self.stage_diff("m", 0, LONG, "color", "swap", "pred", "mean")
        return {
            "V-E1": {"failed": bad1, "ok": not bad1},
            "V-E2": v2 | {"ok": v2["d"] > v2["delta_E"]},
            "V-E3": v3 | {"ok": v3["d"] <= v3["delta_E"], "color": v3c},
        }

    def condition1(self, arm: str, target: str) -> dict[str, Any]:
        per = {
            m: [self.stage_diff(arm, s, LONG, target, m, "pred", "mean") for s in SEEDS]
            for m in ("id_easy", "id_hard", "swap")
        }
        ok = {m: all(x["d"] > x["delta_E"] for x in v) for m, v in per.items()}
        if all(ok.values()):
            verdict = "met"
        elif ok["id_easy"] and ok["id_hard"]:
            verdict = "undecided (swap out of distribution)"
        else:
            verdict = "not met"
        return {"per": per, "ok": ok, "verdict": verdict}

    def condition4(self, arm: str, target: str) -> dict[str, Any]:
        per = [
            self.step_diff(arm, s, 6000, LONG, target, "pred", "id_easy") for s in SEEDS
        ]
        return {"per": per, "met": all(x["d"] >= -x["delta_E"] for x in per)}

    def e1e_compare(self, arm: str, target: str) -> dict[str, Any]:
        """E1e (``Q``): the 60000 code's mean over the seeds against ``init``
        (the step-0 rows' mean), ``color_V``, ``pca_V``, ``ext`` and E1's
        ``interp``, in ``id_easy`` and ``r2`` (delta_Ey)."""
        a = [self.e1e(arm, s, LONG, target, "Q") for s in SEEDS]
        out: dict[str, Any] = {
            "value": {m: _mean([r[m] for r in a]) for m in ("r2", "id_easy")}
        }
        for m in ("r2", "id_easy"):
            sig = self.sigma_seed(lambda x, s, t: self.e1e(x, s, t, target, "Q"), m)
            refs = {
                "init": [self.e1e(arm, s, 0, target, "Q") for s in SEEDS],
                "color_V": [self.e1e_y("color_V", target, "Q")],
                "pca_V": [self.e1e_y("pca_V", target, "Q")],
                "ext": [self.e1e_y("ext", target, "Q")],
                "interp": [self.e1(arm, 0, LONG, target, "Q", "interp")],
            }
            for y, b in refs.items():
                out.setdefault(m, {})[y] = self.vs(m, a, b, sig)
                out[m][y]["ref"] = _mean([r[m] for r in b])
        return out

    def e2_compare(self, arm: str, code: str, target: str, band: str) -> dict[str, Any]:
        a = [self.e2(arm, s, LONG, code, target, band) for s in SEEDS]
        out: dict[str, Any] = {
            "value": {m: _mean([r[m] for r in a]) for m in ("r2", "id_easy", "id_hard")}
        }
        for m in ("r2", "id_easy", "id_hard"):
            sig = self.sigma_seed(
                lambda x, s, t: self.e2(x, s, t, code, target, band), m
            )
            refs = {"init": [self.e2(arm, s, 0, code, target, band) for s in SEEDS]} | {
                y: [self.e2_y(y, target, band)] for y in self.side_yard
            }
            for y, b in refs.items():
                out.setdefault(m, {})[y] = self.vs(m, a, b, sig)
                out[m][y]["ref"] = _mean([r[m] for r in b])
        return out

    @staticmethod
    def beats(cmp: dict[str, Any], ys: Sequence[str]) -> bool:
        return all(cmp[m][y]["class"] == "up" for m in ("r2", "id_easy") for y in ys)

    def conditions(self, arm: str, target: str) -> dict[str, Any]:
        c1 = self.condition1(arm, target)
        e1e = self.e1e_compare(arm, target)
        e2 = {
            (code, band): self.e2_compare(arm, code, target, band)
            for code in CODES
            for band in ("side", "outer")
        }
        c2_e1e = self.beats(e1e, ("init", "color_V", "pca_V"))
        c2_e2 = {
            code: self.beats(e2[(code, "side")], ("init", "color", "pca"))
            for code in CODES
        }
        c3_e1e = c2_e1e and self.beats(e1e, ("interp",))
        c3_e2 = {
            code: c2_e2[code]
            and self.beats(e2[(code, "side")], ("interp",))
            and self.beats(e2[(code, "outer")], ("init", "color", "pca"))
            for code in CODES
        }
        return {
            "c1": c1,
            "c2": {"e1e": c2_e1e, "e2": c2_e2, "met": c2_e1e or any(c2_e2.values())},
            "c3": {"e1e": c3_e1e, "e2": c3_e2, "met": c3_e1e or any(c3_e2.values())},
            "c4": self.condition4(arm, target),
            "e1e": e1e,
            "e2": {f"{c}/{b}": v for (c, b), v in e2.items()},
        }

    def judge(self) -> dict[str, Any]:
        stages = {
            f"{arm}/s{s}/{step}/{target}": self.stage_table(arm, s, step, target, "Q")
            for (arm, s, step) in self.part["e1"]
            for target in ("incep", "color")
        }
        in_dist = {
            f"r{r}/{step}/{target}/{mask}": {
                st: {
                    m: self.e1(f"r{r}", 0, step, target, mask, st)[m]
                    for m in ("r2", "id_easy", "id_hard")
                }
                for st in ("oracle", "pred", "mean", "interp")
            }
            for r, mask in ((0.5, "H"), (0.75, "R"))
            for step in (0, 6000, LONG)
            for target in ("incep", "color")
        }
        trend = {
            f"{arm}/s{s}/{target}/{stage}/{m}": self.step_diff(
                arm, s, 6000, LONG, target, stage, m
            )
            for arm in ARMS
            for s in SEEDS
            for target in ("incep", "color")
            for stage in ("oracle", "pred")
            for m in ("id_easy", "id_hard", "r2", "swap")
        }
        pred_vs_mean = {
            f"{arm}/s{s}/{step}/{target}/{m}": self.stage_diff(
                arm, s, step, target, m, "pred", "mean"
            )
            for (arm, s, step) in self.part["e1"]
            for target in ("incep", "color")
            for m in ("id_easy", "id_hard", "swap", "r2")
        }
        swap_r05 = {
            f"{step}/{target}/{st}": self.e1("r0.5", 0, step, target, "S", st)["swap"]
            for step in (0, 6000, LONG)
            for target in ("incep", "color")
            for st in ("oracle", "pred")
        }
        e1g = {
            f"{arm}/s{s}/{step}/{target}/{r['mask']}/{r['stage']}": {
                m: r[m] for m in ("id_easy", "id_hard", "vis_easy")
            }
            for (arm, s, step), rows in self.part["e1g"].items()
            for r in rows
            for target in ("incep", "color")
            if r["target"] == target and r["mask"] in ("Q_br", "R", "H_l")
        }
        return {
            "validity": self.validity(),
            "stages_Q": stages,
            "in_dist_other_r": in_dist,
            "pred_vs_mean": pred_vs_mean,
            "trend_6000_60000": trend,
            "swap_r0.5": swap_r05,
            "conditions": {
                f"{arm}/{t}": self.conditions(arm, t)
                for arm in ARMS
                for t in ("incep", "color")
            },
            "e1g": e1g,
        }


# -- printing --------------------------------------------------------------------------


def _print_g(g: G, res: dict[str, Any]) -> None:
    print("## G\n")
    print("refs:", res["refs"])
    print("sigma_dec:", {k: round(v, 4) for k, v in res["sigma_dec"].items()})
    print("sigma_seed:", {k: round(v, 4) for k, v in res["sigma_seed"].items()})
    print("validity:", json.dumps(res["validity"]))
    print()
    print(
        "| run | step | code | top1 | top5 | hard_top1 | pick_max | FID | feat_ratio |"
    )
    print("| -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    for (arm, s, step, code, dec), r in sorted(g.rows.items()):
        print(
            f"| {arm} s{s} | {step} | {code} d{dec} | {_fmt(r['top1'])} | "
            f"{_fmt(r['top5'], 3)} | {_fmt(r['hard_top1'], 3)} | "
            f"{_fmt(r['pick_max'], 3)} | {r['fid']:.1f} | {_fmt(r['feat_ratio'], 3)} |"
        )
    for y, r in g.yard.items():
        print(
            f"| {y} | — | — | {_fmt(r['top1'])} | {_fmt(r['top5'], 3)} | "
            f"{_fmt(r['hard_top1'], 3)} | {_fmt(r['pick_max'], 3)} | {r['fid']:.1f} | "
            f"{_fmt(r['feat_ratio'], 3)} |"
        )
    print()
    for name in ("q1", "q2", "q3", "yardsticks"):
        print(f"### {name}\n")
        for key, v in res[name].items():
            if not key.startswith(("top1/", "hard_top1/", "fid/", "feat_ratio/")):
                continue
            parts = []
            for k, x in v.items():
                d = x["d"]
                ds = "/".join(_fmt(e) for e in d) if isinstance(d, list) else _fmt(d)
                thr = x.get("delta_w", x.get("delta_y"))
                ts = (
                    "/".join(_fmt(e) for e in thr)
                    if isinstance(thr, list)
                    else _fmt(thr)
                )
                parts.append(f"{k}: {ds} (δ {ts}) {x['class']}")
            print(f"- {key}: " + "; ".join(parts))
        print()
    print(
        "arms (F - P16):",
        {
            k: (round(v["d"], 4), round(v["delta_a"], 4), v["class"])
            for k, v in res["arms"].items()
        },
    )
    print("above_uncond:", {k: v["class"] for k, v in res["above_uncond"].items()})
    print("views:", res["views"], "overfit:", res["overfit"])
    print()


def _print_fill(res: dict[str, Any]) -> None:
    print("## E1\n")
    print("validity:", json.dumps(res["validity"]))
    print()
    print(
        "| ckpt | target | oracle e/h/r2/swap | pred e/h/r2/swap | mean e/h | interp e/h/r2 | gain e/h/swap |"
    )
    print("| -- | -- | -- | -- | -- | -- | -- |")
    for key, t in res["stages_Q"].items():
        o, p, mn, it, gn = t["oracle"], t["pred"], t["mean"], t["interp"], t["gain"]
        print(
            f"| {key.rsplit('/', 1)[0]} | {key.rsplit('/', 1)[1]} | "
            f"{_fmt(o['id_easy'])}/{_fmt(o['id_hard'], 3)}/{_fmt(o['r2'], 3)}/{_fmt(o['swap'], 3)} | "
            f"{_fmt(p['id_easy'])}/{_fmt(p['id_hard'], 3)}/{_fmt(p['r2'], 3)}/{_fmt(p['swap'], 3)} | "
            f"{_fmt(mn['id_easy'])}/{_fmt(mn['id_hard'], 3)} | "
            f"{_fmt(it['id_easy'])}/{_fmt(it['id_hard'], 3)}/{_fmt(it['r2'], 3)} | "
            f"{_fmt(gn['id_easy'], 2)}/{_fmt(gn['id_hard'], 2)}/{_fmt(gn['swap'], 2)} |"
        )
    print()
    print("### pred - mean (60000)\n")
    for k, v in res["pred_vs_mean"].items():
        if "/60000/" in k:
            print(f"- {k}: {_fmt(v['d'])} (δ_E {_fmt(v['delta_E'])})")
    print("\n### 6000 -> 60000\n")
    for k, v in res["trend_6000_60000"].items():
        print(f"- {k}: {_fmt(v['d'])} (δ_E {_fmt(v['delta_E'])})")
    print("\n### other r, in distribution\n")
    for k, v in res["in_dist_other_r"].items():
        print(
            f"- {k}: "
            + "; ".join(
                f"{st} " + "/".join(_fmt(x, 3) for x in m.values())
                for st, m in v.items()
            )
        )
    print("\nswap r0.5:", {k: round(v, 3) for k, v in res["swap_r0.5"].items()})
    print("\n### conditions\n")
    for key, c in res["conditions"].items():
        print(
            f"- {key}: c1 {c['c1']['verdict']} {c['c1']['ok']}; c2 {c['c2']}; "
            f"c3 {c['c3']}; c4 {c['c4']['met']}"
        )
        for m in ("r2", "id_easy"):
            print(
                f"  - E1e {m} {_fmt(c['e1e']['value'][m])}: "
                + "; ".join(
                    f"{y} {_fmt(v['ref'])} ({_fmt(v['d'])}, δ {_fmt(v['delta'])}) {v['class']}"
                    for y, v in c["e1e"][m].items()
                )
            )
        for cb, e in c["e2"].items():
            for m in ("r2", "id_easy", "id_hard"):
                print(
                    f"  - E2 {cb} {m} {_fmt(e['value'][m])}: "
                    + "; ".join(
                        f"{y} {_fmt(v['ref'])} {v['class']}" for y, v in e[m].items()
                    )
                )
    print("\n### E1g\n")
    for k, v in res["e1g"].items():
        print(f"- {k}: " + " ".join(f"{m} {_fmt(x)}" for m, x in v.items()))


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()
    g = G(args.runs)
    res_g = g.judge()
    res_f = Fill(args.runs).judge()
    _print_g(g, res_g)
    _print_fill(res_f)
    if args.json is not None:
        args.json.write_text(
            json.dumps({"g": res_g, "fill": res_f}, indent=1, default=str) + "\n"
        )


if __name__ == "__main__":
    main()
