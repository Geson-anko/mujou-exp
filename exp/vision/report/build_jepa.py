"""Build ``exp/vision/report-jepa.html``, the report of V18, V18b and V18c (JEPA
with BP and i.i.d. minibatches), from the runs and the page template
``jepa.html.in`` next to this file.

The judged numbers come from the two ``judge.json`` of ``results/v18`` and
``results/v18b`` (the rules applied by their ``judge.py``). V18c's numbers come
from the JSON of ``results/v18c`` (``judge_screen``, ``judge_long``,
``judge_self``, ``judge``, ``judge_m_checks``, ``judge_scene``, ``judge_geom``,
``judge_width_yard``, ``judge_width_apply``, ``judge_content_real3`` and
``judge_content_models3``; ``judge_geom`` and
``judge_width_apply`` are not in git and are rebuilt by their commands in the
README), and G's
``top1`` from the rows its ``judge.py`` reads. The rest is read
from ``runs/``: the yardsticks (``v18-baselines``, ``v18b-baselines``), V18's
diagnosis (``v18-diag``: ``h_flat``/``h_mean`` and the Imagenette probe) and,
for P16 in V18b's figure, V18's long runs as ``results/v18b/figs.py`` reads
them. The zurich pictures of ``IMAGES`` are shrunk by ``ffmpeg`` and
embedded::

    uv run python -m exp.vision.report.build_jepa
"""

import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from exp.vision.report.build_eval_guide import geom_name
from exp.vision.report.build_optim import PLACEHOLDER, _jsonl, _r
from exp.vision.results.v18c.judge import ARMS as G_ARMS, G

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "exp" / "vision" / "results"
P16_LONG = ["v18-long-lr1e-4-lam0.5-r0.25-s0", "v18-long-lr1e-4-lam0.5-r0.25-s1"]
F0 = "v18b-long/v18b-long-f-lr1e-4-lam0.5-r0.25-s0"
M0 = "v18b-long/v18b-long-m-lr3e-5-lam0.8-r0.25-s0"
P0 = "v18-long/v18-long-lr1e-4-lam0.5-r0.25-s0"
STEPS_G = [0, 6000, 20000, 60000]
# key: the zurich picture under runs/ (top row: the true last frame, bottom
# row: the posthoc decoder's drawing). The stem is stacks / 10.
IMAGES = {
    "init": "v18-baselines/posthoc_jepa/init-zurich.png",
    "color": "v18-baselines/posthoc_jepa/color-zurich.png",
    "p16_6000": f"{P0}/posthoc_jepa/76800.0-zurich.png",
    "p16_60000": f"{P0}/posthoc_jepa/768000.0-zurich.png",
    "hflat_0": "v18-diag/img/r0.5-00000.0-h_flat-zurich.png",
    "hflat_60000": "v18-diag/img/r0.5-768000.0-h_flat-zurich.png",
    "z05_60000": "v18-diag/img/r0.5-768000.0-z-zurich.png",
    "init_f": "v18b-baselines/posthoc_jepa/init_f-zurich.png",
    "f_6000": f"{F0}/posthoc_jepa/76800.0-z-zurich.png",
    "f_60000": f"{F0}/posthoc_jepa/768000.0-z-zurich.png",
    "fpool_0": f"{F0}/posthoc_diag/00000.0-h_pool-zurich.png",
    "fpool_60000": f"{F0}/posthoc_diag/768000.0-h_pool-zurich.png",
    "init_m": "v18b-baselines/posthoc_jepa/init_m-zurich.png",
    "m_60000": f"{M0}/posthoc_jepa/768000.0-zpca-zurich.png",
}


def image(path: Path, scale: str = "iw/2") -> str:
    """The picture shrunk to ``scale`` wide, as a JPEG data URI."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x.jpg"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(path)]
            + ["-vf", f"scale={scale}:-1", "-q:v", "6", str(out)],
            check=True,
        )
        return "data:image/jpeg;base64," + base64.b64encode(out.read_bytes()).decode()


def yardsticks(runs: Path) -> dict[str, list[float | None]]:
    """``recon_blur`` and ``dyn_1/copy_1`` of each yardstick on zurich."""
    out = {}
    for b in ["v18-baselines", "v18b-baselines"]:
        for p in _jsonl(runs / b / "posthoc_jepa" / "probes.jsonl"):
            if p["group"] == "zurich":
                out[p["code"]] = [_r(p["recon_blur"]), _r(p["dyn_1/copy_1"])]
    return out


def status(row: dict[str, Any]) -> str:
    """The verdict of a screen run: collapse / pass / fail, or none for M
    (it has no usable pair for ``dyn_1/copy_1``)."""
    if row["collapsed"]:
        return "collapse"
    if row.get("judgeable") is False:
        return "none"
    return "pass" if row["passed"] else "fail"


def v18_long(runs: Path, judge: dict[str, Any]) -> list[dict[str, Any]]:
    """V18's long runs by step, with the diagnosis (``h_flat``, ``h_mean``,
    Imagenette) and ``pred_ratio_calib`` added."""
    img = _jsonl(runs / "v18-diag/v18-diag/imagenette/imagenette.jsonl")
    out = []
    for name, course in judge["courses"].items():
        diag = _jsonl(runs / "v18-diag/v18-long" / name / "posthoc_diag/tokens.jsonl")
        calib = {
            r["step"]: r["pred_ratio_calib"]
            for r in _jsonl(runs / "v18-long" / name / "posthoc_jepa/jepa.jsonl")
            if r["step"] > 0 and r.get("code", "z") == "z"
        }

        def tok(code: str) -> list[list[float | None]]:
            rows = [r for r in diag if r["group"] == "zurich" and r["code"] == code]  # noqa: B023
            return [[r["step"], _r(r["recon_blur"])] for r in rows]

        def acc(code: str) -> list[list[float | None]]:
            rows = [r for r in img if r.get("run") == name and r["code"] == code]  # noqa: B023
            return [[r["step"], _r(100 * r["acc"])] for r in rows]

        conf = next(x for x in judge["long_runs"] if x["name"] == name)
        out.append(
            {
                "id": name,
                "r": conf["mask"],
                "seed": conf["seed"],
                "step": [c["step"] for c in course],
                "recon": [_r(c["recon_blur"]) for c in course],
                "span": [_r(c["span_recon_blur"]) for c in course],
                "dyn": [_r(c["dyn_1/copy_1"]) for c in course],
                "pred_ratio": [_r(c["pred_ratio"]) for c in course],
                "calib": [_r(calib.get(c["step"])) for c in course],
                "S": [_r(c["sigreg_heldin"]) for c in course],
                "token_r2": [_r(c["token_r2"]) for c in course],
                "rank": [_r(c["rank"]) for c in course],
                "z_diag": tok("z"),
                "h_flat": tok("h_flat"),
                "h_mean": tok("h_mean"),
                "img_h_mean": acc("h_mean"),
                "img_z": acc("z"),
            }
        )
    return out


def v18b_long(judge: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for conf in judge["long_runs"]:
        course = judge["long_courses"][conf["name"]]
        pick = {
            "recon": "recon_blur",
            "span": "span_recon_blur",
            "dyn": "dyn_1/copy_1",
            "copy": "copy_1",
            "calib": "pred_ratio_calib",
            "S": "sigreg_heldin",
            "token_r2": "token_r2",
            "rho_in": "rho_in",
            "rank": "rank",
            "pool": "h_pool",
            "img": "img_h_mean",
        }
        row: dict[str, Any] = {
            "id": conf["name"],
            "arm": conf["arm"],
            "seed": conf["seed"],
            "step": [c["step"] for c in course],
        }
        for k, src in pick.items():
            scale = 100.0 if k == "img" else 1.0
            row[k] = [None if c[src] is None else _r(scale * c[src]) for c in course]
        out.append(row)
    return out


# ---- V18c ----
V18C = RESULTS / "v18c"
LONG_STEPS = ["0", "2000", "6000", "10000", "20000", "30000", "40000", "50000", "60000"]


def _r8(x: float | None) -> float | None:
    """Eight significant digits, so the page's own rounding to 3 decimals
    matches the README's rounding of the exact value."""
    return _r(x, 8)


def _load(name: str) -> Any:
    return json.loads((V18C / name).read_text())


def v18c_screen() -> list[list[Any]]:
    """Each screen run: stage, lr, lambda, r, ``pred_ratio_calib``,
    ``sigreg_heldin``, ``token_r2``, ``rank``, pending, passed, Imagenette's
    ``h_mean`` at step 6000 (%)."""
    j = _load("judge_screen.json")
    out = []
    for stage, rows in [(1, j["first"]), (2, j["second"])]:
        for r in rows:
            out.append(
                [stage, r["lr"], r["lam"], r["r"]]
                + [_r8(r[k]) for k in ["pred_ratio_calib", "sigreg_heldin"]]
                + [_r8(r["token_r2"]), _r8(r["rank"])]
                + [bool(r["pending"]), bool(r["passed"])]
                + [_r8(100 * r["in_h_mean_6000"])]
            )
    return out


def v18c_long() -> dict[str, Any]:
    """The 12 long runs by checkpoint (``judge_long``), with L1, its premise
    and the lines of K1 to K3."""
    j = _load("judge_long.json")
    pick = {
        "calib": ("pred_ratio_calib", 1.0),
        "S": ("sigreg_heldin", 1.0),
        "token_r2": ("token_r2", 1.0),
        "rank": ("rank", 1.0),
        "part": ("participation", 1.0),
        "drift": ("block_drift", 1.0),
        "img_h": ("in_h_mean", 100.0),
        "img_z": ("in_z", 100.0),
        "recon": ("recon_blur", 1.0),
        "pred": ("train_pred", 1.0),
    }
    runs = []
    for r in j["runs"]:
        t = r["trajectory"]
        row: dict[str, Any] = {
            "lr": r["lr"],
            "lam": r["lam"],
            "seed": r["seed"],
            "premise": r["premise"],
            "L1": r["L1"],
            "K": [r["K1"], r["K2"], r["K3"]],
        }
        for k, (src, scale) in pick.items():
            row[k] = [
                None if t[s].get(src) is None else _r8(scale * t[s][src])
                for s in LONG_STEPS
            ]
        runs.append(row)
    return {"steps": [int(s) for s in LONG_STEPS], "lines": j["lines"], "runs": runs}


def v18c_self() -> dict[str, Any]:
    """The self-space identification (``judge_self``): V18c's long runs by
    checkpoint, and the rows of V18 and V18b compared with them."""
    j = _load("judge_self.json")

    def row(x: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {k: _r8(v) for k, v in x["hard"].items()}
        for k in ["pool", "flat", "interp"]:
            d = x[f"hard_pred-{k}"]
            out[f"d_{k}"] = [_r8(d["d"]), _r8(d["se2"])]
        out["easy"] = _r8(x["easy_pred"]["v"])
        out["swap"] = _r8(x["swap"]["v"])
        out["swap_in"] = x["swap_in_dist"]
        if x.get("fwd_pred") is not None:
            out["fwd"] = _r8(x["fwd_pred"]["v"])
            out["rev"] = _r8(x["rev_pred"]["v"])
            out["L"] = _r8(x["local"]["L"])
            out["rho_V"] = _r8(x["local"]["rho_V"])
        return out

    long = []
    for name, by_step in j["groups"]["long"].items():
        conf = name.split("/")[-1]
        long.append(
            {
                "lr": 1e-4 if "lr1e-4" in conf else 1e-5,
                "lam": float(conf.split("-lam")[1].split("-")[0]),
                "seed": int(conf[-1]),
                "steps": [int(s) for s in by_step],
                "rows": [row(by_step[s]) for s in by_step],
            }
        )
    old = {}
    for group in ["v18", "v18b"]:
        for name, by_step in j["groups"][group].items():
            old[name.split("/")[-1]] = {s: row(by_step[s]) for s in ["6000", "60000"]}
    return {"long": long, "old": old}


def v18c_gen(runs: Path) -> dict[str, Any]:
    """G's ``top1`` by checkpoint (P16 and F, ``z`` and ``z_w``, decoder seed
    0), the yardsticks, and E1's ``color`` target by checkpoint (reference)."""
    g = G(runs)
    top1 = {
        f"{arm}/{code}/s{s}": [_r8(g.row(arm, s, t, code)["top1"]) for t in STEPS_G]
        for arm in G_ARMS
        for code in ["z", "z_w"]
        for s in [0, 1]
    }
    fill = json.loads((V18C / "judge.json").read_text())["fill"]["stages_Q"]
    e1 = {
        f"{arm}/s{s}": {
            st: [_r8(fill[f"{arm}/s{s}/{t}/color"][st]["id_easy"]) for t in STEPS_G]
            for st in ["oracle", "pred"]
        }
        for arm in G_ARMS
        for s in [0, 1]
    }
    return {
        "steps": STEPS_G,
        "top1": top1,
        "yard": {c: _r8(r["top1"]) for c, r in g.yard.items()},
        "init": [_r8(g.row("p16", s, 0, "z")["top1"]) for s in [0, 1]],
        "e1": e1,
    }


def v18c_metric() -> dict[str, Any]:
    """The yardsticks of the second main metric (``m`` on sel) and the audit
    of the third (agreement by group)."""
    m = _load("judge_m_checks.json")["yardsticks"]["rows"]
    sc = _load("judge_scene.json")["validity"]["V-c2"]
    return {
        "m2": {k: _r8(v["m_sel"]) for k, v in m.items()},
        "c2": {k: _r8(v) for k, v in sc["agree"].items()},
    }


GEOM_LAGS = ["8", "16", "32", "64", "128", "256"]
GEOM_JUDGED = ["8", "16", "32"]


def v18c_geom() -> dict[str, Any]:
    """Time linearity and the width of information (``judge_geom.json``):
    the ``in`` curves of every code, the paired differences of each model
    with ``pca`` and with its own step 0, and TwoNN (``pre``, k = 64)."""
    j = _load("judge_geom.json")
    init_of = j["init_of"]
    curves = {
        place: {
            code: [[_r8(c[lag][k]) for k in ("G", "s", "se_G")] for lag in GEOM_LAGS]
            for code, c in j["curves"][place].items()
        }
        for place in ("own", "h_mean")
    }
    pairs: dict[str, list[dict[str, Any]]] = {}
    for place in ("own", "h_mean"):
        rows = []
        for x in j["vs_yard"][place]:
            a, b = x["a"], x["b"]
            if a not in init_of or (b != "pca" and b != init_of[a]):
                continue
            lags = x["G"]["lags"]
            d = x.get("d")
            rows.append(
                {
                    "a": a,
                    "b": "pca" if b == "pca" else "init",
                    "init": b,
                    "c": _r8(x["G"]["c"]),
                    "G": [
                        [_r8(lags[g]["diff"]), _r8(lags[g]["se"])]
                        + [lags[g]["cls"], lags[g]["cls_sim"]]
                        for g in GEOM_JUDGED
                    ],
                    "d": None
                    if d is None
                    else [_r8(d["diff"]), _r8(d["sd"]), d["cls"]],
                }
            )
        pairs[place] = rows
    twonn = {
        place: {
            code: [_r8(v["pre-k64"]["8"][k]) for k in ("mean", "min", "max")]
            for code, v in j["twonn"]["codes"][place].items()
        }
        for place in ("own", "h_mean")
    }
    codes = set(curves["own"]) | set(curves["h_mean"])
    return {
        "curves": curves,
        "pairs": pairs,
        "twonn": twonn,
        "init_of": init_of,
        "names": {c: geom_name(c) for c in codes},
        "delta": [j["delta_g"], j["delta_d"]],
    }


WIDTH_YARD = [
    "pca",
    "mean8",
    "color",
    "init",
    "init_f",
    "init_m",
    "init_c",
    "init_c1",
    "init_v16",
]


def _width_row(summary: dict[str, Any], at: dict[str, Any]) -> list[Any]:
    """One code's linear horizon on its own latent: the upper ends of the
    first window below 0.8 s (bounds 1/4 and 1/2; ``None``: no window),
    ``p`` of both bounds and the pooled ``G`` at 0.8 s, the main width
    ``mean1`` and the bound-1/2 mean width with its interval."""
    pt = summary["point"]

    def hi(q: str) -> float | None:
        w = pt[q]["whole_p"]["first"]
        return None if w is None else _r8(w["hi"])

    lo50, hi50 = summary["draws"]["50"]["mean"]
    return [
        hi("25"),
        hi("50"),
        _r8(at["8"]["p25"]),
        _r8(at["8"]["p50"]),
        _r8(at["8"]["G"]),
        _r8(pt["25"]["mean1"]),
        _r8(pt["50"]["mean"]),
        _r8(lo50),
        _r8(hi50),
    ]


def v18c_width() -> dict[str, Any]:
    """The linear horizon of the exploration (``judge_width_yard.json`` for
    the yardsticks, ``judge_width_apply.json`` for the models; the latter is
    not in git and is rebuilt by its command in the README)."""
    yard = _load("judge_width_yard.json")["codes"]
    apply = _load("judge_width_apply.json")
    codes = {
        c: _width_row(yard[f"{c}/own"], apply["yard_at"][f"{c}/own"])
        for c in WIDTH_YARD
    }
    for c, m in apply["models"].items():
        codes[c] = _width_row(m["own"]["summary"], m["own"]["at"])
    return {
        "codes": codes,
        "names": {c: geom_name(c) for c in codes},
        "yard": WIDTH_YARD,
        "content": v18c_content(),
    }


def _interval(d: dict[str, Any] | None) -> list[float | None]:
    if d is None:
        return [None, None, None]
    return [_r8(d["point"]), _r8(d["lo"]), _r8(d["hi"])]


def _content_row(m: dict[str, Any]) -> list[Any]:
    """One code's content under the guard v3 (exploration): E and E^d of the
    chosen readouts on zurich, least squares' E, the chosen level readout's E
    in the training pairs minus the validation intervals, and the intervals
    of D and D^d against ``pca`` and against the code's own init (``None``
    when absent)."""
    v = m["values"]
    pairs = m.get("pairs", {})
    init = pairs.get(m.get("init")) if m.get("init") else None
    pca = pairs.get("pca")
    return [
        _r8(v["sel/eval"]["E"]),
        _r8(v["sel/eval"]["Ed"]),
        _r8(v["lin/eval"]["E"]),
        _r8(v["sel/train"]["E"] - v["sel/val"]["E"]),
        *_interval(pca["D"] if pca else None),
        *_interval(pca["Dd"] if pca else None),
        *_interval(init["D"] if init else None),
        *_interval(init["Dd"] if init else None),
        m["chosen"]["level"],
    ]


def v18c_content() -> dict[str, list[Any]]:
    """The content guard v3 read on the yardsticks (``judge_content_real3``,
    ``base`` is ``pca``) and the 34 learned codes (``judge_content_models3``)."""
    real = _load("judge_content_real3.json")["codes"]
    out = {c: _content_row(real["base" if c == "pca" else c]) for c in WIDTH_YARD}
    for c, m in _load("judge_content_models3.json")["models"].items():
        out[c] = _content_row(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "exp" / "vision" / "report-jepa.html"
    )
    args = ap.parse_args()
    j18 = json.loads((RESULTS / "v18" / "judge.json").read_text())
    j18b = json.loads((RESULTS / "v18b" / "judge.json").read_text())
    data = {
        "yard": yardsticks(args.runs),
        "img_yard": {
            r["code"]: _r(100 * r["acc"])
            for r in _jsonl(args.runs / "v18-diag/v18-diag/imagenette/imagenette.jsonl")
            if "run" not in r
        },
        "v18_screen": [
            [r["lr"], r["lam"], r["r"], status(r), _r(r["recon_blur"])]
            for r in j18["rows"]
        ],
        "v18_long": v18_long(args.runs, j18),
        "v18b_screen": [
            [r["arm"], r["lr"], r["lam"], status(r), _r(r["recon_blur"])]
            + [bool(r["pending"])]
            for r in j18b["first"]
        ],
        "v18b_long": v18b_long(j18b),
        "p16_long": P16_LONG,
        "q1": j18b["question1"],
        "img": {k: image(args.runs / p) for k, p in IMAGES.items()},
        "v18c_screen": v18c_screen(),
        "v18c_long": v18c_long(),
        "v18c_self": v18c_self(),
        "v18c_gen": v18c_gen(args.runs),
        "v18c_metric": v18c_metric(),
        "v18c_geom": v18c_geom(),
        "v18c_width": v18c_width(),
    }
    data["img"]["e1g"] = image(V18C / "e1g.jpg", "720")
    page = (HERE / "jepa.html.in").read_text()
    if page.count(PLACEHOLDER) != 1:
        raise ValueError(f"the template must hold {PLACEHOLDER} once")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    args.out.write_text(page.replace(PLACEHOLDER, blob.replace("</", "<\\/")))
    print(f"{args.out}: {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
