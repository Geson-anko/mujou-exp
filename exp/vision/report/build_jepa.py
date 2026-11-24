"""Build ``exp/vision/report-jepa.html``, the report of V18 and V18b (JEPA with
BP and i.i.d. minibatches), from the runs and the page template
``jepa.html.in`` next to this file.

The judged numbers come from the two ``judge.json`` of ``results/v18`` and
``results/v18b`` (the rules applied by their ``judge.py``). The rest is read
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

from exp.vision.report.build_optim import PLACEHOLDER, _jsonl, _r

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "exp" / "vision" / "results"
P16_LONG = ["v18-long-lr1e-4-lam0.5-r0.25-s0", "v18-long-lr1e-4-lam0.5-r0.25-s1"]
F0 = "v18b-long/v18b-long-f-lr1e-4-lam0.5-r0.25-s0"
M0 = "v18b-long/v18b-long-m-lr3e-5-lam0.8-r0.25-s0"
P0 = "v18-long/v18-long-lr1e-4-lam0.5-r0.25-s0"
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


def image(path: Path) -> str:
    """The picture at half size, as a JPEG data URI."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x.jpg"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(path)]
            + ["-vf", "scale=iw/2:-1", "-q:v", "6", str(out)],
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
    }
    page = (HERE / "jepa.html.in").read_text()
    if page.count(PLACEHOLDER) != 1:
        raise ValueError(f"the template must hold {PLACEHOLDER} once")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    args.out.write_text(page.replace(PLACEHOLDER, blob.replace("</", "<\\/")))
    print(f"{args.out}: {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
