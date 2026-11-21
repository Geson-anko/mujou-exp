"""Build ``exp/vision/report-optim.html``, the report of V11-V16, from the
runs and the page template ``optim.html.in`` next to this file.

It reads, for each run of the batches ``BATCHES`` under ``runs/``: the
config, the stream evaluation (``posthoc_stream/summary.jsonl`` and
``probes.jsonl``, judged by ``exp.vision.tables.stream_learned``), the last
row of ``z_use.jsonl``, the relative step of a frame of ``enc2`` and ``dec2``
(``grad.jsonl``, 300 s on) and the 60 s drift of ``exp.vision.drift`` (one
JSON line per run in ``--drift``, which needs the checkpoints on the NAS).
V16 (``V16_RUNS``, the VAE of ``V16_STREAM`` learned the plain way by
``exp.vision.iid``) adds the span and after of every checkpoint against the
stacks seen ($t$ / ``time.frame``) and the last probes and ``z_use``
(``v16_row``). The decoded images of ``IMAGES`` (``posthoc_stream/<t>-<group>.png``)
are shrunk by ``ffmpeg`` and embedded::

    uv run python -m exp.vision.report.build_optim --drift runs/optim-drift.jsonl
"""

import argparse
import base64
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from exp.vision.tables import stream_learned

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
BATCHES = [
    "v8-adapt",
    "v8-rms",
    "v11-zoo",
    "v12-dec",
    "v13-ogd",
    "v14-width",
    "v15-speed",
    "v15-seeds",
]
# the V8 runs kept: RMSProp 1e4, the 3 seeds of the reference
V8_KEEP = {"v0-base-142ba038-s0", "v0-base-142ba038-s1", "v0-base-142ba038-s2"}
V16_STREAM = "v14-width/v0-base-b2006f0a-s0"
V16_RUNS = [
    f"v16-iid/iid-{d}-lr{lr}-s0"
    for d in ("all", "span")
    for lr in ("1e-4", "3e-4", "1e-3")
]
# key: (run, picture of posthoc_stream/)
IMAGES = {
    "sgd": ("v11-zoo/v0-base-43ddd19d-s0", "01800.0-span.png"),
    "rms": ("v11-zoo/v0-base-64ae5ffa-s0", "01800.0-span.png"),
    "sgd_dec100": ("v12-dec/v0-base-dcbe054c-s0", "01800.0-span.png"),
    "ang_300_300": ("v15-speed/v0-base-5a45baf7-s0", "01800.0-span.png"),
    "ang_1000_1000": ("v15-speed/v0-base-0a32295e-s0", "01800.0-span.png"),
    "v16_stream": (V16_STREAM, "01800.0-span.png"),
    "v16_all_1800": ("v16-iid/iid-all-lr1e-3-s0", "01800.0-span.png"),
    "v16_all": ("v16-iid/iid-all-lr3e-4-s0", "600000.0-span.png"),
    "v16_all_after": ("v16-iid/iid-all-lr3e-4-s0", "600000.0-after.png"),
    "v16_span": ("v16-iid/iid-span-lr3e-4-s0", "600000.0-span.png"),
    "v16_span_after": ("v16-iid/iid-span-lr3e-4-s0", "600000.0-after.png"),
}
PLACEHOLDER = "/*__DATA__*/"


def _r(x: float | None, digits: int = 6) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _step_seconds(run: Path, node: str) -> float | None:
    """0.1 s over the mean relative step of a frame of ``node`` from 300 s."""
    path = run / "grad.jsonl"
    if not path.exists():
        return None
    rel = [
        r["dtheta"] / r["theta"]
        for r in _jsonl(path)
        if r["node"] == node and r["t"] >= 300 and r.get("theta")
    ]
    return _r(0.1 / (sum(rel) / len(rel)), 3) if rel else None


def run_row(run: Path, drift: dict[str, Any]) -> dict[str, Any]:
    conf = yaml.safe_load((run / "config.yaml").read_text())
    o, m = conf["optim"], conf["model"]
    res, st = run / "result.json", json.loads((run / "status.json").read_text())
    # V11's nesterov 1e4 and adadelta 0.1 diverged but are "failed" (README)
    status = (
        json.loads(res.read_text()).get("status", "?") if res.exists() else "diverged"
    )
    row: dict[str, Any] = {
        "b": run.parent.name,
        "id": run.name.removeprefix("v0-base-"),
        "seed": conf["seed"],
        "opt": o["kind"],
        "tau": o["tau"],
        "dec": o.get("dec_scale", 1.0),
        "ogd": o.get("ogd_every") if o.get("ogd_n", 0) else None,
        "w": m.get("width"),
        "p": m.get("param", "sp"),
        "status": status,
    }
    d = run / "posthoc_stream"
    if status == "diverged" or not (d / "summary.jsonl").exists():
        # the last logged second, as the README (status.json may lag)
        logs = _jsonl(run / "stream.jsonl") if (run / "stream.jsonl").exists() else []
        row["diverged_at"] = _r(logs[-1]["t"] if logs else st.get("t"), 3)
        return row
    summ = sorted(_jsonl(d / "summary.jsonl"), key=lambda r: r["t"])
    probes = sorted(
        (r for r in _jsonl(d / "probes.jsonl") if r["group"] == "span"),
        key=lambda r: r["t"],
    )
    s0, s1, p0, p1 = summ[0], summ[-1], probes[0], probes[-1]
    c = stream_learned(s0, s1, p0, p1)
    k = len(p1["recon_blur_k"])
    row |= {
        "ok": c["all"],
        "why": [x for x, v in c.items() if not v and x != "all"],
        "t": [r["t"] for r in summ],
        "span": [_r(r["span"]) for r in summ],
        "after0": _r(s0["after"]),
        "after": _r(s1["after"]),
        "now": _r(s1.get("now")),
        "pre": _r(s1.get("pre_now")),
        "forget": [
            _r(s1[g] - s1[f"pre_{g}"]) if s1.get(g) is not None else None
            for g in ("past_10_60", "past_60_300", "past_300_")
        ],
        "rb": _r(p1["recon_blur"]),
        "rb_drop": _r(p0["recon_blur"] - p1["recon_blur"]),
        "rb_thr": _r(
            2 * math.sqrt((p0["recon_blur_sd"] ** 2 + p1["recon_blur_sd"] ** 2) / k)
        ),
        "dyn0": _r(p0["dyn_1/copy_1"]),
        "dyn": _r(p1["dyn_1/copy_1"]),
        "step_enc2": _step_seconds(run, "enc2"),
        "step_dec2": _step_seconds(run, "dec2"),
    }
    if (d / "z_use.jsonl").exists():
        z = _jsonl(d / "z_use.jsonl")[-1]
        row["z"] = {k2: _r(z[k2]) for k2 in ("span", "span_shifted", "span_mean_z")}
        row["zd"] = _r(z["span_shifted"] - z["span"])  # before rounding
    if run.name in drift:
        row["drift"] = {
            "enc": _r(drift[run.name]["encoder"]["matrices"], 3),
            "dec": _r(drift[run.name]["decoder"]["matrices"], 3),
        }
    return row


def v16_row(run: Path) -> dict[str, Any]:
    """The span and after of every checkpoint of ``run`` against the stacks
    seen, and its last probes of span and ``z_use``."""
    conf = yaml.safe_load((run / "config.yaml").read_text())
    frame = conf["time"]["frame"]
    d = run / "posthoc_stream"
    summ = sorted(_jsonl(d / "summary.jsonl"), key=lambda r: r["t"])
    p = max(
        (r for r in _jsonl(d / "probes.jsonl") if r["group"] == "span"),
        key=lambda r: r["t"],
    )
    z = _jsonl(d / "z_use.jsonl")[-1]
    iid = run / "iid.json"
    return {
        "id": run.name,
        "data": json.loads(iid.read_text())["data"] if iid.exists() else "stream",
        "lr": json.loads(iid.read_text())["lr"] if iid.exists() else None,
        "stacks": [round(r["t"] / frame) for r in summ],
        "span": [_r(r["span"]) for r in summ],
        "after": [_r(r["after"]) for r in summ],
        "t_last": summ[-1]["t"],
        "rb": _r(p["recon_blur"]),
        "dyn": _r(p["dyn_1/copy_1"]),
        "zd": _r(z["span_shifted"] - z["span"]),
    }


def image(run: Path, picture: str) -> str:
    """The 3 rows of the decoded images ``posthoc_stream/<picture>``, half
    size, as a JPEG data URI."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(run / "posthoc_stream" / picture),
                "-vf",
                "scale=iw/2:-1",
                "-q:v",
                "4",
                str(out),
            ],
            check=True,
        )
        return "data:image/jpeg;base64," + base64.b64encode(out.read_bytes()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--drift", type=Path, required=True)
    ap.add_argument(
        "--out", type=Path, default=ROOT / "exp" / "vision" / "report-optim.html"
    )
    args = ap.parse_args()
    drift = {r["run"]: r for r in _jsonl(args.drift)}
    runs = []
    for b in BATCHES:
        for run in sorted((args.runs / b).glob("v0-base-*")):
            if b.startswith("v8") and run.name not in V8_KEEP:
                continue
            runs.append(run_row(run, drift))
    data = {
        "runs": runs,
        "v16": [v16_row(args.runs / r) for r in [V16_STREAM, *V16_RUNS]],
        "img": {k: image(args.runs / r, f) for k, (r, f) in IMAGES.items()},
    }
    page = (HERE / "optim.html.in").read_text()
    if page.count(PLACEHOLDER) != 1:
        raise ValueError(f"the template must hold {PLACEHOLDER} once")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    args.out.write_text(page.replace(PLACEHOLDER, blob.replace("</", "<\\/")))
    print(f"{args.out}: {len(runs)} runs, {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
