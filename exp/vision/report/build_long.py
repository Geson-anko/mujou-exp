"""Build ``exp/vision/report-long.html``, the report of V17, from the runs and
the page template ``long.html.in`` next to this file.

For each run of ``BATCHES`` under ``runs/`` that has posthoc rows, it reads
the config (the number ``c`` of consecutive stacks a batch, ``tau``, the
seed) and ``posthoc_stream/summary.jsonl`` (span, after, zurich),
``z_use.jsonl`` and ``probes.jsonl`` (``recon_blur``, ``dyn_1/copy_1`` of
span). The times are those of the single stream (``eval.stream_view``), all
evaluated with ``--span 6000``. V16's plain run ``V16_RUN`` is added for
reference (its span is of another stretch: 0-1800 s). The decoded images of
``IMAGES`` are shrunk by ``ffmpeg`` and embedded::

    uv run python -m exp.vision.report.build_long
"""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from exp.vision.report.build_optim import PLACEHOLDER, _jsonl, _r, image

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
BATCHES = [
    "v17-b1",
    "v17-b1-long",
    "v17-b1-seeds",
    "v17-chunk",
    "v17-chunk-x",
    "v17-c10",
    "v17-c30",
    "v17-c10-long",
]
V16_RUN = "v16-iid/iid-all-lr3e-4-s0"
MAIN = "v17-chunk/v0-base-49b81800-s0"  # tau 1e3, chosen by the registered rule
# key: (run, picture of posthoc_stream/; the stem is the run's own seconds)
IMAGES = {
    "b1_6000": ("v17-b1/v0-base-e7f6713d-s0", "06000.0-span.png"),
    "b1_48000": ("v17-b1-long/v0-base-7cdd64bd-s1", "48000.0-span.png"),
    "c100_6000": (MAIN, "00060.0-span.png"),
    "c100_18000": (MAIN, "00180.0-span.png"),
    "c100_180000": (MAIN, "01800.0-span.png"),
    "c100_480000": (MAIN, "04800.0-span.png"),
    "c100_480000_after": (MAIN, "04800.0-after.png"),
    "c10_120000": ("v17-c10-long/v0-base-966b62e1-s0", "12000.0-span.png"),
    "v16_all": (V16_RUN, "600000.0-span.png"),
}


def run_row(run: Path) -> dict[str, Any]:
    conf = yaml.safe_load((run / "config.yaml").read_text())
    data = conf["data"]
    d = run / "posthoc_stream"
    summ = sorted(_jsonl(d / "summary.jsonl"), key=lambda r: r["t"])
    z = sorted(_jsonl(d / "z_use.jsonl"), key=lambda r: r["t"])
    probes = sorted(
        (p for p in _jsonl(d / "probes.jsonl") if p["group"] == "span"),
        key=lambda p: p["t"],
    )
    return {
        "batch": run.parent.name,
        "id": run.name,
        "c": int(data["streams"]) if data.get("layout") == "chunk" else 1,
        "tau": float(conf["optim"]["tau"]),
        "seed": int(conf["seed"]),
        "t": [r["t"] for r in summ],
        "span": [_r(r["span"]) for r in summ],
        "after": [_r(r["after"]) for r in summ],
        "zurich": [_r(r.get("zurich")) for r in summ],
        "z": [[r["t"], _r(r["span_shifted"] - r["span"], 4)] for r in z],
        "probes": [
            [p["t"], _r(p["recon_blur"], 4), _r(p["dyn_1/copy_1"], 4)] for p in probes
        ],
    }


def v16_row(run: Path) -> dict[str, Any]:
    summ = sorted(
        _jsonl(run / "posthoc_stream" / "summary.jsonl"), key=lambda r: r["t"]
    )
    return {
        "id": run.name,
        "t": [r["t"] for r in summ],
        "span": [_r(r["span"]) for r in summ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "exp" / "vision" / "report-long.html"
    )
    args = ap.parse_args()
    runs = [
        run_row(p.parent.parent)
        for b in BATCHES
        for p in sorted((args.runs / b).glob("*/posthoc_stream/summary.jsonl"))
    ]
    data = {
        "runs": runs,
        "v16": v16_row(args.runs / V16_RUN),
        "img": {k: image(args.runs / r, f) for k, (r, f) in IMAGES.items()},
    }
    page = (HERE / "long.html.in").read_text()
    if page.count(PLACEHOLDER) != 1:
        raise ValueError(f"the template must hold {PLACEHOLDER} once")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    args.out.write_text(page.replace(PLACEHOLDER, blob.replace("</", "<\\/")))
    print(f"{args.out}: {len(runs)} runs, {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
