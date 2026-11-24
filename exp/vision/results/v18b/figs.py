# pyright: reportMissingImports=false
"""The figure of V18b's long runs (README, V18b: results): by the stacks seen,
the zurich ``recon_blur`` of each arm's judge code, that of the pooled encoder
output (F and M's ``h_pool``, P16's ``h_flat``), ``pred_ratio_calib`` and the
Imagenette probe of ``h_mean``, for F and M (``runs/v18b-long``) and V18's two
long runs of P16. Reads the numbers that ``judge`` wrote to ``judge.json``.

matplotlib is not a dependency of the project, so run it with::

    uv run --no-project --with matplotlib \
        python exp/vision/results/v18b/figs.py exp/vision/results/v18b
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("runs")
BATCH = 128
ARMS = {  # color, marker, label
    "f": ("#2a78d6", "o", "F (patch 4, flat)"),
    "m": ("#eb6834", "s", "M (patch 4, token mean)"),
    "p16": ("#1baf7a", "^", "P16 (V18)"),
}
SEED_STYLE = {0: "-", 1: "--"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def _p16(name: str) -> dict[str, dict[int, float]]:
    """V18's long run of P16: zurich recon_blur of z, h_flat, pred_ratio_calib
    and the Imagenette h_mean, by step."""
    post = RUNS / "v18-long" / name / "posthoc_jepa"
    diag = RUNS / "v18-diag/v18-long" / name / "posthoc_diag/tokens.jsonl"
    img = RUNS / "v18-diag/v18-diag/imagenette/imagenette.jsonl"
    return {
        "recon": {
            r["step"]: r["recon_blur"]
            for r in _jsonl(post / "probes.jsonl")
            if r["group"] == "zurich" and r.get("code", "z") == "z"
        },
        "pool": {
            r["step"]: r["recon_blur"]
            for r in _jsonl(diag)
            if r["group"] == "zurich" and r["code"] == "h_flat"
        },
        "calib": {
            r["step"]: r["pred_ratio_calib"]
            for r in _jsonl(post / "jepa.jsonl")
            if r["step"] > 0 and r.get("code", "z") == "z"
        },
        "img": {
            r["step"]: 100 * r["acc"]
            for r in _jsonl(img)
            if r.get("run") == name and r["code"] == "h_mean"
        },
    }


def _v18b(course: list[dict]) -> dict[str, dict[int, float]]:
    def pick(key: str, scale: float = 1.0) -> dict[int, float]:
        return {r["step"]: scale * r[key] for r in course if r[key] is not None}

    calib = {s: v for s, v in pick("pred_ratio_calib").items() if s > 0}
    return {
        "recon": pick("recon_blur"),
        "pool": pick("h_pool"),
        "calib": calib,
        "img": pick("img_h_mean", 100.0),
    }


def main(out: Path) -> None:
    j = json.loads((out / "judge.json").read_text())
    series = []
    for name, course in j["long_courses"].items():
        arm = "f" if "-f-" in name else "m"
        series.append((arm, int(name[-1]), _v18b(course)))
    for x in j["p16_long"]:
        series.append(("p16", int(x["name"][-1]), _p16(x["name"])))

    panels = (
        ("recon", "zurich recon_blur of the judged code\n(lower is better)"),
        ("pool", "zurich recon_blur of the encoder output\n(h_pool; P16: h_flat)"),
        ("calib", "pred_ratio_calib\n(training objective, lower = predicts better)"),
        ("img", "Imagenette linear probe of h_mean (%)"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (key, title) in zip(axes, panels):
        for arm, seed, s in series:
            color, marker, label = ARMS[arm]
            pts = sorted(s[key].items())
            ax.plot(
                [st * BATCH / 1e6 for st, _ in pts],
                [v for _, v in pts],
                SEED_STYLE[seed],
                color=color,
                marker=marker,
                markersize=6,
                linewidth=2,
                label=f"{label}, seed {seed}",
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("stacks seen (millions)")
        ax.grid(color="#e4e4e0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].axhline(0.417, color="#5a5a55", linewidth=1, linestyle=":")
    axes[0].text(7.7, 0.421, "color", ha="right", fontsize=8, color="#5a5a55")
    axes[0].axhline(0.316, color="#5a5a55", linewidth=1, linestyle=":")
    axes[0].text(7.7, 0.320, "init_f", ha="right", fontsize=8, color="#5a5a55")
    axes[0].axhline(0.223, color="#5a5a55", linewidth=1, linestyle=":")
    axes[0].text(7.7, 0.227, "init (P16)", ha="right", fontsize=8, color="#5a5a55")
    axes[2].set_yscale("log")
    axes[2].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[2].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].set_yticks([0.2, 0.3, 0.4, 0.6])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "long.png", dpi=90)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
