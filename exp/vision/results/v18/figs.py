# pyright: reportMissingImports=false
"""The figure of V18's long runs (README, V18: results): the zurich
``recon_blur``, ``dyn_1/copy_1`` and ``pred_ratio`` by the stacks seen, for the
four long runs and the three passing screen runs, with the yardsticks.

matplotlib is not a dependency of the project, so run it with::

    uv run --no-project --with matplotlib \
        python exp/vision/results/v18/figs.py exp/vision/results/v18
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("runs")
COLORS = {"0.25": "#2a78d6", "0.5": "#eb6834", "0.75": "#1baf7a"}
YARD = {"init": "#6250d6", "color": "#e34948", "v17": "#5a5a55"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def _series(run: Path) -> tuple[list[float], list[float], list[float], dict]:
    """Stacks seen (millions), zurich recon_blur and dyn of the probed
    checkpoints, and pred_ratio by step."""
    batch = json.loads((run / "jepa.json").read_text())["batch"]
    rows = [
        r
        for r in _jsonl(run / "posthoc_jepa/probes.jsonl")
        if r["group"] == "zurich" and r["step"] > 0
    ]
    rows.sort(key=lambda r: r["step"])
    x = [r["step"] * batch / 1e6 for r in rows]
    pred = {
        r["step"] * batch / 1e6: r["pred_ratio"]
        for r in _jsonl(run / "posthoc_jepa/jepa.jsonl")
        if r["step"] > 0
    }
    return x, [r["recon_blur"] for r in rows], [r["dyn_1/copy_1"] for r in rows], pred


def main(out: Path) -> None:
    ys = {
        r["code"]: r
        for r in _jsonl(RUNS / "v18-baselines/posthoc_jepa/probes.jsonl")
        if r["group"] == "zurich"
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, key, title in zip(
        axes[:2],
        ("recon_blur", "dyn_1/copy_1"),
        (
            "zurich recon_blur (lower is better)",
            "zurich dyn_1/copy_1 (lower is better)",
        ),
    ):
        for name, color in YARD.items():
            ax.axhline(ys[name][key], color=color, lw=1, ls=":")
            va = "top" if name == "v17" else "bottom"
            ax.text(7.9, ys[name][key], name, color="#333", fontsize=8, va=va)
        ax.set_title(title, fontsize=10)
    for r in ("0.25", "0.5", "0.75"):
        seeds = ["s0", "s1"] if r == "0.25" else ["s0"]
        for seed in seeds:
            x, rec, dyn, pred = _series(
                RUNS / f"v18-long/v18-long-lr1e-4-lam0.5-r{r}-{seed}"
            )
            ls = "-" if seed == "s0" else "--"
            label = f"long r={r} {seed}"
            axes[0].plot(
                x, rec, ls, color=COLORS[r], lw=2, marker="o", ms=5, label=label
            )
            axes[1].plot(x, dyn, ls, color=COLORS[r], lw=2, marker="o", ms=5)
            px = sorted(pred)
            axes[2].plot(
                px, [pred[k] for k in px], ls, color=COLORS[r], lw=2, label=label
            )
        x, rec, dyn, _ = _series(RUNS / f"v18-screen/v18-lr1e-4-lam0.5-r{r}-s0")
        kw = dict(color=COLORS[r], lw=1, marker="s", ms=6, mfc="white")
        axes[0].plot(x, rec, ":", label=f"screen r={r} (6000 steps)", **kw)
        axes[1].plot(x, dyn, ":", **kw)
    axes[2].set_title("zurich pred_ratio (prediction / position mean)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("stacks seen (millions)")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center", ncol=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "long.png", dpi=110)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
