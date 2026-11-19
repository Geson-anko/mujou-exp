# pyright: reportMissingImports=false
"""The V10 supplementary measures (README, V10: results) and figures from
``runs/v10-model`` and the ViT-S runs of V8.

matplotlib is not a dependency of the project, so run it with::

    uv run --no-project --with numpy --with matplotlib \
        python exp/vision/results/v10/figs.py exp/vision/results/v10

The registered judgement uses the 1800-s checkpoint (``exp.vision.tables
--stream``). This script adds what that single checkpoint hides: the span
curve of every seed, the mean of the checkpoints in (1500, 1800] s, and the
lowest span of each run with its time.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

V10 = "runs/v10-model/v0-base-"
ARMS = {  # name: (runs, color); ViT blue shades, ResNet orange shades
    "ViT-S": (
        [
            "runs/v8-adapt/v0-base-142ba038-s0",
            "runs/v8-rms/v0-base-142ba038-s1",
            "runs/v8-rms/v0-base-142ba038-s2",
        ],
        "#8fb8ea",
    ),
    "ViT-M": ([f"{V10}4a3e0888-s{i}" for i in range(3)], "#2a78d6"),
    "ViT-L": ([f"{V10}201c6eb4-s{i}" for i in range(3)], "#163f75"),
    "ResNet-S": ([f"{V10}5c3150b2-s{i}" for i in range(3)], "#f2b27a"),
    "ResNet-M": ([f"{V10}38dd4598-s{i}" for i in range(3)], "#e07a1f"),
    "ResNet-L": ([f"{V10}65b937cf-s{i}" for i in range(3)], "#8a4310"),
}
KEYS = ["span", "after", "past_300_"]
WIN = (1500.0, 1800.0)


def load(run: str) -> dict[str, np.ndarray]:
    rows = sorted(
        (
            json.loads(x)
            for x in (Path(run) / "posthoc_stream" / "summary.jsonl")
            .read_text()
            .splitlines()
        ),
        key=lambda r: r["t"],
    )
    out = {"t": np.array([r["t"] for r in rows])}
    for k in KEYS:
        out[k] = np.array([np.nan if r[k] is None else r[k] for r in rows])
    return out


def table() -> None:
    print(
        "| arm | span (1500,1800] mean, 3 seeds | lowest span (t) | after (1500,1800] |"
    )
    for arm, (runs, _) in ARMS.items():
        L = [load(r) for r in runs]
        w = [(d["t"] > WIN[0]) & (d["t"] <= WIN[1]) for d in L]
        span = [f"{d['span'][i].mean():.3f}" for d, i in zip(L, w, strict=True)]
        after = [f"{d['after'][i].mean():.3f}" for d, i in zip(L, w, strict=True)]
        low = []
        for d in L:
            j = int(np.nanargmin(d["span"]))
            low.append(f"{d['span'][j]:.3f} ({d['t'][j]:g})")
        print(f"| {arm} | {', '.join(span)} | {', '.join(low)} | {', '.join(after)} |")


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
        }
    )
    for f in ("Hiragino Sans", "Noto Sans CJK JP"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = f
        break


def fig_span(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for arm, (runs, c) in ARMS.items():
        ax = axes[0] if arm.startswith("ViT") else axes[1]
        for i, r in enumerate(runs):
            d = load(r)
            ax.plot(d["t"], d["span"], color=c, lw=1.4, label=arm if i == 0 else None)
    for ax, title in zip(axes, ["ViT", "ResNet"], strict=True):
        ax.axhline(1, color="#8a8984", lw=1, ls="--")
        ax.set_yscale("log")
        ticks = [0.5, 0.7, 1, 1.5, 2, 4, 7]
        ax.set_yticks(ticks, [f"{k:g}" for k in ticks])
        ax.minorticks_off()
        ax.set_title(f"{title}（RMSProp τ=1e4、各 3 seed）", fontsize=10)
        ax.set_xlabel("チェックポイント [秒]")
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylabel("span の誤差（平均の stack = 1）")
    fig.tight_layout()
    fig.savefig(out / "span.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    style()
    fig_span(out)
    table()
