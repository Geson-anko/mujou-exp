# pyright: reportMissingImports=false
"""The V8 figures and the offset table from ``runs/v8-*/*/posthoc_stream``.

matplotlib is not a dependency of the project, so run it with::

    uv run --no-project --with numpy --with matplotlib \
        python exp/vision/results/v8/figs.py exp/vision/results/v8

The offset of a position $k$ from a checkpoint $T$ is $t_k - T$ seconds
(negative: past, learned; positive: future). A bin of offsets pools
$\\sum e_{T,k} / \\sum b_k$ over the checkpoints $T$ in [600, 1740] s (so
that every offset up to 60 s has a future inside the run's span).
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T0, T1 = 600.0, 1740.0
RUNS = [  # label, run dir, color (categorical slots in a fixed order)
    ("SGD τ=1e5", "runs/v8-tau/v0-base-ba9d09f2-s0", "#2a78d6"),
    ("SGD+momentum τm=10", "runs/v8-mom/v0-base-5384f9c6-s0", "#eb6834"),
    ("RMSProp τ=1e4", "runs/v8-adapt/v0-base-142ba038-s0", "#1baf7a"),
    ("RMSProp τ=1e5", "runs/v8-rms/v0-base-27b254cc-s0", "#eda100"),
    ("AdamW τ=1e5", "runs/v8-rms/v0-base-edf05c88-s0", "#6250d6"),
]
TABLE_BINS = [  # name, low, high (both ends kept when low == high)
    ("-1800..-300", -1800, -300),
    ("-300..-60", -300, -60),
    ("-60..-15", -60, -15),
    ("-10", -10, -10),
    ("-5", -5, -5),
    ("0", 0, 0),
    ("+5", 5, 5),
    ("+10", 10, 10),
    ("+15..+60", 15, 60),
    ("+60..+300", 60, 300),
    ("+300..+3000", 300, 3000),
]


def load(run: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Seconds and base of the positions, matrix rows sorted by ``t``."""
    d = Path(run) / "posthoc_stream"
    p = json.loads((d / "positions.json").read_text())
    rows = [json.loads(x) for x in (d / "matrix.jsonl").read_text().splitlines()]
    return np.array(p["t"]), np.array(p["base"]), sorted(rows, key=lambda r: r["t"])


def offsets(run: str) -> tuple[np.ndarray, np.ndarray]:
    """The pooled ratio at every offset (5 s apart in the span, 10 s after)."""
    t, b, rows = load(run)
    acc: dict[int, list[float]] = {}
    for r in rows:
        if not T0 <= r["t"] <= T1:
            continue
        e = np.array(r["err"])
        for j, o in enumerate(np.round(t - r["t"]).astype(int)):
            s = acc.setdefault(int(o), [0.0, 0.0])
            s[0] += e[j]
            s[1] += b[j]
    o = np.array(sorted(acc))
    return o, np.array([acc[k][0] / acc[k][1] for k in o])


def binned(o: np.ndarray, v: np.ndarray, edges: list[float]) -> tuple[list, list]:
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        i = (o > lo) & (o <= hi)
        if i.any():
            xs.append(float(np.median(o[i])))
            ys.append(float(v[i].mean()))
    return xs, ys


def table() -> None:
    print("| run | " + " | ".join(b[0] for b in TABLE_BINS) + " |")
    for label, run, _ in RUNS:
        t, b, rows = load(run)
        cells = []
        for _, lo, hi in TABLE_BINS:
            num = den = 0.0
            for r in rows:
                if not T0 <= r["t"] <= T1:
                    continue
                o = t - r["t"]
                i = (
                    (o == lo)
                    if lo == hi
                    else ((o > lo) & (o <= hi) if lo >= 0 else (o >= lo) & (o < hi))
                )
                num += float(np.array(r["err"])[i].sum())
                den += float(b[i].sum())
            cells.append(f"{num / den:.2f}")
        print(f"| {label} | " + " | ".join(cells) + " |")


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#8a8984",
            "axes.labelcolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    )
    for f in ("Hiragino Sans", "Noto Sans CJK JP"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = f
        break


def fig_offset(out: Path) -> None:
    edges = [-1800, -900, -600, -300, -180, -120, -60, -30, -20, -15, -10, -5, -0.5]
    edges += [0.5, 5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1800, 3000]
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    for label, run, c in RUNS:
        o, v = offsets(run)
        near = np.abs(o) <= 60
        axes[0].plot(o[near], v[near], color=c, lw=2, marker="o", ms=3.5, label=label)
        x, y = binned(o, v, edges)
        axes[1].plot(x, y, color=c, lw=2, marker="o", ms=3.5, label=label)
    for ax in axes:
        ax.axvline(0, color="#8a8984", lw=1)
        ax.axhline(1, color="#8a8984", lw=1, ls="--")
        ax.set_ylabel("誤差（平均の stack = 1）")
        ax.set_ylim(0, 2.0)
    axes[0].set_xlabel("stack の時刻 − チェックポイントの時刻 [秒]（±60 秒）")
    axes[0].text(-58, 1.92, "← 過去（学習済み）", color="#52514e", fontsize=9)
    axes[0].text(4, 1.92, "未来 →", color="#52514e", fontsize=9)
    axes[1].set_xscale("symlog", linthresh=10)
    ticks = [-1800, -300, -60, -10, 0, 10, 60, 300, 3000]
    axes[1].set_xticks(ticks, [str(k) for k in ticks])
    axes[1].set_xlabel("同じ（全体、対数）")
    axes[1].legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.suptitle(
        "チェックポイントから見た過去と未来の誤差（600〜1740 秒のチェックポイントで平均）",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "offset.png", dpi=110)
    plt.close(fig)


def fig_span_after(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for label, run, c in RUNS:
        path = Path(run) / "posthoc_stream" / "summary.jsonl"
        s = sorted(
            (json.loads(x) for x in path.read_text().splitlines()),
            key=lambda r: r["t"],
        )
        t = [r["t"] for r in s]
        axes[0].plot(t, [r["span"] for r in s], color=c, lw=2, label=label)
        axes[1].plot(t, [r["after"] for r in s], color=c, lw=2, label=label)
    titles = (
        "学習区間（0〜1800 秒）の誤差",
        "run の後の Amsterdam（1800〜4800 秒）の誤差",
    )
    for ax, title in zip(axes, titles, strict=True):
        ax.set_yscale("log")
        ax.axhline(1, color="#8a8984", lw=1, ls="--")
        ax.set_xlabel("チェックポイントの時刻 [秒]")
        ax.set_title(title, fontsize=10)
        ticks = [0.5, 0.7, 1, 2, 4, 7]
        ax.set_yticks(ticks, [f"{k:g}" for k in ticks])
        ax.minorticks_off()
    axes[0].set_ylabel("誤差（平均の stack = 1）")
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "span_after.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    style()
    fig_offset(out)
    fig_span_after(out)
    table()
