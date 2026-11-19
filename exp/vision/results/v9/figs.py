# pyright: reportMissingImports=false
"""The V9 measures (README, V9: A, B, C and the stall) and figures from
``runs/v9-long/*/posthoc_stream``.

matplotlib is not a dependency of the project, so run it with::

    uv run --no-project --with numpy --with matplotlib \
        python exp/vision/results/v9/figs.py exp/vision/results/v9

A group's error at a checkpoint pools $\\sum e_{T,k} / \\sum b_k$ over its
positions $k$ (the mean stack scores 1). Amsterdam is 0-4912 s of the
stream; the next video starts at 4912 s.
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BOUNDARY = 4912.0  # the end of Amsterdam on the stream (s)
RUNS = [  # label, run dir, color (the optimizer colors of the report)
    ("SGD τ=1e5", "runs/v9-long/v0-base-e6572532-s0", "#2a78d6"),
    ("RMSProp τ=1e4", "runs/v9-long/v0-base-39f691a8-s0", "#1baf7a"),
    ("AdamW τ=1e5", "runs/v9-long/v0-base-cde03409-s0", "#6250d6"),
]
GROUPS = {  # name: (lo, hi) of the stack time, lo <= t < hi
    "ams_0_1800": (0.0, 1800.0),
    "ams_1800_end": (1800.0, BOUNDARY),
    "next_seen": (BOUNDARY, 6000.0),
    "next_after": (6000.0, 9000.0),
}
AT = [1800.0, 3000.0, 4860.0, 6000.0]


def load(run: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Seconds and base of the positions, checkpoint times and the matrix."""
    d = Path(run) / "posthoc_stream"
    if not (d / "matrix.jsonl").exists():
        return None
    p = json.loads((d / "positions.json").read_text())
    rows = sorted(
        (json.loads(x) for x in (d / "matrix.jsonl").read_text().splitlines()),
        key=lambda r: r["t"],
    )
    return (
        np.array(p["t"]),
        np.array(p["base"]),
        np.array([r["t"] for r in rows]),
        np.array([r["err"] for r in rows]),
    )


def group_curves(t, b, T, E) -> dict[str, np.ndarray]:
    out = {}
    for g, (lo, hi) in GROUPS.items():
        i = (t >= lo) & (t < hi)
        out[g] = E[:, i].sum(1) / b[i].sum()
    return out


def offsets(t, b, T, E, T0=600.0, T1=5940.0):
    acc: dict[int, list[float]] = {}
    for Ti, e in zip(T, E, strict=True):
        if not T0 <= Ti <= T1:
            continue
        for j, o in enumerate(np.round(t - Ti).astype(int)):
            s = acc.setdefault(int(o), [0.0, 0.0])
            s[0] += e[j]
            s[1] += b[j]
    o = np.array(sorted(acc))
    return o, np.array([acc[k][0] / acc[k][1] for k in o])


OFF_BINS = [
    ("-6000..-300", -6000, -300),
    ("-300..-60", -300, -60),
    ("-10", -10, -10),
    ("-5", -5, -5),
    ("0", 0, 0),
    ("+5", 5, 5),
    ("+10", 10, 10),
    ("+60..+300", 60, 300),
    ("+300..", 300, 9000),
]


def table() -> None:
    print("| run | group | " + " | ".join(f"T={a:g}" for a in AT) + " |")
    for label, run, _ in RUNS:
        L = load(run)
        if L is None:
            print(f"| {label} | (no posthoc_stream) |")
            continue
        t, b, T, E = L
        cur = group_curves(t, b, T, E)
        for g, v in cur.items():
            cells = [f"{v[np.argmin(np.abs(T - a))]:.3f}" for a in AT]
            # the mean of the 5 checkpoints up to a (a - 240 .. a), against the swings
            win = [f"{v[(T > a - 300) & (T <= a)].mean():.3f}" for a in AT]
            print(
                f"| {label} | {g} | "
                + " | ".join(f"{c} ({w})" for c, w in zip(cells, win, strict=True))
                + " |"
            )
        span = (E[:, t < 6000].sum(1)) / b[t < 6000].sum()
        i48, i60 = np.argmin(np.abs(T - 4800)), np.argmin(np.abs(T - 6000))
        print(
            f"  span 4800 -> 6000: {span[i48]:.3f} -> {span[i60]:.3f} (drop {span[i48] - span[i60]:.3f})"
        )
        o, v = offsets(t, b, T, E)
        cells = []
        for _, lo, hi in OFF_BINS:
            i = (
                (o == lo)
                if lo == hi
                else ((o >= lo) & (o < hi) if lo < 0 else (o > lo) & (o <= hi))
            )
            cells.append(f"{v[i].mean():.2f}")
        print(
            "  offsets "
            + " ".join(f"{n}={c}" for (n, _, _), c in zip(OFF_BINS, cells, strict=True))
        )


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


def fig_groups(out: Path) -> None:
    names = {
        "ams_0_1800": "Amsterdam 0〜1800 秒",
        "ams_1800_end": "Amsterdam 1800〜4912 秒",
        "next_seen": "次の動画 4912〜6000 秒",
        "next_after": "次の動画 6000〜9000 秒（見ていない）",
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for label, run, c in RUNS:
        L = load(run)
        if L is None:
            continue
        cur = group_curves(*L)
        for ax, g in zip(axes, GROUPS, strict=True):
            ax.plot(L[2], cur[g], color=c, lw=1.8, label=label)
    for ax, g in zip(axes, GROUPS, strict=True):
        lo, hi = GROUPS[g]
        ax.axvspan(lo, min(hi, 6000), color="#8a8984", alpha=0.12, lw=0)
        ax.axvline(BOUNDARY, color="#8a8984", lw=1, ls=":")
        ax.axhline(1, color="#8a8984", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_title(names[g], fontsize=10)
        ax.set_xlabel("チェックポイント [秒]")
        ticks = [0.3, 0.5, 1, 2, 4]
        ax.set_yticks(ticks, [f"{k:g}" for k in ticks])
        ax.minorticks_off()
    axes[0].set_ylabel("誤差（平均の stack = 1）")
    axes[-1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "groups.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    style()
    fig_groups(out)
    table()
