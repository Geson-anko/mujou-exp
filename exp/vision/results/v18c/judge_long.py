"""L1 and the description of V18c's long runs (exp/vision/README.md, the V18c
sections on the long runs).

Reads ``runs/v18c-long/<run>/`` (``jepa.json``, ``train.jsonl`` and
``posthoc_jepa/{jepa,probes}.jsonl``), the run's times in
``runs/v18c-long/logs/<run>.time`` and the Imagenette probes in
``runs/v18c-long/imagenette/<run>/imagenette.jsonl``. Applies only L1 of the
long-run judgement (no collapse K1-K3 at step 60000, with the screen's lines
from ``judge_screen.json``) and its premise (``pred_ratio_calib`` falls from
step 6000 to 60000; a run without it is not judged, its K1-K3 still laid
out); L2 and L3 are not judged (the user's decision: V18c has no main
measure). The other
checkpoints are laid out, not judged, and the seed-0 runs' steps 6000 and
60000 next to the screen's run of the same setting (``judge_screen.json``).
Prints markdown tables; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_long [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, override

from exp.vision.results.v18c.judge_screen import Run, _flag, _fmt, _tag, run_name

STEPS = 60000  # the long run's last step
MID = 6000  # the screen's last step: L1's premise compares with it
CKPTS = (0, 2000, 6000, 10000, 20000, 30000, 40000, 50000, 60000)  # LONG_CKPT
PROBE_STEPS = (6000, 20000, 60000)  # jepa_eval --probe-steps
LRS = (1e-4, 1e-5)
LAMS = (0.5, 0.8, 0.95)
SEEDS = (0, 1)
R = 0.5
CODE = "z"
LINES = Path(__file__).with_name("judge_screen.json")  # K1-K3 of the screen

STAT_KEYS = (
    "sigreg_heldin",
    "token_r2",
    "rho_in",
    "participation",
    "pred_ratio",
    "pred_ratio_calib",
    "sigreg_zurich",
    "embed_drift",
    "block_drift",
)
PROBE_KEYS = ("rank", "recon_blur", "dyn_1/copy_1")
TRAIN_KEYS = ("loss", "pred", "sigreg", "mix_var", "rho_in", "grad_norm")


class LongRun(Run):
    """A long run; its time file marks ``train_start`` ... ``img_end``."""

    @staticmethod
    @override
    def _times(path: Path) -> dict[str, float]:
        """Seconds of training, evaluation and Imagenette (the last marks)."""
        if not path.exists():
            return {}
        marks: dict[str, float] = {}
        for line in path.read_text().splitlines():
            key, stamp = line.split()[:2]
            marks[key] = float(stamp)
        out = {}
        for part in ("train", "eval", "img"):
            if f"{part}_start" in marks and f"{part}_end" in marks:
                out[part] = marks[f"{part}_end"] - marks[f"{part}_start"]
        return out


def trajectory(run: LongRun) -> dict[int, dict[str, float | None]]:
    """Step -> the numbers laid out at that checkpoint (the training row is
    the mean of the 100 steps up to it; none at step 0)."""
    out: dict[int, dict[str, float | None]] = {}
    for step in CKPTS:
        row: dict[str, float | None] = {}
        for key in TRAIN_KEYS:
            row[f"train_{key}"] = run.train_at(key, step)
        for key in STAT_KEYS:
            row[key] = run.stat(key, step)
        for key in PROBE_KEYS:
            row[key] = run.probe(key, step) if step in PROBE_STEPS else None
        for code in ("h_mean", "z"):
            row[f"in_{code}"] = run.acc.get((step, code))
        out[step] = row
    return out


def judge(run: LongRun, ln: dict[str, float]) -> dict[str, Any]:
    """L1 (K1-K3 at step 60000), divergence and L1's premise of ``run``."""
    traj = trajectory(run)
    last, mid = traj[STEPS], traj[MID]
    row: dict[str, Any] = {
        "name": run.name,
        "lr": run.lr,
        "lam": run.lam,
        "r": run.r,
        "seed": int(run.name.rsplit("-s", 1)[1]),
        "snap": run.snap,
        "finished": run.finished(),
        "diverged": run.diverged(),
        "sec_per_step": run.sec_per_step(),
        "max_mem": max((float(x["max_mem"]) for x in run.train), default=None),
        **{f"{k}_sec": v for k, v in run.times.items()},
        "trajectory": {str(k): v for k, v in traj.items()},
    }
    s, rank, r2 = last["sigreg_heldin"], last["rank"], last["token_r2"]
    q6, q60 = mid["pred_ratio_calib"], last["pred_ratio_calib"]
    row["premise"] = None if q6 is None or q60 is None else q60 < q6
    row["complete"] = run.finished() and None not in (s, rank, r2)
    if not row["complete"] or run.diverged():
        row |= {"K1": None, "K2": None, "K3": None, "collapsed": True}
        row["L1"] = False if row["premise"] else None
        return row
    assert s is not None and rank is not None and r2 is not None
    row["K1"] = s >= ln["k1"]
    row["K2"] = rank < ln["rank_floor"]
    row["K3"] = r2 >= ln["theta3"]
    row["collapsed"] = bool(row["K1"] or row["K2"] or row["K3"])
    row["L1"] = not row["collapsed"] if row["premise"] else None
    return row


def _get(x: dict[str, Any], step: int, key: str) -> float | None:
    return x["trajectory"][str(step)][key]


def seed_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (lr, lambda), the mean over the seeds of every trajectory number
    (None when a seed lacks it)."""
    out = []
    for lr in LRS:
        for lam in LAMS:
            same = [x for x in rows if x["lr"] == lr and x["lam"] == lam]
            if not same:
                continue
            traj: dict[str, dict[str, float | None]] = {}
            for step in CKPTS:
                keys = same[0]["trajectory"][str(step)].keys()
                vals: dict[str, float | None] = {}
                for k in keys:
                    v = [y for x in same if (y := _get(x, step, k)) is not None]
                    vals[k] = mean(v) if len(v) == len(SEEDS) else None
                traj[str(step)] = vals
            out.append({"lr": lr, "lam": lam, "n": len(same), "trajectory": traj})
    return out


def _l1(b: bool | None) -> str:
    return "判定しない" if b is None else "**満たす**" if b else "満たさない"


def _l1_table(rows: list[dict[str, Any]]) -> str:
    out = [
        "| run | 完走 | 発散 | `sigreg_heldin` | `rank` | `token_r2` | K1 | K2 | K3 "
        "| L1 | `calib` 6000 | `calib` 60000 | 前提 | 秒 / step | 学習 (時間) "
        "| 評価 (分) | Imagenette (分) |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- "
        "| -- | -- | -- |",
    ]
    for x in rows:
        out.append(
            f"| `{x['name']}` | {'○' if x['finished'] else '×'} "
            f"| {'×' if x['diverged'] else '·'} "
            f"| {_fmt(_get(x, STEPS, 'sigreg_heldin'))} "
            f"| {_fmt(_get(x, STEPS, 'rank'), 1)} "
            f"| {_fmt(_get(x, STEPS, 'token_r2'))} | {_flag(x['K1'])} "
            f"| {_flag(x['K2'])} | {_flag(x['K3'])} "
            f"| {_l1(x['L1'])} "
            f"| {_fmt(_get(x, MID, 'pred_ratio_calib'))} "
            f"| {_fmt(_get(x, STEPS, 'pred_ratio_calib'))} "
            f"| {'—' if x['premise'] is None else '○' if x['premise'] else '×'} "
            f"| {_fmt(x['sec_per_step'])} "
            f"| {_fmt(x.get('train_sec', math.nan) / 3600, 2)} "
            f"| {_fmt(x.get('eval_sec', math.nan) / 60, 1)} "
            f"| {_fmt(x.get('img_sec', math.nan) / 60, 1)} |"
        )
    return "\n".join(out)


def _step_table(
    rows: list[dict[str, Any]], key: str, d: int, pct: bool, label: str
) -> str:
    steps = [s for s in CKPTS if any(_get(x, s, key) is not None for x in rows)]
    out = [
        f"| {label} | " + " | ".join(str(s) for s in steps) + " |",
        "| -- | " + " | ".join("--" for _ in steps) + " |",
    ]
    for x in rows:
        name = x.get("name") or f"{_tag(x['lr'])} λ{x['lam']:g}"
        vals = []
        for s in steps:
            v = _get(x, s, key)
            vals.append(_fmt(None if v is None else 100 * v if pct else v, d))
        out.append(f"| `{name}` | " + " | ".join(vals) + " |")
    return "\n".join(out)


SCREEN_KEYS = (  # key in judge_screen.json, key of a long run's trajectory
    ("train_pred", "train_pred"),
    ("pred_ratio_calib", "pred_ratio_calib"),
    ("sigreg_heldin", "sigreg_heldin"),
    ("token_r2", "token_r2"),
    ("rank", "rank"),
    ("participation", "participation"),
    ("in_h_mean_6000", "in_h_mean"),
    ("in_z_6000", "in_z"),
)


def screen_rows(
    rows: list[dict[str, Any]], screen: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per seed-0 long run, the screen's run of the same setting (6000 steps,
    its cosine ending there) next to the long run's steps 6000 and 60000."""
    by_name = {x["name"]: x for x in screen}
    out = []
    for x in rows:
        if x["seed"] != 0 or x["name"] not in by_name:
            continue
        y = by_name[x["name"]]
        row: dict[str, Any] = {"name": x["name"]}
        for ks, kl in SCREEN_KEYS:
            row[kl] = (y[ks], _get(x, MID, kl), _get(x, STEPS, kl))
        out.append(row)
    return out


def _screen_table(rows: list[dict[str, Any]]) -> str:
    head = "| run | " + " | ".join(f"{k} 掃引 / 6000 / 60000" for _, k in SCREEN_KEYS)
    out = [head + " |", "| -- |" + " -- |" * len(SCREEN_KEYS)]
    for x in rows:
        cells = []
        for _, k in SCREEN_KEYS:
            pct = k.startswith("in_")
            d = 1 if pct or k in ("rank", "participation") else 3
            cells.append(
                " / ".join(
                    _fmt(None if v is None else 100 * v if pct else v, d) for v in x[k]
                )
            )
        out.append(f"| `{x['name']}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


TABLES = (  # key, digits, percent, title
    ("train_loss", 3, False, "loss（学習の行）"),
    ("train_pred", 3, False, "pred（学習の行）"),
    ("train_sigreg", 3, False, "S（学習の行）"),
    ("pred_ratio_calib", 3, False, "`pred_ratio_calib`"),
    ("sigreg_heldin", 3, False, "`sigreg_heldin`（K1）"),
    ("token_r2", 3, False, "`token_r2`（K3）"),
    ("rank", 1, False, "`rank`（K2、zurich）"),
    ("rho_in", 3, False, "`rho_in`"),
    ("participation", 1, False, "`participation`"),
    ("embed_drift", 4, False, "`embed_drift`"),
    ("block_drift", 4, False, "`block_drift`"),
    ("recon_blur", 3, False, "`recon_blur`（zurich）"),
    ("in_h_mean", 1, True, "Imagenette `h_mean`（%）"),
    ("in_z", 1, True, "Imagenette `z`（%）"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--lines", type=Path, default=LINES)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    screen = json.loads(args.lines.read_text())
    ln = screen["lines"]
    root = args.runs / "v18c-long"
    names = [run_name(lr, lam, R, s) for lam in LAMS for lr in LRS for s in SEEDS]
    rows = [
        judge(LongRun(root / n), ln) for n in names if (root / n / "jepa.json").exists()
    ]
    means = seed_means(rows)
    vs_screen = screen_rows(rows, screen["first"])

    print(
        f"K1 ≥ {ln['k1']:.2f}、K2 の `rank` の床 {ln['rank_floor']:.2f}、"
        f"K3 の θ3 = {ln['theta3']:.3f}（掃引と同じ）\n"
    )
    print(f"## L1（step {STEPS}）\n")
    print(_l1_table(rows))
    judged = [x for x in rows if x["L1"] is not None]
    print(
        f"\nL1 を満たす: {sum(x['L1'] for x in judged)} / {len(judged)}"
        f"（前提が成り立たず判定しない: {len(rows) - len(judged)}、"
        f"どの線にも当たらない: {sum(not x['collapsed'] for x in rows)} / "
        f"{len(names)}）\n"
    )
    for key, d, pct, title in TABLES:
        print(f"## {title}\n")
        print(_step_table(rows, key, d, pct, "run"))
        print()
        print(_step_table(means, key, d, pct, "2 seed の平均"))
        print()
    print("## 掃引の 6000 step との比べ（seed 0）\n")
    print(_screen_table(vs_screen))
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "lines": ln,
                    "runs": rows,
                    "seed_means": means,
                    "screen": vs_screen,
                },
                indent=1,
                ensure_ascii=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
