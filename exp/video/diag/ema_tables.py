r"""DIAGNOSIS ONLY (M17): the tables of 「EMA のターゲットモデル（2026-11-13、BP）」
in ``exp/video/results/m17.md``, from the run directories pulled into
``./runs`` (``just fleet pull --batch m17-ema-v1 --batch m17-ema-post-v1
--batch m17-ema-base-v1``). Reads only; prints markdown.

- the training evals (``evals.jsonl``; time 0 is the untrained network of the
  run's ``baselines.json``) and the posthoc lines (``post/decode/metrics.jsonl``)
  of each arm (``m17/ema_t512``, ``ema_t128``, ``ema_t64``) and $B$
  (``data.streams``), a cell the mean of the seeds then each seed's value
- the yardsticks of each arm (``task=baselines``: the untrained ViT, ``init``,
  and the pixels' PCA of the same $D$, ``pca``), over the seeds
- the verdict of the criteria fixed before the runs (``verdict``), each seed at
  the last checkpoint: the context used beyond the position
  (``comp_<v>`` < ``comp_<v>_pos``), no collapse across the stacks
  (posthoc ``z_spread`` at least ``COLLAPSE``), the unmasked tokens restore
  better than the mean image (``restore_target_<v>_mask`` < 1), and no
  collapse within the stack (``ncomp_<k>`` below ``ncomp_<k>_vis`` and
  ``ncomp_<k>_mean``)

Run from the repository root: ``python -m exp.video.diag.ema_tables
[--runs runs] [--t 1800]``.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from exp.video.diag.m17_tables import at, cell, jl
from exp.video.programs.vision import COLLAPSE

TRAIN_BATCH = "m17-ema-v1"
POST_BATCH = "m17-ema-post-v1"
BASE_BATCH = "m17-ema-base-v1"
ARMS = ("t512", "t128", "t64")
TIMES = [300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0]
VIEWS = ("short", "long")
KINDS = ("short", "long", "future")

Key = tuple[str, int]  # (arm, B)

TRAIN_KEYS = [
    "pred",
    "comp_short",
    "comp_short_pos",
    "comp_short_mean",
    "comp_long",
    "comp_long_pos",
    "comp_long_mean",
]
NCOMP_KEYS = [f"ncomp_{k}{s}" for k in KINDS for s in ("", "_vis", "_mean")]
POST_KEYS = [
    "z_spread",
    "rank",
    "recon_blur",
    "recon",
    "restore_short_mask",
    "restore_target_short_mask",
    "restore_short_mask_blur",
    "restore_target_short_mask_blur",
    "restore_long_mask",
    "restore_target_long_mask",
    "restore_long_mask_blur",
    "restore_target_long_mask_blur",
    "restore_target_all",
    "dyn_1/copy_1",
]
TEST_COMP_KEYS = ["comp_short", "comp_short_pos", "comp_long", "comp_long_pos"]


class EmaRuns(NamedTuple):
    """The training runs by id, the posthoc lines by the parent's id, the
    yardstick lines by (arm, seed), and the run ids of each (arm, B) by seed."""

    train: dict[str, dict[str, Any]]
    post: dict[str, list[dict[str, Any]]]
    base: dict[tuple[str, int], list[dict[str, Any]]]
    groups: dict[Key, list[str]]


def _jobs(root: Path, batch: str) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for run in sorted((root / batch).glob("*/")):
        if (run / "job.json").exists():
            out.append((run, json.loads((run / "job.json").read_text())))
    return out


def arm_of(job: dict[str, Any]) -> str:
    """The arm of a job of preset ``m17/ema_<arm>``."""
    return job["name"].rsplit("_", 1)[-1]


def load(root: Path) -> EmaRuns:
    """The runs of the three batches under ``root``."""
    train: dict[str, dict[str, Any]] = {}
    for run, j in _jobs(root, TRAIN_BATCH):
        base_p = run / "baselines.json"
        train[j["run_id"]] = dict(
            arm=arm_of(j),
            B=j["config"]["data"]["streams"],
            seed=j["seed"],
            ev=jl(run / "evals.jsonl"),
            init=json.loads(base_p.read_text()) if base_p.exists() else {},
            dir=run,
        )
    post = {
        Path(j["parent"]).name: jl(run / "post/decode/metrics.jsonl")
        for run, j in _jobs(root, POST_BATCH)
    }
    base = {
        (arm_of(j), j["seed"]): jl(run / "post/decode/baselines.jsonl")
        + jl(run / "post/decode/metrics.jsonl")
        for run, j in _jobs(root, BASE_BATCH)
    }
    groups: dict[Key, list[str]] = defaultdict(list)
    for rid, r in train.items():
        groups[(r["arm"], r["B"])].append(rid)
    for k in groups:
        groups[k].sort(key=lambda rid: train[rid]["seed"])
    return EmaRuns(train, post, base, dict(groups))


def _order(k: Key) -> tuple[int, int]:
    return (ARMS.index(k[0]) if k[0] in ARMS else len(ARMS), -k[1])


def ev(runs: EmaRuns, rid: str, t: float, key: str, split: str) -> Any:
    """The training eval of ``split`` (time 0: the untrained network)."""
    if t == 0:
        return runs.train[rid]["init"].get(split, {}).get("init_" + key)
    return at(runs.train[rid]["ev"], t, split, key)


def ph(runs: EmaRuns, rid: str, t: float, key: str, split: str) -> Any:
    """The posthoc line of ``split`` (that of the ViT, not of the pca)."""
    for r in runs.post.get(rid, []):
        if (
            r.get("split") == split
            and r.get("code", "vit") == "vit"
            and abs(r.get("t", -1) - t) < 1e-6
        ):
            return r.get(key)
    return None


def base(runs: EmaRuns, arm: str, code: str, key: str, split: str) -> list[Any]:
    """``key`` of the yardstick ``code`` (``init`` or ``pca``) of ``arm``, a
    value per seed."""
    out = []
    for (a, _), lines in sorted(runs.base.items()):
        if a != arm:
            continue
        vals = [
            r[key]
            for r in lines
            if r.get("split") == split and r.get("code") == code and key in r
        ]
        out.append(vals[0] if vals else None)
    return out


Getter = Callable[[EmaRuns, str, float, str, str], Any]


def table(
    runs: EmaRuns,
    keys: Sequence[str],
    get: Getter,
    times: Sequence[float],
    split: str,
    yardsticks: bool = False,
) -> str:
    """A markdown table: a row per (arm, B, time), with the arm's yardsticks
    first if ``yardsticks``."""
    out = [
        "| run | t | " + " | ".join(f"`{k}`" for k in keys) + " |",
        "| -- | -- | " + " | ".join("--" for _ in keys) + " |",
    ]
    done: set[str] = set()
    for k in sorted(runs.groups, key=_order):
        arm, b = k
        if yardsticks and arm not in done:
            done.add(arm)
            for code, name in (("pca", "PCA-1024"), ("init", "学習前の ViT")):
                vals = [base(runs, arm, code, key, split) for key in keys]
                out.append(
                    f"| `{arm}` {name} | – | "
                    + " | ".join(cell(v) for v in vals)
                    + " |"
                )
        rids = runs.groups[k]
        lab = f"`{arm}` $B$={b} `{rids[0].split('-')[2]}`"
        for t in times:
            vals = [[get(runs, rid, t, key, split) for rid in rids] for key in keys]
            if all(v is None for vs in vals for v in vs):
                continue
            out.append(f"| {lab} | {t:g} | " + " | ".join(cell(v) for v in vals) + " |")
    return "\n".join(out)


def _lt(a: Any, b: Any) -> bool | None:
    return None if a is None or b is None else bool(a < b)


def _all(xs: Sequence[bool | None]) -> bool | None:
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def verdict(ev_row: dict[str, Any], post_row: dict[str, Any]) -> dict[str, bool | None]:
    """The four criteria of one seed at one checkpoint (None: a value is
    missing): ``context`` (``comp_<v>`` < ``comp_<v>_pos`` for both views),
    ``spread`` (posthoc ``z_spread`` >= ``COLLAPSE``), ``target``
    (``restore_target_<v>_mask`` < 1), ``ncomp`` (``ncomp_<k>`` below both
    trivial predictions for every kind present), and ``all`` of them."""
    g = ev_row.get
    out: dict[str, bool | None] = {
        "context": _all([_lt(g(f"comp_{v}"), g(f"comp_{v}_pos")) for v in VIEWS]),
        "spread": None
        if post_row.get("z_spread") is None
        else post_row["z_spread"] >= COLLAPSE,
        "target": _all(
            [_lt(post_row.get(f"restore_target_{v}_mask"), 1.0) for v in VIEWS]
        ),
        "ncomp": _all(
            [
                _lt(g(f"ncomp_{k}"), g(f"ncomp_{k}{s}"))
                for k in KINDS
                if g(f"ncomp_{k}") is not None
                for s in ("_vis", "_mean")
            ]
            or [None]
        ),
    }
    out["all"] = _all(list(out.values()))
    return out


def _row(rows: list[dict[str, Any]], t: float, split: str) -> dict[str, Any]:
    for r in rows:
        if (
            r.get("split") == split
            and r.get("code", "vit") == "vit"
            and abs(r.get("t", -1) - t) < 1e-6
        ):
            return r
    return {}


def verdict_table(runs: EmaRuns, t: float, split: str = "val") -> str:
    """The verdict of each seed of each (arm, B) at ``t``."""
    mark = {True: "○", False: "×", None: "?"}
    names = ("context", "spread", "target", "ncomp", "all")
    out = [
        "| run | seed | " + " | ".join(names) + " |",
        "| -- | -- | " + " | ".join("--" for _ in names) + " |",
    ]
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            v = verdict(
                _row(runs.train[rid]["ev"], t, split),
                _row(runs.post.get(rid, []), t, split),
            )
            out.append(
                f"| `{k[0]}` $B$={k[1]} `{rid.split('-')[2]}` | "
                f"{runs.train[rid]['seed']} | "
                + " | ".join(mark[v[n]] for n in names)
                + " |"
            )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    """Every table of the module docstring."""
    p = argparse.ArgumentParser(description="The tables of the EMA runs of M17.")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--t", type=float, default=1800.0, help="the verdict's time")
    args = p.parse_args(argv)
    runs = load(args.runs)
    print(f"### verdict（val、{args.t:g} 秒）\n")
    print(verdict_table(runs, args.t))
    print("\n### train val\n")
    print(table(runs, TRAIN_KEYS, ev, [0.0, *TIMES], "val"))
    print("\n### train val ncomp\n")
    print(table(runs, NCOMP_KEYS, ev, [0.0, *TIMES], "val"))
    print("\n### post val\n")
    print(table(runs, POST_KEYS, ph, TIMES, "val", yardsticks=True))
    print("\n### post test\n")
    print(table(runs, POST_KEYS, ph, [600.0, 1800.0], "test", yardsticks=True))
    print("\n### post test comp\n")
    print(table(runs, ["pred", *TEST_COMP_KEYS], ph, [600.0, 1800.0], "test"))
    print(f"\n### verdict（test、{args.t:g} 秒、posthoc の comp）\n")
    print(verdict_table_post(runs, args.t, "test"))


def verdict_table_post(runs: EmaRuns, t: float, split: str) -> str:
    """``verdict_table`` with the completion errors of the posthoc line, which
    holds them also on test (the training evals have val only)."""
    post_only = EmaRuns(
        {rid: {**r, "ev": runs.post.get(rid, [])} for rid, r in runs.train.items()},
        runs.post,
        runs.base,
        runs.groups,
    )
    return verdict_table(post_only, t, split)


if __name__ == "__main__":
    main()
