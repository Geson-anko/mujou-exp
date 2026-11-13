r"""DIAGNOSIS ONLY (M17): the tables of 「ストリームの SIGReg の結果」 in
``exp/video/results/m17.md``, the SIGReg of the data points on one stream
(``m17/ssig_t512``, ``ssig_t128``, ``ssig_t64``), from the run directories
pulled into ``./runs`` (``just fleet pull --batch m17-ssig-v1 --batch
m17-ssig-post-v1 --batch m17-ssig-base-v1``, and for the EMA $B$ = 1 runs set
beside them ``m17-ema-v1``, ``m17-ema-post-v1`` and ``m17-ema-post-v2``).
Reads only; prints markdown.

- the training evals (``evals.jsonl``; time 0 is the untrained network of the
  run's ``baselines.json``) and the posthoc lines (``post/decode/metrics.jsonl``)
  of each arm, span $k$ (``vision.datapoint_every``) and $\lambda$
  (``vision.sigreg``), a cell the mean of the seeds then each seed's value
- the yardsticks of each arm (``task=baselines``: the untrained ViT, ``init``,
  and the pixels' PCA of the same $D$, ``pca``), a value per seed
- the verdict of the five criteria fixed before the runs (``verdict``), each
  seed at the last checkpoint: the four of the EMA runs
  (``ema_tables.verdict``: ``context``, ``spread``, ``target``, ``ncomp``)
  and ``sens``: ``sens_dither`` and ``sens_shift`` of the run at most those
  of the untrained ViT of the same arm and seed (the network the run started
  from, its yardstick line of the same split; the dither of the same seed)
- beside them the pixels' ratios (``sens_*_pix``) and the same arm's EMA
  $B$ = 1 run (``m17/ema_<arm>``, ``data.streams`` 1; its sensitivities from
  the posthoc of the last checkpoint ``m17-ema-post-v2``, the rest from
  ``m17-ema-post-v1``)
- the comparison with the EMA $B$ = 1 run of the same arm on ``recon_blur``
  and ``dyn_1/copy_1``: a difference only when the four pairs of seeds agree
  (``direction``)

Run from the repository root: ``python -m exp.video.diag.ssig_tables
[--runs runs] [--t 1800]``.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from exp.video.diag.ema_tables import verdict as ema_verdict
from exp.video.diag.m17_tables import cell, jl

TRAIN_BATCH = "m17-ssig-v1"
POST_BATCH = "m17-ssig-post-v1"
BASE_BATCH = "m17-ssig-base-v1"
EMA_TRAIN = "m17-ema-v1"
EMA_POSTS = ("m17-ema-post-v2", "m17-ema-post-v1")  # the sensitivities first
ARMS = ("t512", "t128", "t64")
TIMES = [300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0]
SENS = ("dither", "shift")
NAMES = ("context", "spread", "target", "ncomp", "sens", "all")

Key = tuple[str, int, float]  # (arm, k, lambda)

TRAIN_KEYS = [
    "pred",
    "comp_short",
    "comp_short_pos",
    "comp_short_mean",
    "comp_long",
    "comp_long_pos",
    "comp_long_mean",
]
NCOMP_KEYS = [
    f"ncomp_{k}{s}" for k in ("short", "long", "future") for s in ("", "_vis", "_mean")
]
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
SENS_KEYS = ["sens_dither", "sens_shift", "sens_dither_pix", "sens_shift_pix"]
TEST_COMP_KEYS = ["comp_short", "comp_short_pos", "comp_long", "comp_long_pos"]


class SsigRuns(NamedTuple):
    """The training runs by id, the posthoc lines by the parent's id, the
    yardstick lines by (arm, seed), the run ids of each (arm, k, lambda) by
    seed, and the EMA $B$ = 1 run ids of each arm by seed."""

    train: dict[str, dict[str, Any]]
    post: dict[str, list[dict[str, Any]]]
    base: dict[tuple[str, int], list[dict[str, Any]]]
    groups: dict[Key, list[str]]
    ema: dict[str, list[str]]


def _jobs(root: Path, batch: str) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for run in sorted((root / batch).glob("*/")):
        if (run / "job.json").exists():
            out.append((run, json.loads((run / "job.json").read_text())))
    return out


def arm_of(job: dict[str, Any]) -> str:
    """The arm of a job of preset ``m17/ssig_<arm>`` or ``m17/ema_<arm>``."""
    return job["name"].rsplit("_", 1)[-1]


def _train(run: Path, j: dict[str, Any]) -> dict[str, Any]:
    base_p = run / "baselines.json"
    v = j["config"]["vision"]
    return dict(
        arm=arm_of(j),
        k=int(v.get("datapoint_every", 0)),
        lam=float(v["sigreg"]),
        seed=j["seed"],
        ev=jl(run / "evals.jsonl"),
        init=json.loads(base_p.read_text()) if base_p.exists() else {},
    )


def load(root: Path) -> SsigRuns:
    """The runs of the batches of the module docstring under ``root``."""
    train = {j["run_id"]: _train(run, j) for run, j in _jobs(root, TRAIN_BATCH)}
    ema: dict[str, list[str]] = defaultdict(list)
    for run, j in _jobs(root, EMA_TRAIN):
        if j["config"]["data"]["streams"] == 1:
            train[j["run_id"]] = _train(run, j)
            ema[arm_of(j)].append(j["run_id"])
    post: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in (POST_BATCH, *EMA_POSTS):
        for run, j in _jobs(root, batch):
            post[Path(j["parent"]).name] += jl(run / "post/decode/metrics.jsonl")
    base = {
        (arm_of(j), j["seed"]): jl(run / "post/decode/baselines.jsonl")
        + jl(run / "post/decode/metrics.jsonl")
        for run, j in _jobs(root, BASE_BATCH)
    }
    groups: dict[Key, list[str]] = defaultdict(list)
    for rid, r in train.items():
        if rid.startswith("m17-ssig"):
            groups[(r["arm"], r["k"], r["lam"])].append(rid)
    for g in (*groups.values(), *ema.values()):
        g.sort(key=lambda rid: train[rid]["seed"])
    return SsigRuns(train, dict(post), base, dict(groups), dict(ema))


def _order(k: Key) -> tuple[int, int, float]:
    return (ARMS.index(k[0]) if k[0] in ARMS else len(ARMS), k[1], k[2])


def row(lines: Sequence[dict[str, Any]], t: float, split: str) -> dict[str, Any]:
    """The merged line of the ViT of ``split`` at ``t`` (a key from the first
    line that has it: the EMA runs' sensitivities and the rest come from two
    posthoc batches)."""
    out: dict[str, Any] = {}
    for r in lines:
        if (
            r.get("split") == split
            and r.get("code", "vit") == "vit"
            and abs(r.get("t", -1) - t) < 1e-6
        ):
            out = {**{k: v for k, v in r.items() if v is not None}, **out}
    return out


def ev(runs: SsigRuns, rid: str, t: float, key: str, split: str) -> Any:
    """The training eval of ``split`` (time 0: the untrained network)."""
    if t == 0:
        return runs.train[rid]["init"].get(split, {}).get("init_" + key)
    return row(runs.train[rid]["ev"], t, split).get(key)


def ph(runs: SsigRuns, rid: str, t: float, key: str, split: str) -> Any:
    """The posthoc line of ``split`` at ``t``."""
    return row(runs.post.get(rid, []), t, split).get(key)


def base_line(runs: SsigRuns, arm: str, seed: int, code: str, split: str) -> dict:
    """The yardstick line of ``code`` (``init`` or ``pca``) of ``arm`` and
    ``seed`` on ``split`` ({} if there is none)."""
    for r in runs.base.get((arm, seed), []):
        if r.get("split") == split and r.get("code") == code:
            return r
    return {}


def _le(a: Any, b: Any) -> bool | None:
    return None if a is None or b is None else bool(a <= b)


def _all(xs: Sequence[bool | None]) -> bool | None:
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def verdict(
    ev_row: dict[str, Any], post_row: dict[str, Any], init_row: dict[str, Any]
) -> dict[str, bool | None]:
    """The five criteria of one seed at one checkpoint (None: a value is
    missing): the four of ``ema_tables.verdict``, ``sens`` (``sens_<kind>``
    of ``post_row`` at most that of ``init_row``, the untrained ViT, for the
    dither and the shift), and ``all`` of them."""
    v = ema_verdict(ev_row, post_row)
    del v["all"]
    v["sens"] = _all(
        [_le(post_row.get(f"sens_{s}"), init_row.get(f"sens_{s}")) for s in SENS]
    )
    v["all"] = _all(list(v.values()))
    return v


def _label(k: Key, rids: Sequence[str]) -> str:
    return f"`{k[0]}` k={k[1]} λ={k[2]:g} `{rids[0].split('-')[2]}`"


def run_verdict(runs: SsigRuns, rid: str, t: float, split: str) -> dict:
    """``verdict`` of run ``rid`` at ``t``, the completion errors from the
    training evals on val and from the posthoc line on test (the training
    evals have val only)."""
    r = runs.train[rid]
    lines = r["ev"] if split == "val" else runs.post.get(rid, [])
    return verdict(
        row(lines, t, split),
        row(runs.post.get(rid, []), t, split),
        base_line(runs, r["arm"], r["seed"], "init", split),
    )


def verdict_table(runs: SsigRuns, t: float, split: str = "val") -> str:
    """The verdict of each seed of each (arm, k, lambda) at ``t``, with the
    sensitivities of the run, the untrained ViT of the seed, the pixels and
    the EMA $B$ = 1 run of the arm and seed."""
    mark = {True: "○", False: "×", None: "?"}
    head = [
        "run",
        "seed",
        *NAMES,
        "`sens_dither`（ViT / 画素 / EMA）",
        "`sens_shift`（ViT / 画素 / EMA）",
    ]
    out = ["| " + " | ".join(head) + " |", "|" + " -- |" * len(head)]
    for k in sorted(runs.groups, key=_order):
        rids = runs.groups[k]
        for rid in rids:
            r = runs.train[rid]
            v = run_verdict(runs, rid, t, split)
            init = base_line(runs, k[0], r["seed"], "init", split)
            ema = [
                e for e in runs.ema.get(k[0], []) if runs.train[e]["seed"] == r["seed"]
            ]
            sens = []
            for s in SENS:
                mine = ph(runs, rid, t, f"sens_{s}", split)
                pix = ph(runs, rid, t, f"sens_{s}_pix", split)
                em = ph(runs, ema[0], t, f"sens_{s}", split) if ema else None
                sens.append(
                    f"{cell([mine])}（{cell([init.get(f'sens_{s}')])} / "
                    f"{cell([pix])} / {cell([em])}）"
                )
            out.append(
                f"| {_label(k, rids)} | {r['seed']} | "
                + " | ".join(mark[v[n]] for n in NAMES)
                + " | "
                + " | ".join(sens)
                + " |"
            )
    return "\n".join(out)


def summary(runs: SsigRuns, t: float, split: str = "val") -> str:
    """Per (arm, k, lambda): supported (every criterion on both seeds) or the
    criteria that failed on a seed."""
    out = ["| arm | k | λ | 判定 | 外れた基準（seed） |", "| -- | -- | -- | -- | -- |"]
    for k in sorted(runs.groups, key=_order):
        failed = []
        ok = True
        for rid in runs.groups[k]:
            v = run_verdict(runs, rid, t, split)
            ok = ok and v["all"] is True
            seed = runs.train[rid]["seed"]
            failed += [f"{n}（{seed}）" for n in NAMES[:-1] if v[n] is not True]
        n_seeds = len(runs.groups[k])
        word = "支持" if ok and n_seeds == 2 else "棄却"
        out.append(
            f"| `{k[0]}` | {k[1]} | {k[2]:g} | {word} | {'、'.join(failed) or '–'} |"
        )
    return "\n".join(out)


def table(
    runs: SsigRuns,
    keys: Sequence[str],
    get: Any,
    times: Sequence[float],
    split: str,
    yardsticks: bool = False,
    ema: bool = False,
) -> str:
    """A markdown table: a row per (arm, k, lambda, time), with the arm's
    yardsticks (``yardsticks``) and its EMA $B$ = 1 run (``ema``) first."""
    out = [
        "| run | t | " + " | ".join(f"`{k}`" for k in keys) + " |",
        "| -- | -- | " + " | ".join("--" for _ in keys) + " |",
    ]
    done: set[str] = set()
    for k in sorted(runs.groups, key=_order):
        arm = k[0]
        if arm not in done:
            done.add(arm)
            seeds = sorted(runs.train[r]["seed"] for r in runs.groups[k])
            if yardsticks:
                for code, name in (("pca", "PCA-1024"), ("init", "学習前の ViT")):
                    vals = [
                        [base_line(runs, arm, s, code, split).get(key) for s in seeds]
                        for key in keys
                    ]
                    out.append(
                        f"| `{arm}` {name} | – | "
                        + " | ".join(cell(v) for v in vals)
                        + " |"
                    )
            if ema and runs.ema.get(arm):
                e = runs.ema[arm]
                for t in times[-1:]:
                    vals = [[get(runs, r, t, key, split) for r in e] for key in keys]
                    lab = f"`{arm}` EMA $B$=1 `{e[0].split('-')[2]}`"
                    out.append(
                        f"| {lab} | {t:g} | " + " | ".join(cell(v) for v in vals) + " |"
                    )
        rids = runs.groups[k]
        for t in times:
            vals = [[get(runs, rid, t, key, split) for rid in rids] for key in keys]
            if all(v is None for vs in vals for v in vs):
                continue
            out.append(
                f"| {_label(k, rids)} | {t:g} | "
                + " | ".join(cell(v) for v in vals)
                + " |"
            )
    return "\n".join(out)


def direction(a: Sequence[float | None], b: Sequence[float | None]) -> str:
    """``<`` when every pair of the seeds' values has ``a`` below ``b``,
    ``>`` when above, else ``–`` (no difference: the rule of
    「ストリームごとのバッファ」)."""
    pairs = [(x, y) for x in a for y in b if x is not None and y is not None]
    if not pairs or len(pairs) < len(a) * len(b):
        return "–"
    if all(x < y for x, y in pairs):
        return "<"
    if all(x > y for x, y in pairs):
        return ">"
    return "–"


def ema_comparison(runs: SsigRuns, t: float, split: str = "val") -> str:
    """Per (arm, k, lambda): ``recon_blur`` and ``dyn_1/copy_1`` against the
    EMA $B$ = 1 run of the arm, and the direction of the four pairs."""
    keys = ("recon_blur", "dyn_1/copy_1")
    out = [
        "| run | "
        + " | ".join(f"`{k}`（SIGReg / EMA $B$=1 / 向き）" for k in keys)
        + " |",
        "| -- |" + " -- |" * len(keys),
    ]
    for k in sorted(runs.groups, key=_order):
        rids = runs.groups[k]
        e = runs.ema.get(k[0], [])
        cells = []
        for key in keys:
            a = [ph(runs, r, t, key, split) for r in rids]
            b = [ph(runs, r, t, key, split) for r in e]
            cells.append(f"{cell(a)} / {cell(b)} / {direction(a, b)}")
        out.append(f"| {_label(k, rids)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    """Every table of the module docstring."""
    p = argparse.ArgumentParser(description="The tables of the stream SIGReg of M17.")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--t", type=float, default=1800.0, help="the verdict's time")
    args = p.parse_args(argv)
    runs = load(args.runs)
    t = args.t
    print(f"### summary（val、{t:g} 秒）\n")
    print(summary(runs, t))
    print(f"\n### verdict（val、{t:g} 秒）\n")
    print(verdict_table(runs, t))
    print(f"\n### summary（test、{t:g} 秒、posthoc の comp）\n")
    print(summary(runs, t, "test"))
    print(f"\n### verdict（test、{t:g} 秒、posthoc の comp）\n")
    print(verdict_table(runs, t, "test"))
    print(f"\n### EMA B=1 との比較（val、{t:g} 秒）\n")
    print(ema_comparison(runs, t))
    print("\n### train val\n")
    print(table(runs, TRAIN_KEYS, ev, [0.0, *TIMES], "val"))
    print("\n### train val ncomp\n")
    print(table(runs, NCOMP_KEYS, ev, [0.0, *TIMES], "val"))
    print("\n### post val\n")
    print(table(runs, POST_KEYS, ph, TIMES, "val", yardsticks=True, ema=True))
    print("\n### post val sens\n")
    print(table(runs, SENS_KEYS, ph, TIMES, "val", yardsticks=True, ema=True))
    print("\n### post test\n")
    print(table(runs, POST_KEYS, ph, [t], "test", yardsticks=True, ema=True))
    print("\n### post test comp\n")
    print(table(runs, TEST_COMP_KEYS, ph, [t], "test"))


if __name__ == "__main__":
    main()
