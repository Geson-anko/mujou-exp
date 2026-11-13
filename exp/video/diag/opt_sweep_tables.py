r"""DIAGNOSIS ONLY (M17): the tables of 「学習の設定の掃引」 in
``exp/video/results/m17.md``, the sweep of the encoder's optimizer and its
learning rate per frame for the SIGReg of the data points on one stream
(``m17/ssig_t128``, ``vision.datapoint_every=6``, seed 100), from the run
directories pulled into ``./runs`` (``just fleet pull --batch m17-opt-v1
--batch m17-opt-post-v1 --batch m17-opt-base-v1``; other batch names by the
options). Reads only; prints markdown.

A run is the (kind, lr, warmup) of its ``optim_vision``. Its criteria, fixed
before the runs, at ``--t`` (1800 s) on val:

- ``L``: the held-out objective $J = (1 - \lambda)\,\mathrm{pred} + \lambda S$
  (``loss`` of ``evals.jsonl``) is below its value at time 0, the untrained
  network (``init_loss`` of the run's ``baselines.json``)
- ``U``: ``recon_blur`` and ``dyn_1/copy_1`` of the posthoc line both below
  those of the untrained ViT of the same arm and seed (the ``init`` line of
  ``task=baselines``)
- ``C``: the five criteria of the stream SIGReg (``ssig_tables.verdict``:
  context, spread, target, ncomp, sens against the same untrained ViT)
- ``learns``: L and U and C

A missing value is unknown (``?``), not a pass; a value that is not finite
fails (a comparison with NaN is false). The loop writes a non-finite value as
``null`` and stops a run that diverged (``status`` ``diverged`` in its
``result.json``, at ``t_end``), so the $J$ of such a run that is missing or
``null`` at ``--t`` is taken as not finite: L and learns fail.

Shown beside them and not judged: the curves of the held-out $J$, pred and $S$ at each checkpoint (time 0 from
``baselines.json``), the training loss (``stream.jsonl``, the mean over the
300 s before each checkpoint), the posthoc ``rank`` and ``z_spread``, and,
given ``--drift`` (the outputs of ``python -m exp.video.diag.stream_stats
drift`` of the runs), the drift between consecutive checkpoints of the
current encoder's code of fixed held-in stacks: a stand-in for the staleness
of the stored $z$, which it does not read.

Run from the repository root: ``python -m exp.video.diag.opt_sweep_tables
[--runs runs] [--t 1800] [--drift drift.json]``.
"""

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from exp.video.diag.m17_tables import cell, fmt, jl
from exp.video.diag.ssig_tables import NAMES as C_NAMES, row, verdict

TRAIN_BATCH = "m17-opt-v1"
POST_BATCH = "m17-opt-post-v1"
BASE_BATCH = "m17-opt-base-v1"
KINDS = ("sgd", "rmsprop", "adamw")
TIMES = [300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0]
U_KEYS = ("recon_blur", "dyn_1/copy_1")
CRITERIA = ("L", "U", "C", "learns")
MARK = {True: "○", False: "×", None: "?"}

Key = tuple[str, float, float]  # (kind, lr, warmup)


class OptRuns(NamedTuple):
    """The training runs by id, the posthoc lines by the parent's id, the
    yardstick lines by (arm, seed), and the run ids of each (kind, lr,
    warmup)."""

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


def key_of(job: dict[str, Any]) -> Key:
    """(kind, lr, warmup) of the job's ``optim_vision``."""
    o = job["config"]["optim_vision"]
    return (o["kind"], float(o["lr"]), float(o["warmup"]))


def _arm(job: dict[str, Any]) -> str:
    return job["name"].rsplit("_", 1)[-1]


def load(
    root: Path,
    train_batch: str = TRAIN_BATCH,
    post_batch: str = POST_BATCH,
    base_batch: str = BASE_BATCH,
) -> OptRuns:
    """The runs of the three batches under ``root``."""
    train: dict[str, dict[str, Any]] = {}
    groups: dict[Key, list[str]] = defaultdict(list)
    for run, j in _jobs(root, train_batch):
        base_p, res_p = run / "baselines.json", run / "result.json"
        train[j["run_id"]] = dict(
            key=key_of(j),
            arm=_arm(j),
            seed=j["seed"],
            ev=jl(run / "evals.jsonl"),
            stream=jl(run / "stream.jsonl"),
            init=json.loads(base_p.read_text()) if base_p.exists() else {},
            result=json.loads(res_p.read_text()) if res_p.exists() else {},
        )
        groups[key_of(j)].append(j["run_id"])
    post: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run, j in _jobs(root, post_batch):
        post[Path(j["parent"]).name] += jl(run / "post/decode/metrics.jsonl")
    base = {
        (_arm(j), j["seed"]): jl(run / "post/decode/baselines.jsonl")
        + jl(run / "post/decode/metrics.jsonl")
        for run, j in _jobs(root, base_batch)
    }
    for g in groups.values():
        g.sort(key=lambda rid: train[rid]["seed"])
    return OptRuns(train, dict(post), base, dict(groups))


def _order(k: Key) -> tuple[int, float, float]:
    return (KINDS.index(k[0]) if k[0] in KINDS else len(KINDS), k[1], k[2])


def ev(runs: OptRuns, rid: str, t: float, key: str, split: str = "val") -> Any:
    """The held-out eval of ``split`` (time 0: the untrained network)."""
    if t == 0:
        return runs.train[rid]["init"].get(split, {}).get("init_" + key)
    return row(runs.train[rid]["ev"], t, split).get(key)


def ph(runs: OptRuns, rid: str, t: float, key: str, split: str = "val") -> Any:
    """The posthoc line of ``split`` at ``t``."""
    return row(runs.post.get(rid, []), t, split).get(key)


def untrained(runs: OptRuns, rid: str, split: str = "val") -> dict[str, Any]:
    """The ``init`` line (the untrained ViT) of the run's arm and seed on
    ``split`` ({} if there is none)."""
    r = runs.train[rid]
    for line in runs.base.get((r["arm"], r["seed"]), []):
        if line.get("split") == split and line.get("code") == "init":
            return line
    return {}


def train_loss(stream: Sequence[dict[str, Any]], t: float, key: str = "loss") -> Any:
    """The mean of ``key`` of the training rows in the 300 s before ``t``
    (None if there are none)."""
    xs = [
        r[key]
        for r in stream
        if t - 300 < r.get("t", -1) <= t and r.get(key) is not None
    ]
    return sum(xs) / len(xs) if xs else None


def _lt(a: Any, b: Any) -> bool | None:
    return None if a is None or b is None else bool(a < b)


def _all(xs: Sequence[bool | None]) -> bool | None:
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def criteria(
    j0: Any,
    j: Any,
    ev_row: dict[str, Any],
    post_row: dict[str, Any],
    init_row: dict[str, Any],
) -> dict[str, bool | None]:
    """``L`` ($J$ = ``j`` below ``j0``), ``U`` (``U_KEYS`` of ``post_row``
    below those of ``init_row``, the untrained ViT), ``C`` (``all`` of
    ``ssig_tables.verdict``, with its five beside) and ``learns``."""
    c = verdict(ev_row, post_row, init_row)
    out: dict[str, bool | None] = {n: c[n] for n in C_NAMES[:-1]}
    out["L"] = _lt(j, j0)
    out["U"] = _all([_lt(post_row.get(k), init_row.get(k)) for k in U_KEYS])
    out["C"] = c["all"]
    out["learns"] = _all([out["L"], out["U"], out["C"]])
    return out


def diverged(runs: OptRuns, rid: str) -> bool:
    """Whether the run stopped because it diverged (its ``result.json``)."""
    return runs.train[rid]["result"].get("status") == "diverged"


def objective(runs: OptRuns, rid: str, t: float, split: str = "val") -> Any:
    """$J$ at ``t``; NaN when it is missing or ``null`` in a run that
    diverged (not finite)."""
    j = ev(runs, rid, t, "loss", split)
    return math.nan if j is None and diverged(runs, rid) else j


def run_criteria(
    runs: OptRuns, rid: str, t: float, split: str = "val"
) -> dict[str, bool | None]:
    """``criteria`` of run ``rid`` at ``t``."""
    return criteria(
        ev(runs, rid, 0.0, "loss", split),
        objective(runs, rid, t, split),
        row(runs.train[rid]["ev"], t, split),
        row(runs.post.get(rid, []), t, split),
        untrained(runs, rid, split),
    )


def label(k: Key) -> str:
    """``kind`` lr=``lr`` (and the warm-up when there is one)."""
    w = f" warmup {k[2]:g} 秒" if k[2] else ""
    return f"`{k[0]}` lr={k[1]:g}{w}"


def _md(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + " -- |" * len(head)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def verdict_table(runs: OptRuns, t: float, split: str = "val") -> str:
    """Per run, sorted by optimizer then learning rate (the curve of each
    criterion over lr): $J$ at 0 and ``t``, their difference and L, ``U_KEYS`` of the run and
    of the untrained ViT and U, the five of C and C, and learns."""
    head = ["run", "seed", "終わり", "J(0)", f"J({t:g})", f"J({t:g}) − J(0)", "L"]
    head += [f"`{k}`（run / 学習前）" for k in U_KEYS] + ["U"]
    head += [*C_NAMES[:-1], "C", "learns"]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            c = run_criteria(runs, rid, t, split)
            init = untrained(runs, rid, split)
            cells = [label(k), str(runs.train[rid]["seed"]), end(runs, rid)]
            j0, j = ev(runs, rid, 0.0, "loss", split), ev(runs, rid, t, "loss", split)
            dj = None if j0 is None or j is None else j - j0
            cells += [fmt(j0), fmt(j), fmt(dj), MARK[c["L"]]]
            cells += [
                f"{fmt(ph(runs, rid, t, u, split))} / {fmt(init.get(u))}"
                for u in U_KEYS
            ]
            cells += [MARK[c["U"]]]
            cells += [MARK[c[n]] for n in (*C_NAMES[:-1], "C", "learns")]
            rows.append(cells)
    return "\n".join(_md(head, rows))


def end(runs: OptRuns, rid: str) -> str:
    """How the run ended: its ``status`` and ``t_end`` (``?`` without
    ``result.json``)."""
    r = runs.train[rid]["result"]
    if not r:
        return "?"
    return f"{r.get('status')} t={fmt(r.get('t_end'))}"


def summary(runs: OptRuns, t: float, split: str = "val") -> str:
    """Per optimizer: the learning rates whose runs learn, those whose
    criteria are unknown, and those whose runs diverged."""
    rows = []
    for kind in sorted({k[0] for k in runs.groups}, key=lambda x: _order((x, 0, 0))):
        learn, unknown, div = [], [], []
        for k in sorted(runs.groups, key=_order):
            if k[0] != kind:
                continue
            got = [run_criteria(runs, r, t, split)["learns"] for r in runs.groups[k]]
            name = f"{k[1]:g}" + (f"（warmup {k[2]:g} 秒）" if k[2] else "")
            if got and all(g is True for g in got):
                learn.append(name)
            elif any(g is None for g in got) and not any(g is False for g in got):
                unknown.append(name)
            if any(diverged(runs, r) for r in runs.groups[k]):
                div.append(name)
        rows.append(
            [
                f"`{kind}`",
                "、".join(learn) or "なし",
                "、".join(unknown) or "–",
                "、".join(div) or "–",
            ]
        )
    head = ["optimizer", "学べた lr", "不明な lr", "発散した lr"]
    return "\n".join(_md(head, rows))


def curve_table(
    runs: OptRuns, getter: Any, key: str, times: Sequence[float], split: str = "val"
) -> str:
    """A row per run, a column per time, of ``getter(runs, rid, t, key,
    split)`` (the seeds' mean then each)."""
    head = ["run", *[f"{t:g}" for t in times]]
    rows = []
    for k in sorted(runs.groups, key=_order):
        rids = runs.groups[k]
        rows.append(
            [label(k)]
            + [cell([getter(runs, r, t, key, split) for r in rids]) for t in times]
        )
    return "\n".join(_md(head, rows))


def _train_getter(runs: OptRuns, rid: str, t: float, key: str, split: str) -> Any:
    del split  # the training rows have no split
    return train_loss(runs.train[rid]["stream"], t, key)


def drift_table(runs: OptRuns, paths: Sequence[Path]) -> str:
    """The drift of the code of each run between consecutive checkpoints
    (``stream_stats drift``; the pairs 300 s apart), a column per later
    checkpoint."""
    by_run: dict[str, dict[float, float]] = {}
    for p in paths:
        for r in json.loads(p.read_text())["runs"]:
            by_run[r["run"]] = {
                q["t2"]: q["drift"] for q in r["pairs"] if abs(q["delta"] - 300) < 1e-6
            }
    head = ["run", *[f"{t - 300:g}→{t:g}" for t in TIMES]]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            d = by_run.get(rid, {})
            rows.append([label(k)] + [fmt(d.get(t)) for t in TIMES])
    return "\n".join(_md(head, rows))


def base_table(runs: OptRuns, split: str = "val") -> str:
    """The yardsticks of time 0: per run, the untrained network's $J$, pred
    and $S$ (``baselines.json``), and the untrained ViT's ``U_KEYS`` and
    sensitivities (``task=baselines``)."""
    keys = ("loss", "pred", "sigreg")
    yard = (*U_KEYS, "sens_dither", "sens_shift")
    head = ["run", *[f"`init_{k}`" for k in keys], *[f"`{k}`" for k in yard]]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            init = untrained(runs, rid, split)
            rows.append(
                [label(k)]
                + [fmt(ev(runs, rid, 0.0, x, split)) for x in keys]
                + [fmt(init.get(x)) for x in yard]
            )
    return "\n".join(_md(head, rows))


def main(argv: Sequence[str] | None = None) -> None:
    """Every table of the module docstring."""
    p = argparse.ArgumentParser(description="The tables of the M17 optimizer sweep.")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--train", default=TRAIN_BATCH)
    p.add_argument("--post", default=POST_BATCH)
    p.add_argument("--base", default=BASE_BATCH)
    p.add_argument("--t", type=float, default=1800.0, help="the verdict's time")
    p.add_argument("--drift", type=Path, nargs="*", default=[])
    args = p.parse_args(argv)
    runs = load(args.runs, args.train, args.post, args.base)
    t = args.t
    print(f"### summary（val、{t:g} 秒）\n")
    print(summary(runs, t))
    print(f"\n### L・U・C（val、{t:g} 秒）\n")
    print(verdict_table(runs, t))
    reasons = [
        f"- {label(k)}: {runs.train[r]['result'].get('reason')}"
        for k in sorted(runs.groups, key=_order)
        for r in runs.groups[k]
        if diverged(runs, r)
    ]
    if reasons:
        print("\n発散の理由（`result.json` の `reason`）:\n")
        print("\n".join(reasons))
    print("\n### 時刻 0 の値（学習前）\n")
    print(base_table(runs))
    times = [0.0, *TIMES]
    for key in ("loss", "pred", "sigreg"):
        print(f"\n### held-out の `{key}`（val）\n")
        print(curve_table(runs, ev, key, times))
    print("\n### 学習の損失（stream.jsonl、各時刻の前 300 秒の平均）\n")
    print(curve_table(runs, _train_getter, "loss", TIMES))
    for key in ("rank", "z_spread"):
        print(f"\n### posthoc の `{key}`（val）\n")
        print(curve_table(runs, ph, key, TIMES))
    if args.drift:
        print("\n### 符号の drift（固定した積み、300 秒あいた checkpoint の対）\n")
        print(drift_table(runs, args.drift))


if __name__ == "__main__":
    main()
