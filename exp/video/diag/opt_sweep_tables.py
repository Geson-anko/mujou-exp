r"""DIAGNOSIS ONLY (M17): the tables of 「学習の設定の掃引」 in
``exp/video/results/m17.md``, the sweep of the encoder's optimizer and its
learning rate per frame for the SIGReg of the data points on one stream
(``m17/ssig_t128``, ``vision.datapoint_every=6``, seed 100), from the run
directories pulled into ``./runs`` (``just fleet pull --batch m17-opt-v1
--batch m17-opt-post-v1 --batch m17-opt-base-v1``; other batch names by the
options). Reads only; prints markdown.

A run is the (kind, lr, warmup, scale) of its ``optim_vision`` (``scale``
``"none"`` when the job predates it); under ``scale`` ``fan_in`` the lr is
$\eta$ (the rate of the vectors, $\eta / n$ of a matrix of fan-in $n$) and the
labels say so. Its criteria, fixed
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

Shown beside them and not judged: per run, the ratios $J(t)/J(0)$ and of
``U_KEYS`` to the untrained ViT with the posthoc ``rank``, ``z_spread`` and
the last drift (``lr_table``, the curves over lr of each optimizer); the curves of the held-out $J$, pred and $S$ at each checkpoint (time 0 from
``baselines.json``), the training loss (``stream.jsonl``, the mean over the
300 s before each checkpoint), the posthoc ``rank`` and ``z_spread``, and,
given ``--drift`` (the outputs of ``python -m exp.video.diag.stream_stats
drift`` of the runs), the drift between consecutive checkpoints of the
current encoder's code of fixed held-in stacks: a stand-in for the staleness
of the stored $z$, which it does not read.

Also shown, read after the runs and not criteria: the course of the held-out
$J$ over all the evals (``course_table``), ``pred`` beside the spread of the
tokens within a stack ``tok_within`` at time 0, the first eval and ``--t``
(``collapse_table``), the values behind C (``comp_table``, ``sens_table``),
the same L, U and C at each checkpoint (``times_table``) and the two parts of
the last drift (``drift_parts_table``).

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
VIEWS = ("short", "long")
CRITERIA = ("L", "U", "C", "learns")
MARK = {True: "○", False: "×", None: "?"}

Key = tuple[str, float, float, str]  # (kind, lr, warmup, scale)


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
    """(kind, lr, warmup, scale) of the job's ``optim_vision``."""
    o = job["config"]["optim_vision"]
    return (o["kind"], float(o["lr"]), float(o["warmup"]), o.get("scale", "none"))


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


def _order(k: Key) -> tuple[int, float, float, str]:
    return (KINDS.index(k[0]) if k[0] in KINDS else len(KINDS), k[1], k[2], k[3])


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
    return _init_line(runs, r["arm"], r["seed"], split)


def _init_line(runs: OptRuns, arm: str, seed: int, split: str) -> dict[str, Any]:
    for line in runs.base.get((arm, seed), []):
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


def rate(k: Key) -> str:
    """``lr=<lr>``, or ``η=<lr>`` under ``scale`` ``fan_in``."""
    return f"{'η' if k[3] == 'fan_in' else 'lr'}={k[1]:g}"


def label(k: Key) -> str:
    """``kind`` and ``rate`` (and the warm-up when there is one)."""
    w = f" warmup {k[2]:g} 秒" if k[2] else ""
    return f"`{k[0]}` {rate(k)}{w}"


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
    for kind in sorted(
        {k[0] for k in runs.groups}, key=lambda x: _order((x, 0, 0, ""))
    ):
        learn, unknown, div = [], [], []
        for k in sorted(runs.groups, key=_order):
            if k[0] != kind:
                continue
            got = [run_criteria(runs, r, t, split)["learns"] for r in runs.groups[k]]
            name = (f"η={k[1]:g}" if k[3] == "fan_in" else f"{k[1]:g}") + (
                f"（warmup {k[2]:g} 秒）" if k[2] else ""
            )
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


def drifts(paths: Sequence[Path], key: str = "drift") -> dict[str, dict[float, float]]:
    """Per run id, ``key`` (``drift``, or its numerator ``change`` and
    denominator ``apart``) of each pair of checkpoints 300 s apart by its
    later time (the outputs of ``stream_stats drift``)."""
    by_run: dict[str, dict[float, float]] = {}
    for p in paths:
        for r in json.loads(p.read_text())["runs"]:
            by_run[r["run"]] = {
                q["t2"]: q[key] for q in r["pairs"] if abs(q["delta"] - 300) < 1e-6
            }
    return by_run


def _ratio(a: Any, b: Any) -> Any:
    return None if a is None or b is None else a / b


def lr_table(
    runs: OptRuns, t: float, paths: Sequence[Path] = (), split: str = "val"
) -> str:
    """Per run, sorted by optimizer then learning rate (the curves over lr):
    $J(t)/J(0)$ (NaN for a run that diverged), ``U_KEYS`` of the posthoc
    over those of the untrained ViT, the posthoc ``rank`` and ``z_spread``
    with the untrained ViT's in a last row, and the drift of the pair
    ending at ``t``. Ratios to show, not criteria: L and U are the
    comparisons of these ratios with 1."""
    by_run = drifts(paths)
    head = ["run", f"J({t:g})/J(0)", *[f"`{k}` / 学習前" for k in U_KEYS]]
    head += ["`rank`", "`z_spread`", f"drift {t - 300:g}→{t:g}"]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            init = untrained(runs, rid, split)
            j0 = ev(runs, rid, 0.0, "loss", split)
            rows.append(
                [label(k), fmt(_ratio(objective(runs, rid, t, split), j0))]
                + [fmt(_ratio(ph(runs, rid, t, u, split), init.get(u))) for u in U_KEYS]
                + [fmt(ph(runs, rid, t, x, split)) for x in ("rank", "z_spread")]
                + [fmt(by_run.get(rid, {}).get(t))]
            )
    seeds = {(r["arm"], r["seed"]) for r in runs.train.values()}
    for arm, seed in sorted(seeds):
        init = _init_line(runs, arm, seed, split)
        rows.append(
            [f"学習前の ViT（{arm}、seed {seed}）", "1", "1", "1"]
            + [fmt(init.get(x)) for x in ("rank", "z_spread")]
            + ["–"]
        )
    return "\n".join(_md(head, rows))


def drift_table(runs: OptRuns, paths: Sequence[Path]) -> str:
    """The drift of the code of each run between consecutive checkpoints
    (``stream_stats drift``; the pairs 300 s apart), a column per later
    checkpoint."""
    by_run = drifts(paths)
    head = ["run", *[f"{t - 300:g}→{t:g}" for t in TIMES]]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            d = by_run.get(rid, {})
            rows.append([label(k)] + [fmt(d.get(t)) for t in TIMES])
    return "\n".join(_md(head, rows))


def course(
    runs: OptRuns, rid: str, split: str = "val", after: float = 300.0
) -> dict[str, Any] | None:
    """Over the held-out evals of the run with a value of $J$ (time 0 not
    among them; None if there is none): the least and the greatest $J$ with
    their times, the number of evals below $J(0)$, the number of consecutive
    pairs of evals in which $J$ rises and of all the pairs, and the greatest
    over the least $J$ of the evals at or after ``after``."""
    js = sorted(
        (r["t"], r["loss"])
        for r in runs.train[rid]["ev"]
        if r.get("split") == split and r.get("loss") is not None
    )
    if not js:
        return None
    j0 = ev(runs, rid, 0.0, "loss", split)
    late = [j for t, j in js if t >= after]
    return dict(
        least=min(js, key=lambda x: x[1]),
        greatest=max(js, key=lambda x: x[1]),
        below=None if j0 is None else sum(j < j0 for _, j in js),
        rises=sum(b[1] > a[1] for a, b in zip(js, js[1:])),
        pairs=len(js) - 1,
        late_ratio=max(late) / min(late) if late else None,
    )


def course_table(runs: OptRuns, split: str = "val", after: float = 300.0) -> str:
    """``course`` of each run."""
    head = [
        "run",
        "評価の数",
        "最小の J（秒）",
        "最大の J（秒）",
        "J(0) を下回った評価",
    ]
    head += ["J が上がった区間", f"{after:g} 秒以降の最大 / 最小"]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            c = course(runs, rid, split, after)
            if c is None:
                rows.append([label(k), "0", "–", "–", "–", "–", "–"])
                continue
            (t1, j1), (t2, j2) = c["least"], c["greatest"]
            rows.append(
                [
                    label(k),
                    str(c["pairs"] + 1),
                    f"{fmt(j1)}（{t1:g}）",
                    f"{fmt(j2)}（{t2:g}）",
                    "–" if c["below"] is None else str(c["below"]),
                    f"{c['rises']} / {c['pairs']}",
                    fmt(c["late_ratio"]),
                ]
            )
    return "\n".join(_md(head, rows))


def collapse_table(
    runs: OptRuns, t: float, first: float = 60.0, split: str = "val"
) -> str:
    """Per run, ``pred`` and ``tok_within`` (the mean squared deviation of a
    token from its stack's mean token) at time 0, ``first`` and ``t``."""
    times = (0.0, first, t)
    head = ["run"] + [f"`{x}` {u:g}" for x in ("pred", "tok_within") for u in times]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            rows.append(
                [label(k)]
                + [
                    fmt(ev(runs, rid, u, x, split))
                    for x in ("pred", "tok_within")
                    for u in times
                ]
            )
    return "\n".join(_md(head, rows))


def comp_table(runs: OptRuns, t: float, split: str = "val") -> str:
    """The values behind context, ncomp and target at ``t``: per view,
    ``comp_<v>`` with its position-only and copy-the-visible predictions
    (``_pos``, ``_vis``), ``ncomp_<v>`` with ``_pos``, ``_vis`` and
    ``_mean``, and the posthoc ``restore_target_<v>_mask``."""
    head = ["run"]
    for v in VIEWS:
        head += [
            f"`comp_{v}`（run / pos / vis）",
            f"`ncomp_{v}`（run / pos / vis / mean）",
            f"`restore_target_{v}_mask`",
        ]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            cells = [label(k)]
            for v in VIEWS:
                cells.append(
                    " / ".join(
                        fmt(ev(runs, rid, t, f"comp_{v}{s}", split))
                        for s in ("", "_pos", "_vis")
                    )
                )
                cells.append(
                    " / ".join(
                        fmt(ev(runs, rid, t, f"ncomp_{v}{s}", split))
                        for s in ("", "_pos", "_vis", "_mean")
                    )
                )
                cells.append(fmt(ph(runs, rid, t, f"restore_target_{v}_mask", split)))
            rows.append(cells)
    return "\n".join(_md(head, rows))


def sens_table(runs: OptRuns, t: float, split: str = "val") -> str:
    """Per run at ``t``, ``sens_dither`` and ``sens_shift`` of the run, of the
    untrained ViT (the yardstick of sens) and of the pixels (``_pix``)."""
    kinds = ("dither", "shift")
    head = ["run", *[f"`sens_{s}`（run / 学習前 / 画素）" for s in kinds]]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            init = untrained(runs, rid, split)
            rows.append(
                [label(k)]
                + [
                    " / ".join(
                        [
                            fmt(ph(runs, rid, t, f"sens_{s}", split)),
                            fmt(init.get(f"sens_{s}")),
                            fmt(ph(runs, rid, t, f"sens_{s}_pix", split)),
                        ]
                    )
                    for s in kinds
                ]
            )
    return "\n".join(_md(head, rows))


def times_table(
    runs: OptRuns, times: Sequence[float] = TIMES, split: str = "val"
) -> str:
    """``run_criteria`` at each of ``times``, a cell the marks of L, U and C
    in that order (learns is all three)."""
    head = ["run", *[f"{t:g}" for t in times]]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            cells = [label(k)]
            for t in times:
                c = run_criteria(runs, rid, t, split)
                cells.append("".join(MARK[c[n]] for n in ("L", "U", "C")))
            rows.append(cells)
    return "\n".join(_md(head, rows))


def drift_parts_table(runs: OptRuns, paths: Sequence[Path], t: float) -> str:
    """Per run, the pair of checkpoints ending at ``t``: ``change`` (the
    numerator: the code of the same stacks by the two encoders), ``apart``
    (the denominator: different stacks by one encoder) and ``drift``."""
    parts = ("change", "apart", "drift")
    by_key = {x: drifts(paths, x) for x in parts}
    head = ["run", *[f"`{x}` {t - 300:g}→{t:g}" for x in parts]]
    rows = []
    for k in sorted(runs.groups, key=_order):
        for rid in runs.groups[k]:
            rows.append(
                [label(k)] + [fmt(by_key[x].get(rid, {}).get(t)) for x in parts]
            )
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
    print(f"\n### lr に対する曲線（val、{t:g} 秒）\n")
    print(lr_table(runs, t, args.drift))
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
        print(f"\n### drift の分子と分母（{t - 300:g}→{t:g} 秒）\n")
        print(drift_parts_table(runs, args.drift, t))
    print("\n### held-out の J の経過（val、全部の評価）\n")
    print(course_table(runs))
    print("\n### pred と積みの中のトークンの広がり（val）\n")
    print(collapse_table(runs, t))
    print(f"\n### C の値（val、{t:g} 秒）\n")
    print(comp_table(runs, t))
    print(f"\n### sens の値（val、{t:g} 秒）\n")
    print(sens_table(runs, t))
    print("\n### 各 checkpoint の L・U・C（val）\n")
    print(times_table(runs))


if __name__ == "__main__":
    main()
