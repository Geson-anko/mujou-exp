r"""DIAGNOSIS ONLY (M18 stage 1): the tables of 「段 1 の結果」 of
``exp/video/results/m18.md`` from the run directories pulled into ``./runs``
(``just fleet pull --batch ...``): the training runs of ``m18-ae-v1``
(``job.json``, ``status.json``, ``baselines.json``, ``evals.jsonl``,
``stream.jsonl``, ``timing.jsonl``), their posthoc runs of ``m18-ae-post-v1``
(``post/decode/metrics.jsonl``) and the yardsticks of ``m18-ae-base-v1``
(``post/decode/baselines.jsonl`` and ``refs.jsonl``). Reads only; prints
markdown.

The criteria are those written before the runs (results/m18.md, 「段 1：自己
符号化器（回す前に決めたこと）」), at t = 1800 s on val:

- R: ``stack_mse`` < ``R_BAR`` (the PCA-1024 of stage 0);
- U: the posthoc ``recon_blur`` and ``dyn_1/copy_1`` of the code ``ae`` both
  below those of the untrained ViT (the yardsticks' ``baselines.jsonl``
  names it ``code=init``; the pre-registration wrote ``code=vit``);
- the overfit control: ``pool_stack_mse`` < ``POOL_BAR``;
- a missing value is 不明 (``None``), a diverged run fails its criteria.

The reading of the B = 1 runs of an optimizer is the table of the
pre-registration (``reading``); Q5 compares B = 1 at 1800 s with B = 32 at 60 s
(the same number of stacks within 7 %), judged only when B = 1 went below its
own time 0 (``q5``). Q1, Q2 and Q5 are predictions of RMSProp and AdamW only
(``PREDICTED``); SGD's row reads 予測なし.

The wall-clock of training (``train_rate``) is a run's units over the wall
time of ``timing.jsonl`` less the time of its evaluations (``eval_s``); the
real-time target of the plan is ``10 / B`` units a second (a unit is
``advance`` = 3 frames of the 30 fps video, 0.1 s of it).

Run from the repository root: ``python -m exp.video.diag.m18_tables
[--runs runs]``.
"""

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from exp.video.diag.m17_tables import at, fmt, jl

R_BAR = 0.177  # stage 0: the PCA-1024 of val
POOL_BAR = 0.196  # stage 0: the PCA-1024 of the 256 stacks of the pool
T_END = 1800.0
T_Q5 = 60.0  # B = 32 at 60 s has seen 57600 stacks, B = 1 at 1800 s 54000
CKPTS = [300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0]
OPTS = ("rmsprop", "adamw", "sgd")
PREDICTED = ("rmsprop", "adamw")  # Q1, Q2 and Q5 are of these only
ARMS = (("stream", 1), ("shuffle", 1), ("pool", 1), ("stream", 32))
UNITS_PER_VIDEO_SECOND = 10  # advance 3 of 30 fps
TRAIN, POST, BASE = "m18-ae-v1", "m18-ae-post-v1", "m18-ae-base-v1"

Key = tuple[str, str, int]  # (optimizer, sampling, B)


class Runs(NamedTuple):
    """The run directories of stage 1 by ``Key``, and the yardsticks'."""

    train: dict[Key, Path]
    post: dict[Key, Path]
    base: Path | None


def _key(job: dict[str, Any]) -> Key:
    c = job["config"]
    return (
        c["optim_vision"]["kind"],
        c["data"]["tours"]["sampling"],
        c["data"]["streams"],
    )


def load(root: Path) -> Runs:
    """The runs of the three batches under ``root`` (module docstring); a
    posthoc run is keyed by its parent's key (``job["after"]``)."""

    def jobs(batch: str) -> list[tuple[Path, dict[str, Any]]]:
        d = root / batch
        runs = sorted(d.iterdir()) if d.exists() else []
        return [
            (r, json.loads((r / "job.json").read_text()))
            for r in runs
            if (r / "job.json").exists()
        ]

    train = {_key(j): r for r, j in jobs(TRAIN)}
    by_id = {r.name: k for k, r in train.items()}
    post = {by_id[j["after"][0]]: r for r, j in jobs(POST) if j["after"][0] in by_id}
    base = [r for r, _ in jobs(BASE)]
    return Runs(train, post, base[0] if base else None)


# --- the criteria ------------------------------------------------------------------


def below(x: float | None, bar: float | None, *, diverged: bool) -> bool | None:
    """``x < bar``; False for a diverged run, None (不明) if a value is
    missing."""
    if diverged:
        return False
    if x is None or bar is None:
        return None
    return x < bar


def both(a: bool | None, b: bool | None) -> bool | None:
    """The conjunction of two verdicts: False if either is False, else None
    if either is unknown."""
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def mark(v: bool | None) -> str:
    """○, × or 不明."""
    return "不明" if v is None else ("○" if v else "×")


def reading(overfit: bool | None, shuffle: bool | None, stream: bool | None) -> str:
    """The row of the reading table of the pre-registration (``shuffle`` and
    ``stream``: R and U both)."""
    if overfit is None:
        return "不明（過学習の確かめ）"
    if not overfit:
        return "実装かモデルが原因。ストリームの主張の前に直す"
    if shuffle is None or stream is None:
        return "不明（並べ替えかストリーム）"
    match shuffle, stream:
        case False, False:
            return "学習の設定かモデルが原因。段 3 の掃引を自己符号化器で回す"
        case True, False:
            return "時間相関のある順序が原因"
        case True, True:
            return "段 2 に進む"
        case _:
            return "予想していない（並べ替え ×、ストリーム ○）"


def q5(b1_end: float | None, b1_t0: float | None, b32: float | None) -> str:
    """Q5: B = 1 at 1800 s below B = 32 at 60 s; judged only if B = 1 at
    1800 s went below its own time 0."""
    if b1_end is None or b1_t0 is None or b32 is None:
        return "不明"
    if not b1_end < b1_t0:
        return "判定しない（B = 1 が時刻 0 より下がらない）"
    return "当たり" if b1_end < b32 else "外れ"


def q2(
    shuffle: tuple[bool | None, bool | None], stream: tuple[bool | None, bool | None]
) -> str:
    """Q2: the shuffle and the stream reach the same verdicts of R and of U
    (each a pair ``(R, U)``); 不明 if a verdict is unknown."""
    if None in shuffle or None in stream:
        return "不明"
    return "当たり" if shuffle == stream else "外れ"


# --- the speed -----------------------------------------------------------------------


def train_rate(timing: Sequence[dict[str, Any]]) -> float | None:
    """Units a wall-clock second of training: the units between the first
    and last rows with ``unit`` over their wall time less the ``eval_s`` of
    the rows after the first (None with fewer than two such rows)."""
    rows = [(i, r) for i, r in enumerate(timing) if "unit" in r]
    if len(rows) < 2:
        return None
    (i0, r0), (i1, r1) = rows[0], rows[-1]
    wall = _wall(r1) - _wall(r0)
    evals = sum(r.get("eval_s", 0.0) for r in timing[i0 + 1 : i1 + 1])
    return (r1["unit"] - r0["unit"]) / (wall - evals)


def _wall(r: dict[str, Any]) -> float:
    return datetime.fromisoformat(r["wall"]).timestamp()


# --- the tables ----------------------------------------------------------------------


class Facts(NamedTuple):
    """A training run's values of the criteria (module docstring)."""

    state: str
    t0: float | None  # val stack_mse at time 0
    end: float | None  # val stack_mse at 1800 s
    pool: float | None
    recon_blur: float | None
    dyn: float | None
    r: bool | None
    u: bool | None
    overfit: bool | None


def facts(runs: Runs, key: Key, vit: dict[str, Any]) -> Facts:
    """The values and verdicts of the run ``key`` against the yardstick
    line ``vit``."""
    d = runs.train.get(key)
    if d is None:
        return Facts("無い", None, None, None, None, None, None, None, None)
    state = json.loads((d / "status.json").read_text())["state"]
    dv = state == "diverged"
    ev = jl(d / "evals.jsonl")
    b = d / "baselines.json"
    t0 = json.loads(b.read_text())["val"]["stack_mse"] if b.exists() else None
    end = at(ev, T_END, "val", "stack_mse")
    pool = at(ev, T_END, "val", "pool_stack_mse")
    p = runs.post.get(key)
    m = jl(p / "post/decode/metrics.jsonl") if p else []
    rb, dyn = at(m, T_END, "val", "recon_blur"), at(m, T_END, "val", "dyn_1/copy_1")
    u = both(
        below(rb, vit.get("recon_blur"), diverged=dv),
        below(dyn, vit.get("dyn_1/copy_1"), diverged=dv),
    )
    overfit = below(pool, POOL_BAR, diverged=dv) if key[1] == "pool" else None
    r = below(end, R_BAR, diverged=dv)
    return Facts(state, t0, end, pool, rb, dyn, r, u, overfit)


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [_row(head), _row(["--"] * len(head)), *map(_row, rows)]
    return "\n".join(lines)


def _base_line(path: Path | None, name: str, code: str) -> dict[str, Any]:
    if path is None:
        return {}
    for r in jl(path / "post/decode" / name):
        if r.get("split") == "val" and r.get("code") == code:
            return r
    return {}


def report(root: Path) -> str:
    """The markdown of the tables (module docstring)."""
    runs = load(root)
    vit = _base_line(runs.base, "baselines.jsonl", "init")
    tube = _base_line(runs.base, "refs.jsonl", "tube")
    pca = _base_line(runs.base, "baselines.jsonl", "pca")
    out = ["## 物差し（m18-ae-base-v1、val）"]
    keys = ("recon", "recon_blur", "dyn_1/copy_1")
    out.append(
        _table(
            ["符号", *keys],
            [
                [n, *(fmt(line.get(k)) for k in keys)]
                for n, line in (
                    ("init（学習前の ViT）", vit),
                    ("tube", tube),
                    ("pca", pca),
                )
            ],
        )
    )
    # posthoc unit 0 recomputes the same yardstick: it should agree
    agree = []
    for key, p in sorted(runs.post.items()):
        line = _base_line(p, "baselines.jsonl", "init")
        agree.append([" ".join(map(str, key)), *(fmt(line.get(k)) for k in keys)])
    out += ["", "posthoc の unit 0 の init（一致の確かめ）", ""]
    out.append(_table(["run", *keys], agree))

    out += ["", "## 判定（val、1800 秒）", ""]
    rows, f = [], {}
    for o in OPTS:
        for s, b in ARMS:
            f[o, s, b] = x = facts(runs, (o, s, b), vit)
            rows.append(
                [
                    o,
                    f"{s} B={b}",
                    x.state,
                    fmt(x.t0),
                    fmt(x.end),
                    mark(x.r),
                    fmt(x.recon_blur),
                    fmt(x.dyn),
                    mark(x.u) if s != "pool" else "－",
                    fmt(x.pool),
                    mark(x.overfit) if s == "pool" else "－",
                ]
            )
    head = ["opt", "腕", "状態", "時刻 0", "stack_mse", "R", "recon_blur"]
    head += ["dyn_1/copy_1", "U", "pool_stack_mse", "過学習"]
    out.append(_table(head, rows))

    out += ["", "## 読み方の表（B = 1）と予測", ""]
    rows = []
    for o in OPTS:
        sh, st, po = f[o, "shuffle", 1], f[o, "stream", 1], f[o, "pool", 1]
        b32 = runs.train.get((o, "stream", 32))
        b32v = at(jl(b32 / "evals.jsonl"), T_Q5, "val", "stack_mse") if b32 else None
        learned_sh, learned_st = both(sh.r, sh.u), both(st.r, st.u)
        rows.append(
            [
                o,
                mark(po.overfit),
                mark(learned_sh),
                mark(learned_st),
                reading(po.overfit, learned_sh, learned_st),
                *(
                    ["予測なし"] * 3  # the pre-registration predicts nothing of SGD
                    if o not in PREDICTED
                    else [
                        mark(po.overfit).replace("○", "当たり").replace("×", "外れ"),
                        q2((sh.r, sh.u), (st.r, st.u)),
                        q5(st.end, st.t0, b32v),
                    ]
                ),
            ]
        )
    head = ["opt", "過学習", "並べ替え", "ストリーム", "読み", "Q1", "Q2", "Q5"]
    out.append(_table(head, rows))

    out += ["", "## checkpoint ごと（val）", ""]
    keys = ("stack_mse", "recon_blur", "dyn_1/copy_1", "own_recon", "own_recon_blur")
    rows = []
    for k, p in sorted(runs.post.items()):
        m = jl(p / "post/decode/metrics.jsonl")
        for t in CKPTS:
            rows.append([" ".join(map(str, k)), f"{t:.0f}"])
            rows[-1] += [fmt(at(m, t, "val", key)) for key in keys]
    out.append(_table(["run", "t", *keys], rows))

    out += ["", "## 学習の効率（val の stack_mse、学習の壁時計）", ""]
    rows = []
    for k, d in sorted(runs.train.items()):
        ev = jl(d / "evals.jsonl")
        rate = train_rate(jl(d / "timing.jsonl"))
        b = k[2]
        for t in (60.0, 300.0, 900.0, 1800.0):
            units = round(t * 30)
            wall = units / rate if rate else None
            rows.append(
                [
                    " ".join(map(str, k)),
                    f"{t:.0f}",
                    str(units * b),
                    fmt(wall),
                    fmt(at(ev, t, "val", "stack_mse")),
                ]
            )
    out.append(_table(["run", "t", "積み", "壁時計 s", "stack_mse"], rows))

    out += ["", "## 速さ（学習だけ、評価を除く）", ""]
    rows = []
    for k, d in sorted(runs.train.items()):
        rate = train_rate(jl(d / "timing.jsonl"))
        target = UNITS_PER_VIDEO_SECOND / k[2]
        host = json.loads((d / "provenance/attempt-1.json").read_text()).get("host")
        rows.append(
            [
                " ".join(map(str, k)),
                str(host),
                fmt(rate),
                fmt(rate * k[2] if rate else None),
                fmt(target),
                fmt(rate / target if rate else None),
            ]
        )
    head = ["run", "host", "単位/秒", "積み/秒", "目標 10/B", "目標との比"]
    out.append(_table(head, rows))
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs"))
    a = p.parse_args(argv)
    print(report(a.runs))


if __name__ == "__main__":
    main()
