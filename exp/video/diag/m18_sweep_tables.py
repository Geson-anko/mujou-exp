r"""DIAGNOSIS ONLY (M18): the tables of 「η の掃引」 and 「デコーダの大きさ」 of
``exp/video/results/m18.md`` from the run directories pulled into ``./runs``
(``just fleet pull --batch ...``). Reads only; prints markdown.

The B = 1 runs of the autoencoder are keyed by (optimizer, learning rate,
sampling, decoder depth): those of stage 1 (``m18-ae-v1``, ``m18-ae-post-v1``:
the point of each optimizer in the window, reused), of the sweep
(``m18-sweep-v1``, ``m18-sweep-post-v1``) and of the decoder axis
(``m18-dec-v1``, ``m18-dec-post-v1``); a posthoc run is keyed by its parent's
key (``job["after"]``). The yardstick of U is ``m18-ae-base-v1``'s untrained
ViT (``code=init``): the encoder's initial weights do not depend on the
decoder's depth (``modules.build_ae`` draws the decoder after it).

The criteria are those of stage 1 (``m18_tables``), at t = 1800 s on val: R
(the stream's ``stack_mse`` < ``R_BAR``), U (the posthoc ``recon_blur`` and
``dyn_1/copy_1`` both below the untrained ViT's), the overfit check (the pool
arm's ``pool_stack_mse`` < ``POOL_BAR``). ``select`` is the rule written
before the runs (revised before the results to drop the overfit gate): of
the stream runs that did not diverge and hold U, the lowest posthoc
``recon_blur``; the overfit check reads the system's decoder, not the code,
and is shown beside it.

``scaled`` reads a run of a rate below the stage 1 rate against the stage 1
run of the same optimizer and arm at the time ``t * lr / lr_ref``: if a step
moves the weights by a fixed multiple of the rate (the Adam family at its
per-element bound, or SGD's gradient flow), the two follow one path.

Run from the repository root: ``python -m exp.video.diag.m18_sweep_tables
[--runs runs]``.
"""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from exp.video.diag.m17_tables import at, fmt, jl
from exp.video.diag.m18_tables import (
    CKPTS,
    POOL_BAR,
    R_BAR,
    T_END,
    below,
    both,
    mark,
)

# The stage 1 rates (per frame): the geometric mean of the window of the
# Adam family, and SGD's 1 / (1.02 rho_max) of stage 0's Hessian.
LR_REF = {"rmsprop": 2.9e-5, "adamw": 2.9e-5, "sgd": 2.05e-7}
# SGD's boundary 2 / (1.02 rho_max), rho_max = 4.793e6 (results/m18-stage0/
# hessian.json), rounded down to 3 digits so that it stays below it.
SGD_TOP = 4.09e-7
GRID = {
    "rmsprop": (1e-5, 1e-4, 1e-3, 1e-2),
    "adamw": (1e-5, 1e-4, 1e-3, 1e-2),
    "sgd": (4.09e-10, 4.09e-9, 4.09e-8, SGD_TOP),
}
OPTS = ("rmsprop", "adamw", "sgd")
DEPTHS = (2, 4, 8)  # 1, 2 and 4 nodes of 2 blocks; 8 is the encoder's depth
DEPTH_REF = 2  # stage 1's decoder
TRAIN = ("m18-ae-v1", "m18-sweep-v1", "m18-dec-v1")
POST = ("m18-ae-post-v1", "m18-sweep-post-v1", "m18-dec-post-v1")
BASE = "m18-ae-base-v1"

Key = tuple[str, float, str, int]  # (optimizer, lr, sampling, decoder depth)


class Runs(NamedTuple):
    """The B = 1 run directories by ``Key``, and the yardsticks'."""

    train: dict[Key, Path]
    post: dict[Key, Path]
    base: Path | None


def _key(job: dict[str, Any]) -> Key | None:
    """The key of a job's config; None for B > 1."""
    c = job["config"]
    if c["data"]["streams"] != 1:
        return None
    o = c["optim_vision"]
    return (
        o["kind"],
        o["lr"],
        c["data"]["tours"]["sampling"],
        c["ae"]["decoder_depth"],
    )


def _jobs(root: Path, batch: str) -> list[tuple[Path, dict[str, Any]]]:
    d = root / batch
    runs = sorted(d.iterdir()) if d.exists() else []
    return [
        (r, json.loads((r / "job.json").read_text()))
        for r in runs
        if (r / "job.json").exists()
    ]


def load(root: Path) -> Runs:
    """The B = 1 runs of the batches under ``root`` (module docstring)."""
    train: dict[Key, Path] = {}
    for batch in TRAIN:
        for r, j in _jobs(root, batch):
            if (k := _key(j)) is not None:
                train[k] = r
    by_id = {r.name: k for k, r in train.items()}
    post = {
        by_id[j["after"][0]]: r
        for batch in POST
        for r, j in _jobs(root, batch)
        if j["after"] and j["after"][0] in by_id
    }
    base = [r for r, _ in _jobs(root, BASE)]
    return Runs(train, post, base[0] if base else None)


# --- the values and the rule ----------------------------------------------------


class Facts(NamedTuple):
    """A run's values at 1800 s on val and its verdicts (module docstring);
    the posthoc's values are of a stream run."""

    state: str
    end: float | None  # stack_mse
    pool: float | None  # pool_stack_mse
    recon: float | None
    recon_blur: float | None
    dyn: float | None
    rank: float | None
    r: bool | None
    u: bool | None
    overfit: bool | None
    own: float | None  # the system's decoder g: own_recon, own_recon_blur
    own_blur: float | None


def facts(runs: Runs, key: Key, vit: dict[str, Any]) -> Facts:
    """The values and verdicts of the run ``key`` against the yardstick
    line ``vit``."""
    d = runs.train.get(key)
    if d is None:
        return Facts("無い", *(None,) * 11)
    state = json.loads((d / "status.json").read_text())["state"]
    dv = state == "diverged"
    ev = jl(d / "evals.jsonl")
    end = at(ev, T_END, "val", "stack_mse")
    pool = at(ev, T_END, "val", "pool_stack_mse")
    p = runs.post.get(key)
    m = jl(p / "post/decode/metrics.jsonl") if p else []

    def post(name: str) -> float | None:
        return at(m, T_END, "val", name)

    rb, dyn = post("recon_blur"), post("dyn_1/copy_1")
    u = both(
        below(rb, vit.get("recon_blur"), diverged=dv),
        below(dyn, vit.get("dyn_1/copy_1"), diverged=dv),
    )
    overfit = below(pool, POOL_BAR, diverged=dv) if key[2] == "pool" else None
    r = below(end, R_BAR, diverged=dv)
    return Facts(
        state,
        end,
        pool,
        post("recon"),
        rb,
        dyn,
        post("rank"),
        r,
        u,
        overfit,
        post("own_recon"),
        post("own_recon_blur"),
    )


class Candidate(NamedTuple):
    """A stream run for ``select``: its posthoc ``recon_blur`` and U, and
    whether it diverged."""

    key: Any
    recon_blur: float | None
    u: bool | None
    diverged: bool


def select(cands: Sequence[Candidate]) -> Any:
    """The rule written before the runs: of the candidates that did not
    diverge, hold U and have a ``recon_blur``, the key of the one of the
    lowest ``recon_blur``. None if no candidate is eligible."""
    ok = [
        (c.recon_blur, c.key)
        for c in cands
        if not c.diverged and c.u is True and c.recon_blur is not None
    ]
    return min(ok, key=lambda vk: vk[0])[1] if ok else None


def first_pass(rates: Sequence[float], verdicts: Sequence[bool | None]) -> float | None:
    """The lowest rate whose verdict is True (None if none is)."""
    passed = [lr for lr, v in sorted(zip(rates, verdicts, strict=True)) if v is True]
    return passed[0] if passed else None


def scaled(rows: Sequence[dict[str, Any]], t0: float | None, t: float, key: str) -> Any:
    """``key`` of val at time ``t`` by linear interpolation between the
    evaluations of ``rows`` and the time 0 value ``t0`` (None outside them)."""
    pts = sorted(
        (r["t"], r[key])
        for r in rows
        if r.get("split") == "val" and key in r and r["t"] > 0
    )
    if t0 is not None:
        pts.insert(0, (0.0, t0))
    for (ta, va), (tb, vb) in zip(pts, pts[1:], strict=False):
        if ta <= t <= tb:
            return va + (vb - va) * (t - ta) / (tb - ta)
    return None


def last_t(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    """The last time of val with a finite ``key`` (None if there is none): how
    far a run that diverged got."""
    ts = [
        r["t"]
        for r in rows
        if r.get("split") == "val"
        and isinstance(r.get(key), (int, float))
        and math.isfinite(r[key])
    ]
    return max(ts) if ts else None


# --- the tables -----------------------------------------------------------------


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join([_row(head), _row(["--"] * len(head)), *map(_row, rows)])


def _base_line(path: Path | None, code: str) -> dict[str, Any]:
    for r in jl(path / "post/decode/baselines.jsonl") if path else []:
        if r.get("split") == "val" and r.get("code") == code:
            return r
    return {}


def _rates(opt: str) -> list[float]:
    return sorted({*GRID[opt], LR_REF[opt]})


def _t0(d: Path | None, key: str) -> float | None:
    b = d / "baselines.json" if d else None
    return json.loads(b.read_text())["val"].get(key) if b and b.exists() else None


def report(root: Path) -> str:
    """The markdown of the tables (module docstring)."""
    runs = load(root)
    vit = _base_line(runs.base, "init")
    out = [
        "## η の掃引（B = 1、デコーダ 2 ブロック、val、1800 秒）",
        "",
        f"U の物差し（学習前の ViT）: recon_blur {fmt(vit.get('recon_blur'))}、"
        f"dyn_1/copy_1 {fmt(vit.get('dyn_1/copy_1'))}",
        "",
    ]
    rows, cands = [], []
    for o in OPTS:
        for lr in _rates(o):
            st = facts(runs, (o, lr, "stream", DEPTH_REF), vit)
            po = facts(runs, (o, lr, "pool", DEPTH_REF), vit)
            rows.append(
                [
                    o,
                    f"{lr:g}",
                    f"{st.state} / {po.state}",
                    fmt(st.end),
                    mark(st.r),
                    fmt(st.recon),
                    fmt(st.recon_blur),
                    fmt(st.dyn),
                    fmt(st.rank),
                    mark(st.u),
                    fmt(po.pool),
                    mark(po.overfit),
                ]
            )
            dv = st.state == "diverged"
            cands.append(Candidate((o, lr), st.recon_blur, st.u, dv))
    head = ["opt", "lr", "状態（stream / pool）", "stack_mse", "R", "recon"]
    head += ["recon_blur", "dyn_1/copy_1", "rank", "U", "pool_stack_mse", "過学習"]
    out.append(_table(head, rows))

    pick = select(cands)
    out += ["", "## 選んだ点", ""]
    if pick is None:
        out.append("選べない（条件を満たすストリームの run が無い）")
    else:
        o, lr = pick
        out.append(f"{o}、lr = {lr:g}")

    out += ["", "## 過学習の確かめを最初に通る学習率（予測の判定用）", ""]
    rows = []
    for o in OPTS:
        rates = _rates(o)
        v = [facts(runs, (o, lr, "pool", DEPTH_REF), vit).overfit for lr in rates]
        lr0 = first_pass(rates, v)
        rows.append([o, "無い" if lr0 is None else f"{lr0:g}"])
    out.append(_table(["opt", "最初に通る lr"], rows))

    out += ["", "## 一歩の大きさの尺度（段 1 の run の t · lr / lr_ref と比べる）", ""]
    rows = []
    for o in OPTS:
        for lr in GRID[o]:
            if lr >= LR_REF[o]:
                continue
            for s, k in (("stream", "stack_mse"), ("pool", "pool_stack_mse")):
                d = runs.train.get((o, lr, s, DEPTH_REF))
                ref = runs.train.get((o, LR_REF[o], s, DEPTH_REF))
                got = at(jl(d / "evals.jsonl"), T_END, "val", k) if d else None
                ts = T_END * lr / LR_REF[o]
                want = (
                    scaled(jl(ref / "evals.jsonl"), _t0(ref, k), ts, k) if ref else None
                )
                rows.append([o, f"{lr:g}", s, k, fmt(got), f"{ts:.1f}", fmt(want)])
    head = ["opt", "lr", "腕", "値", "1800 秒の値", "段 1 の時刻 s", "段 1 の値"]
    out.append(_table(head, rows))

    out += [
        "",
        "## 経過（val、run の evals.jsonl、過学習の確かめは pool_stack_mse）",
        "",
    ]
    times = (60.0, *CKPTS)
    rows = []
    for o in OPTS:
        for lr in _rates(o):
            for s, k in (("stream", "stack_mse"), ("pool", "pool_stack_mse")):
                d = runs.train.get((o, lr, s, DEPTH_REF))
                ev = jl(d / "evals.jsonl") if d else []
                last = last_t(ev, k)
                rows.append(
                    [o, f"{lr:g}", s, fmt(_t0(d, k))]
                    + [fmt(at(ev, t, "val", k)) for t in times]
                    + ["–" if last is None else f"{last:.0f}"]
                )
    head = ["opt", "lr", "腕", "時刻 0", *(f"{t:.0f}" for t in times)]
    out.append(_table([*head, "値のある最後の t"], rows))

    out += ["", "## checkpoint ごとの posthoc（ストリーム、val）", ""]
    keys = ("recon", "recon_blur", "dyn_1/copy_1", "rank", "own_recon")
    rows = []
    for o in OPTS:
        for lr in _rates(o):
            p = runs.post.get((o, lr, "stream", DEPTH_REF))
            m = jl(p / "post/decode/metrics.jsonl") if p else []
            for t in CKPTS:
                rows.append([o, f"{lr:g}", f"{t:.0f}"])
                rows[-1] += [fmt(at(m, t, "val", key)) for key in keys]
    out.append(_table(["opt", "lr", "t", *keys], rows))

    out += ["", "## デコーダの大きさ（val、1800 秒）", ""]
    if pick is None:
        out.append("η の点を選べないので無い")
        return "\n".join(out)
    o, lr = pick
    rows, cands = [], []
    for dep in DEPTHS:
        st = facts(runs, (o, lr, "stream", dep), vit)
        po = facts(runs, (o, lr, "pool", dep), vit)
        rows.append(
            [
                str(dep),
                f"{st.state} / {po.state}",
                fmt(st.recon),
                fmt(st.recon_blur),
                fmt(st.dyn),
                fmt(st.rank),
                mark(st.u),
                fmt(st.end),
                fmt(st.own),
                fmt(st.own_blur),
                fmt(po.pool),
                mark(po.overfit),
            ]
        )
        cands.append(Candidate(dep, st.recon_blur, st.u, st.state == "diverged"))
    head = ["深さ", "状態（stream / pool）", "recon", "recon_blur", "dyn_1/copy_1"]
    head += [
        "rank",
        "U",
        "stack_mse（参考）",
        "own_recon（参考）",
        "own_recon_blur（参考）",
        "pool_stack_mse（参考）",
        "過学習（参考）",
    ]
    out.append(_table(head, rows))
    dpick = select(cands)
    out += [
        "",
        f"{o}、lr = {lr:g} で選んだ深さ: {'選べない' if dpick is None else dpick}",
    ]

    out += [
        "",
        "## デコーダの大きさの checkpoint ごとの posthoc（ストリーム、val）",
        "",
    ]
    keys = ("recon", "recon_blur", "dyn_1/copy_1", "rank", "own_recon")
    keys += ("own_recon_blur", "stack_mse")
    rows = []
    for dep in DEPTHS:
        p = runs.post.get((o, lr, "stream", dep))
        m = jl(p / "post/decode/metrics.jsonl") if p else []
        for t in CKPTS:
            rows.append([str(dep), f"{t:.0f}"])
            rows[-1] += [fmt(at(m, t, "val", key)) for key in keys]
    out.append(_table(["深さ", "t", *keys], rows))

    out += ["", "## デコーダの大きさの経過（val、run の evals.jsonl）", ""]
    times = (60.0, *CKPTS)
    rows = []
    for dep in DEPTHS:
        for s, k in (("stream", "stack_mse"), ("pool", "pool_stack_mse")):
            d = runs.train.get((o, lr, s, dep))
            ev = jl(d / "evals.jsonl") if d else []
            rows.append(
                [str(dep), s, fmt(_t0(d, k))]
                + [fmt(at(ev, t, "val", k)) for t in times]
            )
    out.append(_table(["深さ", "腕", "時刻 0", *(f"{t:.0f}" for t in times)], rows))
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs"))
    a = p.parse_args(argv)
    print(report(a.runs))


if __name__ == "__main__":
    main()
