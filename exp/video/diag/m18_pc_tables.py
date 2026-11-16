r"""DIAGNOSIS ONLY (M18 stage 4): the tables of 「段 4：PC-ALM の結果」 of
``exp/video/results/m18.md`` from the run directories pulled into ``./runs``
(``just fleet pull --batch ...``): the PC-ALM runs of ``m18-pc-v1`` and the
BP runs at 29 ms of ``m18-bp29-v1`` (``job.json``, ``status.json``,
``baselines.json``, ``evals.jsonl``, ``stream.jsonl``, ``timing.jsonl``),
their posthoc runs (``m18-pc-post-v1``, ``m18-bp29-post-v1``:
``post/decode/metrics.jsonl``, ``baselines.jsonl``, ``refs.jsonl``), stage
2's BP at 25 ms (``QP6_REF``), and the outputs of ``m18_pc_grad`` (QP3,
QP4). Reads only; prints markdown, and with ``--json`` writes the values the
tables are made of.

The verdict is the one written before the runs (results/m18.md, 「段 4：
PC-ALM（回す前に決めたこと）」), at t = 1800 s on val, unrounded (``verdict``):

- U: for each PC-ALM run of B = 1 (seeds 100, 0, 1, 2), ``recon_blur`` and
  ``dyn_1/copy_1`` of its posthoc's ``code=ae`` line both below those of the
  same posthoc's ``code=init`` line (the untrained $\mu$ of its seed);
- the difference from BP: for each of the two metrics, $|\bar m_{PC} - \bar
  m_{BP}|$ (the means over the 4 seeds) at most $\max_s m_{BP}(s) - \min_s
  m_{BP}(s)$;
- a missing value of the 8 runs makes a criterion 不明; a diverged run fails
  it.

The predictions (``predictions``): QP1 U; QP2 the difference within BP's
range for both metrics; QP3 no ``loss`` and QP4 no tensor in any
``failures`` of ``m18_pc_grad`` (``grad_summary``); QP5 (speed) is not judged
(the pre-registration) and its ranges are reported; QP6 BP at 29 ms of seed
100 against stage 2's 25 ms: ``stream.jsonl``'s $-\mathcal L$ up to 120 s and
val's ``bpd`` at 60 s and 120 s within a relative ``QP6_REL`` (``qp6``).

The diagnoses (not judged): the first ``stream.jsonl`` row of each pair, the
likelihood every 60 s per pair (``pair_track``: the largest $|PC - BP|$ and
the first time it exceeds the range of BP's 4 seeds at that time), B = 32,
the speed (``m18_tables.train_rate``) and the posthoc per checkpoint per pair.

Run from the repository root: ``python -m exp.video.diag.m18_pc_tables
[--runs runs] [--grad DIR] [--json F]``.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from exp.video.diag.m17_tables import at, fmt, jl
from exp.video.diag.m18_tables import CKPTS, T_END, below, both, mark, train_rate

TRAIN = {"pc": "m18-pc-v1", "bp": "m18-bp29-v1"}
POST = {"pc": "m18-pc-post-v1", "bp": "m18-bp29-post-v1"}
SEEDS = (100, 0, 1, 2)
METRICS = ("recon_blur", "dyn_1/copy_1")
QP6_REF = "m18-vae-v1/m18-vae_t128-eb82006a-s100"  # stage 2, BP at 25 ms
QP6_REL, QP6_T = 1e-6, 120.0
QP5 = {1: (10.0, 11.3), 32: (2.3, 2.6)}  # units a second, B = 1 and B = 32
BENCH = {("pc", 1): 10.70, ("pc", 32): 2.41, ("bp", 1): 70.1, ("bp", 32): 23.3}
LIKELIHOOD = ("bpd", "kl", "sigma", "sigma_star", "stack_mse")
POSTHOC = ("recon", "recon_blur", "dyn_1/copy_1", "rank", "own_recon_blur")
POSTHOC += ("stack_mse",)
R_BAR, COLLAPSE = 6.887, 8.127  # stage 2: the pPCA-1024 and the mean stack, val
NAMES = {"pc": "PC-ALM", "bp": "BP（29 ms）"}

Key = tuple[str, int, int]  # (rule, B, seed)


class Runs(NamedTuple):
    """The run directories of stage 4 by ``Key``."""

    train: dict[Key, Path]
    post: dict[Key, Path]


def _jobs(root: Path, batch: str) -> list[tuple[Path, dict[str, Any]]]:
    d = root / batch
    runs = sorted(d.iterdir()) if d.exists() else []
    return [
        (r, json.loads((r / "job.json").read_text()))
        for r in runs
        if (r / "job.json").exists()
    ]


def load(root: Path) -> Runs:
    """The runs of the four batches under ``root``; a posthoc run is keyed by
    its parent's key (``job["after"]``)."""
    train: dict[Key, Path] = {}
    post: dict[Key, Path] = {}
    for rule in ("pc", "bp"):
        mine = {}
        for r, j in _jobs(root, TRAIN[rule]):
            mine[r.name] = (rule, j["config"]["data"]["streams"], j["config"]["seed"])
            train[mine[r.name]] = r
        for r, j in _jobs(root, POST[rule]):
            if j["after"] and j["after"][0] in mine:
                post[mine[j["after"][0]]] = r
    return Runs(train, post)


# --- the verdict -----------------------------------------------------------------------


def verdict(
    pc: Mapping[int, Mapping[str, float | None]],
    bp: Mapping[int, Mapping[str, float | None]],
    init: Mapping[int, Mapping[str, float | None]],
    *,
    diverged: frozenset[int] = frozenset(),
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    r"""The verdict of the module docstring of the posthoc values at 1800 s:
    ``pc``, ``bp`` and ``init`` (the untrained $\mu$ of each seed) map a
    seed to its ``METRICS``; ``diverged`` the seeds whose PC-ALM or BP run
    diverged. Returns U per seed (``u``) and in all (``U``), per metric the
    means, the difference, BP's range and whether it is within (``diff``),
    their conjunction (``D``) and the two together (``passed``)."""
    u = {}
    for s in seeds:
        dv = s in diverged
        p, i = pc.get(s, {}), init.get(s, {})
        u[s] = _u(p, i, diverged=dv)
    big_u = _all(list(u.values()))
    diff: dict[str, dict[str, Any]] = {}
    for m in METRICS:
        a = [pc.get(s, {}).get(m) for s in seeds]
        b = [bp.get(s, {}).get(m) for s in seeds]
        if diverged:
            diff[m] = {"within": False}
            continue
        if any(x is None for x in a + b):
            diff[m] = {"within": None}
            continue
        aa = [float(x) for x in a if x is not None]
        bb = [float(x) for x in b if x is not None]
        mean_pc, mean_bp = sum(aa) / len(aa), sum(bb) / len(bb)
        d, rng = abs(mean_pc - mean_bp), max(bb) - min(bb)
        diff[m] = {
            "mean_pc": mean_pc,
            "mean_bp": mean_bp,
            "diff": d,
            "range_bp": rng,
            "within": d <= rng,
        }
    big_d = _all([diff[m]["within"] for m in METRICS])
    return {"u": u, "U": big_u, "diff": diff, "D": big_d, "passed": both(big_u, big_d)}


def _u(
    post: Mapping[str, float | None],
    init: Mapping[str, float | None],
    *,
    diverged: bool,
) -> bool | None:
    """U of one run: both ``METRICS`` of ``post`` below ``init``'s."""
    rb, dyn = METRICS
    return both(
        below(post.get(rb), init.get(rb), diverged=diverged),
        below(post.get(dyn), init.get(dyn), diverged=diverged),
    )


def _all(vs: Sequence[bool | None]) -> bool | None:
    out: bool | None = True
    for v in vs:
        out = both(out, v)
    return out


# --- the likelihood per pair -----------------------------------------------------------


def pair_track(
    pc: Mapping[float, float],
    bp: Mapping[float, float],
    bp_all: Sequence[Mapping[float, float]],
) -> dict[str, Any]:
    """Of a pair's values by time (``pc``, ``bp``) and BP's of every seed
    (``bp_all``): the largest $|PC - BP|$ and its time over the times both
    have (``max_abs``, ``t_max``), and the first time it exceeds the range
    of ``bp_all`` at that time (``first_out``; None if never; only times
    where every seed of ``bp_all`` has a value; None without ``bp_all``)."""
    ts = sorted(t for t in pc if t in bp)
    best, t_best, first = 0.0, None, None
    for t in ts:
        d = abs(pc[t] - bp[t])
        if t_best is None or d > best:
            best, t_best = d, t
        vals = [r[t] for r in bp_all if t in r]
        full = bool(bp_all) and len(vals) == len(bp_all)
        if first is None and full and d > max(vals) - min(vals):
            first = t
    return {
        "max_abs": best if t_best is not None else None,
        "t_max": t_best,
        "first_out": first,
    }


# --- the gradient check (QP3, QP4) -----------------------------------------------------


def grad_summary(grad: Mapping[str, Any]) -> dict[str, Any]:
    r"""Of an output of ``m18_pc_grad`` (points by name): per point its
    ``failures``, the loss's gap $|\mathrm{nll} + \mathrm{kl} -
    \mathrm{exact}|$ of float64 PC-ALM over the bound $2^{-23}
    (|\mathrm{nll}| + |\mathrm{kl}|)$ (``loss_ratio``), the largest ratio
    over the tensors of float64 PC-ALM's relative error to float32 BP's
    (``rel_ratio``), and the worst ``rel`` of float32 BP, float32 PC-ALM,
    float64 PC-ALM and of PC-ALM against BP in float32 (``worst``); QP3
    (no ``loss``) and QP4 (no tensor) over all points."""
    points = {}
    qp3 = qp4 = True
    for name, p in grad.items():
        st = p["loss"]["pc64"]
        bound = 2.0**-23 * (abs(st["nll"]) + abs(st["kl"]))
        gap = abs(st["nll"] + st["kl"] - p["loss"]["exact"])
        ratios = [
            c["rel"] / f["rel"]
            for f, c in zip(p["tensors"]["bp"], p["tensors"]["pc64"], strict=True)
            if f["rel"]
        ]
        fails = list(p["failures"])
        qp3 &= "loss" not in fails
        qp4 &= not [f for f in fails if f != "loss"]
        points[name] = {
            "t": p["t"],
            "failures": fails,
            "loss_ratio": gap / bound,
            "rel_ratio": max(ratios),
            "worst": {k: p[k]["max_rel"] for k in ("bp", "pc", "pc64", "pc_bp")},
        }
    return {"points": points, "QP3": qp3, "QP4": qp4}


# --- QP6 ------------------------------------------------------------------------------


def qp6(
    stream: Sequence[Mapping[str, Any]],
    ref_stream: Sequence[Mapping[str, Any]],
    evals: Sequence[dict[str, Any]],
    ref_evals: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    r"""QP6 of a run's and the reference's ``stream.jsonl`` and
    ``evals.jsonl`` rows: the largest relative difference of $-\mathcal L$
    (``loss``) at the seconds up to ``QP6_T`` that both have, and of val's
    ``bpd`` at 60 s and 120 s; ``held`` when both are within ``QP6_REL``
    (None when a value is missing)."""
    ref = {r["t"]: r["loss"] for r in ref_stream if r["t"] <= QP6_T}
    pairs = [(r["loss"], ref[r["t"]]) for r in stream if r["t"] in ref]
    loss = max((abs(a - b) / abs(b) for a, b in pairs), default=None)
    bpd = []
    for t in (60.0, QP6_T):
        a, b = at(list(evals), t, "val", "bpd"), at(list(ref_evals), t, "val", "bpd")
        bpd.append(None if a is None or b is None else abs(a - b) / abs(b))
    vals = [loss, *bpd]
    held = None if None in vals else all(v <= QP6_REL for v in vals if v is not None)
    return {"loss_rel": loss, "rows": len(pairs), "bpd_rel": bpd, "held": held}


def judge(v: bool | None) -> str:
    """○, × or 判定不能."""
    return "判定不能" if v is None else ("○" if v else "×")


# --- the tables ------------------------------------------------------------------------


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join([_row(head), _row(["--"] * len(head)), *map(_row, rows)])


def _line(path: Path, split: str, code: str) -> dict[str, Any]:
    for r in jl(path):
        if r.get("split") == split and r.get("code") == code:
            return r
    return {}


def _series(d: Path | None, key: str) -> dict[float, float]:
    """``key`` of val by time: ``baselines.json`` at 0, then ``evals.jsonl``."""
    if d is None:
        return {}
    out: dict[float, float] = {}
    b = d / "baselines.json"
    if b.exists() and (v := json.loads(b.read_text())["val"].get(key)) is not None:
        out[0.0] = v
    for r in jl(d / "evals.jsonl"):
        if r.get("split") == "val" and r.get(key) is not None:
            out[float(r["t"])] = r[key]
    return out


def _state(d: Path | None) -> str | None:
    return json.loads((d / "status.json").read_text())["state"] if d else None


def _g(x: float | None) -> str:
    return "–" if x is None else f"{x:.6g}"


def report(root: Path, grad_dir: Path | None) -> tuple[str, dict[str, Any]]:
    """The markdown of the tables (module docstring), and their values."""
    runs = load(root)
    data: dict[str, Any] = {}
    out = ["## 実行の確かめ", ""]
    rows = []
    for key in sorted(runs.train, key=lambda k: (k[0], -k[1], SEEDS.index(k[2]))):
        for kind, d in (("学習", runs.train[key]), ("posthoc", runs.post.get(key))):
            if d is None:
                continue
            j = json.loads((d / "job.json").read_text())
            s = json.loads((d / "status.json").read_text())
            prov = d / "provenance/attempt-1.json"
            host = json.loads(prov.read_text())["host"] if prov.exists() else "–"
            c = j["code"]
            rows.append(
                [NAMES[key[0]], str(key[1]), str(key[2]), kind, d.name, s["state"]]
                + [f"{s['t']:g} / {s['of']:g}", c["snap"][:12], c["mujou_snap"][:7]]
                + [str(c["dirty"]), host]
            )
    head = ["学習則", "B", "seed", "種類", "run", "状態", "t / of", "snapshot"]
    out.append(_table(head + ["mujou", "dirty", "host"], rows))

    # the posthoc values at 1800 s and the verdict
    post: dict[Key, dict[str, Any]] = {}
    init: dict[Key, dict[str, Any]] = {}
    yard: dict[Key, dict[str, Any]] = {}
    for key, p in runs.post.items():
        m = jl(p / "post/decode/metrics.jsonl")
        post[key] = {x: at(m, T_END, "val", x) for x in POSTHOC}
        init[key] = _line(p / "post/decode/baselines.jsonl", "val", "init")
        pca = _line(p / "post/decode/baselines.jsonl", "val", "pca")
        tube = _line(p / "post/decode/refs.jsonl", "val", "tube")
        yard[key] = {
            "pca_recon_blur": pca.get("recon_blur"),
            "tube_dyn": tube.get("dyn_1/copy_1"),
        }
    dv = frozenset(
        s for (r, b, s), d in runs.train.items() if b == 1 and _state(d) == "diverged"
    )
    v = verdict(
        {s: post.get(("pc", 1, s), {}) for s in SEEDS},
        {s: post.get(("bp", 1, s), {}) for s in SEEDS},
        {s: init.get(("pc", 1, s), {}) for s in SEEDS},
        diverged=dv,
    )
    data["verdict"] = v
    out += ["", "## 判定（val、1800 秒、posthoc、丸める前）", ""]
    rows = [
        [str(s), _g(post.get(("pc", 1, s), {}).get("recon_blur"))]
        + [_g(init.get(("pc", 1, s), {}).get("recon_blur"))]
        + [_g(post.get(("pc", 1, s), {}).get("dyn_1/copy_1"))]
        + [_g(init.get(("pc", 1, s), {}).get("dyn_1/copy_1")), mark(v["u"][s])]
        for s in SEEDS
    ]
    out.append(
        _table(["seed", "recon_blur", "学習前", "dyn_1/copy_1", "学習前", "U"], rows)
    )
    out.append("")
    rows = [
        [m, _g(d.get("mean_pc")), _g(d.get("mean_bp")), _g(d.get("diff"))]
        + [_g(d.get("range_bp")), mark(d["within"])]
        for m, d in v["diff"].items()
    ]
    out.append(
        _table(["指標", "PC の平均", "BP の平均", "差", "BP の範囲", "範囲内"], rows)
    )
    out.append(
        f"\nU {mark(v['U'])}、BP との差 {mark(v['D'])}、合わせて {mark(v['passed'])}"
    )

    # per seed, with the yardsticks
    out += ["", "## seed ごと（val、1800 秒）", ""]
    rows = []
    for s in SEEDS:
        for r in ("pc", "bp"):
            k = (r, 1, s)
            p = post.get(k, {})
            bpd = (
                at(jl(runs.train[k] / "evals.jsonl"), T_END, "val", "bpd")
                if k in runs.train
                else None
            )
            data.setdefault("per_seed", {}).setdefault(str(s), {})[r] = (
                p
                | {
                    "bpd": bpd,
                    "init": {m: init.get(k, {}).get(m) for m in METRICS},
                }
                | yard.get(k, {})
            )
            rows.append(
                [str(s), NAMES[r], *(fmt(p.get(x)) for x in POSTHOC), fmt(bpd)]
                + [mark(below(bpd, R_BAR, diverged=False))]
                + [mark(below(bpd, COLLAPSE, diverged=False))]
                + [
                    fmt(init.get(k, {}).get("recon_blur")),
                    fmt(init.get(k, {}).get("dyn_1/copy_1")),
                ]
                + [
                    fmt(yard.get(k, {}).get("pca_recon_blur")),
                    fmt(yard.get(k, {}).get("tube_dyn")),
                ]
            )
    head = ["seed", "学習則", *POSTHOC, "bpd", "R", "崩壊なし"]
    head += ["学習前 rb", "学習前 dyn", "画素 PCA rb", "tube dyn"]
    out.append(_table(head, rows))
    out += ["", "## seed ごとの差（PC − BP、val、1800 秒）", ""]
    rows = []
    for s in SEEDS:
        a, b = post.get(("pc", 1, s), {}), post.get(("bp", 1, s), {})
        diffs = [
            None if a.get(x) is None or b.get(x) is None else a[x] - b[x]
            for x in POSTHOC
        ]
        data.setdefault("per_seed_diff", {})[str(s)] = dict(
            zip(POSTHOC, diffs, strict=True)
        )
        rows.append([str(s), *map(_g, diffs)])
    out.append(_table(["seed", *POSTHOC], rows))

    # the first stream row
    out += ["", "## stream.jsonl の最初の行の −L（診断、閾値なし）", ""]
    rows = []
    for s in SEEDS:
        a = (
            jl(runs.train[("pc", 1, s)] / "stream.jsonl")[:1]
            if ("pc", 1, s) in runs.train
            else []
        )
        b = (
            jl(runs.train[("bp", 1, s)] / "stream.jsonl")[:1]
            if ("bp", 1, s) in runs.train
            else []
        )
        la, lb = (a[0]["loss"] if a else None), (b[0]["loss"] if b else None)
        rel = None if la is None or lb is None else (la - lb) / abs(lb)
        data.setdefault("first_row", {})[str(s)] = {"pc": la, "bp": lb, "rel": rel}
        rows.append([str(s), _g(la), _g(lb), _g(rel)])
    out.append(_table(["seed", "PC-ALM", "BP", "相対の差"], rows))

    # the likelihood every 60 s
    series = {
        (r, s, x): _series(runs.train.get((r, 1, s)), x)
        for r in ("pc", "bp")
        for s in SEEDS
        for x in LIKELIHOOD
    }
    out += ["", "## 尤度の経過（val、60 秒ごと、対ごとの |PC − BP|）", ""]
    rows = []
    for x in LIKELIHOOD:
        bp_all = [series[("bp", s, x)] for s in SEEDS]
        for s in SEEDS:
            tr = pair_track(series[("pc", s, x)], series[("bp", s, x)], bp_all)
            data.setdefault("track", {}).setdefault(x, {})[str(s)] = tr
            rows.append(
                [x, str(s), _g(tr["max_abs"]), fmt(tr["t_max"]), fmt(tr["first_out"])]
            )
    out.append(
        _table(
            ["量", "seed", "最大の |差|", "その時刻", "BP の範囲を初めて超えた時刻"],
            rows,
        )
    )
    times = (0.0, 60.0, *CKPTS)
    out += ["", "## 尤度の経過（val、4 seed の平均、PC / BP）", ""]
    rows = []
    for t in times:
        cells = [fmt(t)]
        for x in ("bpd", "kl", "sigma", "sigma_star"):
            pair = []
            for r in ("pc", "bp"):
                vals = [series[(r, s, x)].get(t) for s in SEEDS]
                ok = [y for y in vals if y is not None]
                mean = sum(ok) / len(ok) if len(ok) == len(SEEDS) else None
                data.setdefault("mean_track", {}).setdefault(x, {}).setdefault(r, {})[
                    str(t)
                ] = mean
                pair.append(fmt(mean))
            cells.append(" / ".join(pair))
        rows.append(cells)
    out.append(_table(["t", "bpd", "kl", "sigma", "sigma_star"], rows))

    # B = 32
    out += ["", "## B = 32（seed 100）", ""]
    rows = []
    b32 = {r: runs.train.get((r, 32, 100)) for r in ("pc", "bp")}
    for r in ("pc", "bp"):
        k = (r, 32, 100)
        p, i = post.get(k, {}), init.get(k, {})
        dv32 = _state(b32[r]) == "diverged"
        u = _u(p, i, diverged=dv32) if k in runs.post else None
        bpd = _series(b32[r], "bpd").get(T_END)
        data.setdefault("b32", {})[r] = p | {"bpd": bpd, "U": u}
        rows.append([NAMES[r], *(fmt(p.get(x)) for x in POSTHOC), fmt(bpd), mark(u)])
    out.append(_table(["学習則", *POSTHOC, "bpd", "U"], rows))
    tr32 = pair_track(_series(b32["pc"], "bpd"), _series(b32["bp"], "bpd"), [])
    data["b32"]["bpd_track"] = tr32
    out.append(
        f"\nval の bpd の最大の |PC − BP|: {_g(tr32['max_abs'])}（{fmt(tr32['t_max'])} 秒）"
    )

    # speed
    out += ["", "## 速さ（timing.jsonl、評価を除く）", ""]
    rows = []
    for key in sorted(runs.train, key=lambda k: (k[0], -k[1], SEEDS.index(k[2]))):
        rate = train_rate(jl(runs.train[key] / "timing.jsonl"))
        target = 10 / key[1]
        lo, hi = QP5[key[1]]
        inside = None if rate is None or key[0] != "pc" else lo <= rate <= hi
        data.setdefault("speed", {})[f"{key[0]}-{key[1]}-{key[2]}"] = {
            "rate": rate,
            "target": target,
            "bench": BENCH[key[:2]],
            "qp5_range": inside,
        }
        rows.append(
            [NAMES[key[0]], str(key[1]), str(key[2]), fmt(rate), fmt(target)]
            + [fmt(None if rate is None else rate / target), fmt(BENCH[key[:2]])]
            + [mark(inside) if key[0] == "pc" else "－"]
        )
    out.append(
        _table(
            [
                "学習則",
                "B",
                "seed",
                "単位/秒",
                "目標",
                "目標との比",
                "bench",
                "QP5 の範囲",
            ],
            rows,
        )
    )

    # the posthoc per checkpoint per pair
    out += ["", "## checkpoint ごとの posthoc（val、PC − BP）", ""]
    rows = []
    for s in SEEDS:
        a = (
            jl(runs.post[("pc", 1, s)] / "post/decode/metrics.jsonl")
            if ("pc", 1, s) in runs.post
            else []
        )
        b = (
            jl(runs.post[("bp", 1, s)] / "post/decode/metrics.jsonl")
            if ("bp", 1, s) in runs.post
            else []
        )
        for t in CKPTS:
            va = {x: at(a, t, "val", x) for x in POSTHOC}
            vb = {x: at(b, t, "val", x) for x in POSTHOC}
            d = {
                "recon_blur": _sub(va["recon_blur"], vb["recon_blur"]),
                "dyn_1/copy_1": _sub(va["dyn_1/copy_1"], vb["dyn_1/copy_1"]),
                "stack_mse_rel": _rel(va["stack_mse"], vb["stack_mse"]),
                "own_recon_blur_rel": _rel(va["own_recon_blur"], vb["own_recon_blur"]),
            }
            data.setdefault("per_ckpt", {}).setdefault(str(s), {})[str(t)] = {
                "pc": va,
                "bp": vb,
                "diff": d,
            }
            rows.append([str(s), fmt(t), *(_g(y) for y in d.values())])
    out.append(
        _table(
            ["seed", "t", "recon_blur の差", "dyn_1/copy_1 の差"]
            + ["stack_mse の相対の差", "own_recon_blur の相対の差"],
            rows,
        )
    )

    # QP3, QP4
    grads = {}
    for s in SEEDS:
        rid = runs.train.get(("pc", 1, s))
        f = grad_dir / f"grad-{rid.name}.json" if grad_dir and rid else None
        if f is not None and f.exists():
            grads[s] = grad_summary(json.loads(f.read_text()))
    data["grad"] = grads
    out += ["", "## BP の勾配との一致（m18_pc_grad、val の最初の 8 個の積み）", ""]
    rows = []
    for s, gs in grads.items():
        for name, p in gs["points"].items():
            w = p["worst"]
            rows.append(
                [str(s), name, ",".join(p["failures"]) or "なし", _g(p["loss_ratio"])]
                + [
                    _g(p["rel_ratio"]),
                    _g(w["bp"]),
                    _g(w["pc"]),
                    _g(w["pc64"]),
                    _g(w["pc_bp"]),
                ]
            )
    head = ["seed", "点", "failures", "loss の差 / 閾値", "最大の rel 比（PC64/BP32）"]
    head += ["BP32 の最大 rel", "PC32 の最大 rel", "PC64 の最大 rel", "PC32 対 BP32"]
    out.append(_table(head, rows))

    # QP6
    ref = root / QP6_REF
    bp100 = runs.train.get(("bp", 1, 100))
    q6 = (
        qp6(
            jl(bp100 / "stream.jsonl"),
            jl(ref / "stream.jsonl"),
            jl(bp100 / "evals.jsonl"),
            jl(ref / "evals.jsonl"),
        )
        if bp100 and ref.exists()
        else {"held": None}
    )
    data["qp6"] = q6

    # the predictions
    speeds = [
        x["qp5_range"]
        for x in data.get("speed", {}).values()
        if x["qp5_range"] is not None
    ]
    pred = {
        "QP1": judge(v["U"]),
        "QP2": judge(v["D"]),
        "QP3": judge(
            _all([g["QP3"] for g in grads.values()])
            if len(grads) == len(SEEDS)
            else None
        ),
        "QP4": judge(
            _all([g["QP4"] for g in grads.values()])
            if len(grads) == len(SEEDS)
            else None
        ),
        "QP5": "判定しない（範囲に入った run: " + f"{sum(speeds)} / {len(speeds)}）",
        "QP6": judge(q6["held"]),
    }
    data["predictions"] = pred
    out += ["", "## 予測", ""]
    out.append(_table(["予測", "判定"], [[q, x] for q, x in pred.items()]))
    out.append(f"\nQP6: {json.dumps(q6)}")
    return "\n".join(out), data


def _sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _rel(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None or b == 0 else a / b - 1


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument(
        "--grad",
        type=Path,
        default=Path("runs/m18-stage4-grad"),
        help="the outputs of m18_pc_grad (grad-<run id>.json)",
    )
    p.add_argument("--json", type=Path, help="write the values of the tables here")
    a = p.parse_args(argv)
    text, data = report(a.runs, a.grad)
    print(text)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
