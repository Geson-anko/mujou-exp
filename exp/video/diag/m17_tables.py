r"""DIAGNOSIS ONLY (M17): the tables of ``exp/video/results/m17.md`` from the
run directories pulled into ``./runs`` (``just fleet pull --batch ...``): the
``evals.jsonl``, ``stream.jsonl``, ``baselines.json`` and ``result.json`` of the
training runs, and the ``post/decode/metrics.jsonl`` and ``baselines.jsonl`` of
their posthoc runs. Reads only; prints markdown (or plain rows).

- ``buf``: the tables of 「ストリームごとのバッファ」 → 「結果（2026-11-13）」
  (training evals and posthoc on val and test of ``m17-buf-v1`` /
  ``m17-buf-post-v1``, the P check of the six earlier runs ``m17-tok-v2`` /
  ``m17-pcheck-v1``, the yardsticks of ``m17-tok-base-v6``). A cell of several
  seeds is their mean, then each seed's value in parentheses
- ``buf-raw``: every key and checkpoint of the same runs, the yardsticks'
  lines and the learner's $S$ (``stream.jsonl``, mean over the 60 s before
  each checkpoint), from which the text of that section quotes
- ``tok``: the tables of 「トークンの SIGReg」 → 「学習の評価」 and
  「posthoc」 (``m17-tok-v2`` with its posthoc ``m17-tok-post-v3`` / ``-v5`` and
  yardsticks ``m17-tok-base-v3`` / ``-v5``); the learner's $S$ is the mean of the
  last 60 rows of ``stream.jsonl``
- ``tok-raw [train|post|all]``: the same runs, every key, plain rows

Run from the repository root: ``python -m exp.video.diag.m17_tables buf
[--runs runs]``.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

Getter = Callable[[str, float, str], Any]
Group = tuple[str, int, int, float]  # (batch, B, d, lambda)

# --- shared ------------------------------------------------------------------------


def jl(p: Path) -> list[dict[str, Any]]:
    """The rows of the JSON-lines file ``p`` ([] if it is missing)."""
    return [json.loads(x) for x in open(p) if x.strip()] if p.exists() else []


def fmt(v: float | None) -> str:
    """A number in about 3 significant digits ("–" for None)."""
    if v is None:
        return "–"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01:
        return f"{v:.3f}"
    if a >= 0.001:
        return f"{v:.4f}"
    return f"{v:.1e}"


def cell(vals: Sequence[float | None]) -> str:
    """The mean of the seeds' values, then each value in parentheses."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    if len(vals) == 1:
        return fmt(vals[0])
    m = sum(vals) / len(vals)
    return f"{fmt(m)}（{' / '.join(fmt(v) for v in vals)}）"


def at(rows: list[dict[str, Any]], t: float, split: str, key: str) -> Any:
    """``key`` of the row of ``split`` at time ``t`` (None if there is none)."""
    for r in rows:
        if r.get("split") == split and abs(r.get("t", -1) - t) < 1e-6:
            return r.get(key)
    return None


# --- the per-stream buffer runs (buf, buf-raw) -----------------------------------

TIMES = [300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0]
TRAIN_BATCHES = ("m17-buf-v1", "m17-tok-v2")
POST_BATCHES = ("m17-buf-post-v1", "m17-pcheck-v1")
BASE_BATCH = "m17-tok-base-v6"
BASE_RUNS = {16: "m17-tok_n16-9b04e6d6-s100", 32: "m17-tok_n16-d197eb69-s100"}


class BufRuns(NamedTuple):
    """The training runs by id, the posthoc lines by the parent's id, and the
    run ids of each (batch, B, d, lambda), by seed."""

    root: Path
    train: dict[str, dict[str, Any]]
    post: dict[str, dict[str, Any]]
    groups: dict[Group, list[str]]


def load_buf(root: Path) -> BufRuns:
    """The runs of ``buf`` under ``root``."""
    train: dict[str, dict[str, Any]] = {}
    for batch in TRAIN_BATCHES:
        for run in sorted((root / batch).glob("*/")):
            if not (run / "job.json").exists():
                continue
            j = json.loads((run / "job.json").read_text())
            c = j["config"]
            base_p = run / "baselines.json"
            train[j["run_id"]] = dict(
                batch=batch,
                B=c["data"]["streams"],
                lam=c["vision"]["sigreg"],
                seed=j["seed"],
                d=c["vision"]["token_dim"],
                ev=jl(run / "evals.jsonl"),
                base=json.loads(base_p.read_text()) if base_p.exists() else {},
                dir=run,
            )
    post: dict[str, dict[str, Any]] = {}
    for batch in POST_BATCHES:
        for run in sorted((root / batch).glob("*/")):
            if not (run / "job.json").exists():
                continue
            j = json.loads((run / "job.json").read_text())
            parent = Path(j["parent"]).name
            post[parent] = dict(
                lines=jl(run / "post/decode/metrics.jsonl"),
                dir=run,
                base=jl(run / "post/decode/baselines.jsonl"),
            )
    groups: dict[Group, list[str]] = defaultdict(list)
    for rid, r in train.items():
        groups[(r["batch"], r["B"], r["d"], r["lam"])].append(rid)
    for k in groups:
        groups[k].sort(key=lambda rid: train[rid]["seed"])
    return BufRuns(root, train, post, groups)


def _order(k: Group) -> tuple[bool, int, int, float]:
    return (k[0] != "m17-buf-v1", -k[1], k[2], k[3])


def ev_getter(runs: BufRuns, split: str) -> Getter:
    """The training eval of ``split`` (time 0: the untrained network)."""

    def get(rid: str, t: float, key: str) -> Any:
        if t == 0:
            return runs.train[rid]["base"].get(split, {}).get("init_" + key)
        return at(runs.train[rid]["ev"], t, split, key)

    return get


def post_getter(runs: BufRuns, split: str) -> Getter:
    """The posthoc line of ``split``."""

    def get(rid: str, t: float, key: str) -> Any:
        return at(runs.post.get(rid, {}).get("lines", []), t, split, key)

    return get


def raw_table(
    runs: BufRuns,
    keys: Sequence[str],
    getter: Getter,
    title: str,
    times: Sequence[float],
) -> None:
    print(f"\n#### {title}\n")
    print("| run | t | " + " | ".join(f"`{k}`" for k in keys) + " |")
    print("| -- | -- | " + " | ".join("--" for _ in keys) + " |")
    for k in sorted(runs.groups, key=_order):
        rids = runs.groups[k]
        batch, b, d, lam = k
        tag = "旧（8192 行）" if batch == "m17-tok-v2" else f"$B$ = {b}"
        label = f"{tag}、$d$ = {d}、$\\lambda$ = {lam:g}"
        ids = " / ".join(
            f"`{rid.split('-')[2]}`-s{runs.train[rid]['seed']}" for rid in rids
        )
        for t in times:
            vals = [[getter(rid, t, key) for rid in rids] for key in keys]
            if all(v is None for vs in vals for v in vs):
                continue
            print(
                f"| {label} {ids} | {t:g} | " + " | ".join(cell(v) for v in vals) + " |"
            )


RAW_TRAIN_KEYS = [
    "pred",
    "sigreg",
    "comp_short",
    "comp_short_pos",
    "comp_short_mean",
    "comp_long",
    "comp_long_pos",
    "comp_long_mean",
    "comp_future",
    "comp_future_pos",
    "ncomp_short_pos",
    "tok_within",
]
RAW_POST_KEYS = [
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
    "restore_short_all",
    "restore_long_all",
    "restore_target_all",
    "restore_target_all_blur",
    "dyn_1/copy_1",
    "comp_short",
    "comp_short_pos",
    "comp_long",
    "comp_long_pos",
    "comp_future",
    "comp_future_pos",
]


def learner_s(runs: BufRuns, rid: str, t: float) -> float | None:
    """Mean learner-side S (stream.jsonl, with the buffer) over (t-60, t]."""
    run = runs.train[rid]["dir"]
    rows = [
        r
        for r in jl(run / "stream.jsonl")
        if t - 60 < r.get("t", -1) <= t and "sigreg" in r
    ]
    return sum(r["sigreg"] for r in rows) / len(rows) if rows else None


def buf_raw(root: Path) -> None:
    """``buf-raw`` of the module docstring."""
    runs = load_buf(root)
    for sp in ("val", "test"):
        raw_table(
            runs,
            RAW_TRAIN_KEYS,
            ev_getter(runs, sp),
            f"学習の評価（{sp}）",
            [0.0] + TIMES,
        )
        raw_table(runs, RAW_POST_KEYS, post_getter(runs, sp), f"posthoc（{sp}）", TIMES)
    for run in sorted((root / BASE_BATCH).glob("*/")):
        j = json.loads((run / "job.json").read_text())
        print(f"\n#### 物差し {j['run_id']} d={j['config']['vision']['token_dim']}\n")
        for line in jl(run / "post/decode/baselines.jsonl") + jl(
            run / "post/decode/metrics.jsonl"
        ):
            keys = RAW_POST_KEYS + ["recon_lin", "recon_lin_blur", "pca_rank"]
            print(
                line.get("split"),
                line.get("code"),
                {
                    k: fmt(line[k])
                    for k in keys
                    if k in line and isinstance(line[k], (int, float))
                },
            )
        base = run / "baselines.json"
        if base.exists():
            b = json.loads(base.read_text())
            for sp, bb in b.items():
                parts = (
                    "pred",
                    "sigreg",
                    "comp_short",
                    "comp_long",
                    "comp_future",
                    "tok_within",
                )
                print(
                    "init",
                    sp,
                    {
                        k: fmt(v)
                        for k, v in bb.items()
                        if isinstance(v, (int, float)) and any(s in k for s in parts)
                    },
                )
    print("\n#### 学ぶ側の S（stream.jsonl、各時刻の直前 60 秒の平均）\n")
    for k in sorted(runs.groups, key=_order):
        batch, b, d, lam = k
        tag = "旧（8192 行）" if batch == "m17-tok-v2" else f"$B$ = {b}"
        label = f"{tag}、$d$ = {d}、$\\lambda$ = {lam:g}"
        print(
            label,
            " | ".join(
                f"{t:g}: {cell([learner_s(runs, r, t) for r in runs.groups[k]])}"
                for t in TIMES
            ),
        )


def base_lines(root: Path, d: int, split: str, code: str) -> Iterator[dict[str, Any]]:
    run = root / BASE_BATCH / BASE_RUNS[d]
    for line in jl(run / "post/decode/baselines.jsonl") + jl(
        run / "post/decode/metrics.jsonl"
    ):
        if line.get("split") == split and line.get("code", "init") == code:
            yield line


def base_get(root: Path, d: int, split: str, code: str, key: str) -> Any:
    """``key`` of the yardstick ``code`` (``pca`` or ``init``) of width ``d``."""
    vals = [line[key] for line in base_lines(root, d, split, code) if key in line]
    return vals[0] if vals else None


def md_table(
    runs: BufRuns,
    keys: Sequence[str],
    getter: Callable[[str, float, str, str], Any],
    times: Sequence[float],
    sel: Callable[[Group], bool],
    split: str,
    base_rows: Sequence[tuple[str, Callable[[str], Any]]] = (),
) -> str:
    out = [
        "| run | t | " + " | ".join(f"`{k}`" for k in keys) + " |",
        "| -- | -- | " + " | ".join("--" for _ in keys) + " |",
    ]
    for name, f in base_rows:
        out.append(
            f"| {name} | – | "
            + " | ".join(fmt(f(k)) if f(k) is not None else "–" for k in keys)
            + " |"
        )
    for k in sorted([k for k in runs.groups if sel(k)], key=_order):
        batch, b, d, lam = k
        if batch == "m17-tok-v2":
            lab = f"旧、$d$={d}、$\\lambda$={lam:g}"
        else:
            lab = f"$B$={b}、$\\lambda$={lam:g}"
        ids = " / ".join(f"`{r.split('-')[2]}`" for r in runs.groups[k][:1])
        for t in times:
            vals = [
                [getter(rid, t, key, split) for rid in runs.groups[k]] for key in keys
            ]
            if all(v is None for vs in vals for v in vs):
                continue
            out.append(
                f"| {lab} {ids} | {t:g} | " + " | ".join(cell(v) for v in vals) + " |"
            )
    return "\n".join(out)


def buf_md(root: Path) -> None:
    """``buf`` of the module docstring."""
    runs = load_buf(root)

    def ev(rid: str, t: float, key: str, split: str) -> Any:
        return ev_getter(runs, split)(rid, t, key)

    def ph(rid: str, t: float, key: str, split: str) -> Any:
        return post_getter(runs, split)(rid, t, key)

    def oldget(rid: str, t: float, key: str, split: str) -> Any:
        v = ph(rid, t, key, split)
        return v if v is not None else ev(rid, t, key, split)

    def new(k: Group) -> bool:
        return k[0] == "m17-buf-v1"

    def old(k: Group) -> bool:
        return k[0] == "m17-tok-v2"

    def basep(split: str) -> list[tuple[str, Callable[[str], Any]]]:
        return [
            ("PCA（8192 成分）", lambda k: base_get(root, 16, split, "pca", k)),
            ("学習前の ViT", lambda k: base_get(root, 16, split, "init", k)),
        ]

    T = [0.0, *TIMES]
    TK = [
        "pred",
        "sigreg",
        "comp_short",
        "comp_short_pos",
        "comp_short_mean",
        "comp_long",
        "comp_long_pos",
        "comp_long_mean",
        "tok_within",
    ]
    PK = [
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
        "restore_short_all",
        "restore_long_all",
        "restore_target_all",
        "restore_target_all_blur",
        "dyn_1/copy_1",
    ]
    TT = ["comp_short", "comp_short_pos", "comp_long", "comp_long_pos"]
    PT = [
        "z_spread",
        "recon_blur",
        "recon",
        "restore_short_mask",
        "restore_target_short_mask",
        "restore_short_mask_blur",
        "restore_target_short_mask_blur",
        "restore_target_all",
        "dyn_1/copy_1",
    ]
    OK = [
        "pred",
        "comp_short",
        "comp_short_pos",
        "comp_long",
        "comp_long_pos",
        "comp_future",
        "comp_future_pos",
        "z_spread",
        "restore_short_mask",
        "restore_target_short_mask",
        "restore_target_all",
    ]
    print("### T1 train val\n")
    print(md_table(runs, TK, ev, T, new, "val"))
    print("\n### T2 post val\n")
    print(md_table(runs, PK, ph, T[1:], new, "val", basep("val")))
    print("\n### T3 train test\n")
    print(md_table(runs, TT, ph, [600.0, 1800.0], new, "test"))
    print("\n### T4 post test\n")
    print(md_table(runs, PT, ph, [600.0, 1800.0], new, "test", basep("test")))
    print("\n### T5 old pcheck val\n")
    print(md_table(runs, OK, oldget, [300.0, 600.0], old, "val"))
    print("\n### T5b new at 600 (same cols)\n")
    print(md_table(runs, OK, oldget, [300.0, 600.0], new, "val"))


# --- the first runs of the token SIGReg (tok, tok-raw) ---------------------------

TOK_RUNS = {  # run id: (lambda, d), of m17-tok-v2
    "8fe992ab": (0.0017, 16),
    "f8cef62c": (0.0026, 16),
    "a873a30a": (0.0038, 16),
    "a39ccabc": (0.0017, 32),
    "6cc6b7f4": (0.0026, 32),
    "d5167420": (0.0038, 32),
}
TOK_POST = {  # run id: its posthoc (batch, id)
    "8fe992ab": ("m17-tok-post-v3", "19ec9084"),
    "f8cef62c": ("m17-tok-post-v3", "422d93aa"),
    "a873a30a": ("m17-tok-post-v3", "5fc10931"),
    "a39ccabc": ("m17-tok-post-v5", "da9dc201"),
    "6cc6b7f4": ("m17-tok-post-v5", "5256ae56"),
    "d5167420": ("m17-tok-post-v5", "b80bb1c2"),
}
TOK_BASE = {16: ("m17-tok-base-v3", "179c52a1"), 32: ("m17-tok-base-v5", "971239ea")}


def run_dir(root: Path, batch: str, rid: str) -> Path:
    return root / batch / f"m17-tok_n16-{rid}-s100"


def fmt_raw(x: float | None, n: int = 3) -> str:
    """The number format of ``tok-raw``."""
    if x is None:
        return "–"
    if isinstance(x, int):
        return str(x)
    a = abs(x)
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}"
    return f"{x:.{n}g}" if a < 0.01 else f"{x:.{n}f}"


def fmt_tok(x: float | None, p: int = 3) -> str:
    """The number format of ``tok``."""
    if x is None:
        return "–"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) < 0.01:
        return f"{x:.4f}"
    return f"{x:.{p}f}"


def tok_raw_train(root: Path, split: str = "val") -> None:
    f = fmt_raw
    keys = [
        "pred",
        "sigreg",
        "tok_within",
        "tok_between",
        "tok_temporal",
        "rank",
        "z_spread",
        "ncomp_short",
        "ncomp_long",
        "ncomp_future",
    ]
    print(
        f"\n## training evals ({split}); init from baselines.json; columns: t "
        + " ".join(keys)
    )
    for rid, (lam, d) in TOK_RUNS.items():
        rd = run_dir(root, "m17-tok-v2", rid)
        ev = [r for r in jl(rd / "evals.jsonl") if r.get("split") == split]
        base = (
            json.loads((rd / "baselines.json").read_text())
            if (rd / "baselines.json").exists()
            else {}
        )
        init = base.get(split, {})
        st = jl(rd / "stream.jsonl")
        res = (
            json.loads((rd / "result.json").read_text())
            if (rd / "result.json").exists()
            else {}
        )
        print(
            f"\n{rid} lam={lam} d={d} status={res.get('status')} stream_last={ {k: round(v, 4) for k, v in st[-1].items()} if st else None }"
        )
        print("  init", " ".join(f(init.get("init_" + k)) for k in keys))
        for r in ev:
            if r["t"] in (60.0, 120.0, 300.0, 450.0, 600.0) or r is ev[-1]:
                print(f"  t={r['t']:.0f}", " ".join(f(r.get(k)) for k in keys))
        if ev:
            best = min(ev, key=lambda r: r["pred"])
            print(f"  min pred at t={best['t']:.0f}: {best['pred']:.4f}")
        # stream sigreg at the end (mean of last 60 rows)
        if st:
            tail = st[-60:]
            print(
                "  stream mean last 60: pred %.4f sigreg %.2f"
                % (
                    sum(x["pred"] for x in tail) / len(tail),
                    sum(x["sigreg"] for x in tail) / len(tail),
                )
            )


def tok_raw_post(root: Path, split: str) -> None:
    f = fmt_raw
    keys = [
        "recon_blur",
        "recon",
        "recon_fit",
        "dyn_1/copy_1",
        "dyn_5/copy_5",
        "dyn_10/copy_10",
        "dyn_1",
        "dynw_1",
        "rank",
        "z_spread",
        "restore_short_mask",
        "restore_long_mask",
        "restore_short_all",
        "restore_target_all",
        "alpha",
    ]
    print(f"\n## posthoc ({split}); columns: " + " ".join(keys))
    for d, b in TOK_BASE.items():
        for line in jl(run_dir(root, *b) / "post" / "decode" / "baselines.jsonl"):
            if line["split"] == split:
                print(
                    f"  base d={d} {line['code']:5s}",
                    " ".join(f(line.get(k)) for k in keys),
                    f"pca_rank={line.get('pca_rank')}",
                )
    for rid, (lam, d) in TOK_RUNS.items():
        pd = run_dir(root, *TOK_POST[rid]) / "post" / "decode"
        for line in jl(pd / "baselines.jsonl"):
            if line["split"] == split and line["code"] == "init":
                print(
                    f"  {rid} lam={lam} d={d} init ",
                    " ".join(f(line.get(k)) for k in keys),
                )
        for line in jl(pd / "metrics.jsonl"):
            if line["split"] == split:
                print(
                    f"  {rid} lam={lam} d={d} t={line['t']:.0f}",
                    " ".join(f(line.get(k)) for k in keys),
                )


def tok_md_train(root: Path) -> None:
    g = fmt_tok
    print(
        "| $\\lambda$ | $d$ | run | `pred`（学習前 → 300 → 600 秒） | `sigreg` val（学習前 → 300 → 600 秒） | 学ぶ側の $S$（最後の 60 秒） | `tok_within` / `tok_between` / `tok_temporal`（300 → 600 秒） | `ncomp` 短 / 長 / 未来（600 秒） | rank（$z$、600 秒） |"
    )
    print("| -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    for rid, (lam, d) in sorted(TOK_RUNS.items(), key=lambda x: (x[1][1], x[1][0])):
        rd = run_dir(root, "m17-tok-v2", rid)
        ev = {r["t"]: r for r in jl(rd / "evals.jsonl") if r["split"] == "val"}
        init = json.loads((rd / "baselines.json").read_text())["val"]
        st = jl(rd / "stream.jsonl")[-60:]
        s_stream = sum(x["sigreg"] for x in st) / len(st)
        a, b = ev[300.0], ev[600.0]
        print(
            f"| {lam} | {d} | `{rid}` | {g(init['init_pred'])} → {g(a['pred'])} → {g(b['pred'])} | {g(init['init_sigreg'])} → {g(a['sigreg'])} → {g(b['sigreg'])} | {g(s_stream)} | "
            f"{g(a['tok_within'])} / {g(a['tok_between'])} / {g(a['tok_temporal'])} → {g(b['tok_within'])} / {g(b['tok_between'])} / {g(b['tok_temporal'])} | "
            f"{g(b['ncomp_short'])} / {g(b['ncomp_long'])} / {g(b['ncomp_future'])} | {g(b['rank'], 2)} |"
        )
    for d, rid in ((16, "8fe992ab"), (32, "a39ccabc")):
        init = json.loads(
            (run_dir(root, "m17-tok-v2", rid) / "baselines.json").read_text()
        )["val"]
        keys = (
            "pred",
            "sigreg",
            "tok_within",
            "tok_between",
            "tok_temporal",
            "ncomp_short",
            "ncomp_long",
            "ncomp_future",
            "rank",
        )
        print(f"init d={d}:", {k: round(init["init_" + k], 4) for k in keys})


def tok_md_post(root: Path, split: str = "val", other: str = "test") -> None:
    g = fmt_tok
    keys = [
        "recon_blur",
        "recon",
        "dyn_1/copy_1",
        "dyn_5/copy_5",
        "dyn_10/copy_10",
        "rank",
        "z_spread",
        "restore_short_mask",
        "restore_long_mask",
        "restore_target_all",
    ]
    print("| 符号 | $d$ | " + " | ".join(f"`{k}`" for k in keys) + " |")
    print("| -- | -- | " + " | ".join("--" for _ in keys) + " |")

    def line(name: str, d: int, r: dict[str, Any], o: dict[str, Any]) -> None:
        cells = []
        for k in keys:
            if r.get(k) is None:
                cells.append("–")
                continue
            both = k in ("recon_blur", "recon", "dyn_1/copy_1")
            cells.append(f"{g(r[k])}（{g(o.get(k))}）" if both else g(r[k]))
        print(f"| {name} | {d} | " + " | ".join(cells) + " |")

    for d in (16, 32):
        rows = jl(run_dir(root, *TOK_BASE[d]) / "post" / "decode" / "baselines.jsonl")
        b = {(r["split"], r["code"]): r for r in rows}
        if b:
            line(f"PCA（{512 * d} 成分）", d, b[(split, "pca")], b[(other, "pca")])
            line("学習前の ViT", d, b[(split, "init")], b[(other, "init")])
        for rid, (lam, dd) in sorted(TOK_RUNS.items(), key=lambda x: x[1][0]):
            if dd != d:
                continue
            m = jl(run_dir(root, *TOK_POST[rid]) / "post" / "decode" / "metrics.jsonl")
            by = {(r["split"], r["t"]): r for r in m}
            for t in (300.0, 600.0):
                if (split, t) in by:
                    line(
                        f"$\\lambda$ = {lam}、{t:.0f} 秒",
                        d,
                        by[(split, t)],
                        by[(other, t)],
                    )


def main(argv: Sequence[str] | None = None) -> None:
    """One command of the module docstring."""
    p = argparse.ArgumentParser(description="The tables of results/m17.md.")
    p.add_argument("command", choices=("buf", "buf-raw", "tok", "tok-raw"))
    p.add_argument("what", nargs="?", default="all", choices=("all", "train", "post"))
    p.add_argument("--runs", type=Path, default=Path("runs"))
    args = p.parse_args(argv)
    root = args.runs
    if args.command == "buf":
        buf_md(root)
    elif args.command == "buf-raw":
        buf_raw(root)
    elif args.command == "tok":
        tok_md_train(root)
        print()
        tok_md_post(root)
    else:
        if args.what in ("all", "train"):
            tok_raw_train(root, "val")
        if args.what in ("all", "post"):
            tok_raw_post(root, "val")
            tok_raw_post(root, "test")


if __name__ == "__main__":
    main()
