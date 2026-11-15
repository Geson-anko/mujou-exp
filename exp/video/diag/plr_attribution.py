r"""DIAGNOSIS ONLY (M17, 2026-11-14): the predictions P1, P2 and P3 of
「層ごとの学習率（回す前に決めたこと）」 in ``exp/video/results/m17.md``, from
the ``rows.jsonl`` of ``early_shrink live`` (one directory per run of the
diagnosis, named ``<arm>-<kind>-<lr>-s<seconds>``: ``arm`` ``plr`` for the
per-layer rate ``optim_vision.scale=fan_in`` of ``m17-plr-v1``, ``opt`` for
the sweep ``m17-opt-v1`` at the same numbers; ``lr`` the run's
``optim_vision.lr``, $\eta$ under ``fan_in``).

- ``q`` of a linear map at time $t$: its ``output_change`` over $\eta F$, $F$
  the frames to $t$ (``frame`` of the row), with ``step_bound`` (time 0) and
  ``q / step_bound``
- P1 (1 s): the patch embedding's ``q`` is below the largest ``q`` of the
  other linear maps
- P2 (60 s): ``tok_within`` with the embedding back at time 0 (``revert``) is
  at most the largest with any one other module back
- P3 (600 steps, 15 s): ``tok_within`` of ``plr`` above that of ``opt`` at the
  same kind, number and time
- additivity of the one-module attribution (a reading after the runs, not a
  prediction): if ``tok_within`` were $f_0 + \sum_{i \in S} \delta_i$ over
  the set $S$ of moved modules, every encoder module $i$ would have
  $(\text{revert}_i + \text{only}_i) / (f_{\text{all}} + f_0) = 1$; the
  minimum and maximum of this ratio over the encoder's modules and the
  embedding's (the predictor does not act on ``tok_within``)

With ``--runs`` (the pulled run tree), each entry also gets the held-out
values of the fleet's run of the same arm (``m17-plr-v1`` or ``m17-opt-v1``,
kind, lr, warm-up 0) at 60 s (``fleet_60``), to compare the reproduction on
the diagnosis's GPU with the run's.

Reads only; writes the summary JSON and prints markdown::

    python -m exp.video.diag.plr_attribution <out_root> <summary.json> [--runs runs]
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EMBED = "encoder.stages.0.embed.proj"
BATCHES = {"plr": "m17-plr-v1", "opt": "m17-opt-v1"}
SCALES = {"plr": "fan_in", "opt": "none"}
TIMES = (1.0, 10.0, 15.0, 60.0)
P1_T, P2_T, P3_T = 1.0, 60.0, 15.0


def parse_tag(tag: str) -> tuple[str, str, float, float]:
    """(arm, kind, lr, seconds) of ``<arm>-<kind>-<lr>-s<seconds>``."""
    arm, kind, rest = tag.split("-", 2)
    lr, s = rest.rsplit("-s", 1)
    return arm, kind, float(lr), float(s)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """The rows of a ``rows.jsonl``."""
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def q_of(row: Mapping[str, Any], lr: float) -> dict[str, float]:
    """``output_change`` over ``lr`` times the row's frames, per linear map."""
    f = row["frame"]
    return {k: v / (lr * f) for k, v in row["output_change"].items()}


def embed_below_others(values: Mapping[str, float]) -> bool | None:
    """The embedding's value below the largest of the others (P1); None
    without the embedding or another map."""
    others = [v for k, v in values.items() if k != EMBED]
    if EMBED not in values or not others:
        return None
    return values[EMBED] < max(others)


def revert_summary(att: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict:
    """``tok_within`` with the embedding back (``revert``), the largest with
    one other module back and its module, the embedding alone (``only``),
    and P2: the embedding's revert at most the others' largest."""
    rev = {m: a["revert"]["tok_within"] for m, a in att.items()}
    others = {m: v for m, v in rev.items() if m != EMBED}
    top = max(others, key=lambda m: others[m])
    return {
        "revert_embed": rev[EMBED],
        "revert_other_max": others[top],
        "revert_other_module": top,
        "only_embed": att[EMBED]["only"]["tok_within"],
        "p2": rev[EMBED] <= others[top],
    }


def additivity(
    att: Mapping[str, Mapping[str, Mapping[str, float]]], moved: float, t0: float
) -> dict[str, Any]:
    """``(revert + only) / (moved + t0)`` of ``tok_within`` per encoder
    module, 1 for all when the attribution is additive: its minimum and
    maximum with their modules, and the embedding's."""
    r = {
        m: (a["revert"]["tok_within"] + a["only"]["tok_within"]) / (moved + t0)
        for m, a in att.items()
        if m.startswith("encoder.")
    }
    lo, hi = min(r, key=lambda m: r[m]), max(r, key=lambda m: r[m])
    return {
        "add_min": r[lo],
        "add_min_module": lo,
        "add_max": r[hi],
        "add_max_module": hi,
        "add_embed": r[EMBED],
    }


def summarize(root: Path) -> dict[str, dict[str, Any]]:
    """Per ``<arm>-<kind>-<lr>`` the held-out values at ``TIMES`` and time 0,
    ``q`` and ``q / step_bound`` at 1, 10 and 60 s with P1 at 1 s, and the
    attribution at the end of each run (by its seconds) with P2."""
    out: dict[str, dict[str, Any]] = {}
    for d in sorted(p for p in root.iterdir() if (p / "rows.jsonl").exists()):
        arm, kind, lr, s = parse_tag(d.name)
        rows = read_rows(d / "rows.jsonl")
        e = out.setdefault(
            f"{arm}-{kind}-{lr:g}",
            {"arm": arm, "kind": kind, "lr": lr, "values": {}, "q": {}, "att": {}},
        )
        bound = next((r["step_bound"] for r in rows if "step_bound" in r), None)
        if bound is not None:
            e["step_bound"] = bound
        tw = {float(r["t"]): r["tok_within"] for r in rows if "frame" in r}
        for r in rows:
            if "frame" not in r:  # the attribution at the end: modules by name
                att = {k: v for k, v in r.items() if k != "t"}
                e["att"][f"{r['t']:g}"] = {
                    **revert_summary(att),
                    **additivity(att, tw[float(r["t"])], tw[0.0]),
                }
                continue
            t = float(r["t"])
            if t == 0.0 or t in TIMES:
                e["values"][f"{t:g}"] = {
                    k: r[k] for k in ("tok_within", "pred", "loss") if k in r
                }
            if t in (1.0, 10.0, 60.0) and "output_change" in r:
                q = q_of(r, lr)
                e["q"][f"{t:g}"] = {
                    "q": q,
                    "q_over_bound": (
                        {k: v / e["step_bound"][k] for k, v in q.items()}
                        if "step_bound" in e
                        else {}
                    ),
                    "p1": embed_below_others(q) if t == P1_T else None,
                }
    for e in out.values():
        if e["arm"] != "plr":
            continue
        ref = out.get(f"opt-{e['kind']}-{e['lr']:g}")
        mine = e["values"].get(f"{P3_T:g}", {}).get("tok_within")
        theirs = ref["values"].get(f"{P3_T:g}", {}).get("tok_within") if ref else None
        e["p3"] = (
            None
            if mine is None or theirs is None
            else bool(mine > theirs)  # nan compares false: not slower
        )
        e["p3_values"] = {"plr": mine, "opt": theirs}
    return out


def add_fleet(s: Mapping[str, dict[str, Any]], runs: Path, t: float = 60.0) -> None:
    """Each entry's ``fleet_60``: ``tok_within`` and ``pred`` at ``t`` on val
    of the fleet run of its arm's batch with its kind and lr, warm-up 0 and
    its arm's scale (None when there is no such run)."""
    found: dict[tuple[str, str, float], dict[str, Any]] = {}
    for arm, batch in BATCHES.items():
        for job_p in sorted((runs / batch).glob("*/job.json")):
            o = json.loads(job_p.read_text())["config"]["optim_vision"]
            if float(o["warmup"]) or o.get("scale", "none") != SCALES[arm]:
                continue
            rows = read_rows(job_p.parent / "evals.jsonl")
            row = next(
                (r for r in rows if r.get("split") == "val" and r["t"] == t), None
            )
            if row is not None:
                found[(arm, o["kind"], float(o["lr"]))] = {
                    k: row[k] for k in ("tok_within", "pred")
                }
    for e in s.values():
        e["fleet_60"] = found.get((e["arm"], e["kind"], e["lr"]))


def _f(x: Any) -> str:
    if x is None:
        return "–"
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return f"{x:.3g}"


def _mark(b: bool | None) -> str:
    return {True: "○", False: "×", None: "?"}[b]


def _short(m: str) -> str:
    return m.replace("encoder.", "").replace("stages.", "s").replace("blocks.", "b")


def _pairs(s: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, float, Any, Any]]:
    """(kind, lr, plr entry, opt entry) of each kind and lr of ``plr``."""
    out = []
    for e in s.values():
        if e["arm"] == "plr":
            out.append((e["kind"], e["lr"], e, s.get(f"opt-{e['kind']}-{e['lr']:g}")))
    return sorted(out, key=lambda x: (x[0], x[1]))


def _p1_cells(e: Mapping[str, Any] | None) -> list[str]:
    if e is None or "1" not in e["q"]:
        return ["–", "–", "–"]
    q = e["q"]["1"]["q"]
    top = max((k for k in q if k != EMBED), key=lambda k: q[k])
    return [_f(q[EMBED]), f"{_f(q[top])}（`{_short(top)}`）", _mark(e["q"]["1"]["p1"])]


def _p2_cells(e: Mapping[str, Any] | None) -> list[str]:
    if e is None or "60" not in e["att"]:
        return ["–", "–", "–", "–"]
    a = e["att"]["60"]
    return [
        _f(e["values"].get("60", {}).get("tok_within")),
        _f(a["revert_embed"]),
        f"{_f(a['revert_other_max'])}（`{_short(a['revert_other_module'])}`）",
        _mark(a["p2"]),
    ]


def _tw(e: Mapping[str, Any] | None) -> str:
    if e is None:
        return "–"
    return " / ".join(
        _f(e["values"].get(t, {}).get("tok_within")) for t in ("1", "10", "15", "60")
    )


def tables(s: Mapping[str, Mapping[str, Any]]) -> str:
    """Markdown of 「層ごとの学習率の結果」: P1, P2 and P3 beside the sweep
    at the same number, the attribution over time, and (with ``fleet_60``)
    the reproduction against the runs."""
    pairs = _pairs(s)
    md = [
        "**P1（1 秒）**: `output_change` $/(\\eta F)$。埋め込みと、ほかの線形写像の最大"
        "（どの写像か）",
        "",
        "| optimizer | η / lr | 層ごと: 埋め込み | 層ごと: ほかの最大 | P1 | "
        "掃引: 埋め込み | 掃引: ほかの最大 | 掃引で同じ比べ方 |",
        "|" + " -- |" * 8,
    ]
    for kind, lr, p, o in pairs:
        md.append(
            f"| `{kind}` | {lr:g} | " + " | ".join(_p1_cells(p) + _p1_cells(o)) + " |"
        )
    md += [
        "",
        "**P2（60 秒）**: 1 つのモジュールだけを時刻 0 に戻したときの `tok_within`。"
        "埋め込みと、ほかの最大（どのモジュールか）。すべて動かしたときの値も並べる",
        "",
        "| optimizer | η / lr | 層ごと: すべて動かす | 層ごと: 埋め込みを戻す | "
        "層ごと: ほかの最大 | P2 | 掃引: すべて動かす | 掃引: 埋め込みを戻す | "
        "掃引: ほかの最大 | 掃引で同じ比べ方 |",
        "|" + " -- |" * 10,
    ]
    for kind, lr, p, o in pairs:
        md.append(
            f"| `{kind}` | {lr:g} | " + " | ".join(_p2_cells(p) + _p2_cells(o)) + " |"
        )
    md += [
        "",
        "**P3（600 ステップ = 15 秒）**: `tok_within`（時刻 0 は 0.474）",
        "",
        "| optimizer | η / lr | 層ごと 1 / 10 / 15 / 60 秒 | 掃引 1 / 10 / 15 / 60 秒 | "
        "P3（15 秒で層ごとが大きい） |",
        "|" + " -- |" * 5,
    ]
    for kind, lr, p, o in pairs:
        md.append(f"| `{kind}` | {lr:g} | {_tw(p)} | {_tw(o)} | {_mark(p.get('p3'))} |")
    md += [
        "",
        "**帰属の時間の経過（層ごとの学習率）**: 埋め込みを戻す / ほかの最大を戻す / "
        "埋め込みだけ動かす（`tok_within`）",
        "",
        "| optimizer | η | 1 秒 | 10 秒 | 60 秒 |",
        "|" + " -- |" * 5,
    ]
    for kind, lr, p, _ in pairs:
        cells = []
        for t in ("1", "10", "60"):
            a = p["att"].get(t)
            cells.append(
                "–"
                if a is None
                else f"{_f(a['revert_embed'])} / {_f(a['revert_other_max'])} / "
                f"{_f(a['only_embed'])}"
            )
        md.append(f"| `{kind}` | {lr:g} | " + " | ".join(cells) + " |")
    md += [
        "",
        "**加法性**: エンコーダのモジュールごとの "
        "(戻す + それだけ動かす) / (すべて動かす + 時刻 0)（`tok_within`）。"
        "加法的ならどれも 1。最小（モジュール）/ 最大（モジュール）/ 埋め込み",
        "",
        "| optimizer | η / lr | 層ごと 1 秒 | 層ごと 10 秒 | 層ごと 60 秒 | 掃引 60 秒 |",
        "|" + " -- |" * 6,
    ]

    def add(e: Mapping[str, Any] | None, t: str) -> str:
        a = None if e is None else e["att"].get(t)
        if a is None or "add_min" not in a:
            return "–"
        return (
            f"{_f(a['add_min'])}（`{_short(a['add_min_module'])}`）/ "
            f"{_f(a['add_max'])}（`{_short(a['add_max_module'])}`）/ "
            f"{_f(a['add_embed'])}"
        )

    for kind, lr, p, o in pairs:
        md.append(
            f"| `{kind}` | {lr:g} | {add(p, '1')} | {add(p, '10')} | "
            f"{add(p, '60')} | {add(o, '60')} |"
        )
    if any("fleet_60" in e for e in s.values()):
        md += [
            "",
            "**mini-00 の再現と run の値（60 秒）**: `tok_within`（mini-00 / run）",
            "",
            "| optimizer | η / lr | 層ごと | 掃引 |",
            "|" + " -- |" * 4,
        ]

        def rep(e: Mapping[str, Any] | None) -> str:
            if e is None or e.get("fleet_60") is None:
                return "–"
            mine = e["values"].get("60", {}).get("tok_within")
            return f"{_f(mine)} / {_f(e['fleet_60']['tok_within'])}"

        for kind, lr, p, o in pairs:
            md.append(f"| `{kind}` | {lr:g} | {rep(p)} | {rep(o)} |")
    return "\n".join(md)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="P1, P2, P3 of the M17 per-layer lr")
    p.add_argument("root", type=Path)
    p.add_argument("summary", type=Path)
    p.add_argument("--runs", type=Path)
    args = p.parse_args(argv)
    s = summarize(args.root)
    if args.runs is not None:
        add_fleet(s, args.runs)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(s, indent=1) + "\n")
    print(tables(s))


if __name__ == "__main__":
    main()
