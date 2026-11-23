"""The judgement of V18's screen (exp/vision/README.md, the V18 section).

Reads ``runs/v18-screen/<run>/`` (``jepa.json``, ``train.jsonl`` and
``posthoc_jepa/{jepa,probes}.jsonl``) and ``runs/v18-baselines/posthoc_jepa``
(the yardsticks) and applies the rules registered before the runs, as they
are: the thresholds, the usable pairs (metric, yardstick), the collapses
K1-K3, the pass of each run at its last checkpoint (step 6000) and the rule
choosing the long runs. With ``runs/v18-long/`` present it also applies rule 5
to the long runs (60000 steps): the pass at step 60000 and a ``recon_blur``
below that of step 6000 by more than delta_r, and prints their course by the
stacks seen. Prints markdown tables; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18.judge [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

DELTA = {"recon_blur": 0.01, "dyn_1/copy_1": 0.02}
METRICS = ("recon_blur", "dyn_1/copy_1")
LOWER = ("init", "color")  # the lower yardsticks
LEARNED = ("v16", "v17")  # the VAEs known to learn
LAST_STEP = 6000
LONG_STEPS = 60000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def _fmt(x: float | None, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def _tag(x: float) -> str:
    """``1e-3`` style of a learning rate, ``0.05`` of the others."""
    if x < 0.001 or x == 0.001:
        return f"{x:.0e}".replace("e-0", "e-")
    return f"{x:g}"


class Code:
    """The zurich numbers of one code (a run's checkpoint or a yardstick)."""

    def __init__(self, stats: dict[str, Any], zurich: dict[str, Any]) -> None:
        self.stats = stats
        self.zurich = zurich

    def __getitem__(self, key: str) -> float:
        if key in ("recon_blur", "recon_blur_sd", "dyn_1/copy_1", "rank"):
            return float(self.zurich[key])
        return float(self.stats[key])


def yardsticks(root: Path) -> dict[str, Code]:
    post = root / "posthoc_jepa"
    stats = {r["code"]: r for r in _jsonl(post / "jepa.jsonl")}
    zurich = {
        r["code"]: r for r in _jsonl(post / "probes.jsonl") if r["group"] == "zurich"
    }
    return {name: Code(stats[name], zurich[name]) for name in stats if name in zurich}


def delta(metric: str, a: Code, b: Code) -> float:
    """The threshold of comparing codes ``a`` and ``b`` on ``metric``.

    ``recon_blur`` takes twice the joint standard error of the 4 decoder
    seeds (V1's rule) when it is larger than delta_r.
    """
    if metric != "recon_blur":
        return DELTA[metric]
    se2 = 2.0 * math.sqrt((a["recon_blur_sd"] ** 2 + b["recon_blur_sd"] ** 2) / 4.0)
    return max(DELTA[metric], se2)


def usable_pairs(ys: dict[str, Code]) -> list[tuple[str, str]]:
    """(metric, yardstick) pairs that some learned VAE goes below."""
    out = []
    for m in METRICS:
        for y in LOWER:
            if any(ys[v][m] < ys[y][m] - delta(m, ys[v], ys[y]) for v in LEARNED):
                out.append((m, y))
    return out


class Run:
    """One screen run: its config, training rows and last-checkpoint code."""

    def __init__(self, path: Path, last: int = LAST_STEP) -> None:
        self.path = path
        self.last = last
        self.name = path.name
        conf = json.loads((path / "jepa.json").read_text())
        self.lr = float(conf["lr"])
        self.lam = float(conf["lam"])
        self.r = float(conf["mask_ratio"])
        self.k = int(conf["k"])
        self.train = _jsonl(path / "train.jsonl")
        post = path / "posthoc_jepa"
        self.stats = {r["step"]: r for r in _jsonl(post / "jepa.jsonl")}
        self.probes = {
            (r["step"], r["group"]): r for r in _jsonl(post / "probes.jsonl")
        }

    def finished(self) -> bool:
        return bool(self.train) and self.train[-1]["step"] == self.last

    def diverged(self) -> bool:
        return any(
            not math.isfinite(float(row[k]))
            for row in self.train
            for k in ("loss", "pred", "sigreg")
        )

    def code(self, step: int) -> Code | None:
        z = self.probes.get((step, "zurich"))
        s = self.stats.get(step)
        return None if z is None or s is None else Code(s, z)


def collapse(code: Code, ys: dict[str, Code]) -> dict[str, bool]:
    k1_line = (code["sigreg_heldin_gauss"] + code["sigreg_heldin_const"]) / 2.0
    rank_floor = min(ys["color"]["rank"], ys["init"]["rank"])
    theta3 = (1.0 + max(ys["color"]["token_r2"], ys["init"]["token_r2"])) / 2.0
    return {
        "K1": code["sigreg_heldin"] >= k1_line,
        "K2": code["rank"] < rank_floor,
        "K3": code["token_r2"] >= theta3,
    }


def judge(run: Run, ys: dict[str, Code], pairs: list[tuple[str, str]]) -> dict:
    code = run.code(run.last)
    row: dict[str, Any] = {
        "name": run.name,
        "lr": run.lr,
        "lam": run.lam,
        "r": run.r,
        "k": run.k,
        "finished": run.finished(),
        "diverged": run.diverged(),
    }
    if code is None or not run.finished() or run.diverged():
        row["passed"] = False
        row["collapsed"] = True
        return row
    ks = collapse(code, ys)
    beats = {f"{m}<{y}": code[m] < ys[y][m] - delta(m, code, ys[y]) for m, y in pairs}
    collapsed = any(ks.values())
    row.update(ks)
    row.update(beats)
    row["collapsed"] = collapsed
    row["passed"] = not collapsed and all(beats.values())
    for key in (
        "recon_blur",
        "recon_blur_sd",
        "dyn_1/copy_1",
        "rank",
        "token_r2",
        "sigreg_heldin",
        "sigreg_zurich",
        "participation",
        "rho_in",
        "z_var",
        "pred_ratio",
        "pred_ratio_calib",
    ):
        row[key] = code[key]
    row["train_pred"] = float(run.train[-1]["pred"])
    row["train_sigreg"] = float(run.train[-1]["sigreg"])
    row["recon_minus_init"] = code["recon_blur"] - ys["init"]["recon_blur"]
    two = run.code(2000)
    row["recon_blur_2000"] = None if two is None else two["recon_blur"]
    return row


def choose(rows: list[dict]) -> dict | None:
    """Rule 1 among ``rows``: least recon_blur, ties within delta_r by dyn."""
    passed = [r for r in rows if r["passed"]]
    if not passed:
        return None
    best = min(passed, key=lambda r: r["recon_blur"])
    near = [
        r for r in passed if r["recon_blur"] - best["recon_blur"] <= DELTA["recon_blur"]
    ]
    return min(near, key=lambda r: r["dyn_1/copy_1"])


def long_runs(rows: list[dict]) -> list[dict[str, Any]]:
    """Rules 1-3: a chosen setting per mask ratio at seed 0, one more at seed 1."""
    chosen = [
        c
        for r in sorted({x["r"] for x in rows})
        if (c := choose([x for x in rows if x["r"] == r])) is not None
    ]
    out = [_long(c, 0) for c in chosen]
    top = choose(chosen)
    if top is not None:
        out.append(_long(top, 1))
    return out


def _long(row: dict, seed: int) -> dict[str, Any]:
    name = f"v18-long-lr{_tag(row['lr'])}-lam{row['lam']:g}-r{row['r']:g}-s{seed}"
    return {
        "name": name,
        "lr": row["lr"],
        "lam": row["lam"],
        "mask": row["r"],
        "seed": seed,
        "steps": LONG_STEPS,
    }


def judge_long(
    run: Run, ys: dict[str, Code], pairs: list[tuple[str, str]]
) -> dict[str, Any]:
    """Rule 5 on a long run: passes at its last step and its ``recon_blur``
    there is below that of step 6000 by more than delta_r."""
    row = judge(run, ys, pairs)
    row["seed"] = int(run.name.rsplit("-s", 1)[1])
    first, last = run.code(LAST_STEP), run.code(run.last)
    if first is None or last is None:
        row["rule5"] = False
        return row
    drop = first["recon_blur"] - last["recon_blur"]
    row["recon_blur_6000"] = first["recon_blur"]
    row["recon_drop"] = drop
    row["recon_drop_delta"] = delta("recon_blur", first, last)
    row["rule5"] = row["passed"] and drop > row["recon_drop_delta"]
    return row


def course(run: Run) -> list[dict[str, Any]]:
    """The run's checkpoints by stacks seen: the JEPA numbers at every one,
    the zurich probe where it was taken, and the training means there."""
    batch = int(json.loads((run.path / "jepa.json").read_text())["batch"])
    train = {r["step"]: r for r in run.train}
    out = []
    for step, s in sorted(run.stats.items()):
        z = run.probes.get((step, "zurich"))
        span = run.probes.get((step, "span"))
        t = train.get(step, {})
        out.append(
            {
                "step": step,
                "stacks": step * batch,
                "recon_blur": None if z is None else z["recon_blur"],
                "recon_blur_sd": None if z is None else z["recon_blur_sd"],
                "dyn_1/copy_1": None if z is None else z["dyn_1/copy_1"],
                "rank": None if z is None else z["rank"],
                "span_recon_blur": None if span is None else span["recon_blur"],
                "span_dyn_1/copy_1": None if span is None else span["dyn_1/copy_1"],
                "sigreg_heldin": s["sigreg_heldin"],
                "token_r2": s["token_r2"],
                "participation": s["participation"],
                "pred_ratio": s["pred_ratio"],
                "train_pred": t.get("pred"),
                "train_sigreg": t.get("sigreg"),
                "lr": t.get("lr"),
            }
        )
    return out


def print_long(rows: list[dict], courses: dict[str, list[dict]]) -> None:
    print("\n## long runs (zurich): rule 5\n")
    print(
        "| run | K1 | K2 | K3 | recon_blur 6000 | recon_blur 60000 | drop | δ"
        " | dyn 60000 | passed | rule 5 |"
    )
    print("| -- " * 11 + "|")
    flag = lambda b: "×" if b else "·"  # noqa: E731
    for x in rows:
        print(
            f"| {x['name']} | {flag(x['K1'])} | {flag(x['K2'])} | {flag(x['K3'])}"
            f" | {_fmt(x['recon_blur_6000'], 4)} | {_fmt(x['recon_blur'], 4)}"
            f" | {x['recon_drop']:+.4f} | {_fmt(x['recon_drop_delta'], 4)}"
            f" | {_fmt(x['dyn_1/copy_1'])} | {'pass' if x['passed'] else 'fail'}"
            f" | {'pass' if x['rule5'] else 'fail'} |"
        )
    for name, c in courses.items():
        print(f"\n### {name}\n")
        print(
            "| step | stacks (万) | recon_blur | dyn | span recon_blur | span dyn"
            " | rank | sigreg_heldin | token_r2 | participation | pred_ratio"
            " | train pred | train sigreg |"
        )
        print("| -- " * 13 + "|")
        for r in c:
            print(
                f"| {r['step']} | {r['stacks'] / 1e4:g} | {_fmt(r['recon_blur'], 4)}"
                f" | {_fmt(r['dyn_1/copy_1'])} | {_fmt(r['span_recon_blur'], 4)}"
                f" | {_fmt(r['span_dyn_1/copy_1'])} | {_fmt(r['rank'], 1)}"
                f" | {_fmt(r['sigreg_heldin'], 2)} | {_fmt(r['token_r2'])}"
                f" | {_fmt(r['participation'], 1)} | {_fmt(r['pred_ratio'])}"
                f" | {_fmt(r['train_pred'])} | {_fmt(r['train_sigreg'])} |"
            )


def print_tables(
    ys: dict[str, Code], pairs: list[tuple[str, str]], rows: list[dict]
) -> None:
    print("## yardsticks (zurich)\n")
    print("| code | recon_blur | sd | dyn_1/copy_1 | rank | token_r2 | sigreg_heldin |")
    print("| -- | -- | -- | -- | -- | -- | -- |")
    for name, c in ys.items():
        tr = c.stats.get("token_r2")
        print(
            f"| `{name}` | {_fmt(c['recon_blur'], 4)} | {_fmt(c['recon_blur_sd'], 4)}"
            f" | {_fmt(c['dyn_1/copy_1'])} | {_fmt(c['rank'], 1)}"
            f" | {_fmt(tr)} | {_fmt(c['sigreg_heldin'], 1)} |"
        )
    theta3 = (1.0 + max(ys["color"]["token_r2"], ys["init"]["token_r2"])) / 2.0
    floor = min(ys["color"]["rank"], ys["init"]["rank"])
    print(f"\nusable pairs: {pairs}; theta3 = {theta3:.4f}; rank floor = {floor:.2f}\n")

    print("## runs (zurich, step 6000)\n")
    print(
        "| r | lr | λ | K1 | K2 | K3 | recon_blur | −init | dyn | rank | token_r2"
        " | sigreg_heldin | pred_ratio | pred_ratio_calib | train pred | passed |"
    )
    print("| -- " * 16 + "|")
    for x in sorted(rows, key=lambda x: (x["r"], x["lr"], x["lam"])):
        if "recon_blur" not in x:
            print(
                f"| {x['r']:g} | {_tag(x['lr'])} | {x['lam']:g} | not finished/diverged |"
            )
            continue
        flag = lambda b: "×" if b else "·"  # noqa: E731
        print(
            f"| {x['r']:g} | {_tag(x['lr'])} | {x['lam']:g}"
            f" | {flag(x['K1'])} | {flag(x['K2'])} | {flag(x['K3'])}"
            f" | {_fmt(x['recon_blur'])} | {x['recon_minus_init']:+.3f}"
            f" | {_fmt(x['dyn_1/copy_1'])} | {_fmt(x['rank'], 1)}"
            f" | {_fmt(x['token_r2'])} | {_fmt(x['sigreg_heldin'], 2)}"
            f" | {_fmt(x['pred_ratio'])} | {_fmt(x['pred_ratio_calib'])}"
            f" | {_fmt(x['train_pred'])} | {'pass' if x['passed'] else 'fail'} |"
        )
    print("\n## by mask ratio\n")
    print(
        "| r | k | collapsed | passed | min recon_blur (uncollapsed) | min dyn (uncollapsed) | mean pred_ratio (uncollapsed) |"
    )
    print("| -- | -- | -- | -- | -- | -- | -- |")
    for r in sorted({x["r"] for x in rows}):
        g = [x for x in rows if x["r"] == r]
        ok = [x for x in g if not x["collapsed"]]
        print(
            f"| {r:g} | {g[0]['k']} | {sum(x['collapsed'] for x in g)}/{len(g)}"
            f" | {sum(x['passed'] for x in g)}/{len(g)}"
            f" | {_fmt(min((x['recon_blur'] for x in ok), default=None))}"
            f" | {_fmt(min((x['dyn_1/copy_1'] for x in ok), default=None))}"
            f" | {_fmt(sum(x['pred_ratio'] for x in ok) / len(ok) if ok else None)} |"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()
    ys = yardsticks(a.runs / "v18-baselines")
    pairs = usable_pairs(ys)
    if not any(m == "recon_blur" for m, _ in pairs) or not any(
        m == "dyn_1/copy_1" for m, _ in pairs
    ):
        raise SystemExit(f"no usable pair for a metric: {pairs}; check the evaluation")
    runs = [Run(d) for d in sorted((a.runs / "v18-screen").glob("v18-*-s0"))]
    rows = [judge(r, ys, pairs) for r in runs]
    print_tables(ys, pairs, rows)
    longs = long_runs(rows)
    print(f"\nlong runs: {json.dumps(longs, indent=1)}")
    out: dict[str, Any] = {"pairs": pairs, "rows": rows, "long_runs": longs}
    long_dirs = [a.runs / "v18-long" / x["name"] for x in longs]
    if all(d.exists() for d in long_dirs):
        long = [Run(d, last=LONG_STEPS) for d in long_dirs]
        long_rows = [judge_long(r, ys, pairs) for r in long]
        courses = {r.name: course(r) for r in long}
        print_long(long_rows, courses)
        top = long_rows[-1]  # rule 3's setting is the one with a seed-1 run
        both = [
            x
            for x in long_rows
            if (x["lr"], x["lam"], x["r"]) == (top["lr"], top["lam"], top["r"])
        ]
        verdict = {
            "setting": {"lr": top["lr"], "lam": top["lam"], "r": top["r"]},
            "rule5": {x["seed"]: x["rule5"] for x in both},
            "to_v19": all(x["rule5"] for x in both),
        }
        print(f"\nverdict: {json.dumps(verdict)}")
        out |= {"long": long_rows, "courses": courses, "verdict": verdict}
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
