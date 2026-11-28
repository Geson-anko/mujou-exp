"""The judgement of V18c's screen (exp/vision/README.md, the V18c section).

Reads ``runs/v18c-screen/<run>/`` (``jepa.json``, ``train.jsonl`` and
``posthoc_jepa/{jepa,probes}.jsonl``), the run's times in
``runs/v18c-screen/logs/<run>.time``, the Imagenette probes in
``runs/v18c-screen/imagenette/<run>/imagenette.jsonl``, the yardsticks of V18c
(``runs/v18c-baselines``: ``init_c``, ``color8``) and of V18
(``runs/v18-baselines``: ``color``), and applies the rules registered before
the runs, as they are: the collapses K1-K3 and divergence at step 6000, the
pending rule of the first stage, the short-run pass, the second-stage list and
the predictions that the screen can answer. The new evaluation (generation and
Inception, E1) is not read: the long run's setting is not chosen here.
Prints markdown tables; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18c.judge_screen [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

STEPS = 6000  # the screen's last step
FIRST_R = 0.5
SECOND_R = (0.25, 0.75)
LRS = (1e-5, 3e-5, 1e-4)
LAMS = (0.5, 0.8, 0.95)
PENDING = 0.05  # pred_ratio_calib and sigreg_heldin over the reference
PASS_PRED = 0.95  # pred_ratio_calib below this
PASS_S = 2.0  # sigreg_heldin below this (between 1.05 and K3's 2.95)
DELTA_A = 0.023  # Imagenette accuracy, as a fraction (prediction 9)
CODE = "z"  # the code of every V18c rule: z = W vec(t)
YARD_CODES = ("init_c", "init_c_h_mean", "color8")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def _fmt(x: float | None, d: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:.{d}f}"


def _tag(x: float) -> str:
    """``1e-4`` / ``3e-5`` style of a learning rate."""
    return f"{x:.0e}".replace("e-0", "e-")


def _flag(b: bool | None) -> str:
    return "—" if b is None else "×" if b else "·"


def run_name(lr: float, lam: float, r: float, seed: int = 0) -> str:
    return f"v18c-lr{_tag(lr)}-lam{lam:g}-r{r:g}-s{seed}"


def yardsticks(runs: Path) -> dict[str, dict[str, float]]:
    """Code -> its step-0 stats with the zurich ``rank`` (V18's ``color`` and
    V18c's codes; the names do not clash)."""
    out: dict[str, dict[str, float]] = {}
    for root in (runs / "v18-baselines", runs / "v18c-baselines"):
        post = root / "posthoc_jepa"
        zurich = {
            r["code"]: r
            for r in _jsonl(post / "probes.jsonl")
            if r.get("group") == "zurich"
        }
        for r in _jsonl(post / "jepa.jsonl"):
            row = {k: float(v) for k, v in r.items() if isinstance(v, (int, float))}
            if r["code"] in zurich:
                row["rank"] = float(zurich[r["code"]]["rank"])
                row["recon_blur"] = float(zurich[r["code"]]["recon_blur"])
                row["dyn_1/copy_1"] = float(zurich[r["code"]]["dyn_1/copy_1"])
            out[r["code"]] = row
    return out


def lines(ys: dict[str, dict[str, float]]) -> dict[str, float]:
    """K1's line, K2's rank floor and K3's theta_3."""
    init = ys["init_c"]
    return {
        "k1": (init["sigreg_heldin_gauss"] + init["sigreg_heldin_const"]) / 2.0,
        "rank_floor": min(ys["color"]["rank"], init["rank"]),
        "theta3": (1.0 + max(ys["color8"]["token_r2"], init["token_r2"])) / 2.0,
    }


class Run:
    """One run: its config, training rows, codes, times and Imagenette."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        conf = json.loads((path / "jepa.json").read_text())
        self.lr = float(conf["lr"])
        self.lam = float(conf["lam"])
        self.r = float(conf["mask_ratio"])
        self.steps = int(conf["steps"])
        self.snap = str(conf["provenance"]["code"]["snap"])[:12]
        self.train = _jsonl(path / "train.jsonl")
        post = path / "posthoc_jepa"
        self.stats = {(x["code"], x["step"]): x for x in _jsonl(post / "jepa.jsonl")}
        self.zurich = {
            (x["code"], x["step"]): x
            for x in _jsonl(post / "probes.jsonl")
            if x["group"] == "zurich"
        }
        imagenette = path.parent / "imagenette" / self.name / "imagenette.jsonl"
        self.acc = {
            (int(x["step"]), str(x["code"])): float(x["acc"])
            for x in _jsonl(imagenette)
            if "run" in x
        }
        self.color_acc = next(
            (float(x["acc"]) for x in _jsonl(imagenette) if x.get("code") == "color"),
            None,
        )
        self.times = self._times(path.parent / "logs" / f"{self.name}.time")

    @staticmethod
    def _times(path: Path) -> dict[str, float]:
        """Seconds of training and of the evaluation (the last attempt)."""
        if not path.exists():
            return {}
        marks: dict[str, float] = {}
        for line in path.read_text().splitlines():
            key, stamp = line.split()[:2]
            marks[key] = float(stamp)
        if "train_end" not in marks:
            return {}
        out = {"train": marks["train_end"] - marks["start"]}
        if "eval_end" in marks:
            out["eval"] = marks["eval_end"] - marks["train_end"]
        return out

    def finished(self) -> bool:
        return bool(self.train) and self.train[-1]["step"] == self.steps

    def diverged(self) -> bool:
        return any(
            not math.isfinite(float(row[k]))
            for row in self.train
            for k in ("loss", "pred", "sigreg")
        )

    def stat(self, key: str, step: int = STEPS) -> float | None:
        s = self.stats.get((CODE, step))
        return None if s is None or key not in s else float(s[key])

    def probe(self, key: str, step: int = STEPS) -> float | None:
        z = self.zurich.get((CODE, step))
        return None if z is None else float(z[key])

    def train_at(self, key: str, step: int = STEPS) -> float | None:
        row = next((x for x in self.train if x["step"] == step), None)
        return None if row is None else float(row[key])

    def sec_per_step(self) -> float | None:
        """The mean of the logged ``sec`` over the rows past step 100."""
        rows = [x for x in self.train if x["step"] > 100]
        if len(rows) < 2:
            return None
        return (rows[-1]["sec"] - rows[0]["sec"]) / (rows[-1]["step"] - rows[0]["step"])


def judge(run: Run, ln: dict[str, float]) -> dict[str, Any]:
    """Collapse, divergence and the pass's two value tests of ``run`` at step
    6000, with the numbers the README lays out next to them."""
    row: dict[str, Any] = {
        "name": run.name,
        "lr": run.lr,
        "lam": run.lam,
        "r": run.r,
        "snap": run.snap,
        "finished": run.finished(),
        "diverged": run.diverged(),
        "sec_per_step": run.sec_per_step(),
        **{f"{k}_sec": v for k, v in run.times.items()},
    }
    for key in ("pred", "sigreg", "mix_var", "rho_in", "grad_norm"):
        row[f"train_{key}"] = run.train_at(key)
        row[f"train_{key}_1000"] = run.train_at(key, 1000)
    for key in (
        "sigreg_heldin",
        "token_r2",
        "rho_in",
        "participation",
        "pred_ratio_calib",
        "sigreg_zurich",
        "embed_drift",
        "block_drift",
    ):
        row[key] = run.stat(key)
    row["pred_ratio_calib_2000"] = run.stat("pred_ratio_calib", 2000)
    row["sigreg_heldin_2000"] = run.stat("sigreg_heldin", 2000)
    for key in ("rank", "recon_blur", "dyn_1/copy_1"):
        row[key] = run.probe(key)
    for step in (0, STEPS):
        for code in ("h_mean", "z"):
            row[f"in_{code}_{step}"] = run.acc.get((step, code))
    row["in_color"] = run.color_acc
    s, rank, r2, q = (
        run.stat("sigreg_heldin"),
        run.probe("rank"),
        run.stat("token_r2"),
        run.stat("pred_ratio_calib"),
    )
    row["complete"] = run.finished() and None not in (s, rank, r2, q)
    if s is None or rank is None or r2 is None or q is None:
        row |= {"K1": None, "K2": None, "K3": None, "collapsed": True}
        return row
    if not run.finished() or run.diverged():
        row |= {"K1": None, "K2": None, "K3": None, "collapsed": True}
        return row
    row["K1"] = s >= ln["k1"]
    row["K2"] = rank < ln["rank_floor"]
    row["K3"] = r2 >= ln["theta3"]
    row["collapsed"] = bool(row["K1"] or row["K2"] or row["K3"])
    row["beats_pred"] = q < PASS_PRED
    row["healthy_s"] = s < PASS_S
    return row


def apply_pending(rows: list[dict[str, Any]]) -> None:
    """First stage: per lambda, the non-collapsed run of the largest learning
    rate is the reference; another run is pending when its pred_ratio_calib or
    sigreg_heldin exceeds the reference's by more than PENDING."""
    for row in rows:
        row["pending"] = False
        row["reference"] = None
    for lam in LAMS:
        same = [x for x in rows if x["lam"] == lam and not x["collapsed"]]
        if not same:
            continue
        ref = max(same, key=lambda x: x["lr"])
        for x in same:
            x["reference"] = ref["name"]
            if x is ref:
                continue
            x["pending"] = (
                x["pred_ratio_calib"] > ref["pred_ratio_calib"] + PENDING
                or x["sigreg_heldin"] > ref["sigreg_heldin"] + PENDING
            )


def apply_pass(rows: list[dict[str, Any]]) -> None:
    for x in rows:
        x["passed"] = bool(
            not x["collapsed"]
            and not x["diverged"]
            and not x.get("pending", False)
            and x["beats_pred"]
            and x["healthy_s"]
        )


def second_stage(first: list[dict[str, Any]]) -> list[str]:
    """Per lambda, the first stage's pass of the largest learning rate, at
    r = 0.25 and 0.75 (none when nothing passed)."""
    names = []
    for lam in LAMS:
        passed = [x for x in first if x["lam"] == lam and x["passed"]]
        if not passed:
            continue
        best = max(passed, key=lambda x: x["lr"])
        names += [run_name(best["lr"], lam, r) for r in SECOND_R]
    return names


def predictions(
    ys: dict[str, dict[str, float]],
    ln: dict[str, float],
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    stage2: list[str],
) -> dict[str, Any]:
    """The registered predictions 2 and 4-10 (2 and 3 from the yardsticks;
    3's step-0 h_mean from the first run of the first stage)."""
    lr4 = sorted((x for x in first if x["lr"] == 1e-4), key=lambda x: x["lam"])
    in0 = next((x["in_h_mean_0"] for x in first if x["in_h_mean_0"] is not None), None)
    p: dict[str, Any] = {
        "2_init_c_sigreg_in_45_75": 45 <= ys["init_c"]["sigreg_heldin"] <= 75,
        "2_init_c_token_r2_in_-0.05_0.2": -0.05 <= ys["init_c"]["token_r2"] <= 0.2,
        "2_color8_token_r2_in_-0.1_0.2": -0.1 <= ys["color8"]["token_r2"] <= 0.2,
        "2_theta3_in_0.45_0.6": 0.45 <= ln["theta3"] <= 0.6,
        "3_h_mean_0_in_42_50": None if in0 is None else 0.42 <= in0 <= 0.50,
        "4_no_divergence": not any(x["diverged"] for x in first),
        "5_no_collapse": not any(x["collapsed"] for x in first),
        "5_token_r2_le_0.1": all(
            x["token_r2"] is not None and x["token_r2"] <= 0.1 for x in first
        ),
        "5_rho_in_ge_0.9": all(
            x["rho_in"] is not None and x["rho_in"] >= 0.9 for x in first
        ),
    }
    if len(lr4) == 3 and all(x["complete"] for x in lr4):
        s = [x["sigreg_heldin"] for x in lr4]
        q = [x["pred_ratio_calib"] for x in lr4]
        p["6_lr1e-4_sigreg_in_1.10_1.35"] = all(1.10 <= v <= 1.35 for v in s)
        p["6_sigreg_decreases_with_lam"] = s[0] > s[1] > s[2]
        p["6_lr1e-4_pred_ratio_in_0.60_0.85"] = all(0.60 <= v <= 0.85 for v in q)
        p["6_pred_ratio_increases_with_lam"] = q[0] < q[1] < q[2]
        p["8_lr1e-4_all_pass"] = all(x["passed"] for x in lr4)
        gains = [
            x["in_h_mean_6000"] - x["in_h_mean_0"]
            for x in lr4
            if x["in_h_mean_6000"] is not None and x["in_h_mean_0"] is not None
        ]
        p["9_h_mean_gain_ge_delta_a_in_2_of_3"] = (
            None if len(gains) < 3 else sum(g >= DELTA_A for g in gains) >= 2
        )
    p["7_pending_lr1e-5_all_3"] = sum(
        x["pending"] for x in first if x["lr"] == 1e-5
    ) == len([x for x in first if x["lr"] == 1e-5])
    p["7_pending_lr3e-5_ge_1"] = any(x["pending"] for x in first if x["lr"] == 3e-5)
    p["8_second_stage_is_lr1e-4"] = stage2 == [
        run_name(1e-4, lam, r) for lam in LAMS for r in SECOND_R
    ]
    if second:
        p["10_no_collapse"] = not any(x["collapsed"] for x in second)
        order = []
        for lam in LAMS:
            lr = {x["lr"] for x in second if x["lam"] == lam}
            if not lr:
                continue
            by_r = {
                x["r"]: x for x in first + second if x["lam"] == lam and x["lr"] in lr
            }
            q = [by_r[r]["pred_ratio_calib"] for r in (0.25, 0.5, 0.75) if r in by_r]
            order.append(len(q) == 3 and q[0] < q[1] < q[2])
        p["10_pred_ratio_order_r"] = all(order) if order else None
    return p


def _table(rows: list[dict[str, Any]]) -> str:
    head = (
        "| run | 学習率 | λ | r | pred | S | `pred_ratio_calib` | `sigreg_heldin` "
        "| `token_r2` | `rho_in` | `rank` | `participation` | `embed_drift` "
        "| `block_drift` | K1 | K2 | K3 | 保留 | 合格 |\n"
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- "
        "| -- | -- | -- | -- | -- |"
    )
    out = [head]
    for x in rows:
        out.append(
            f"| `{x['name']}` | {_tag(x['lr'])} | {x['lam']:g} | {x['r']:g} "
            f"| {_fmt(x['train_pred'])} | {_fmt(x['train_sigreg'])} "
            f"| {_fmt(x['pred_ratio_calib'])} | {_fmt(x['sigreg_heldin'])} "
            f"| {_fmt(x['token_r2'])} | {_fmt(x['rho_in'])} | {_fmt(x['rank'], 1)} "
            f"| {_fmt(x['participation'], 1)} | {_fmt(x['embed_drift'], 4)} "
            f"| {_fmt(x['block_drift'], 4)} | {_flag(x['K1'])} | {_flag(x['K2'])} "
            f"| {_flag(x['K3'])} | {_flag(x.get('pending'))} | "
            f"{'**合格**' if x['passed'] else '·'} |"
        )
    return "\n".join(out)


def _side_table(rows: list[dict[str, Any]]) -> str:
    def pct(v: float | None) -> str:
        return "—" if v is None else f"{100 * v:.1f}"

    out = [
        "| run | `h_mean` 0 | `h_mean` 6000 | `z` 0 | `z` 6000 | `recon_blur` "
        "| `dyn_1/copy_1` | `sigreg_zurich` | `mix_var` | 秒 / step | 学習 (分) "
        "| 評価 (分) |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for x in rows:
        out.append(
            f"| `{x['name']}` | {pct(x['in_h_mean_0'])} | {pct(x['in_h_mean_6000'])} "
            f"| {pct(x['in_z_0'])} | {pct(x['in_z_6000'])} | {_fmt(x['recon_blur'])} "
            f"| {_fmt(x['dyn_1/copy_1'])} | {_fmt(x['sigreg_zurich'], 1)} "
            f"| {_fmt(x['train_mix_var'])} | {_fmt(x['sec_per_step'])} "
            f"| {_fmt(x.get('train_sec', math.nan) / 60, 1)} "
            f"| {_fmt(x.get('eval_sec', math.nan) / 60, 1)} |"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    screen = args.runs / "v18c-screen"
    ys = yardsticks(args.runs)
    ln = lines(ys)
    runs = [Run(p.parent) for p in sorted(screen.glob("v18c-*/jepa.json"))]
    rows = {r.name: judge(r, ln) for r in runs}
    first = [
        rows[n]
        for lr in (1e-4, 1e-5, 3e-5)
        for lam in LAMS
        if (n := run_name(lr, lam, FIRST_R)) in rows
    ]
    apply_pending(first)
    apply_pass(first)
    stage2 = second_stage(first)
    second = [rows[n] for n in stage2 if n in rows]
    for x in second:
        x["pending"] = None  # the pending rule is for the first stage only
    apply_pass(second)
    preds = predictions(ys, ln, first, second, stage2)

    print("## 物差し\n")
    print(
        "| 符号 | `sigreg_heldin` | `token_r2` | `rho_in` | `rank` | `participation` |"
    )
    print("| -- | -- | -- | -- | -- | -- |")
    for c in ("color", *YARD_CODES):
        y = ys.get(c, {})
        print(
            f"| `{c}` | {_fmt(y.get('sigreg_heldin'))} | {_fmt(y.get('token_r2'))} "
            f"| {_fmt(y.get('rho_in'))} | {_fmt(y.get('rank'), 1)} "
            f"| {_fmt(y.get('participation'), 1)} |"
        )
    print(
        f"\nK1 ≥ {ln['k1']:.2f}、K2 の `rank` の床 {ln['rank_floor']:.2f}、"
        f"K3 の θ3 = {ln['theta3']:.3f}\n"
    )
    print("## 第 1 段\n")
    print(_table(first))
    print()
    print(_side_table(first))
    print(f"\n第 2 段の一覧: {stage2 or '（なし）'}\n")
    if second:
        print("## 第 2 段\n")
        print(_table(second))
        print()
        print(_side_table(second))
        print()
    print("## 予想\n")
    for k, v in preds.items():
        print(f"- {k}: {v}")
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "yardsticks": {c: ys.get(c) for c in ("color", *YARD_CODES)},
                    "lines": ln,
                    "first": first,
                    "second_stage": stage2,
                    "second": second,
                    "predictions": preds,
                },
                indent=1,
                ensure_ascii=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
