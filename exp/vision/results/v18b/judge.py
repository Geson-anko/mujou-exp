"""The judgement of V18b's screen (exp/vision/README.md, the V18b section).

Reads ``runs/v18b-screen/<run>/`` (``jepa.json``, ``train.jsonl`` and
``posthoc_jepa/{jepa,probes}.jsonl``), the Imagenette probes in
``runs/v18b-screen/imagenette/<run>/imagenette.jsonl``, the yardsticks of V18
(``runs/v18-baselines``) and of patch 4 (``runs/v18b-baselines``), and applies
the rules registered before the runs, as they are: the usable pairs per arm,
the collapses K1-K3 per arm, the pass at step 6000, the pending rule of the
first stage, the choice of each arm's best, the second-stage list and, once
the second stage is in, the setting and the long runs of each arm. Prints
markdown tables; ``--json`` also writes the numbers::

    uv run python -m exp.vision.results.v18b.judge [--runs runs] [--json out.json]
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exp.vision.results.v18.judge import (
    DELTA,
    LEARNED,
    METRICS,
    Code,
    delta,
    yardsticks as read_yardsticks,
)

LONG_STEPS = 60000  # s_long, fixed by the trial runs (README, V18b trial runs)
DELTA_A = 0.023  # Imagenette accuracy, as a fraction
PENDING = 0.05  # pred_ratio_calib and sigreg_heldin over the reference
FIRST_R = 0.5
SECOND_R = (0.25, 0.75)
V18_BEST = (1e-4, 0.5)  # P16's setting whose runs V18 already has
V18_LONG = (1e-4, 0.5, 0.25)  # the setting of V18's two long runs


@dataclass(frozen=True)
class Arm:
    """Which code of an arm plays which role in the rules."""

    name: str  # "f", "m" or "p16"
    judge: str  # the code of recon_blur, dyn_1/copy_1 and rank (K2)
    sigreg: str  # the SIGReg data point: K1 and the pending rule's S
    tokens: str  # the code carrying token_r2 (K3) and rho_in
    init: str  # the arm's init yardstick (its step-0 judge code)
    init_tokens: str  # the yardstick carrying the init's token_r2
    color_tokens: str  # the same-grid color for theta_3


ARMS = {
    "f": Arm("f", "z", "z", "z", "init_f", "init_f", "color4"),
    "m": Arm("m", "zpca", "zbar", "zflat", "init_m", "init_m_zflat", "color4"),
    "p16": Arm("p16", "z", "z", "z", "init", "init", "color"),
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def _fmt(x: float | None, d: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def _tag(x: float) -> str:
    """``1e-4`` / ``3e-5`` style of a learning rate."""
    return f"{x:.0e}".replace("e-0", "e-")


def _flag(b: bool) -> str:
    return "×" if b else "·"


def run_name(arm: str, lr: float, lam: float, r: float, seed: int = 0) -> str:
    return f"v18b-{arm}-lr{_tag(lr)}-lam{lam:g}-r{r:g}-s{seed}"


def yardsticks(runs: Path) -> dict[str, Code]:
    """V18's yardsticks with patch 4's next to them (the names do not clash)."""
    return read_yardsticks(runs / "v18-baselines") | read_yardsticks(
        runs / "v18b-baselines"
    )


def usable_pairs(arm: Arm, ys: dict[str, Code]) -> list[tuple[str, str]]:
    """(metric, yardstick) pairs, with the yardsticks V18's ``color`` and the
    arm's init, that some learned VAE goes below by more than delta."""
    return [
        (m, y)
        for m in METRICS
        for y in ("color", arm.init)
        if any(ys[v][m] < ys[y][m] - delta(m, ys[v], ys[y]) for v in LEARNED)
    ]


def lines(arm: Arm, ys: dict[str, Code]) -> dict[str, float]:
    """The arm's rank floor (K2) and theta_3 (K3)."""
    return {
        "rank_floor": min(ys["color"]["rank"], ys[arm.init]["rank"]),
        "theta3": (
            1.0 + max(ys[arm.color_tokens]["token_r2"], ys[arm.init_tokens]["token_r2"])
        )
        / 2.0,
    }


class Run:
    """One run: its config, training rows, codes and Imagenette probes."""

    def __init__(self, path: Path, imagenette: Path) -> None:
        self.path = path
        self.name = path.name
        conf = json.loads((path / "jepa.json").read_text())
        self.arm = ARMS[self.name.split("-")[1]]
        self.lr = float(conf["lr"])
        self.lam = float(conf["lam"])
        self.r = float(conf["mask_ratio"])
        self.steps = int(conf["steps"])
        self.train = _jsonl(path / "train.jsonl")
        post = path / "posthoc_jepa"
        self.stats = {(x["code"], x["step"]): x for x in _jsonl(post / "jepa.jsonl")}
        self.probes = {
            (x["code"], x["step"], x["group"]): x for x in _jsonl(post / "probes.jsonl")
        }
        self.acc = {
            (x["step"], x["code"]): float(x["acc"])
            for x in _jsonl(imagenette / self.name / "imagenette.jsonl")
            if "run" in x
        }

    def finished(self) -> bool:
        return bool(self.train) and self.train[-1]["step"] == self.steps

    def diverged(self) -> bool:
        return any(
            not math.isfinite(float(row[k]))
            for row in self.train
            for k in ("loss", "pred", "sigreg")
        )

    def code(self, name: str, step: int) -> Code | None:
        z = self.probes.get((name, step, "zurich"))
        s = self.stats.get((name, step))
        return None if z is None or s is None else Code(s, z)

    def stat(self, name: str, step: int, key: str) -> float | None:
        s = self.stats.get((name, step))
        return None if s is None or key not in s else float(s[key])

    def train_at(self, step: int) -> dict[str, Any]:
        return next((x for x in self.train if x["step"] == step), {})


def judge(run: Run, ys: dict[str, Code]) -> dict[str, Any]:
    """Collapse and pass of ``run`` at its last step, with the numbers the
    README lays out next to them."""
    arm = run.arm
    row: dict[str, Any] = {
        "name": run.name,
        "arm": arm.name,
        "lr": run.lr,
        "lam": run.lam,
        "r": run.r,
        "finished": run.finished(),
        "diverged": run.diverged(),
    }
    code = run.code(arm.judge, run.steps)
    if code is None or not run.finished() or run.diverged():
        row |= {"collapsed": True, "passed": False}
        return row
    pairs = usable_pairs(arm, ys)
    ln = lines(arm, ys)
    s = run.stats[(arm.sigreg, run.steps)]
    t = run.stats[(arm.tokens, run.steps)]
    ks = {
        "K1": s["sigreg_heldin"]
        >= (s["sigreg_heldin_gauss"] + s["sigreg_heldin_const"]) / 2.0,
        "K2": code["rank"] < ln["rank_floor"],
        "K3": t["token_r2"] >= ln["theta3"],
    }
    beats = {f"{m}<{y}": code[m] < ys[y][m] - delta(m, code, ys[y]) for m, y in pairs}
    judgeable = all(any(m == pm for pm, _ in pairs) for m in METRICS)
    collapsed = any(ks.values())
    row |= ks | beats
    row["collapsed"] = collapsed
    row["judgeable"] = judgeable
    row["passed"] = judgeable and not collapsed and all(beats.values())
    row["recon_blur"] = code["recon_blur"]
    row["recon_blur_sd"] = code["recon_blur_sd"]
    row["dyn_1/copy_1"] = code["dyn_1/copy_1"]
    row["rank"] = code["rank"]
    row["token_r2"] = float(t["token_r2"])
    row["theta3"] = ln["theta3"]
    row["rho_in"] = float(t["rho_in"])
    row["sigreg_heldin"] = float(s["sigreg_heldin"])
    row["sigreg_zurich"] = float(s["sigreg_zurich"])
    row["participation"] = float(code["participation"])
    row["pred_ratio"] = float(s["pred_ratio"])
    row["pred_ratio_calib"] = float(s["pred_ratio_calib"])
    row["embed_drift"] = float(s["embed_drift"])
    row["block_drift"] = float(s["block_drift"])
    last = run.train[-1]
    row["train_pred"] = float(last["pred"])
    row["train_sigreg"] = float(last["sigreg"])
    row["recon_minus_init"] = code["recon_blur"] - ys[arm.init]["recon_blur"]
    two = run.code(arm.judge, 2000)
    row["recon_blur_2000"] = None if two is None else two["recon_blur"]
    row["rho_in_0"] = run.stat(arm.tokens, 0, "rho_in")
    for code_name in ("h_mean", "z", "zbar"):
        a0, a1 = run.acc.get((0, code_name)), run.acc.get((run.steps, code_name))
        row[f"img_{code_name}_0"] = a0
        row[f"img_{code_name}"] = a1
    a0, a1 = row["img_h_mean_0"], row["img_h_mean"]
    row["img_h_mean_up"] = None if a0 is None or a1 is None else a1 - a0 >= DELTA_A
    if arm.name == "m":
        for c in ("zbar", "zflat"):
            cc = run.code(c, run.steps)
            row[f"{c}_recon_blur"] = None if cc is None else cc["recon_blur"]
            row[f"{c}_dyn"] = None if cc is None else cc["dyn_1/copy_1"]
    hm = run.code("h_mean", run.steps)
    row["h_mean_recon_blur"] = None if hm is None else hm["recon_blur"]
    return row


def pending(rows: list[dict[str, Any]]) -> None:
    """The first stage's pending rule, in place: per arm and lambda, the
    uncollapsed run of the largest learning rate is the reference, and any
    other uncollapsed run whose pred_ratio_calib or sigreg_heldin (of the
    SIGReg data point) exceeds the reference's by more than 0.05 is pending."""
    for x in rows:
        x["pending"] = False
        x["reference"] = None
    for arm in ARMS:
        for lam in sorted({x["lam"] for x in rows}):
            group = [
                x
                for x in rows
                if x["arm"] == arm and x["lam"] == lam and not x["collapsed"]
            ]
            if not group:
                continue
            ref = max(group, key=lambda x: x["lr"])
            for x in group:
                x["reference"] = ref["name"]
                if x is ref:
                    continue
                x["pending"] = (
                    x["pred_ratio_calib"] > ref["pred_ratio_calib"] + PENDING
                    or x["sigreg_heldin"] > ref["sigreg_heldin"] + PENDING
                )


def choose(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
    """The choice rule among ``rows``: the passing, not pending runs if any
    (else the uncollapsed, not pending ones, with ``False`` for "no pass");
    least recon_blur, ties within delta_r by least dyn_1/copy_1, ties within
    delta_d of that by the largest Imagenette h_mean."""
    ok = [x for x in rows if not x["collapsed"] and not x.get("pending", False)]
    passed = [x for x in ok if x["passed"]]
    pool = passed or ok
    if not pool:
        return None, False
    best = min(x["recon_blur"] for x in pool)
    near = [x for x in pool if x["recon_blur"] - best <= DELTA["recon_blur"]]
    dyn = min(x["dyn_1/copy_1"] for x in near)
    near = [x for x in near if x["dyn_1/copy_1"] - dyn <= DELTA["dyn_1/copy_1"]]
    if len(near) == 1:
        return near[0], bool(passed)
    return max(near, key=lambda x: x["img_h_mean"] or -1.0), bool(passed)


def second_stage(best: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    """The second-stage runs: each arm's best at r = 0.25 and 0.75. P16 at
    V18's setting reuses V18's screen runs."""
    out = []
    for arm, b in best.items():
        if b is None:
            continue
        for r in SECOND_R:
            reuse = None
            if arm == "p16" and (b["lr"], b["lam"]) == V18_BEST:
                reuse = f"v18-screen/v18-lr{_tag(b['lr'])}-lam{b['lam']:g}-r{r:g}-s0"
            out.append(
                {
                    "name": run_name(arm, b["lr"], b["lam"], r),
                    "arm": arm,
                    "lr": b["lr"],
                    "lam": b["lam"],
                    "mask": r,
                    "reuse": reuse,
                }
            )
    return out


def long_runs(setting: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Rule 2 of the long runs: F and M at seeds 0 and 1, P16 at seed 0 only
    when its setting is not that of V18's long runs."""
    out = []
    for arm, s in setting.items():
        if s is None:
            continue
        if arm == "p16" and (s["lr"], s["lam"], s["r"]) == V18_LONG:
            continue
        for seed in (0, 1) if arm != "p16" else (0,):
            name = (
                f"v18b-long-{arm}-lr{_tag(s['lr'])}-lam{s['lam']:g}-r{s['r']:g}-s{seed}"
            )
            out.append(
                {
                    "name": name,
                    "arm": arm,
                    "lr": s["lr"],
                    "lam": s["lam"],
                    "mask": s["r"],
                    "seed": seed,
                    "steps": LONG_STEPS,
                }
            )
    return out


def course(run: Run) -> list[dict[str, Any]]:
    """The training means and the zurich judge code at each checkpoint."""
    arm = run.arm
    out = []
    for step in sorted({s for c, s in run.stats if c == arm.judge}):
        t = run.train_at(step)
        z = run.probes.get((arm.judge, step, "zurich"))
        out.append(
            {
                "step": step,
                "train_pred": t.get("pred"),
                "train_sigreg": t.get("sigreg"),
                "train_rho_in": t.get("rho_in"),
                "recon_blur": None if z is None else z["recon_blur"],
                "sigreg_heldin": run.stat(arm.sigreg, step, "sigreg_heldin"),
                "pred_ratio_calib": run.stat(arm.judge, step, "pred_ratio_calib"),
                "token_r2": run.stat(arm.tokens, step, "token_r2"),
                "rho_in": run.stat(arm.tokens, step, "rho_in"),
            }
        )
    return out


def print_yardsticks(ys: dict[str, Code]) -> None:
    print("## yardsticks (zurich)\n")
    print("| code | recon_blur | sd | dyn_1/copy_1 | rank | token_r2 | sigreg_heldin |")
    print("| -- | -- | -- | -- | -- | -- | -- |")
    for name, c in ys.items():
        print(
            f"| `{name}` | {_fmt(c['recon_blur'], 4)} | {_fmt(c['recon_blur_sd'], 4)}"
            f" | {_fmt(c['dyn_1/copy_1'])} | {_fmt(c['rank'], 1)}"
            f" | {_fmt(c.stats.get('token_r2'))} | {_fmt(c['sigreg_heldin'], 2)} |"
        )
    print("\n| arm | usable pairs | rank floor | θ3 | judgeable |")
    print("| -- | -- | -- | -- | -- |")
    for arm in ARMS.values():
        pairs = usable_pairs(arm, ys)
        ln = lines(arm, ys)
        ok = all(any(m == pm for pm, _ in pairs) for m in METRICS)
        print(
            f"| {arm.name} | {pairs} | {ln['rank_floor']:.2f} | {ln['theta3']:.4f}"
            f" | {ok} |"
        )


def print_runs(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n## {title} (zurich, step 6000)\n")
    print(
        "| arm | r | lr | λ | K1 | K2 | K3 | recon_blur | sd | −init | dyn"
        " | rank | token_r2 | rho_in | S | pred_ratio_calib | train pred"
        " | train S | img h_mean 0→6000 | embed/block drift | pending | passed |"
    )
    print("| -- " * 22 + "|")
    order = {"f": 0, "m": 1, "p16": 2}
    for x in sorted(rows, key=lambda x: (order[x["arm"]], x["r"], x["lr"], x["lam"])):
        if "recon_blur" not in x:
            print(f"| {x['arm']} | {x['r']:g} | {_tag(x['lr'])} | {x['lam']:g} | —|")
            continue
        img = f"{_fmt(x['img_h_mean_0'])}→{_fmt(x['img_h_mean'])}"
        print(
            f"| {x['arm']} | {x['r']:g} | {_tag(x['lr'])} | {x['lam']:g}"
            f" | {_flag(x['K1'])} | {_flag(x['K2'])} | {_flag(x['K3'])}"
            f" | {_fmt(x['recon_blur'], 4)} | {_fmt(x['recon_blur_sd'], 4)}"
            f" | {x['recon_minus_init']:+.3f} | {_fmt(x['dyn_1/copy_1'])}"
            f" | {_fmt(x['rank'], 1)} | {_fmt(x['token_r2'])} | {_fmt(x['rho_in'])}"
            f" | {_fmt(x['sigreg_heldin'], 3)} | {_fmt(x['pred_ratio_calib'], 3)}"
            f" | {_fmt(x['train_pred'])} | {_fmt(x['train_sigreg'], 2)} | {img}"
            f" | {_fmt(x['embed_drift'])}/{_fmt(x['block_drift'])}"
            f" | {'pending' if x.get('pending') else '·'}"
            f" | {'pass' if x['passed'] else 'fail'} |"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()
    ys = yardsticks(a.runs)
    screen = a.runs / "v18b-screen"
    img = screen / "imagenette"
    runs = [Run(d, img) for d in sorted(screen.glob("v18b-*-s0"))]
    first = [judge(r, ys) for r in runs if r.r == FIRST_R]
    pending(first)
    print_yardsticks(ys)
    print_runs(first, "first stage")

    best: dict[str, dict[str, Any] | None] = {}
    any_pass: dict[str, bool] = {}
    for arm in ARMS:
        best[arm], any_pass[arm] = choose([x for x in first if x["arm"] == arm])
    print("\n## first-stage best per arm\n")
    for arm, b in best.items():
        tag = "—" if b is None else b["name"]
        print(f"- {arm}: {tag} (passed: {any_pass[arm]})")
    stage2 = second_stage(best)
    print(f"\nsecond stage: {json.dumps(stage2, indent=1)}")

    out: dict[str, Any] = {
        "first": first,
        "best": {k: None if v is None else v["name"] for k, v in best.items()},
        "first_passed": any_pass,
        "second_stage": stage2,
        "courses": {r.name: course(r) for r in runs},
    }
    names = {x["name"] for x in stage2}
    second = [judge(r, ys) for r in runs if r.name in names]
    if second:
        print_runs(second, "second stage")
    if len(second) == len(stage2):
        setting: dict[str, dict[str, Any] | None] = {}
        passed: dict[str, bool] = {}
        for arm, b in best.items():
            if b is None:
                setting[arm], passed[arm] = None, False
                continue
            three = [b] + [x for x in second if x["arm"] == arm]
            setting[arm], passed[arm] = choose(three)
        longs = long_runs(setting)
        print("\n## setting per arm (rule 1 of the long runs)\n")
        for arm, s in setting.items():
            tag = "—" if s is None else s["name"]
            print(f"- {arm}: {tag} (passed: {passed[arm]})")
        print(f"\nlong runs: {json.dumps(longs, indent=1)}")
        out |= {
            "second": second,
            "setting": {
                k: None if v is None else v["name"] for k, v in setting.items()
            },
            "setting_passed": passed,
            "long_runs": longs,
        }
    if a.json is not None:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
