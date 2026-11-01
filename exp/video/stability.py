"""M11: the push of SIGReg on |z| and the large rollout10, before stage 3.

    uv run python -m exp.video.stability run --out runs/stability
    uv run python -m exp.video.stability summarize --out runs/stability

The search is 10 minutes of the moving stream with seeds 0-2, from the setting
of M9 (``stop_target``, kappa 0.8, SGD): z on the sphere (``z_norm``) at three
learning rates, and the credit the prediction error sends to its target
(``target_weight``); with ``--open``, the dynamics learning from its own
predictions (``open_loop``). The check (``--confirm``) runs the chosen setting for 30
minutes by PC-ALM and by BP.
"""

import argparse
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path

from exp.lib.record import REPO_ROOT
from exp.lib.sweep import run_all
from exp.video.config import Config, DataConfig, ModelConfig, OptimConfig
from exp.video.stage0 import RUN_MODULE, SEEDS

FRAME = 1 / 30
BASE = ModelConfig(kappa=0.8, stop_target=True)  # M9
WEIGHTS = (0.03, 0.1, 0.3, 1.0)  # 1.0: no stop gradient at all


def _config(name: str, seed: int, model: ModelConfig, lr: float, **kw) -> Config:
    return Config(
        name=f"{name}-lr{lr:g}-s{seed}",
        seed=seed,
        model=model,
        optim=OptimConfig(kind="sgd", tau=FRAME / lr),
        data=DataConfig(kind="video", seconds=kw.pop("seconds", 600.0)),
        **kw,
    )


def configs() -> list[Config]:
    out = []
    for seed in SEEDS:
        for norm in (False, True):
            for lr in (0.01, 0.03, 0.1):
                model = ModelConfig(kappa=0.8, stop_target=True, z_norm=norm)
                out.append(_config(f"m11-{'norm' if norm else 'raw'}", seed, model, lr))
        for w in WEIGHTS:
            model = ModelConfig(
                kappa=0.8, stop_target=w < 1, target_weight=w, z_norm=True
            )
            for lr in (0.01, 0.03):
                out.append(_config(f"m11-norm-w{w:g}", seed, model, lr))
        for lr in (0.01, 0.03):
            model = ModelConfig(kappa=0.8, stop_target=True, z_norm=True)
            out.append(_config("m11-norm-bp", seed, model, lr, bp=True))
    return out


# Added after seed 0 of configs(): the dynamics also learns from its own
# predictions (open_loop of every open_cycle predictions), as a rollout runs.
OPEN = ((0.5, 20), (0.5, 40), (0.25, 40))


def open_configs() -> list[Config]:
    out = []
    for seed in SEEDS:
        for f, cycle in OPEN:
            model = ModelConfig(
                kappa=0.8, stop_target=True, z_norm=True, open_loop=f, open_cycle=cycle
            )
            out.append(_config(f"m11-open{f:g}-c{cycle}", seed, model, 0.03))
        model = ModelConfig(
            kappa=0.8, stop_target=True, z_norm=True, open_loop=0.5, open_cycle=20
        )
        out.append(_config("m11-open0.5-c20", seed, model, 0.01))
        out.append(_config("m11-open0.5-c20-bp", seed, model, 0.03, bp=True))
    return out


# Added after the check by BP (README: M11): the encoder also learns to make z
# predictable (target_weight), with a heavier SIGReg against the collapse;
# screened by BP, which steps as PC-ALM does (stage 0) and runs faster.
PREDICTIVE = ((0.3, 0.8), (1.0, 0.8), (0.3, 0.9), (1.0, 0.95))  # (w, kappa)


def predictive_configs() -> list[Config]:
    out = []
    for seed in SEEDS[:2]:
        for w, kappa in PREDICTIVE:
            model = ModelConfig(
                kappa=kappa,
                stop_target=w < 1,
                target_weight=w,
                z_norm=True,
                open_loop=0.5,
                open_cycle=20,
            )
            name = f"m11-pred-w{w:g}-k{kappa:g}-bp"
            out.append(_config(name, seed, model, 0.03, bp=True))
    return out


# Added after the offline bound (README: M11): with the gradient stopped where
# the nodes stop it, the full credit to the target and a SIGReg about 15 times
# the prediction error (kappa 0.94) reach rollout10 < 1 at rank 40; online,
# over 30 minutes and at two learning rates.
HEAVY = (0.92, 0.94, 0.96)  # kappa, with w = 1


def heavy_configs() -> list[Config]:
    out = []
    for seed in SEEDS[:2]:
        for kappa in HEAVY:
            model = ModelConfig(
                kappa=kappa,
                stop_target=False,
                z_norm=True,
                open_loop=0.5,
                open_cycle=20,
            )
            for lr in (0.03, 0.1):
                name = f"m11-heavy-k{kappa:g}-bp"
                out.append(_config(name, seed, model, lr, bp=True, seconds=1800.0))
    model = replace(CHOSEN, kappa=0.94, stop_target=False)
    for lr in (0.03, 0.1):
        out.append(_config("m11-heavy-k0.94-pc", 0, model, lr, seconds=1800.0))
    return out


# Chosen from the search (README: M11): z on the sphere, the dynamics learning
# from its own predictions half of the time, at the learning rate 0.03.
CHOSEN = ModelConfig(
    kappa=0.8, stop_target=True, z_norm=True, open_loop=0.5, open_cycle=20
)


# Added after the offline bound with a frozen encoder (README: M11): z of
# blurred frames varies smoothly with the positions of the balls, and the
# dynamics alone predicts it beyond the constant at 10 frames.
BLURS = (1.5, 2.0, 3.0)  # sigma, pixels


def blur_configs() -> list[Config]:
    out = []
    for blur in BLURS:
        model = replace(CHOSEN, blur=blur)
        for seed in SEEDS[:2]:
            out.append(
                _config(
                    f"m11-blur{blur:g}-bp", seed, model, 0.03, bp=True, seconds=1800.0
                )
            )
        out.append(_config(f"m11-blur{blur:g}-pc", 0, model, 0.03, seconds=1800.0))
    return out


# Added after the 30 minutes of the heavy SIGReg (README: M11): whether a
# longer stream takes rollout10 below 1, by BP, which runs in real time.
def long_configs() -> list[Config]:
    heavy = ModelConfig(
        kappa=0.96, stop_target=False, z_norm=True, open_loop=0.5, open_cycle=20
    )
    out = [
        _config("m11-long-heavy-k0.96-bp", seed, heavy, 0.03, bp=True, seconds=10800.0)
        for seed in SEEDS[:2]
    ]
    out.append(_config("m11-long-chosen-bp", 0, CHOSEN, 0.03, bp=True, seconds=10800.0))
    return out


# Added with the long runs: AdamW, which collapsed or diverged in M10 when
# SIGReg pushed |z| out, now that z_norm stops that push; per element
# learning rates for the frame, as M10's reference.
def adam_configs() -> list[Config]:
    heavy = ModelConfig(
        kappa=0.96, stop_target=False, z_norm=True, open_loop=0.5, open_cycle=20
    )
    out = []
    for name, model in (("heavy-k0.96", heavy), ("chosen", CHOSEN)):
        for lr in (1e-3, 3e-4):
            optim = OptimConfig(kind="adamw", tau=FRAME / lr, tau_m=0.3, tau_v=30.0)
            for seed in SEEDS[:2]:
                c = _config(
                    f"m11-adam-{name}-bp", seed, model, lr, bp=True, seconds=1800.0
                )
                out.append(replace(c, optim=optim))
    return out


def confirm() -> list[Config]:
    return [
        _config(
            f"m11-confirm-{'bp' if bp else 'pc'}",
            seed,
            CHOSEN,
            0.03,
            bp=bp,
            seconds=1800.0,
        )
        for seed in SEEDS
        for bp in (False, True)
    ]


def summary(path: Path) -> dict[str, float]:
    """Of one run: the last evaluation, the prequential error of the last
    quarter relative to the variance of z, and the largest RMS of z."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    end = max(r["time"] for r in records)
    evals = [r for r in records if "rank" in r]
    late = [r for r in records if r["time"] >= 0.75 * end and "error" in r]
    out = {"time": end, "z_rms_max": max(r.get("z_rms", 0.0) for r in records)}
    if evals:
        last = evals[-1]
        for key, value in last.items():
            if key.startswith(("rank", "rollout", "persist", "mean", "z_var")):
                out[key] = value
        if late:
            out["preq"] = statistics.fmean(r["error"] for r in late) / last["z_var"]
    return out


KEYS = (
    "time",
    "z_rms_max",
    "z_var",
    "rank",
    "preq",
    "rollout1",
    "rollout10",
    "persist1",
    "persist10",
    "mean10",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="The runs of M11.")
    parser.add_argument("what", choices=["run", "summarize"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "stability")
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--open", action="store_true", help="the open-loop runs")
    parser.add_argument("--confirm", action="store_true", help="the 30-minute check")
    parser.add_argument("--predictive", action="store_true", help="w with open loop")
    parser.add_argument("--heavy", action="store_true", help="w = 1, heavy SIGReg")
    parser.add_argument("--blur", action="store_true", help="blurred frames")
    parser.add_argument("--long", action="store_true", help="3 hours by BP")
    parser.add_argument("--adam", action="store_true", help="AdamW by BP")
    args = parser.parse_args()
    if args.what == "run":
        if args.confirm:
            todo = confirm()
        elif args.predictive:
            todo = predictive_configs()
        elif args.heavy:
            todo = heavy_configs()
        elif args.blur:
            todo = blur_configs()
        elif args.long:
            todo = long_configs()
        elif args.adam:
            todo = adam_configs()
        else:
            todo = open_configs() if args.open else configs()
        run_all(RUN_MODULE, todo, args.out, args.jobs)
        return
    for path in sorted(args.out.glob("*/metrics.jsonl")):
        if not path.stat().st_size:  # just started
            continue
        name = path.parent.name.split("-", 3)[3]
        s = summary(path)
        done = (path.parent / "done").exists()
        row = {k: round(s[k], 4) for k in KEYS if k in s and math.isfinite(s[k])}
        print(name, "" if done else "NOT-DONE", json.dumps(row))


if __name__ == "__main__":
    main()
