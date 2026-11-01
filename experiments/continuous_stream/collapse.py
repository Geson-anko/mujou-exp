"""M9: the collapse of the encoder under SGD, and what keeps it from collapsing.

    uv run python -m experiments.continuous_stream.collapse run --out runs/collapse
    uv run python -m experiments.continuous_stream.collapse summarize --out runs/collapse

The search is 10 minutes of the moving stream with seed 0, over the weight of
SIGReg, the learning rate and whether the prediction error sends credit to its
target (``stop_target``). The check (``--confirm``) runs the setting chosen
from it for 30 minutes with seeds 0-2, by PC-ALM and by BP, at the learning
rates of ``CONFIRM_RATES``.
"""

import argparse
import json
from pathlib import Path

from experiments.continuous_stream.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
)
from experiments.continuous_stream.record import REPO_ROOT
from experiments.continuous_stream.stage0 import SEEDS, run_all
from experiments.continuous_stream.stage1 import summary

FRAME = 1 / 30
KAPPAS = (0.05, 0.2, 0.5, 0.8)
RATES = (0.03, 0.1)  # per frame


def configs() -> list[Config]:
    return [
        Config(
            name=f"m9-{'stop' if stop else 'full'}-k{kappa:g}-lr{lr:g}-s0",
            seed=0,
            model=ModelConfig(kappa=kappa, stop_target=stop),
            optim=OptimConfig(kind="sgd", tau=FRAME / lr),
            data=DataConfig(kind="video", seconds=600.0),
        )
        for stop in (False, True)
        for kappa in KAPPAS
        for lr in RATES
    ]


# Chosen from the search (README: M9). 0.1 diverged in two of three seeds.
CHOSEN = ModelConfig(kappa=0.8, stop_target=True)
CONFIRM_RATES = (0.1, 0.03, 0.01)


def confirm() -> list[Config]:
    return [
        Config(
            name=f"m9-confirm-{'bp' if bp else 'pc'}{lr_tag(lr)}-s{seed}",
            seed=seed,
            bp=bp,
            model=CHOSEN,
            optim=OptimConfig(kind="sgd", tau=FRAME / lr),
            data=DataConfig(kind="video", seconds=1800.0),
        )
        for lr in CONFIRM_RATES
        for seed in SEEDS
        for bp in (False, True)
    ]


def lr_tag(lr: float) -> str:
    """The first check (0.1) has no tag in its name."""
    return "" if lr == 0.1 else f"-lr{lr:g}"


KEYS = ("z_var", "rank", "rel_error", "rollout1", "rollout10", "probe_z_pos")


def main() -> None:
    parser = argparse.ArgumentParser(description="The runs of M9.")
    parser.add_argument("what", choices=["run", "summarize"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "collapse")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--confirm", action="store_true", help="the 30-minute check")
    args = parser.parse_args()
    if args.what == "run":
        run_all(args.out, args.jobs, confirm() if args.confirm else configs())
        return
    for done in sorted(args.out.glob("*/done")):
        name = done.parent.name.split("-", 3)[3]
        s = summary(done.parent / "metrics.jsonl")
        print(name, json.dumps({k: round(s[k], 4) for k in KEYS if k in s}))


if __name__ == "__main__":
    main()
