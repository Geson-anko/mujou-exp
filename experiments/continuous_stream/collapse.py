"""M9: the collapse of the encoder under SGD, and what keeps it from collapsing.

    uv run python -m experiments.continuous_stream.collapse run --out runs/collapse
    uv run python -m experiments.continuous_stream.collapse summarize --out runs/collapse

The search is 10 minutes of the moving stream with seed 0, over the weight of
SIGReg, the learning rate and whether the prediction error sends credit to its
target (``stop_target``).
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
from experiments.continuous_stream.stage0 import run_all
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


KEYS = ("z_var", "rank", "rel_error", "rollout1", "rollout10", "probe_z_pos")


def main() -> None:
    parser = argparse.ArgumentParser(description="The runs of M9.")
    parser.add_argument("what", choices=["run", "summarize"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "collapse")
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    if args.what == "run":
        run_all(args.out, args.jobs, configs())
        return
    for done in sorted(args.out.glob("*/done")):
        name = done.parent.name.split("-", 3)[3]
        s = summary(done.parent / "metrics.jsonl")
        print(name, json.dumps({k: round(s[k], 4) for k in KEYS if k in s}))


if __name__ == "__main__":
    main()
