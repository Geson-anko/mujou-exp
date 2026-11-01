"""Stage 3 (PLAN.md, section 8; M12): the world model on EgoDex, with the
encoder in three nodes and the dynamics in three.

    uv run python -m exp.video.stage3 run --variants m9 --out runs/stage3
    uv run python -m exp.video.stage3 summarize --out runs/stage3

The stream is the part 1 of EgoDex, 24 tasks one after another (``egodex.py``),
and the clips to evaluate on are of its test split. Each run steps by hand on a
GPU (one CPU core each): the 4090 takes two runs for each one of the 4060.
"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from exp.lib.record import REPO_ROOT
from exp.lib.sweep import run_all
from exp.video.config import Config, DataConfig, ModelConfig, OptimConfig
from exp.video.stability import KEYS, summary
from exp.video.stage0 import RUN_MODULE, SEEDS

FRAME = 1 / 30


def model(**changes: float | bool) -> ModelConfig:
    """The network of EgoDex: 3 + 3 nodes (latency.py), kappa 0.8 and the
    stop gradient of M9, with ``changes``."""
    base = ModelConfig(
        d_z=128, d_s=512, enc_nodes=3, dyn_nodes=3, kappa=0.8, stop_target=True
    )
    return replace(base, **changes)


FIX = model(z_norm=True, open_loop=0.5, open_cycle=20, buffer=1023, buffer_every=8)
# name: (model, learning rate, bp)
VARIANTS: dict[str, tuple[ModelConfig, float, bool]] = {
    "m9": (model(), 0.01, False),  # diverged within 2 s (README, M12)
    "norm": (model(z_norm=True), 0.01, False),
    # The setting of M11, with the buffer of SIGReg spread over minutes: the
    # frames of EgoDex change slowly (README, M12).
    "fix": (FIX, 0.03, False),
    "fix-bp": (FIX, 0.03, True),
    # And the prediction as a change from z_{k-1}: the frames change slowly.
    "res": (replace(FIX, residual=True), 0.03, False),
    "res-bp": (replace(FIX, residual=True), 0.03, True),
}
DEVICES = ("cuda:0", "cuda:0", "cuda:1")


def configs(variants: list[str], seconds: float) -> list[Config]:
    out = []
    for i, (name, seed) in enumerate((n, s) for n in variants for s in SEEDS):
        m, lr, bp = VARIANTS[name]
        out.append(
            Config(
                name=f"m12-{name}-s{seed}",
                seed=seed,
                bp=bp,
                device=DEVICES[i % len(DEVICES)],
                model=m,
                optim=OptimConfig(kind="sgd", tau=FRAME / lr),
                data=DataConfig(kind="egodex", size=64, seconds=seconds),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="The runs of stage 3.")
    parser.add_argument("what", choices=["run", "summarize"])
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "stage3")
    parser.add_argument("--jobs", type=int, default=9)
    args = parser.parse_args()
    if args.what == "run":
        run_all(RUN_MODULE, configs(args.variants, args.seconds), args.out, args.jobs)
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
