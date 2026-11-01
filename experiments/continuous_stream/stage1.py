"""Stage 1 (PLAN.md, section 8): the world model on 10 minutes of the synthetic
video, by SGD at three learning rates, and the correlation of its gradients.

    uv run python -m experiments.continuous_stream.stage1 run --out runs/stage1
    uv run python -m experiments.continuous_stream.stage1 summarize --out runs/stage1
"""

import argparse
import json
import statistics
from pathlib import Path

from experiments.continuous_stream.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
)
from experiments.continuous_stream.record import REPO_ROOT
from experiments.continuous_stream.stage0 import SEEDS, run_all

FRAME = 1 / 30
RATES = (0.01, 0.03, 0.1)  # per frame
KAPPAS = (0.05, 0.5, 0.8)  # 0.05: LeJEPA's; the others after z collapsed with it


def configs() -> list[Config]:
    return [
        Config(
            name=f"video-k{kappa:g}-lr{lr:g}-s{seed}",
            seed=seed,
            model=ModelConfig(kappa=kappa),
            optim=OptimConfig(kind="sgd", tau=FRAME / lr),
            data=DataConfig(kind="video", seconds=600.0),
        )
        for seed in SEEDS
        for kappa in KAPPAS
        for lr in RATES
    ]


def summary(path: Path) -> dict[str, float]:
    """Of one run: the prequential error of the last 5 minutes relative to the
    variance of z there, and the last evaluation."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    late = [r for r in records if r["time"] >= 300.0]
    evals = [r for r in late if "rank" in r]
    error = statistics.fmean(r["error"] for r in late if "error" in r)
    z_var = statistics.fmean(r["z_var"] for r in evals)
    out = {"error": error, "z_var": z_var, "rel_error": error / z_var}
    for key, value in evals[-1].items():
        if key.startswith(("rank", "rollout", "probe")):
            out[key] = value
    for key in {k for r in evals for k in r if k.startswith("rho_")}:
        out[key] = statistics.fmean(r[key] for r in evals if key in r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="The runs of stage 1.")
    parser.add_argument("what", choices=["run", "summarize"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "stage1")
    parser.add_argument("--jobs", type=int, default=14)
    args = parser.parse_args()
    if args.what == "run":
        run_all(args.out, args.jobs, configs())
        return
    for done in sorted(args.out.glob("*/done")):
        name = done.parent.name.split("-", 3)[3]
        s = summary(done.parent / "metrics.jsonl")
        print(name, json.dumps({k: round(v, 4) for k, v in sorted(s.items())}))


if __name__ == "__main__":
    main()
