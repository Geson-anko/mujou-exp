"""M13: the easier synthetic video, and the model that predicts it.

    uv run python -m exp.video.easy configs --out runs/m13
    uv run python -m exp.video.easy run [--big] --out runs/m13
    uv run python -m exp.video.easy summarize --out runs/m13

The stream never draws new balls (``redraw``), its balls bounce with $e = 1$
under a constant gravity without drag, and the bounces keep the energy
(``conserve``): the frames change continuously and the balls never rest. B
streams start from their own balls. ``configs`` writes the settings of the
offline bound (``offline.py --config``) for the small and the big model;
``run`` runs the online check (the setting of M11, 30 minutes, PC-ALM and BP,
seeds 0-2, with checkpoints to ``decode.py``); ``summarize`` prints, of each
run, the last line of its decoding as ratios to the bars (README: M13).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from exp.lib.record import REPO_ROOT
from exp.lib.sweep import run_all
from exp.video.config import Config, DataConfig, ModelConfig, OptimConfig
from exp.video.stage0 import RUN_MODULE, SEEDS

FRAME = 1 / 30
STREAMS = 128
EASY = DataConfig(
    kind="video",
    seconds=1800.0,
    gamma_amp=0.0,
    e0=1.0,
    e_amp=0.0,
    drag=0.0,
    conserve=True,
    redraw=False,
    streams=STREAMS,
)
# The online setting of M11: w = 1, z on the sphere, open loop half the time,
# kappa 0.13 (stronger by B: README, M11), a buffer of 1023.
SMALL = ModelConfig(kappa=0.13, z_norm=True, open_loop=0.5, open_cycle=20, buffer=1023)
# Twice the channels of the encoder, twice d_s, and three nodes of dynamics.
BIG = replace(SMALL, width=64, d_s=512, dyn_nodes=3)
LR, WARMUP, CHECKPOINT = 0.03, 300.0, 300.0


def offline_configs() -> list[Config]:
    """The model and the stream of the offline bound, small and big."""
    return [
        Config(name=f"offline-{tag}", model=m, data=EASY)
        for tag, m in (("small", SMALL), ("big", BIG))
    ]


def online_configs(model: ModelConfig, tag: str) -> list[Config]:
    """The check of ``model``: PC-ALM and BP over the seeds, on both GPUs."""
    out = []
    for seed in SEEDS:
        for bp in (False, True):
            out.append(
                Config(
                    name=f"m13-{tag}-{'bp' if bp else 'pc'}-s{seed}",
                    seed=seed,
                    bp=bp,
                    checkpoint_interval=CHECKPOINT,
                    device="cuda:1" if (seed + bp) % 2 else "cuda:0",
                    model=model,
                    optim=OptimConfig(kind="sgd", tau=FRAME / LR, warmup=WARMUP),
                    data=EASY,
                )
            )
    return out


HORIZONS = (5, 10, 20)


def summary(run: Path) -> dict[str, float]:
    """Of the last checkpoint decoded: the reconstruction, and at each horizon
    the decoded prediction over the mean frame and over the last frame, and
    the latent rollout over the shrunk last z."""
    lines = (run / "decode" / "metrics.jsonl").read_text().splitlines()
    last = json.loads(lines[-1])
    out = {"time": last["time"], "recon": last["recon"], "rank": last["rank"]}
    for n in HORIZONS:
        out[f"img/mean{n}"] = last[f"img_rollout{n}"] / last[f"pix_mean{n}"]
        out[f"img/copy{n}"] = last[f"img_rollout{n}"] / last[f"pix_copy{n}"]
        out[f"z/shrink{n}"] = last[f"rollout{n}"] / last[f"shrink{n}"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="The runs of M13.")
    parser.add_argument("what", choices=["configs", "run", "summarize"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "m13")
    parser.add_argument("--big", action="store_true", help="the big model")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.what == "configs":
        args.out.mkdir(parents=True, exist_ok=True)
        for c in offline_configs():
            (args.out / f"{c.name}.json").write_text(c.to_json())
    elif args.what == "run":
        todo = (
            online_configs(BIG, "big") if args.big else online_configs(SMALL, "small")
        )
        run_all(RUN_MODULE, todo, args.out, args.jobs)
    else:
        for path in sorted(args.out.glob("*/decode/metrics.jsonl")):
            if not path.stat().st_size:  # being decoded
                continue
            run = path.parent.parent
            row = {k: round(v, 4) for k, v in summary(run).items()}
            print(run.name, json.dumps(row))


if __name__ == "__main__":
    main()
