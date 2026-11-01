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
import statistics
from dataclasses import replace
from pathlib import Path

from exp.lib.record import REPO_ROOT
from exp.lib.sweep import run_all
from exp.video.config import Config, DataConfig, ModelConfig, OptimConfig
from exp.video.egodex import EgoDexStream
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
    tag = "" if seconds == 1800.0 else f"-{seconds:g}s"
    for i, (name, seed) in enumerate((n, s) for n in variants for s in SEEDS):
        m, lr, bp = VARIANTS[name]
        out.append(
            Config(
                name=f"m12-{name}{tag}-s{seed}",
                seed=seed,
                bp=bp,
                device=DEVICES[i % len(DEVICES)],
                model=m,
                optim=OptimConfig(kind="sgd", tau=FRAME / lr),
                data=DataConfig(kind="egodex", size=64, seconds=seconds),
            )
        )
    return out


FIRST = 10.0  # seconds


def retention(path: Path, data: str) -> dict[str, float]:
    """Of a run over both passes of the stream: how much the prequential error
    jumps when the stream switches to a task (the mean error of the first
    ``FIRST`` seconds of its block over that of the last ``FIRST`` seconds of
    the block before), in the second pass over in the first, the mean over the
    tasks. Below 1: coming back to a task costs less than meeting it first.
    The ratios within a pass leave out how the scale of z drifts in time."""
    stream = EgoDexStream(data)
    blocks: list[tuple[str, int, list[float]]] = []  # in the order of the stream
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "error" not in r:
            continue
        key = (stream.task_of(r["frame"]), stream.pass_of(r["frame"]))
        if not blocks or blocks[-1][:2] != key:
            blocks.append((*key, []))
        blocks[-1][2].append(r["error"])
    n = round(FIRST / FRAME)
    jumps: dict[tuple[str, int], float] = {}
    for (_, _, before), (task, p, errors) in zip(blocks, blocks[1:]):
        if len(before) >= n and len(errors) >= n:
            head, tail = statistics.fmean(errors[:n]), statistics.fmean(before[-n:])
            jumps[task, p] = head / tail
    tasks = [t for t, p in jumps if p == 1 and (t, 0) in jumps]
    if not tasks:
        return {}
    return {
        "tasks": len(tasks),
        "jump_first": statistics.fmean(jumps[t, 0] for t in tasks),
        "jump_again": statistics.fmean(jumps[t, 1] for t in tasks),
        "again_over_first": statistics.fmean(jumps[t, 1] / jumps[t, 0] for t in tasks),
    }


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
        if s["time"] > 2600:  # into the second pass
            recorded = json.loads((path.parent / "config.json").read_text())
            config = Config.from_dict(recorded["config"])
            ret = retention(path, config.data.path)
            row.update({f"retention_{k}": round(v, 4) for k, v in ret.items()})
        print(name, "" if done else "NOT-DONE", json.dumps(row))


if __name__ == "__main__":
    main()
