"""The speed of the VAE of PC-ALM at B = 1 for each arch and size: frames
(units) a second of wall time (real time is ``1 / time.frame`` = 10) and the
peak CUDA memory, on synthetic stacks (``RandomStacks``, so that no data is
needed)::

    uv run python -m exp.vision.bench --device cuda [--frames 100] [override ...]

Each row is the program's ``Online`` of ``+experiment=v0/base model=<arch>_<size>``
with the overrides: ``--warmup`` frames (the CUDA graphs are recorded in the
first), then ``--frames`` frames timed. The rows are printed as JSON lines,
and written to ``--out`` as a JSON list.
"""

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

import torch
from torch import Tensor

from exp.lib.tours.stacks import StackStreams
from exp.vision.compose import compose
from exp.vision.models import param_table
from exp.vision.program import Online
from exp.vision.schema.model import ARCHS, BLOCKS
from exp.vision.schema.run import RunConf

__all__ = ["RandomStacks", "bench", "main"]


class RandomStacks(StackStreams):
    """``data.streams`` streams of uniform u8 stacks of ``conf``'s shape,
    those of the ``k``-th ``next()`` drawn from the seed ``seed + k``; every
    stack is ``ready``, ``reset`` only at the first."""

    def __init__(self, conf: RunConf, *, seed: int = 0) -> None:
        t = conf.data.tours
        self._shape = (conf.data.streams, t.frames, 3, t.size, t.size)
        self._seed = seed
        self._k = 0

    @override
    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        g = torch.Generator().manual_seed(self._seed + self._k)
        x = torch.randint(0, 256, self._shape, generator=g, dtype=torch.uint8)
        b = self._shape[0]
        reset = torch.full((b,), self._k == 0)
        self._k += 1
        return x, torch.ones(b, dtype=torch.bool), reset

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"k": self._k}

    @override
    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._k = d["k"]


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def bench(conf: RunConf, *, frames: int, warmup: int) -> dict[str, Any]:
    """Time ``frames`` frames of ``conf``'s ``Online`` after ``warmup``:
    ``{"arch", "size", "units_s", "gpu_mem_mb"}`` (the peak allocated over
    all the frames; 0 on the CPU)."""
    dev = conf.device
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(dev))
    online = Online(conf, None, RandomStacks(conf))
    try:
        for _ in range(warmup):
            online.advance()
        _sync(dev)
        start = time.perf_counter()
        for _ in range(frames):
            online.advance()
        _sync(dev)
        wall = time.perf_counter() - start
    finally:
        online.close()
    gpu = 0.0
    if dev.startswith("cuda"):
        gpu = torch.cuda.max_memory_allocated(torch.device(dev)) / 2**20
    return {
        "arch": conf.model.arch,
        "size": conf.model.size,
        "units_s": frames / wall,
        "gpu_mem_mb": round(gpu, 1),
    }


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.bench",
        description="Frames a second and peak GPU memory of every arch and size.",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--experiment", default="v0/base")
    ap.add_argument("--out", type=Path)
    ap.add_argument("overrides", nargs="*", help="hydra overrides of every row")
    args = ap.parse_args(argv)
    rows = []
    for arch in ARCHS:
        for size in BLOCKS:
            conf = compose(
                [
                    f"+experiment={args.experiment}",
                    f"model={arch}_{size.lower()}",
                    f"device={args.device}",
                    "data.streams=1",
                    *args.overrides,
                ]
            )
            params = {(r["arch"], r["size"]): r["encoder"] for r in param_table(conf)}
            row = bench(conf, frames=args.frames, warmup=args.warmup)
            row["encoder_params"] = params[arch, size]
            print(json.dumps(row), flush=True)
            rows.append(row)
    if args.out is not None:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    main()
