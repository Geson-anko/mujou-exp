r"""How far the weights of each module move over a minute (V15).

The relative step of a frame (``grad_stats``' $\lVert\Delta\theta\rVert /
\lVert\theta\rVert$) does not tell how far the weights get: the steps of
RMSProp along the raw gradient mostly cancel from one frame to the next,
while those of ``Angular`` along a moving average add up. The drift over
$\Delta$ seconds is the net move, from the run's evaluation checkpoints,

$$D_\Delta = \operatorname{mean}_t \frac{\lVert\theta(t + \Delta) -
\theta(t)\rVert}{\lVert\theta(t)\rVert},$$

over the consecutive checkpoints $\Delta$ apart from ``start`` on, $\theta$
the module's floating-point tensors of 2 or more dimensions (``"matrices"``)
or all of them (``"all"``). A module with no matrix (the likelihood's
$\ln\sigma$) has ``"matrices"`` NaN.

``python -m exp.vision.drift <run_dir> ...`` prints a JSON line per run::

    {"run": <name>, "encoder": {"matrices": ..., "all": ...}, ...}
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from exp.lib import ckpt

__all__ = ["drift", "main", "relative_move"]

START = 300.0  # s: as grad_stats' window 300-1800
EVERY = 60.0  # s: measure.ckpt_every of V11-V15


def relative_move(
    a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor], keys: Sequence[str]
) -> float:
    r"""$\lVert b - a\rVert / \lVert a\rVert$ over the tensors ``keys``, in
    float64; NaN without keys."""
    if not keys:
        return math.nan
    num = sum(float((b[k].double() - a[k].double()).pow(2).sum()) for k in keys)
    den = sum(float(a[k].double().pow(2).sum()) for k in keys)
    return math.sqrt(num) / math.sqrt(den)


def drift(
    run_dir: Path, *, start: float = START, every: float = EVERY
) -> dict[str, dict[str, float]]:
    """``{module: {"matrices": D, "all": D}}`` of the run (module docstring).

    Raises:
        ValueError: Fewer than 2 checkpoints ``every`` seconds apart from
            ``start`` on.
    """
    times = sorted(
        t
        for p in (run_dir / "ckpt").glob("*.pt")
        if (t := float(p.stem)) >= start and math.isclose(t % every, 0.0, abs_tol=1e-6)
    )
    if len(times) < 2:
        raise ValueError(f"{run_dir}: fewer than 2 checkpoints from {start} s")
    trees = {
        t: torch.load(ckpt.eval_path(run_dir, t), weights_only=True)["modules"]
        for t in times
    }
    out: dict[str, dict[str, float]] = {}
    for name, first in trees[times[0]].items():
        floats = [k for k, v in first.items() if v.is_floating_point()]
        matrices = [k for k in floats if first[k].ndim >= 2]
        pairs = list(zip(times, times[1:]))
        out[name] = {
            kind: sum(
                relative_move(trees[a][name], trees[b][name], keys) for a, b in pairs
            )
            / len(pairs)
            for kind, keys in (("matrices", matrices), ("all", floats))
        }
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--start", type=float, default=START)
    ap.add_argument("--every", type=float, default=EVERY)
    args = ap.parse_args(argv)
    for run in args.runs:
        row = drift(run, start=args.start, every=args.every)
        print(json.dumps({"run": run.name, **row}), flush=True)


if __name__ == "__main__":
    main()
