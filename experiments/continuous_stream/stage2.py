"""Stage 2 (PLAN.md, section 8): the optimizers of section 5 and the baselines
on 30 minutes of the synthetic video, and the judgement of H2-H4.

    uv run python -m experiments.continuous_stream.stage2 run --out runs/stage2
    uv run python -m experiments.continuous_stream.stage2 summarize --out runs/stage2

Every direction is grafted to the step size of SGD at the learning rate chosen
in stage 1; ``adamw`` (``pc.optim.AdamW`` as it is) is the practical
reference. The baselines learn by SGD.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import replace
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
LR = 0.1  # per frame, chosen in stage 1 with KAPPA (README: the moving stream)
KAPPA = 0.8
TAU = FRAME / LR
TAUS = (0.3, 3.0)  # tau_L, tau_R in seconds: about the correlation time of stage 1
REFRESH = 5.0  # seconds between the eigendecompositions (README: compute time)
SECONDS = 1800.0
PERIOD = 600.0  # T of the physics

BASELINE = "sgd"


def optimizers() -> dict[str, OptimConfig]:
    sgd = OptimConfig(kind="sgd", tau=TAU)
    graft = replace(sgd, graft=True, refresh=REFRESH)
    t1, t2 = TAUS
    return {
        "sgd": sgd,
        "adam-m0.3": replace(graft, kind="adam_direction", tau_m=0.3, tau_v=30.0),
        "adam-m1": replace(graft, kind="adam_direction", tau_m=1.0, tau_v=30.0),
        # lr = 1e-3 per frame, element-wise.
        "adamw": OptimConfig(kind="adamw", tau=FRAME / 1e-3, tau_m=0.3, tau_v=30.0),
        "orth": replace(graft, kind="orthogonalized"),
        f"white-l{t1:g}": replace(graft, kind="whitened", tau_l=t1),
        f"white-r{t1:g}": replace(graft, kind="whitened", tau_r=t1),
        f"white-lr{t1:g}": replace(graft, kind="whitened", tau_l=t1, tau_r=t1),
        f"white-lr{t2:g}": replace(graft, kind="whitened", tau_l=t2, tau_r=t2),
        f"proj4-lr{t1:g}": replace(graft, kind="projected", tau_l=t1, tau_r=t1, rank=4),
        f"proj16-lr{t1:g}": replace(
            graft, kind="projected", tau_l=t1, tau_r=t1, rank=16
        ),
    }


def configs() -> list[Config]:
    data = DataConfig(kind="video", seconds=SECONDS, physics_period=PERIOD)
    model = ModelConfig(kappa=KAPPA)
    out = []
    for seed in SEEDS:
        for name, optim in optimizers().items():
            out.append(
                Config(
                    name=f"v2-{name}-s{seed}",
                    seed=seed,
                    optim=optim,
                    model=model,
                    data=data,
                )
            )
        sgd = optimizers()["sgd"]
        out.append(
            Config(
                name=f"v2-bp-s{seed}",
                seed=seed,
                bp=True,
                optim=sgd,
                model=model,
                data=data,
            )
        )
        for order in ("shuffled", "replay"):
            out.append(
                Config(
                    name=f"v2-{order}-s{seed}",
                    seed=seed,
                    optim=sgd,
                    model=model,
                    data=replace(data, order=order),
                )
            )
    return out


def summary(path: Path) -> dict[str, float]:
    """Of one run, relative to the variance of z of the evaluations:

    - ``preq``: the prequential error of the last 10 minutes;
    - ``rollout<n>``: the rollout errors, averaged over the evaluations of the
      last 10 minutes (already relative);
    - ``retention``: the prequential error in the 10 s after the physics comes
      back to each of 8 phases in its third period, ``2T + i T / 8``;
    - ``rank``, ``probe_*``: of the last evaluation.
    """
    records = [json.loads(line) for line in path.read_text().splitlines()]
    live = [r for r in records if "frame" in r]
    end = live[-1]["time"]
    last = [r for r in live if r["time"] >= end - 600.0]
    evals = [r for r in last if "rank" in r]
    z_var = statistics.fmean(r["z_var"] for r in evals)
    out = {"z_var": z_var}
    out["preq"] = statistics.fmean(r["error"] for r in last if "error" in r) / z_var
    for key in ("rollout1", "rollout5", "rollout10", "rollout20"):
        out[key] = statistics.fmean(r[key] for r in evals)
    windows = [2 * PERIOD + i * PERIOD / 8 for i in range(8)]
    kept = [
        r["error"]
        for r in live
        if "error" in r and any(w <= r["time"] < w + 10.0 for w in windows)
    ]
    if kept:
        out["retention"] = statistics.fmean(kept) / z_var
    for key, value in evals[-1].items():
        if key == "rank" or key.startswith("probe"):
            out[key] = value
    return out


def table(out: Path) -> dict[str, dict[str, list[float]]]:
    """The summary of each setting, a list over ``SEEDS``: NaN for a seed whose
    run did not finish or lacks the metric."""
    t: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [math.nan] * len(SEEDS))
    )
    for done in sorted(out.glob("*/done")):
        name = done.parent.name.split("-", 3)[3].removeprefix("v2-")
        setting, seed = name.rsplit("-s", 1)
        for key, value in summary(done.parent / "metrics.jsonl").items():
            t[setting][key][SEEDS.index(int(seed))] = value
    return t


BASES = ("sgd", "adam-m0.3", "adam-m1", "adamw")


def collapsed(row: dict[str, list[float]], sgd: dict[str, list[float]], s: int) -> bool:
    """Whether the run of seed ``s`` collapsed or did not finish: ``preq`` is
    NaN, or the last effective rank of $z$ is below half of SGD's in the seed.
    A collapsed $z$ predicts itself trivially, so its errors are not
    comparable."""
    if math.isnan(row["preq"][s]):
        return True
    return "rank" in row and not row["rank"][s] >= sgd["rank"][s] / 2


def judge(t: dict[str, dict[str, list[float]]]) -> list[str]:
    """H2-H4 of PLAN.md, section 8, by the thresholds fixed before the runs.

    The baseline of each seed and metric is the best of ``BASES`` (SGD and the
    AdamW kinds) for that seed, among the runs that did not collapse
    (``collapsed``). A whitened or projected optimizer "improves" a metric by
    ``1 - value / baseline``; one that collapsed in a seed improves nothing.

    - H2: an optimizer improves both ``preq`` and ``rollout10`` by >= 10% in
      every seed.
    - H3: the orthogonalized step (Muon) improves each by less than half of the
      best of H2 (by the mean improvement), in every seed.
    - H4: that best one keeps (``retention``) better than the baseline of
      ``preq`` in every seed.
    - The precondition: SGD predicts one frame ahead better than the mean of
      $z$ (``rollout1`` below 1) in every seed. Without it, the lines of H2-H4
      are only for the record.
    """
    seeds = range(len(t["sgd"]["preq"]))
    bases = {
        s: [b for b in BASES if b in t and not collapsed(t[b], t["sgd"], s)]
        for s in seeds
    }

    def base(key: str, s: int) -> float:
        return min(t[b][key][s] for b in bases[s])

    def gain(name: str, key: str, s: int) -> float:
        if collapsed(t[name], t["sgd"], s):
            return -math.inf
        return 1 - t[name][key][s] / base(key, s)

    keys = ("preq", "rollout10")
    candidates = [n for n in t if n.startswith(("white", "proj"))]
    lines = []
    passed = [
        n for n in candidates if all(gain(n, k, s) >= 0.1 for k in keys for s in seeds)
    ]
    for n in sorted(candidates):
        gains = ", ".join(
            f"{k} {'/'.join(f'{gain(n, k, s):+.1%}' for s in seeds)}" for k in keys
        )
        lines.append(f"  {n}: {gains}")
    best = max(
        candidates,
        key=lambda n: sum(gain(n, k, s) for k in keys for s in seeds),
    )
    lines.insert(0, f"H2: {'supported' if passed else 'rejected'} {passed}")
    if "orth" in t:
        h3 = all(
            gain("orth", k, s) < 0.5 * gain(best, k, s) for k in keys for s in seeds
        )
        orth = ", ".join(
            f"{k} {'/'.join(f'{gain("orth", k, s):+.1%}' for s in seeds)}" for k in keys
        )
        lines.append(
            f"H3 (against {best}): {'supported' if h3 else 'rejected'}; orth: {orth}"
        )
    if all("retention" in t[n] for n in (best, *BASES) if n in t):
        b = [min(t[x]["retention"][s] for x in bases[s]) for s in seeds]
        h4 = all(t[best]["retention"][s] < b[s] for s in seeds)
        values = "/".join(f"{t[best]['retention'][s]:.3g} vs {b[s]:.3g}" for s in seeds)
        lines.append(f"H4 ({best}): {'supported' if h4 else 'rejected'}; {values}")
    if "rollout1" in t["sgd"]:
        r = t["sgd"]["rollout1"]
        met = all(r[s] < 1 for s in seeds)
        values = "/".join(f"{r[s]:.3g}" for s in seeds)
        lines.append(
            f"Precondition (sgd rollout1 < 1): {'met' if met else 'not met'}; {values}"
        )
    excluded = [
        f"{b}-s{SEEDS[s]}" for s in seeds for b in BASES if b in t and b not in bases[s]
    ]
    lines.append(f"Collapsed baselines, left out: {excluded}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="The runs of stage 2.")
    parser.add_argument("what", choices=["run", "summarize"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "stage2")
    parser.add_argument("--jobs", type=int, default=14)
    args = parser.parse_args()
    if args.what == "run":
        run_all(args.out, args.jobs, configs())
        return
    for setting, values in sorted(table(args.out).items()):
        cells = "  ".join(
            f"{k}={'/'.join(f'{v:.3g}' for v in vs)}"
            for k, vs in sorted(values.items())
        )
        print(f"{setting:16s} {cells}")
    for line in judge(table(args.out)):
        print(line)


if __name__ == "__main__":
    main()
