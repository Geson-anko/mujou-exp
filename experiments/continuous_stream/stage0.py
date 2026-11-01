"""Stage 0 (PLAN.md, section 8): the sweep on the linear regression stream, and
its summary.

    uv run python -m experiments.continuous_stream.stage0 run --out runs/stage0
    uv run python -m experiments.continuous_stream.stage0 summarize --out runs/stage0

Each run is one process (``run.main``), ``--jobs`` at a time. The summary takes
the second half of each run (after the transient): the time mean of
``||w - w_*||`` for the linear model, and of ``iid_loss`` and the prequential
``loss`` for both.
"""

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from experiments.continuous_stream.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    TimeConfig,
)
from experiments.continuous_stream.record import REPO_ROOT

SEEDS = (0, 1, 2)
SECONDS = 600.0
TAU = 10.0  # SGD: a learning rate of 1/300 per frame
TAU_MS = (0.1, 0.3, 1.0, 3.0)
TAU_XS = (1.0, 10.0, 100.0)


def configs() -> list[Config]:
    """Every run of stage 0."""
    out = []
    optims = {"sgd": OptimConfig(kind="sgd", tau=TAU)}
    for tau_m in TAU_MS:
        for graft in (True, False):
            name = f"adam-m{tau_m:g}-{'graft' if graft else 'raw'}"
            optims[name] = OptimConfig(
                kind="adam_direction", tau=TAU, tau_m=tau_m, tau_v=30.0, graft=graft
            )
    streams = {
        f"x{tau_x:g}": DataConfig(kind="linear", tau_x=tau_x) for tau_x in TAU_XS
    }
    streams["iid"] = DataConfig(kind="linear", iid=True)
    for seed in SEEDS:
        for sname, data in streams.items():
            for oname, optim in optims.items():
                out.append(
                    Config(
                        name=f"lin-{sname}-{oname}-s{seed}",
                        seed=seed,
                        eval_interval=1.0,
                        optim=optim,
                        data=replace(data, seconds=SECONDS),
                    )
                )
        # The MLP: when to learn, and PC-ALM against BP.
        for interval in (25e-3, None):
            for bp in (False, True):
                tag = "li25" if interval else "linone"
                out.append(
                    Config(
                        name=f"mlp-x10-{tag}-{'bp' if bp else 'pc'}-s{seed}",
                        seed=seed,
                        bp=bp,
                        eval_interval=1.0,
                        time=TimeConfig(learn_interval=interval),
                        optim=OptimConfig(kind="sgd", tau=TAU),
                        model=ModelConfig(hidden=32),
                        data=DataConfig(kind="linear", tau_x=10.0, seconds=SECONDS),
                    )
                )
    return out


def run_all(out: Path, jobs: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in configs() if not list(out.glob(f"*-{c.name}/done"))]
    print(f"{len(todo)} runs to go", flush=True)

    def one(c: Config) -> None:
        path = out / f"{c.name}.json"
        path.write_text(c.to_json())
        cmd = [sys.executable, "-m", "experiments.continuous_stream.run"]
        cmd += ["--config", str(path), "--out", str(out)]
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAILED {c.name}\n{result.stderr[-2000:]}", flush=True)
            return
        (Path(result.stdout.strip().splitlines()[-1]) / "done").touch()
        print(f"done {c.name}", flush=True)

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(one, todo))


def _half_means(path: Path) -> dict[str, float]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    half = records[len(records) // 2 :]
    out = {}
    for key in ("dist", "iid_loss", "loss"):
        values = [r[key] for r in half if key in r]
        if values:
            out[key] = statistics.fmean(values)
    return out


def summarize(out: Path) -> dict[str, dict[str, list[float]]]:
    """The second-half means of each setting, a list over the seeds."""
    table: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for done in sorted(out.glob("*/done")):
        name = done.parent.name.split("-", 3)[3]  # drop the date
        setting = name.rsplit("-s", 1)[0]
        for key, value in _half_means(done.parent / "metrics.jsonl").items():
            table[setting][key].append(value)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="The sweep of stage 0.")
    parser.add_argument("what", choices=["run", "summarize"])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "stage0")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.what == "run":
        run_all(args.out, args.jobs)
    else:
        for setting, values in sorted(summarize(args.out).items()):
            cells = "  ".join(
                f"{k}={','.join(f'{v:.4g}' for v in vs)}" for k, vs in values.items()
            )
            print(f"{setting:32s} {cells}")


if __name__ == "__main__":
    main()
