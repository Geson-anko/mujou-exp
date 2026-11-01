"""A sweep: one process per run of a module, ``--jobs`` at a time."""

import json
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exp.lib.config import RunConfig
from exp.lib.record import REPO_ROOT

__all__ = ["run_all"]


def run_all(module: str, configs: Sequence[RunConfig], out: Path, jobs: int) -> None:
    """Run ``python -m <module> --config <file> --out <out>`` for each of
    ``configs`` not done yet in ``out``, ``jobs`` at a time, each in a process of
    its own. The module prints the directory of its run as its last line; a
    ``done`` file in it marks the run as done."""
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in configs if not list(out.glob(f"*-{c.name}/done"))]
    print(f"{len(todo)} runs to go", flush=True)

    def one(c: RunConfig) -> None:
        path = out / f"{c.name}.json"
        path.write_text(json.dumps(c.to_dict(), indent=2))
        cmd = [sys.executable, "-m", module, "--config", str(path), "--out", str(out)]
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAILED {c.name}\n{result.stderr[-2000:]}", flush=True)
            return
        (Path(result.stdout.strip().splitlines()[-1]) / "done").touch()
        print(f"done {c.name}", flush=True)

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(one, todo))
