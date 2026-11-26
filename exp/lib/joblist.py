"""Run the lines of a job list in parallel on one host (``plans/jepa.md``,
「線形の幅」; the parallel form of ``runs/v18c-tlin/scripts/apply.sh``).

A list's line is ``<name> <args...>`` (blank lines and ``#`` lines
skipped); each runs as ``<cmd...> <name> <gpu> <args...>`` (the form of
``runs/v18c-tlin/scripts/job.sh``: its name, its CUDA index or ``""`` of
none, the module's arguments), its output and errors in ``<logs>/<name>.log``
and, when it ends, a line ``<host> <name> rc=<code> <unix time>`` in
``<logs>/<list stem>.done`` (``ALL`` after the last). ``parallel`` of them
run at once, each its own process (the numbers of a job are those of the job
alone: one thread each, the parallel being of processes). Standard library
only, so it runs as a file by the host's ``python3``::

    python3 exp/lib/joblist.py --list <host>.list --logs <dir> --parallel 2 \
        --gpu 0 -- bash <job.sh>
"""

import argparse
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__all__ = ["main", "parse_list", "run_list"]


def parse_list(text: str) -> list[tuple[str, list[str]]]:
    """The jobs ``(name, args)`` of a list's ``text`` (module docstring)."""
    jobs = []
    for line in text.splitlines():
        words = shlex.split(line, comments=True)
        if words:
            jobs.append((words[0], words[1:]))
    return jobs


def run_list(
    cmd: Sequence[str],
    jobs: Sequence[tuple[str, Sequence[str]]],
    logs: Path,
    *,
    parallel: int,
    gpu: str = "",
    done: str = "jobs",
) -> dict[str, int]:
    """Run ``jobs`` (``parse_list``) by ``cmd``, ``parallel`` at once, each
    logged in ``logs`` (module docstring; the ends in ``<done>.done``).
    Returns each job's exit code.

    Raises:
        ValueError: Two jobs share a name (their logs would mix).
    """
    names = [n for n, _ in jobs]
    if len(set(names)) != len(names):
        raise ValueError(f"a name twice in {names}")
    logs.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    lock = threading.Lock()
    ends = logs / f"{done}.done"

    def one(name: str, args: Sequence[str]) -> int:
        with (logs / f"{name}.log").open("wb") as log:
            rc = subprocess.run(
                [*cmd, name, gpu, *args], stdout=log, stderr=subprocess.STDOUT
            ).returncode
        with lock, ends.open("a") as f:
            f.write(f"{host} {name} rc={rc} {int(time.time())}\n")
        return rc

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {n: pool.submit(one, n, a) for n, a in jobs}
        codes = {n: f.result() for n, f in futures.items()}
    with ends.open("a") as f:
        f.write(f"{host} ALL {int(time.time())}\n")
    return codes


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--parallel", type=int, required=True)
    p.add_argument("--gpu", default="", help="the CUDA index of every job ('' none)")
    p.add_argument("cmd", nargs="+", help="the job's command, before its name")
    a = p.parse_args(argv)
    codes = run_list(a.cmd, parse_list(a.list.read_text()), a.logs,
                     parallel=a.parallel, gpu=a.gpu, done=a.list.stem)  # fmt: skip
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
