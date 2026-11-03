"""The entry of a run.

As the fleet's worker starts it (in the snapshot's code tree, with the
environment of the contract)::

    python -m exp.video.main --job <run_dir>/job.json --attempt <n>

and by hand, on this machine::

    python -m exp.video.main +experiment=m14/s0 seconds=60 --out /tmp/m14

The local form composes the overrides (``exp.video.compose``), writes
``<out>/job.json`` as ``fleet submit`` makes it (``exp.lib.fleet.jobs.make_job``,
batch ``"local"``, the code of this checkout: its ``HEAD``, whether it is
dirty, and the diff from ``HEAD`` with mujou's in ``code.diff``, the files git
does not track but does not ignore included, as in a fleet snapshot), and
runs it the same way. ``<out>`` is the run's directory: it must be empty or
hold the same job. Then a run that finished (``result.json``) is left as it
is and its exit code returned again; any other resumes from its ``resume/``,
as after a stop. ``OMP_NUM_THREADS`` is set to 1 unless it is set.

The program of a job is looked up by name in ``PROGRAMS`` and imported only
then; its module's ``PROGRAM`` is a ``loop.Program``. This module imports the
standard library only until it runs a job. The exit code is the loop's
(``exp.lib.rundir.Exit``); 2 for wrong arguments.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["PROGRAMS", "load_program", "local_job", "main"]

PROGRAMS = {
    "dynamics": "exp.video.programs.dynamics",
    "vision": "exp.video.programs.vision",
    "joint": "exp.video.programs.joint",
}
REPO = Path(__file__).resolve().parents[2]
LOCAL_RES = {"cpu": {"class": "cpu", "ram_gb": 2.0, "gpu_mem_gb": 0.0}}


def load_program(name: str) -> Any:
    """The ``PROGRAM`` of the program ``name`` (a key of ``PROGRAMS``).

    Raises:
        ValueError: ``name`` is not a program.
    """
    if name not in PROGRAMS:
        raise ValueError(f"no program {name!r}; one of {sorted(PROGRAMS)}")
    return importlib.import_module(PROGRAMS[name]).PROGRAM


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return out.stdout


def _diff(repo: Path, *prefix: str) -> str:
    """The diff of the tree ``repo`` from its ``HEAD``, the files git does not
    track but does not ignore as new files; nothing is staged."""
    out = _git(repo, "diff", *prefix, "HEAD")
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    for name in filter(None, untracked.split("\0")):
        # --no-index exits 1 when the files differ, which a new file always does
        out += subprocess.run(
            ["git", "-C", str(repo), "diff", "--no-index", *prefix]
            + ["--", "/dev/null", name],
            capture_output=True,
            text=True,
        ).stdout
    return out


def _local_code(repo: Path) -> tuple[dict[str, Any], str]:
    """The ``code`` of a job of the checkout ``repo`` and its ``code.diff``:
    ``snap`` and ``base_head`` are ``HEAD`` (no snapshot is made), ``dirty``
    whether the tree or mujou's differs from it. Without git, ``"unknown"``."""
    try:
        head = _git(repo, "rev-parse", "HEAD").strip()
        status = _git(repo, "status", "--porcelain")
        diff = _diff(repo)
        mujou = repo / "mujou"
        m_head, m_status, m_diff = "unknown", "", ""
        if (mujou / ".git").exists():
            m_head = _git(mujou, "rev-parse", "HEAD").strip()
            m_status = _git(mujou, "status", "--porcelain")
            m_diff = _diff(mujou, "--src-prefix=a/mujou/", "--dst-prefix=b/mujou/")
    except OSError, subprocess.CalledProcessError:
        head, status, diff, m_head, m_status, m_diff = (
            "unknown",
            "",
            "",
            "unknown",
            "",
            "",
        )
    code = {
        "snap": head,
        "base_head": head,
        "dirty": bool(status.strip() or m_status.strip()),
        "mujou_snap": m_head,
        "mujou_pushed": False,
        "envkey": "local",
    }
    return code, diff + m_diff


def local_job(overrides: Sequence[str], out: Path, *, repo: Path = REPO) -> Path:
    """Compose ``overrides``, write the job and the code of the checkout
    ``repo`` into the run's directory ``out``, and return the path of its
    ``job.json``.

    Raises:
        FileExistsError: ``out`` holds files of another job.
    """
    from exp.lib.fleet.jobs import make_job
    from exp.video.compose import compose, to_dict

    config = to_dict(compose(overrides))
    code, diff = _local_code(repo)
    res = (
        LOCAL_RES["cpu"]
        if config["device"] == "cpu"
        else {
            "class": "any_gpu",
            "ram_gb": 2.0,
            "gpu_mem_gb": 2.0,
        }
    )
    job = make_job(
        batch="local", config=config, overrides=overrides, code=code, res=res
    )
    path = out / "job.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("run_id") != job["run_id"]:
            raise FileExistsError(
                f"{out} holds the run {old.get('run_id')!r}, not {job['run_id']!r}"
            )
        return path  # the same job: resume it
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty and has no job.json")
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2) + "\n")
    (out / "CODE.json").write_text(json.dumps(code, indent=2) + "\n")
    (out / "code.diff").write_text(diff)
    return path


def _attempt(run_dir: Path) -> int:
    """The attempt after the last of ``provenance/attempt-<n>.json``."""
    done = [
        int(p.stem.removeprefix("attempt-"))
        for p in (run_dir / "provenance").glob("attempt-*.json")
    ]
    return max(done, default=0) + 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m exp.video.main",
        description="Run a job of the fleet (--job) or compose one and run it here.",
    )
    ap.add_argument("--job", type=Path, help="the job.json the worker wrote")
    ap.add_argument("--attempt", type=int, help="the attempt of --job")
    ap.add_argument("--out", type=Path, help="the run's directory of a local run")
    ap.add_argument("overrides", nargs="*", help="hydra overrides of a local run")
    args = ap.parse_args(argv)
    if args.job is not None:
        if args.out is not None or args.overrides or args.attempt is None:
            ap.error("--job takes --attempt and nothing else")
        job, attempt = args.job, args.attempt
    else:
        if args.out is None:
            ap.error("a local run needs --out (or give --job and --attempt)")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            job = local_job(args.overrides, args.out)
        except FileExistsError as e:
            ap.error(str(e))
        attempt = _attempt(args.out)
    from exp.lib.rundir import Exit, write_json

    if args.job is None and (job.parent / "result.json").exists():
        status = json.loads((job.parent / "result.json").read_text())["status"]
        print(f"{job.parent} has finished ({status}); remove it to run again")
        return Exit.OK if status == "completed" else Exit.DIVERGED
    from exp.video import loop

    try:
        program = load_program(json.loads(job.read_text())["program"])
    except Exception:  # a job this code cannot run: failed, as the loop says
        traceback.print_exc()
        write_json(job.parent / "status.json", {"state": "failed", "attempt": attempt})
        return Exit.ERROR
    return loop.run(program, job, attempt=attempt)


if __name__ == "__main__":
    sys.exit(main())
