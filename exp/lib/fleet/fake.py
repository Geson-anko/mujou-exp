"""A stand-in program that honors the run contract, for the tests and smoke runs.

``python -m exp.lib.fleet.fake --job <run_dir>/job.json --attempt <n>``. Submit
it with ``--entry exp.lib.fleet.fake``. It reads its behavior from
``job["fake"]`` and from the overrides ``fake.<key>=<json>`` (later wins):

``frames`` (50) frames over ``seconds`` (1.0) of wall time; ``exit`` (0): the
exit code, or a list indexed by attempt (the last repeats): 0 completed, 3
diverged, 4 environment error, 1 an exception; ``kill``: SIGKILL itself at the
end (as the OOM killer would); ``no_manifest``: exit 0 without
``manifest.json``; ``hang_at``: stop advancing at that frame (a stall);
``ignore_stop``: do not honor ``stop.request``; ``child``: start a grandchild
that sleeps in the same process group (its pid in ``child.pid``);
``resume_every`` (10): frames between resume checkpoints. It resumes from
``resume/latest`` when that exists, and writes ``status.json`` (the heartbeat,
whose last ``state`` is ``stopped``, ``env`` or ``failed`` when it ends so),
``evals.jsonl``, ``result.json`` and ``manifest.json`` like a run of
``exp.video.main``.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from exp.lib.fleet.contract import (
    EXIT_DIVERGED,
    EXIT_ENV,
    EXIT_OK,
    EXIT_STOPPED,
    write_json,
    write_manifest,
)

__all__ = ["main", "params"]


def params(job: dict[str, Any]) -> dict[str, Any]:
    """The behavior: ``job["fake"]`` updated by the ``fake.<key>=<json>`` overrides."""
    out = dict(job.get("fake") or {})
    for o in job.get("overrides", []):
        key, sep, raw = o.partition("=")
        if sep and key.startswith("fake."):
            try:
                out[key.removeprefix("fake.")] = json.loads(raw)
            except ValueError:
                out[key.removeprefix("fake.")] = raw
    return out


def _exit_code(p: dict[str, Any], attempt: int) -> int:
    code = p.get("exit", EXIT_OK)
    if isinstance(code, list):
        code = code[min(attempt, len(code)) - 1]
    return int(code)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m exp.lib.fleet.fake")
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--attempt", type=int, required=True)
    args = ap.parse_args(argv)
    run_dir: Path = args.job.parent
    job = json.loads(args.job.read_text())
    p = params(job)
    frames = int(p.get("frames", 50))
    dt = float(p.get("seconds", 1.0)) / frames
    resume_every = int(p.get("resume_every", 10))
    hang_at = p.get("hang_at")
    if p.get("child"):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        (run_dir / "child.pid").write_text(f"{child.pid}\n")

    latest = run_dir / "resume" / "latest"
    start = int(latest.read_text()) if latest.exists() else 0
    t0 = time.monotonic()

    def beat(frame: int, state: str = "running") -> None:
        elapsed = max(time.monotonic() - t0, 1e-9)
        fps = (frame - start) / elapsed
        write_json(
            run_dir / "status.json",
            {
                "state": state,
                "attempt": args.attempt,
                "pid": os.getpid(),
                "pgid": os.getpgid(0),
                "frame": frame,
                "t": frame * dt,
                "of": frames * dt,
                "fps": fps,
                "eta": (frames - frame) / fps if fps > 0 else None,
                "rss_mb": 0.0,
                "gpu_mem_mb": 0.0,
            },
        )

    def checkpoint(frame: int) -> None:
        latest.parent.mkdir(exist_ok=True)
        latest.write_text(f"{frame}\n")

    frame = start
    beat(frame)
    while frame < frames:
        if (run_dir / "stop.request").exists() and not p.get("ignore_stop"):
            checkpoint(frame)
            beat(frame, "stopped")
            return EXIT_STOPPED
        time.sleep(dt)
        if hang_at is not None and frame >= int(hang_at):
            beat(frame)
            continue
        frame += 1
        if frame % resume_every == 0:
            checkpoint(frame)
            line = {"t": frame * dt, "frame": frame, "split": "val"}
            line |= {"roll10": 1.0 / frame, "roll10/cv10": 0.5}
            with (run_dir / "evals.jsonl").open("a") as f:
                f.write(json.dumps(line) + "\n")
        beat(frame)

    code = _exit_code(p, args.attempt)
    if p.get("kill"):
        os.kill(os.getpid(), signal.SIGKILL)
    if code == EXIT_ENV:
        beat(frame, "env")
        print(
            "EnvError: the fake program was told to fail its environment",
            file=sys.stderr,
        )
        return EXIT_ENV
    if code not in (EXIT_OK, EXIT_DIVERGED):
        beat(frame, "failed")
        raise RuntimeError(p.get("error", "the fake program was told to fail"))
    status = "completed" if code == EXIT_OK else "diverged"
    result = {"schema": 1, "status": status, "reason": p.get("reason")}
    result |= {"frames": frame, "resumed_from": start, "attempt": args.attempt}
    write_json(run_dir / "result.json", result)
    if not p.get("no_manifest"):
        write_manifest(run_dir)
    return code


if __name__ == "__main__":
    sys.exit(main())
