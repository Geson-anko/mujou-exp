"""``just sweep <cmd>`` (design 5): the optuna coordinator on the head.

On the Mac every command is forwarded to the head's container, where it runs
with ``--local``: ``tools/sweep`` of ``code/latest`` under ``uv`` with its own
venv (``<root>/venv/sweep-<hash of its lock>``, Python 3.14 with the GIL;
optuna, numpy and sqlalchemy never enter the venv of the runs).

``run <study.toml>``: start the coordinator of a study (detached; one per
study, a second start says it runs). The TOML is read from the snapshot the
study is bound to (``code/latest`` for a new study, or ``--snap``); its
storage is ``<root>/optuna/<name>.db`` on the head, copied every hour to the
NAS's ``studies/<name>/``, where ``RUNNING`` stands until the study ends.
``status <study>``: its trials. ``confirm`` / ``bracket``: the follow-up runs
(``sweep.followup``). This module imports only the standard library and
``exp.lib.fleet`` until a local command needs optuna, so the Mac runs it
under its plain Python.
"""

import argparse
import fcntl
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exp.lib.fleet.config import FleetConf, default_path
from exp.lib.fleet.contract import read_json, write_json
from exp.lib.fleet.jobs import GPU_ANY, compose_configs
from exp.lib.fleet.provision import venv_python
from exp.lib.fleet.queue import Nas, nas_retry
from exp.lib.fleet.slots import CPU
from exp.lib.fleet.transport import SshTransport, Transport

if TYPE_CHECKING:
    import optuna

__all__ = ["main", "study_storage"]

_ABOUT = "The optuna coordinator of the fleet (design 5)."
# Runs `python -m sweep --local <args>` from the latest snapshot, in the venv of
# its tools/sweep lock (made by `uv run` on first use).
_FORWARD_SH = (
    'root=$1; shift; cd -P "$root/code/latest" && '
    "key=$(cat tools/sweep/pyproject.toml tools/sweep/uv.lock | "
    "{ sha256sum 2>/dev/null || shasum -a 256; } | cut -c1-12) && "
    'UV_PROJECT_ENVIRONMENT="$root/venv/sweep-$key" PYTHONPATH="$PWD/tools/sweep" '
    "exec uv run --quiet --frozen --no-dev --project tools/sweep --python 3.14+gil "
    'python -m sweep --local --config "$PWD/fleet.toml" "$@"'
)
_BACKUP_EVERY = 3600.0


def main(
    argv: Sequence[str] | None = None, *, transport: Transport | None = None
) -> int:
    """Run the command of ``argv`` (``sys.argv[1:]`` by default); the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(argv)
    conf = FleetConf.load(args.config)
    if not args.local:
        t = transport or SshTransport(conf)
        rest = argv[argv.index(args.cmd) :]
        return t.container(
            conf.head,
            ["bash", "-c", _FORWARD_SH, "_", str(conf.root), *rest],
            env={"PYTHONPYCACHEPREFIX": "/tmp/pycache"},
            capture=False,
        ).returncode
    return _COMMANDS[args.cmd](conf, args)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="just sweep", description=_ABOUT)
    ap.add_argument("--config", type=Path, default=default_path(), help="fleet.toml")
    ap.add_argument("--local", action="store_true", help="run here (on the head)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="start (or resume) the coordinator of a study")
    p.add_argument("toml", help="the study, relative to the checkout")
    p.add_argument("--snap", help="the snapshot (a prefix) of a new study")
    p.add_argument("--foreground", action="store_true", help="do not detach")
    p.add_argument("--every", type=float, default=60.0, help="seconds between steps")

    p = sub.add_parser("status", help="the trials of a study")
    p.add_argument("study")
    p.add_argument("--json", action="store_true")

    for name in ("confirm", "bracket"):
        p = sub.add_parser(name, help=f"queue the {name} runs of a study")
        p.add_argument("study")
        if name == "confirm":
            p.add_argument("--top", type=int, required=True)
        else:
            which = p.add_mutually_exclusive_group(required=True)
            which.add_argument("--best", action="store_true")
            which.add_argument("--trial", type=int)
            p.add_argument("--param", required=True, help="an override key")
            p.add_argument("--factors", type=_floats, default=[0.5, 1.0, 2.0])
        p.add_argument("--learners", type=_strs, required=True, help="bp,pc")
        p.add_argument("--seeds", type=_ints, required=True, help="0,1,2")
        p.add_argument("--batch", required=True)
        p.add_argument("--res", help="cpu | a GPU model (default: the study's)")
        p.add_argument("--ram", type=float, help="GB")
        p.add_argument("--gpu-mem", type=float, help="GB")
        p.add_argument("--prio", type=int, help="default 20: before the searches")
        p.add_argument("--snap", help="the snapshot (default: the study's)")
        p.add_argument("--dry", action="store_true")
    return ap


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def _strs(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


# -- paths on the head -----------------------------------------------------------


def study_storage(root: Path, name: str) -> optuna.storages.RDBStorage:
    """The storage of the study ``name``: ``<root>/optuna/<name>.db``."""
    import optuna

    d = root / "optuna"
    d.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(
        f"sqlite:///{d / f'{name}.db'}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )


def _code_dir(root: Path, snap: str | None) -> Path:
    """``<root>/code/<snap12>`` of the snapshot prefix ``snap`` (the latest
    when ``None``)."""
    code = root / "code"
    if snap is None:
        return (code / "latest").resolve()
    found = [
        p for p in code.iterdir() if len(p.name) == 12 and p.name.startswith(snap[:12])
    ]
    if len(found) != 1:
        raise SystemExit(f"snapshot {snap!r}: {len(found)} matches under {code}")
    return found[0]


def _composer(root: Path, code_dir: Path, info: dict[str, Any]) -> Callable[..., Any]:
    python = venv_python(root, info["envkey"])
    return lambda lists: compose_configs(python, code_dir, lists)


def _load(root: Path, name: str) -> optuna.Study:
    import optuna

    if not (root / "optuna" / f"{name}.db").exists():
        raise SystemExit(f"no study {name} under {root / 'optuna'}")
    return optuna.load_study(study_name=name, storage=study_storage(root, name))


def _bound_snap(root: Path, name: str) -> str | None:
    if not (root / "optuna" / f"{name}.db").exists():
        return None
    storage = study_storage(root, name)
    if name not in {s.study_name for s in storage.get_all_studies()}:
        return None
    import optuna

    return optuna.load_study(study_name=name, storage=storage).user_attrs.get("snap")


def _own_code() -> str | None:
    """The snapshot this module runs from (``CODE.json`` four levels up, in
    ``<root>/code/<snap12>/tools/sweep/sweep/``); ``None`` outside one."""
    info = Path(__file__).resolve().parents[3] / "CODE.json"
    return str(read_json(info)["snap"]) if info.exists() else None


def _locked(path: Path) -> int | None:
    """The pid of the holder of the lock ``path`` (``None``: free)."""
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = path.with_suffix(".pid")
            return int(pid.read_text()) if pid.exists() else -1
        fcntl.flock(f, fcntl.LOCK_UN)
        return None


# -- run ---------------------------------------------------------------------------


def _run(conf: FleetConf, args: argparse.Namespace) -> int:
    from sweep.coordinator import Coordinator
    from sweep.study import StudyConf, StudyError, backup, open_study

    code_dir = _code_dir(conf.root, args.snap)
    name = StudyConf.load(code_dir / args.toml).name
    bound = _bound_snap(conf.root, name)
    if bound is not None:
        if args.snap is not None and not bound.startswith(args.snap):
            raise SystemExit(
                f"study {name} is bound to the snapshot {bound[:12]}, not {args.snap}:"
                " new code is a new study (-v2)"
            )
        code_dir = _code_dir(conf.root, bound)
    study_conf = StudyConf.load(code_dir / args.toml)
    base = conf.root / "optuna" / name
    base.parent.mkdir(parents=True, exist_ok=True)
    lock = base.with_suffix(".lock")
    held = _locked(lock)
    if held is not None:
        print(f"the coordinator of {name} runs (pid {held})")
        return 0
    if not args.foreground:
        return _detach(conf, args, code_dir.name, base)

    lockf = lock.open("a")  # held until this process exits
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"the coordinator of {name} runs (started meanwhile)")
        return 0
    base.with_suffix(".pid").write_text(f"{os.getpid()}\n")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    info = read_json(code_dir / "CODE.json")
    nas = Nas(conf.nas)
    nas.ensure()
    try:
        study = open_study(study_conf, study_storage(conf.root, name), code=info)
        coord = Coordinator(
            study_conf,
            study,
            nas=nas,
            code=info,
            composer=_composer(conf.root, code_dir, info),
        )
    except StudyError as e:
        raise SystemExit(str(e)) from None
    own = _own_code()
    started = time.time()
    # The coordinator's own code (sweep.objective, the pruning) may differ
    # from the study's snapshot: each start is kept in the study.
    starts = list(study.user_attrs.get("coordinators", []))
    study.set_user_attr("coordinators", [*starts, {"code": own, "at": started}])
    shared = nas.root / "studies" / name
    marker = {
        "toml": args.toml,
        "snap": info["snap"],
        "coordinator_code": own,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started": started,
    }
    nas_retry(lambda: shared.mkdir(parents=True, exist_ok=True))
    nas_retry(lambda: write_json(shared / "RUNNING", marker))
    db, copy = base.with_suffix(".db"), shared / f"{name}.db"
    backed = time.monotonic()

    def hourly() -> None:
        nonlocal backed
        if time.monotonic() - backed >= _BACKUP_EVERY:
            backed = time.monotonic()
            try:
                backup(db, copy)
            except OSError:
                logging.getLogger("sweep").exception("the backup to the NAS failed")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    logging.getLogger("sweep").info(
        "study %s on %s, coordinator code %s (pid %d)",
        name,
        info["snap"][:12],
        str(own)[:12],
        os.getpid(),
    )
    coord.serve(stop, every=args.every, after_step=hourly)
    backup(db, copy)
    if coord.finished():
        (shared / "RUNNING").unlink(missing_ok=True)
        logging.getLogger("sweep").info("study %s finished", name)
    return 0


def _detach(conf: FleetConf, args: argparse.Namespace, snap12: str, base: Path) -> int:
    """Start ``run --foreground`` in its own session, logging to
    ``<root>/optuna/<name>.log``; wait until it holds its lock (it writes its
    pid after taking the lock; polling the lock itself could take it from
    the child for a moment)."""
    argv = [sys.executable, "-m", "sweep", "--local", "--config", str(args.config)]
    argv += [
        "run",
        args.toml,
        "--snap",
        snap12,
        "--foreground",
        "--every",
        str(args.every),
    ]
    with base.with_suffix(".log").open("ab") as log:
        proc = subprocess.Popen(
            argv,
            env={**os.environ, "PYTHONPYCACHEPREFIX": "/tmp/pycache"},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid = base.with_suffix(".pid")
    end = time.monotonic() + 60
    while time.monotonic() < end and proc.poll() is None and _pid(pid) != proc.pid:
        time.sleep(0.2)
    if proc.poll() is not None:
        print(f"the coordinator exited at once; see {base.with_suffix('.log')}")
        return 1
    print(f"the coordinator of {base.name} started (pid {proc.pid}, code {snap12})")
    print(f"log {base.with_suffix('.log')}")
    return 0


def _pid(path: Path) -> int | None:
    try:
        return int(path.read_text())
    except FileNotFoundError, ValueError:
        return None


# -- status --------------------------------------------------------------------------


def _status(conf: FleetConf, args: argparse.Namespace) -> int:
    import optuna

    from sweep.fleetio import FleetView

    study = _load(conf.root, args.study)
    trials = study.get_trials(deepcopy=False)
    counts = Counter(t.state.name for t in trials)
    held = _locked(conf.root / "optuna" / f"{args.study}.lock")
    done = sorted(
        (t for t in trials if t.state == optuna.trial.TrialState.COMPLETE),
        key=lambda t: (t.value, t.number),
    )
    running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    view = None
    if running:
        view = FleetView(Nas(conf.nas))
        view.refresh()
    rows = []
    for t in running:
        assert view is not None
        states = Counter(view.seed_run(r).status for r in t.user_attrs.get("runs", []))
        last = t.last_step
        rows.append(
            {
                "trial": t.number,
                "step": last,
                "value": None if last is None else t.intermediate_values[last],
                "runs": dict(states),
                "overrides": t.user_attrs.get("overrides", []),
            }
        )
    best = [
        {
            "trial": t.number,
            "value": t.value,
            "overrides": t.user_attrs.get("overrides", []),
        }
        for t in done[:5]
    ]
    if args.json:
        out = {
            "study": args.study,
            "snap": study.user_attrs.get("snap"),
            "coordinator": held,
            "counts": dict(counts),
            "running": rows,
            "best": best,
        }
        print(json.dumps(out, indent=1))
        return 0
    snap = str(study.user_attrs.get("snap"))[:12]
    alive = f"pid {held}" if held is not None else "not running"
    print(f"{args.study}  snap {snap}  coordinator {alive}")
    print("  ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no trials")
    for b in best:
        print(f"  best #{b['trial']:<4} {b['value']:+.4f}  {' '.join(b['overrides'])}")
    for r in rows:
        at = "-" if r["step"] is None else f"{r['value']:+.4f} @ {r['step']} s"
        runs = " ".join(f"{k} {v}" for k, v in sorted(r["runs"].items()))
        print(f"  running #{r['trial']:<4} {at}  [{runs}]  {' '.join(r['overrides'])}")
    return 0


# -- confirm and bracket ----------------------------------------------------------


def _followup(conf: FleetConf, args: argparse.Namespace) -> int:
    from sweep.followup import FollowupError, bracket, confirm
    from sweep.study import StudyConf

    study = _load(conf.root, args.study)
    study_conf = StudyConf.of_study(study)
    code_dir = _code_dir(conf.root, args.snap or study.user_attrs["snap"])
    info = read_json(code_dir / "CODE.json")
    res = study_conf.res_dict()
    if args.res is not None and args.res != res["class"]:
        # A GPU class for another keeps the study's GPU memory.
        gpu_mem = 0.0 if args.res == CPU else res["gpu_mem_gb"] or 2.0
        res = {**res, "class": args.res, "gpu_mem_gb": gpu_mem}
    if res["class"] not in (CPU, GPU_ANY, *conf.models):
        raise SystemExit(
            f"--res must be cpu or one of {conf.models}, not {res['class']}"
        )
    if args.ram is not None:
        res["ram_gb"] = args.ram
    if args.gpu_mem is not None:
        res["gpu_mem_gb"] = args.gpu_mem
    common: dict[str, Any] = {
        "nas": Nas(conf.nas),
        "code": info,
        "composer": _composer(conf.root, code_dir, info),
        "learners": args.learners,
        "seeds": args.seeds,
        "batch": args.batch,
        "res": res,
        "dry": args.dry,
    }
    if args.prio is not None:
        common["prio"] = args.prio
    Nas(conf.nas).ensure()
    try:
        if args.cmd == "confirm":
            out = confirm(study, top=args.top, **common)
        else:
            out = bracket(
                study,
                trial=None if args.best else args.trial,
                param=args.param,
                factors=args.factors,
                **common,
            )
    except FollowupError as e:
        raise SystemExit(str(e)) from None
    for state, job in out:
        print(f"{state:<9}  {job['run_id']}  {job['label']}")
    n_new = sum(state == "new" for state, _ in out)
    print(f"{n_new} queued{' (dry)' if args.dry else ''}")
    return 0


_COMMANDS: dict[str, Callable[[FleetConf, argparse.Namespace], int]] = {
    "run": _run,
    "status": _status,
    "confirm": _followup,
    "bracket": _followup,
}
