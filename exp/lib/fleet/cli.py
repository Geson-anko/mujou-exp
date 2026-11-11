"""``just fleet <cmd>`` (design 4.7).

On the Mac, ``doctor``, ``push``, ``up``, ``pull``, ``x`` and ``logs`` run here
and reach the hosts by ssh; every other command is forwarded to the head's
container and runs there with ``--local``, against the NAS. ``--local`` runs a
command here against the paths of ``fleet.toml`` (in a container, or in the
tests). ``worker``, ``provision`` and ``up-local`` are the hosts' own.
"""

import argparse
import concurrent.futures
import fcntl
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from exp.lib.fleet import snapshot
from exp.lib.fleet.config import FleetConf, default_path
from exp.lib.fleet.contract import read_json, write_json
from exp.lib.fleet.jobs import (
    DEFAULT_ENTRY,
    GPU_ANY,
    compose_configs,
    expand,
    make_job,
    parse_set,
    seed_of,
)
from exp.lib.fleet.provision import EXTRACT_SH, provision, venv_python
from exp.lib.fleet.queue import FINAL_OK, Nas, QueueName
from exp.lib.fleet.slots import CPU
from exp.lib.fleet.transport import SshTransport, Transport
from exp.lib.fleet.worker import Timing, Worker

__all__ = ["main"]

_ABOUT = "The fleet: run many experiments on the hosts (design 4.7)."
MAC = frozenset({"doctor", "push", "up", "pull", "x", "logs"})
HOST = frozenset({"worker", "provision", "up-local"})
LOST_AFTER = 600.0  # seconds without a heartbeat before a host's runs are "lost?"

# Runs `python -m exp.lib.fleet --local <args>` with the latest snapshot's venv.
_FORWARD_SH = (
    'root=$1; shift; cd "$root/code/latest" && '
    'exec "$root/venv/$(cat .envkey)/bin/python" -m exp.lib.fleet --local '
    '--config "$PWD/fleet.toml" "$@"'
)
# Runs a command in the latest snapshot with its venv first on PATH.
_X_SH = (
    'root=$1; shift; cd "$root/code/latest" && '
    'export PATH="$root/venv/$(cat .envkey)/bin:$PATH" && exec "$@"'
)
# argv: tree root info gpus. The interpreter uv manages (the container has no
# python3) runs this module from the extracted tree, under the host's provision
# lock (fd 9, held until that python exits), so two pushes never sync one venv.
_PROVISION_SH = (
    "tree=$1 root=$2 info=$3 gpus=$4; py=${FLEET_PYTHON:-$(uv python find 3.14t)}; "
    'mkdir -p "$root/locks" && exec 9>"$root/locks/provision.lock" && '
    "{ if command -v flock > /dev/null; then flock 9; fi; } && "
    'cd "$tree" && exec "$py" -m exp.lib.fleet --local --config "$tree/fleet.toml" '
    'provision --tree "$tree" --info "$info" --gpus "$gpus"'
)
_DOCTOR_SH = r"""
nas=$1 root=$2 host=$3
echo "user=$(id -un)"
echo "gpus=$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | tr '\n' ';')"
d="$nas/.doctor"; f="$d/$host.$$"
if mkdir -p "$d" && echo x > "$f.tmp" && mv "$f.tmp" "$f" && [ ! -e "$f.tmp" ] && rm "$f"; then
  echo nas=ok; else echo nas=fail; fi
if mkdir -p "$root" && touch "$root/.doctor" && rm "$root/.doctor"; then echo root=ok; else echo root=fail; fi
echo "clock=$(date +%s.%N)"
for x in git uv flock taskset tar nvidia-smi; do
  if command -v "$x" > /dev/null; then echo "tool_$x=ok"; else echo "tool_$x=missing"; fi; done
echo "python=$(uv python find 3.14t 2>&1)"
echo "cores=$(cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list | sort -u | wc -l)"
echo "cpus=$(nproc)"
"""


def main(
    argv: Sequence[str] | None = None, *, transport: Transport | None = None
) -> int:
    """Run the command of ``argv`` (``sys.argv[1:]`` by default); the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args, extra = parser.parse_known_args(argv)
    if extra and args.cmd != "table":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.table_argv = extra
    conf = FleetConf.load(args.config)
    cmd: str = args.cmd
    if cmd in HOST and not args.local:
        raise SystemExit(f"fleet {cmd} runs on a host: pass --local")
    if cmd in MAC:
        if args.local:
            raise SystemExit(f"fleet {cmd} runs on the Mac (without --local)")
        return _MAC_COMMANDS[cmd](conf, transport or SshTransport(conf), args)
    if not args.local:
        t = transport or SshTransport(conf)
        rest = argv[argv.index(cmd) :]
        return _forward(conf, t, rest, capture=False).returncode
    return _LOCAL_COMMANDS[cmd](conf, args)


# -- argument parsing ------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="just fleet", description=_ABOUT)
    ap.add_argument("--config", type=Path, default=default_path(), help="fleet.toml")
    ap.add_argument("--local", action="store_true", help="run here, against the NAS")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "doctor", help="check ssh, containers, GPUs, NAS, clocks, tools, cores"
    )
    sub.add_parser("push", help="snapshot the working tree, push and provision it")
    p = sub.add_parser("up", help="start the workers that are not running")
    p.add_argument("--restart", action="store_true", help="restart them (new code)")

    p = sub.add_parser("submit", help="compose and queue jobs")
    p.add_argument("--batch", required=True)
    p.add_argument("overrides", nargs="*", help="hydra overrides")
    p.add_argument("--snap", help="the snapshot (a prefix); default: the latest")
    p.add_argument("--sweep", action="append", default=[], metavar="K=A,B")
    p.add_argument("--seeds", type=_ints, help="0,1,2")
    p.add_argument("--res", default=CPU, help="cpu | any_gpu | a GPU model")
    p.add_argument("--ram", type=float, default=2.0, help="GB")
    p.add_argument("--gpu-mem", type=float, help="GB (default 2 for a GPU class)")
    p.add_argument("--prio", type=int, default=50)
    p.add_argument("--after", action="append", default=[], metavar="RUN_ID")
    p.add_argument("--host", help="pin to a host")
    p.add_argument("--avoid-host")
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--entry", default=DEFAULT_ENTRY, help="the program's module")
    p.add_argument("--raw", action="store_true", help="no compose (a fake program)")
    p.add_argument("--name", help="the name of a --raw job (default: the batch)")
    p.add_argument("--set", action="append", default=[], metavar="K=JSON")
    p.add_argument("--dry", action="store_true")

    for name in ("status", "ls"):
        p = sub.add_parser(name, help=f"{name} of the runs")
        p.add_argument("--batch")
        p.add_argument("--json", action="store_true")
        if name == "ls":
            p.add_argument("--state", help="queued | running | <done status>")
    p = sub.add_parser("logs", help="the tail of a run's log")
    p.add_argument("run_id")
    p.add_argument("--err", action="store_true")
    p.add_argument("--tail", type=int, default=40)
    p = sub.add_parser("locate", help="where a run is (JSON)")
    p.add_argument("run_id")

    for name, now in (("stop", False), ("cancel", True)):
        p = sub.add_parser(name, help=f"{name} runs (queued ones go to done/)")
        p.add_argument("run_ids", nargs="*")
        p.add_argument("--batch")
        p.add_argument("--trial", help="STUDY/N")
        if name == "stop":
            p.add_argument("--now", action="store_true", help="SIGTERM at once")
            p.add_argument("--reason", default="user")
    p = sub.add_parser("retry", help="queue a finished run again")
    p.add_argument("run_id")
    p.add_argument("--fresh", action="store_true", help="do not resume")
    p.add_argument("--ram", type=float)
    p.add_argument("--gpu-mem", type=float)
    p = sub.add_parser("requeue", help="put a host's claimed jobs back (from scratch)")
    p.add_argument("--host", required=True)
    for name in ("hold", "release"):
        sub.add_parser(name, help=f"{name} a batch").add_argument("batch")
    for name in ("cordon", "uncordon"):
        p = sub.add_parser(name, help=f"{name} a host (or one of its classes)")
        p.add_argument("host")
        p.add_argument("--res")

    p = sub.add_parser(
        "pull", help="finished runs (from the NAS) or a live one to runs/"
    )
    p.add_argument("--batch")
    p.add_argument("--ckpt", action="store_true", help="with the checkpoints")
    p.add_argument("--live", metavar="RUN_ID", help="a running run, from its host")
    p.add_argument("--dest", type=Path, default=Path("runs"))
    # Its arguments are those of `python -m exp.lib.runs table` (passed through).
    sub.add_parser("table", help="a markdown table (python -m exp.lib.runs table)")
    p = sub.add_parser("x", help="run a command in containers: x <host|all> -- cmd ...")
    p.add_argument("target")
    p.add_argument("cmd_argv", nargs=argparse.REMAINDER)

    p = sub.add_parser("worker", help="(host) the worker loop")
    p.add_argument("--host", required=True)
    for name, default in (
        ("tick", 5.0),
        ("live", 60.0),
        ("grace", 120.0),
        ("stall", 1200.0),
    ):
        p.add_argument(f"--{name}", type=float, default=default)
    p = sub.add_parser("up-local", help="(host) start this host's worker")
    p.add_argument("--host", required=True)
    p.add_argument("--restart", action="store_true")
    p = sub.add_parser("provision", help="(host) finish an extracted snapshot")
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--info", required=True, help="the snapshot (JSON)")
    p.add_argument(
        "--gpus", type=int, default=-1, help="CUDA devices to expect (-1: any)"
    )
    return ap


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


# -- Mac -------------------------------------------------------------------------


def _forward(
    conf: FleetConf,
    t: Transport,
    rest: Sequence[str],
    *,
    host: str | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """``fleet --local <rest>`` in the container of ``host`` (the head by default),
    with the latest snapshot's code and venv."""
    return t.container(
        host or conf.head,
        ["bash", "-c", _FORWARD_SH, "_", str(conf.root), *rest],
        env={"PYTHONPYCACHEPREFIX": "/tmp/pycache"},
        capture=capture,
    )


def _each_host[T](hosts: Sequence[str], fn: Callable[[str], T]) -> dict[str, T]:
    with concurrent.futures.ThreadPoolExecutor(len(hosts)) as ex:
        return dict(zip(hosts, ex.map(fn, hosts), strict=True))


def _text(r: subprocess.CompletedProcess[bytes]) -> str:
    return (r.stdout or b"").decode(errors="replace") + (r.stderr or b"").decode(
        errors="replace"
    )


def _doctor(conf: FleetConf, t: Transport, args: argparse.Namespace) -> int:
    def check(host: str) -> list[tuple[str, bool, str]]:
        rows: list[tuple[str, bool, str]] = []
        r = t.host(host, ["true"], timeout=30)
        rows.append(("ssh", r.returncode == 0, _text(r).strip()))
        if r.returncode != 0:
            return rows
        fmt = "{{.State.Running}} {{.HostConfig.RestartPolicy.Name}}"
        r = t.host(host, ["docker", "inspect", "-f", fmt, conf.container], timeout=30)
        out = _text(r).strip()
        rows.append(("container", out == "true unless-stopped", out))
        if not out.startswith("true"):
            return rows
        t0 = time.time()
        r = t.container(
            host,
            ["bash", "-c", _DOCTOR_SH, "_", str(conf.nas), str(conf.root), host],
            timeout=60,
        )
        t1 = time.time()
        kv = dict(line.split("=", 1) for line in _text(r).splitlines() if "=" in line)
        rows.append(("user", kv.get("user") == conf.user, kv.get("user", "?")))
        gpus = [g for g in kv.get("gpus", "").split(";") if g]
        want = [f"{g.index}: {g.model}" for g in conf.slots.gpus]
        ok = len(gpus) == len(want) and all(
            m.model in g for m, g in zip(conf.slots.gpus, gpus, strict=False)
        )
        rows.append(("gpus", ok, f"{gpus} (want {want})"))
        rows.append(("nas write+rename", kv.get("nas") == "ok", str(conf.nas)))
        rows.append(("root", kv.get("root") == "ok", str(conf.root)))
        skew = float(kv.get("clock", "nan")) - (t0 + t1) / 2
        rows.append(("clock", abs(skew) < 2 + (t1 - t0) / 2, f"{skew:+.2f} s"))
        missing = [k[5:] for k, v in kv.items() if k.startswith("tool_") and v != "ok"]
        rows.append(("tools", not missing, f"missing {missing}" if missing else "ok"))
        rows.append(
            ("python 3.14t", kv.get("python", "").endswith("t"), kv.get("python", ""))
        )
        cores, cpus = int(kv.get("cores", 0)), int(kv.get("cpus", 0))
        rows.append(
            (
                "cores",
                cores >= conf.slots.cpu,
                f"{cores} physical, {cpus} logical (slots.cpu={conf.slots.cpu})",
            )
        )
        return rows

    ok = True
    for host, rows in _each_host(conf.hosts, check).items():
        for name, good, detail in rows:
            ok &= good
            print(f"{host}  {'OK  ' if good else 'FAIL'}  {name:<16} {detail}")
    return 0 if ok else 1


def _push(conf: FleetConf, t: Transport, args: argparse.Namespace) -> int:
    repo = Path.cwd()
    snap = snapshot.make(repo)
    print(
        f"snapshot {snap.snap[:12]} on {snap.base_head[:12]}"
        f"{' (dirty)' if snap.dirty else ''}, mujou {snap.mujou_snap[:12]}"
        f"{'' if snap.mujou_pushed else ' (not pushed)'}"
    )
    info = json.dumps(snap.to_dict())
    tree = str(conf.root / "code" / snap.snap[:12])

    def one(host: str) -> str:
        for name, src, sha in (
            ("repo", repo, snap.snap),
            ("mujou", repo / "mujou", snap.mujou_snap),
        ):
            git_dir = conf.root / "git" / f"{name}.git"
            r = t.host(host, ["git", "init", "--bare", "-q", str(git_dir)])
            if r.returncode != 0:
                return f"FAIL git init: {_text(r)}"
            url, ref = t.git_url(host, git_dir), f"{sha}:refs/fleet/snap/{sha}"
            r = subprocess.run(
                ["git", "-C", str(src), "push", "-q", url, ref],
                capture_output=True,
                env={**os.environ, **t.git_env()},
                check=False,
            )
            if r.returncode != 0:
                return f"FAIL git push {name}: {_text(r)}"
        shas = [snap.snap, snap.mujou_snap, snap.snap[:12]]
        r = t.container(host, ["bash", "-c", EXTRACT_SH, "_", str(conf.root), *shas])
        if r.returncode != 0:
            return f"FAIL extract: {_text(r)}"
        gpus = str(len(conf.slots.gpus))
        r = t.container(
            host,
            ["bash", "-c", _PROVISION_SH, "_", tree, str(conf.root), info, gpus],
            env={"PYTHONPYCACHEPREFIX": "/tmp/pycache"},
        )
        if r.returncode != 0:
            return f"FAIL provision: {_text(r)[-2000:]}"
        return "OK " + r.stdout.decode().strip().splitlines()[-1]

    results = _each_host(conf.hosts, one)
    for host, res in results.items():
        print(f"{host}  {res}")
    print(snap.snap)
    return 0 if all(r.startswith("OK") for r in results.values()) else 1


def _up(conf: FleetConf, t: Transport, args: argparse.Namespace) -> int:
    def one(host: str) -> subprocess.CompletedProcess[bytes]:
        rest = ["up-local", "--host", host] + (["--restart"] if args.restart else [])
        return _forward(conf, t, rest, host=host)

    ok = True
    for host, r in _each_host(conf.hosts, one).items():
        ok &= r.returncode == 0
        print(f"{host}  {_text(r).strip()}")
    return 0 if ok else 1


def _locate_remote(conf: FleetConf, t: Transport, run_id: str) -> dict[str, Any]:
    r = _forward(conf, t, ["locate", run_id])
    if r.returncode != 0:
        raise SystemExit(_text(r).strip())
    return json.loads(r.stdout)


def _logs(conf: FleetConf, t: Transport, args: argparse.Namespace) -> int:
    loc = _locate_remote(conf, t, args.run_id)
    if loc["state"] == "queued":
        print(f"{args.run_id} is queued")
        return 0
    name = f"attempt-{loc['attempt']}.{'err' if args.err else 'out'}"
    running = loc["state"] == "running"
    host = loc["host"] if running else conf.head
    path = Path(loc["run_dir"] if running else loc["nas_dir"]) / "logs" / name
    r = t.container(host, ["tail", "-n", str(args.tail), str(path)], capture=False)
    return r.returncode


def _pull(conf: FleetConf, t: Transport, args: argparse.Namespace) -> int:
    excludes = ["--exclude=resume"] + ([] if args.ckpt else ["--exclude=ckpt"])
    if args.live:
        loc = _locate_remote(conf, t, args.live)
        if loc["state"] != "running":
            raise SystemExit(f"{args.live} is {loc['state']}, not running")
        run_dir = Path(loc["run_dir"])
        host, base, members = loc["host"], run_dir.parent, [run_dir.name]
        dest = args.dest / run_dir.parent.name
    else:
        host, base, dest = conf.head, conf.nas / "runs", args.dest
        members = [args.batch] if args.batch else ["."]
    argv = t.container_argv(
        host, ["tar", "-C", str(base), "-cf", "-", *excludes, *members]
    )
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    # Streamed: a batch with its checkpoints may be larger than the Mac's memory.
    with subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        assert proc.stdout is not None and proc.stderr is not None
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    tar.extract(member, dest, filter="data")
                    n += 1
        except tarfile.ReadError as e:
            proc.kill()
            proc.wait()
            raise SystemExit(
                f"tar: {e}\n{proc.stderr.read().decode()[-2000:]}"
            ) from None
        err = proc.stderr.read()
    if proc.returncode != 0:
        raise SystemExit(err.decode(errors="replace")[-2000:])
    print(f"pulled {n} files and directories to {dest}")
    return 0


def _x(conf: FleetConf, t: Transport, args: argparse.Namespace) -> int:
    cmd = args.cmd_argv[1:] if args.cmd_argv[:1] == ["--"] else args.cmd_argv
    if not cmd:
        raise SystemExit("usage: fleet x <host|all> -- cmd ...")
    hosts = list(conf.hosts) if args.target == "all" else [args.target]
    argv = ["bash", "-c", _X_SH, "_", str(conf.root), *cmd]
    env = {"PYTHONPYCACHEPREFIX": "/tmp/pycache"}
    if len(hosts) == 1:
        return t.container(hosts[0], argv, env=env, capture=False).returncode
    results = _each_host(hosts, lambda h: t.container(h, argv, env=env))
    for host, r in results.items():
        print(f"== {host} (exit {r.returncode})\n{_text(r)}")
    return max(r.returncode for r in results.values())


_MAC_COMMANDS: dict[str, Callable[[FleetConf, Transport, argparse.Namespace], int]] = {
    "doctor": _doctor,
    "push": _push,
    "up": _up,
    "logs": _logs,
    "pull": _pull,
    "x": _x,
}


# -- here (the head's container, a host, or the tests) ------------------------------


def _code_dir(conf: FleetConf, snap: str | None) -> Path:
    code = conf.root / "code"
    if snap is None:
        return (code / "latest").resolve()
    found = [
        p for p in code.iterdir() if p.name.startswith(snap[:12]) and len(p.name) == 12
    ]
    if len(found) != 1:
        raise SystemExit(f"snapshot {snap!r}: {len(found)} matches under {code}")
    return found[0]


def _submit(conf: FleetConf, args: argparse.Namespace) -> int:
    nas = Nas(conf.nas)
    nas.ensure()
    code_dir = _code_dir(conf, args.snap)
    info = read_json(code_dir / "CODE.json")
    if args.res not in (CPU, GPU_ANY, *conf.slots.models):
        raise SystemExit(f"--res must be cpu, any_gpu or one of {conf.slots.models}")
    lists = expand(args.overrides, args.sweep, args.seeds)
    if args.raw:
        configs = [
            {
                "name": args.name or args.batch,
                "program": "fake",
                "seed": seed_of(o),
                "overrides": [x for x in o if not x.startswith("seed=")],
            }
            for o in lists
        ]
    else:
        try:
            python = venv_python(conf.root, info["envkey"])
            configs = compose_configs(python, code_dir, lists)
        except (ValueError, RuntimeError) as e:
            raise SystemExit(str(e)) from None
    if any(c.get("phase") == "confirm" for c in configs) and (
        info["dirty"] or not info["mujou_pushed"]
    ):
        raise SystemExit("phase=confirm needs a clean snapshot with mujou pushed")
    gpu_mem = (
        args.gpu_mem if args.gpu_mem is not None else (0.0 if args.res == CPU else 2.0)
    )
    res = {"class": args.res, "ram_gb": args.ram, "gpu_mem_gb": gpu_mem}
    code = {
        k: info[k]
        for k in ("snap", "base_head", "dirty", "mujou_snap", "mujou_pushed", "envkey")
    }
    taken = _taken(nas)
    extra = parse_set(args.set)
    n_new = 0
    for overrides, config in zip(lists, configs, strict=True):
        job = make_job(
            batch=args.batch,
            config=config,
            overrides=overrides,
            code=code,
            res=res,
            entry=args.entry,
            prio=args.prio,
            after=args.after,
            host=args.host,
            avoid_host=args.avoid_host,
            tags=args.tag,
            extra=extra,
        )
        dup = job["run_id"] in taken
        print(f"{'duplicate' if dup else 'new':<9}  {job['run_id']}  {job['label']}")
        if not dup and not args.dry:
            # A failed or stopped record of the run goes, so the new one shows.
            nas.drop_done(job["run_id"])
            nas.submit(job)
            taken.add(job["run_id"])
            n_new += 1
    print(f"{n_new} queued{' (dry)' if args.dry else ''}")
    return 0


def _taken(nas: Nas) -> set[str]:
    """The run_ids queued, claimed, or done for good (completed or diverged)."""
    taken = {QueueName.parse(p.name).run_id for p in nas.entries()}
    taken |= {QueueName.parse(p.name).run_id for _, p in nas.all_claims()}
    for p in nas.done.glob("*.json"):
        rec = read_json(p)
        if rec.get("status") in FINAL_OK:
            taken.add(rec["run_id"])
    return taken


def _gather(nas: Nas, batch: str | None) -> list[dict[str, Any]]:
    """Every run as ``{"state", "host", "job", "path"}``: queued, running (a
    claim), and done (``state`` is the status). A queue or claim file wins over
    a done record of the same run (one left by an earlier attempt)."""
    rows: list[dict[str, Any]] = []
    done = {}
    for p in sorted(nas.done.glob("*.json")) if nas.done.exists() else []:
        rec = read_json(p)
        done[rec["run_id"]] = rec
    for p in nas.entries() if nas.queue.exists() else []:
        rows.append({"state": "queued", "host": None, "job": read_json(p), "path": p})
        done.pop(QueueName.parse(p.name).run_id, None)
    for host, p in nas.all_claims():
        rows.append({"state": "running", "host": host, "job": read_json(p), "path": p})
        done.pop(QueueName.parse(p.name).run_id, None)
    for rec in done.values():
        rows.append(
            {"state": rec["status"], "host": rec.get("host"), "job": rec, "path": None}
        )
    return [r for r in rows if batch is None or r["job"]["batch"] == batch]


def _heartbeats(conf: FleetConf, nas: Nas) -> dict[str, dict[str, Any] | None]:
    out: dict[str, dict[str, Any] | None] = {}
    for host in conf.hosts:
        try:
            out[host] = read_json(nas.hosts / f"{host}.json")
        except OSError, ValueError:
            out[host] = None
    return out


def _live(nas: Nas, run_id: str) -> dict[str, Any]:
    d = nas.live / run_id
    out: dict[str, Any] = {}
    try:
        out["status"] = read_json(d / "status.json")
    except OSError, ValueError:
        pass
    try:
        lines = (d / "evals.jsonl").read_text().splitlines()
        out["eval"] = json.loads(lines[-1]) if lines else None
    except OSError, ValueError:
        pass
    return out


def _status(conf: FleetConf, args: argparse.Namespace) -> int:
    nas = Nas(conf.nas)
    now = time.time()
    rows = _gather(nas, args.batch)
    beats = _heartbeats(conf, nas)
    stale = {h for h, b in beats.items() if b is None or now - b["time"] > LOST_AFTER}
    for r in rows:
        if r["state"] == "running" and r["host"] in stale:
            r["state"] = "lost?"
    counts = Counter(r["state"] for r in rows)
    running = [r for r in rows if r["state"] in ("running", "lost?")]
    if args.json:
        out = {
            "counts": dict(counts),
            "hosts": beats,
            "cordons": {h: nas.cordoned(h) for h in conf.hosts},
            "holds": sorted(p.name for p in nas.holds.glob("*"))
            if nas.holds.exists()
            else [],
            "running": [
                {
                    "run_id": r["job"]["run_id"],
                    "host": r["host"],
                    **_live(nas, r["job"]["run_id"]),
                }
                for r in running
            ],
        }
        print(json.dumps(out, indent=1))
        return 0
    print("  ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no runs")
    for host, b in beats.items():
        cordon = nas.cordoned(host)
        extra = f"  cordoned {cordon}" if cordon else ""
        if b is None:
            print(f"{host}  no heartbeat{extra}")
            continue
        gpus = " ".join(
            f"{g['model']} {g['jobs'][0]}/{g['jobs'][1]}" for g in b["gpus"]
        )
        print(
            f"{host}  beat {now - b['time']:.0f} s ago  cpu {b['cpu'][0]}/{b['cpu'][1]}  {gpus}"
            f"  ram {b['ram_gb'][0]:.0f}/{b['ram_gb'][1]:.0f} GB  load {b['load'][0]:.1f}"
            f"  snap {b['snap']}{extra}"
        )
    if nas.holds.exists():
        for p in sorted(nas.holds.glob("*")):
            print(f"hold {p.name}: {read_json(p).get('reason')}")
    for r in running:
        job, live = r["job"], _live(nas, r["job"]["run_id"])
        st, ev = live.get("status", {}), live.get("eval") or {}
        progress = (
            f"t {st.get('t', 0):.0f}/{st.get('of', 0):.0f} fps {st.get('fps') or 0:.0f}"
            f" eta {_hms(st.get('eta'))}"
            if st
            else "no live status yet"
        )
        metric = f"  roll10/cv10 {ev['roll10/cv10']:.3f}" if "roll10/cv10" in ev else ""
        print(f"  {r['state']:<7} {r['host']}  {job['run_id']}  {progress}{metric}")
    return 0


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def _ls(conf: FleetConf, args: argparse.Namespace) -> int:
    rows = _gather(Nas(conf.nas), args.batch)
    if args.state:
        rows = [r for r in rows if r["state"] == args.state]
    if args.json:
        print(
            json.dumps(
                [{k: v for k, v in r.items() if k != "path"} for r in rows], indent=1
            )
        )
        return 0
    for r in rows:
        job = r["job"]
        reason = f" ({job['reason']})" if job.get("reason") else ""
        print(
            f"{r['state']:<10} {r['host'] or '-':<7} {job['batch']:<20} {job['run_id']}"
            f"  {job.get('label', '')}{reason}"
        )
    return 0


def _locate(conf: FleetConf, args: argparse.Namespace) -> int:
    nas = Nas(conf.nas)
    for r in _gather(nas, None):
        job = r["job"]
        if job["run_id"] != args.run_id:
            continue
        run_dir = conf.root / "runs" / job["batch"] / job["run_id"]
        loc = {
            "state": r["state"],
            "host": r["host"],
            "attempt": job.get("attempt", 1),
            "run_dir": str(run_dir),
            "nas_dir": str(nas.runs / job["batch"] / job["run_id"]),
        }
        print(json.dumps(loc))
        return 0
    print(f"no run {args.run_id}", file=sys.stderr)
    return 1


def _select(nas: Nas, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not (args.run_ids or args.batch or args.trial):
        raise SystemExit("name runs, --batch or --trial")
    study, _, trial = (args.trial or "").partition("/")
    out = []
    for r in _gather(nas, None):
        job = r["job"]
        if r["state"] not in ("queued", "running"):
            continue
        if (
            job["run_id"] in args.run_ids
            or (args.batch is not None and job["batch"] == args.batch)
            or (
                args.trial
                and job.get("study") == study
                and str(job.get("trial")) == trial
            )
        ):
            out.append(r)
    return out


def _stop(conf: FleetConf, args: argparse.Namespace, *, reason: str, now: bool) -> int:
    nas = Nas(conf.nas)
    for r in _select(nas, args):
        job = r["job"]
        if r["state"] == "queued":
            # Taken out of the queue first: a worker may claim it meanwhile.
            taken = nas.take_back(r["path"])
            if taken is not None:
                record = {**job, "status": "stopped", "reason": reason, "host": None}
                nas.finish(taken, record)
                print(f"stopped (was queued)  {job['run_id']}")
                continue
            claimed = [
                (h, p)
                for h, p in nas.all_claims()
                if QueueName.parse(p.name).run_id == job["run_id"]
            ]
            if not claimed:
                print(f"gone from the queue  {job['run_id']}")
                continue
            r = {**r, "host": claimed[0][0], "path": claimed[0][1]}
        nas.request_stop(r["path"], job["run_id"], reason=reason, now=now)
        print(f"stop requested ({reason})  {job['run_id']} on {r['host']}")
    return 0


def _retry(conf: FleetConf, args: argparse.Namespace) -> int:
    nas = Nas(conf.nas)
    rec = nas.done_record(args.run_id)
    if rec is None:
        raise SystemExit(f"{args.run_id} is not in done/")
    job = {
        k: v
        for k, v in rec.items()
        if k not in ("status", "reason", "host", "exit", "slot", "ended")
    }
    job["attempt"] = int(rec.get("attempt", 1)) + 1
    if args.fresh:
        job["fresh"] = True
    if args.ram is not None:
        job["res"] = {**job["res"], "ram_gb": args.ram}
    if args.gpu_mem is not None:
        job["res"] = {**job["res"], "gpu_mem_gb": args.gpu_mem}
    nas.submit(job)
    (nas.done / f"{args.run_id}.json").unlink()
    print(f"queued again  {args.run_id} (attempt {job['attempt']})")
    return 0


def _requeue(conf: FleetConf, args: argparse.Namespace) -> int:
    nas = Nas(conf.nas)
    for claim in nas.claims(args.host):
        job = read_json(claim)
        job["attempt"] = int(job.get("attempt", 1)) + 1
        job["fresh"] = True
        nas.requeue(claim, job)
        print(f"requeued  {job['run_id']} (from {args.host})")
    return 0


def _hold(conf: FleetConf, args: argparse.Namespace) -> int:
    nas = Nas(conf.nas)
    nas.ensure()
    write_json(nas.holds / args.batch, {"reason": "fleet hold", "at": time.time()})
    return 0


def _release(conf: FleetConf, args: argparse.Namespace) -> int:
    nas = Nas(conf.nas)
    (nas.holds / args.batch).unlink(missing_ok=True)
    shutil.rmtree(nas.failures / args.batch, ignore_errors=True)
    return 0


def _cordon(conf: FleetConf, args: argparse.Namespace) -> int:
    Nas(conf.nas).cordon(args.host, args.res or "all", "fleet cordon")
    return 0


def _uncordon(conf: FleetConf, args: argparse.Namespace) -> int:
    Nas(conf.nas).uncordon(args.host, args.res)
    return 0


def _table(conf: FleetConf, args: argparse.Namespace) -> int:
    """``exp.lib.runs``'s table over the runs on the NAS."""
    from exp.lib import runs  # stdlib only; read only when asked

    argv = ["table", "--root", str(conf.nas), *args.table_argv]
    return runs.main(
        argv, command=shlex.join(["just", "fleet", "table", *args.table_argv])
    )


def _lock_held(path: Path) -> bool:
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def _wait(cond: Callable[[], bool], timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.2)
    return cond()


def _up_local(conf: FleetConf, args: argparse.Namespace) -> int:
    conf.root.mkdir(parents=True, exist_ok=True)
    lock, pidf = conf.root / "worker.lock", conf.root / "worker.pid"
    if _lock_held(lock):
        pid = int(pidf.read_text())
        if not args.restart:
            print(f"worker running (pid {pid})")
            return 0
        os.kill(pid, signal.SIGTERM)
        if not _wait(lambda: not _lock_held(lock), 60):
            print(f"worker pid {pid} did not stop", file=sys.stderr)
            return 1
    code = Path.cwd().resolve()
    with (conf.root / "worker.log").open("ab") as logf:
        fleet = [sys.executable, "-m", "exp.lib.fleet", "--local"]
        proc = subprocess.Popen(
            [*fleet, "--config", str(args.config), "worker", "--host", args.host],
            cwd=code,
            env={**os.environ, "PYTHONPYCACHEPREFIX": "/tmp/pycache"},
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    if (
        not _wait(lambda: _lock_held(lock) or proc.poll() is not None, 20)
        or proc.poll()
    ):
        print(
            f"the worker did not start (see {conf.root / 'worker.log'})",
            file=sys.stderr,
        )
        return 1
    print(f"worker started (pid {proc.pid}, code {code.name})")
    return 0


def _worker(conf: FleetConf, args: argparse.Namespace) -> int:
    conf.root.mkdir(parents=True, exist_ok=True)
    lockf = (conf.root / "worker.lock").open("a")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("a worker is already running here", file=sys.stderr)
        return 0
    (conf.root / "worker.pid").write_text(f"{os.getpid()}\n")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    timing = Timing(tick=args.tick, live=args.live, grace=args.grace, stall=args.stall)
    logging.getLogger("fleet.worker").info(
        "worker %s up (pid %d)", args.host, os.getpid()
    )
    Worker(conf, args.host, timing=timing).serve(stop)
    logging.getLogger("fleet.worker").info("worker %s down", args.host)
    return 0


def _provision(conf: FleetConf, args: argparse.Namespace) -> int:
    key = provision(
        conf.root,
        args.tree,
        json.loads(args.info),
        gpus=None if args.gpus < 0 else args.gpus,
    )
    print(f"code {args.tree.name} venv {key}")
    return 0


_LOCAL_COMMANDS: dict[str, Callable[[FleetConf, argparse.Namespace], int]] = {
    "submit": _submit,
    "status": _status,
    "ls": _ls,
    "locate": _locate,
    "stop": lambda c, a: _stop(c, a, reason=a.reason, now=a.now),
    "cancel": lambda c, a: _stop(c, a, reason="cancel", now=True),
    "retry": _retry,
    "requeue": _requeue,
    "hold": _hold,
    "release": _release,
    "cordon": _cordon,
    "uncordon": _uncordon,
    "table": _table,
    "worker": _worker,
    "up-local": _up_local,
    "provision": _provision,
}
