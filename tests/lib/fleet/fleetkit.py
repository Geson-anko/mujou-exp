"""A fleet in temporary directories: a NAS, host roots with a ready snapshot
whose code is this checkout's ``exp`` and whose venv is this interpreter, and
jobs of the fake program."""

import shutil
import sys
import sysconfig
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from exp.lib.fleet.config import FleetConf
from exp.lib.fleet.jobs import make_job
from exp.lib.fleet.queue import Nas
from exp.lib.fleet.worker import Timing, Worker

REPO = Path(__file__).resolve().parents[3]
SNAP = "ab" * 20
ENVKEY = "e" * 12
CORES = [
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]  # 4 physical cores; slots.cpu=2 gives (2,6), (3,7)
# The stall is long: a loaded machine may take seconds to start an interpreter.
# The stall test sets its own.
FAST = Timing(tick=0.05, live=0.0, grace=0.3, kill=0.3, stall=60.0, absent=2)

TOML = """\
nas = "{nas}"
root = "{root}"
checkout = "{repo}"
container = "c"
user = "u"
head = "h0"
hosts = ["h0", "h1"]

[slots]
cpu = 2
smt = false
ram_gb = 8
gpus = [{{index = 0, model = "4090", mem_gb = 22.0, jobs = 2}},
        {{index = 1, model = "4060", mem_gb = 7.0, jobs = 1}}]
"""


def ready_root(root: Path, config: Path | None = None) -> None:
    """A host root with the snapshot ``SNAP`` and the venv ``ENVKEY`` ready."""
    code = root / "code" / SNAP[:12]
    code.mkdir(parents=True, exist_ok=True)
    if not (code / "exp").exists():
        (code / "exp").symlink_to(REPO / "exp")
    (code / ".ready").write_text(SNAP)
    (code / ".envkey").write_text(ENVKEY + "\n")
    (code / "CODE.json").write_text(
        '{"snap": "%s", "base_head": "%s", "dirty": false, "mujou_snap": "%s",'
        ' "mujou_base": "%s", "mujou_pushed": true, "envkey": "%s"}'
        % (SNAP, SNAP, "cd" * 20, "cd" * 20, ENVKEY)
    )
    if config is not None and not (code / "fleet.toml").exists():
        (code / "fleet.toml").symlink_to(config)
    latest = root / "code" / "latest"
    if not latest.exists():
        latest.symlink_to(code.name)
    venv = (
        root / "venv" / ENVKEY
    )  # this interpreter, with the site-packages of its venv
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    if not (venv / "bin" / "python").exists():
        (venv / "bin" / "python").symlink_to(sys.executable)
        shutil.copyfile(Path(sys.prefix) / "pyvenv.cfg", venv / "pyvenv.cfg")
        site = Path(sysconfig.get_path("purelib"))
        (venv / site.relative_to(sys.prefix)).parent.mkdir(parents=True, exist_ok=True)
        (venv / site.relative_to(sys.prefix)).symlink_to(site)
    (venv / ".ready").write_text(ENVKEY)


class Fleet:
    """The fleet of the tests: ``conf`` (host h0) and ``conf_h1`` share ``nas``."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.toml = tmp / "fleet.toml"
        self.toml.write_text(TOML.format(nas=tmp / "nas", root=tmp / "h0", repo=REPO))
        self.conf = FleetConf.load(self.toml)
        self.conf_h1 = replace(self.conf, root=tmp / "h1")
        for c in (self.conf, self.conf_h1):
            ready_root(c.root, self.toml)
        self.nas = Nas(self.conf.nas)
        self.nas.ensure("h0")
        self.nas.ensure("h1")

    def worker(self, host: str = "h0", timing: Timing = FAST) -> Worker:
        conf = self.conf if host == "h0" else self.conf_h1
        w = Worker(conf, host, timing=timing, cores=CORES)
        w.recover()
        return w

    def submit(self, *, batch: str = "b", seed: int = 0, **fake: Any) -> dict[str, Any]:
        """Queue a fake job; ``res``, ``ram_gb``, ``host`` and ``after`` go to the
        job, the rest to its ``fake`` behavior."""
        res = {
            "class": fake.pop("res", "cpu"),
            "ram_gb": fake.pop("ram_gb", 2.0),
            "gpu_mem_gb": fake.pop("gpu_mem_gb", 0.0),
        }
        job = make_job(
            batch=batch,
            config={"name": fake.pop("name", "t"), "program": "fake", "seed": seed},
            overrides=[f"seed={seed}"],
            code={"snap": SNAP, "envkey": ENVKEY, "dirty": False},
            res=res,
            entry="exp.lib.fleet.fake",
            host=fake.pop("host", None),
            after=fake.pop("after", ()),
            extra={"fake": {"seconds": 0.3, "frames": 10, "resume_every": 2, **fake}},
        )
        self.nas.submit(job)
        return job

    def finished(self, run_id: str) -> bool:
        return self.nas.done_record(run_id) is not None

    def done(self, run_id: str) -> dict[str, Any]:
        rec = self.nas.done_record(run_id)
        assert rec is not None, f"{run_id} is not done"
        return rec


def run_until(
    workers: list[Worker], cond: Callable[[], bool], timeout: float = 30.0
) -> None:
    """Tick ``workers`` until ``cond()``; fail after ``timeout`` seconds."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for w in workers:
            w.tick()
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def wait_for(cond: Callable[[], bool], timeout: float = 30.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")
