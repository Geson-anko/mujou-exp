"""Where a run goes on a host: a pure function of the slots and what is in use.

A slot is a unit of CPU (one physical core, or one SMT thread when
``smt=true``) and, for a GPU run, one GPU. Every run, GPU runs included, takes
one unit of CPU: a run is bound by one core (it launches the GPU's kernels).
"""

import os
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from exp.lib.fleet.config import SlotsConf
from exp.lib.fleet.jobs import GPU_ANY

__all__ = ["CPU", "Slot", "Use", "allocate", "physical_cores", "units"]

CPU = "cpu"  # the resource class of a run without GPU
ALL = "all"  # a cordon of every class of a host


@dataclass(frozen=True)
class Slot:
    """Logical ``cpus`` to pin the run to, and the CUDA index ``gpu`` of model
    ``model`` (``gpu`` is ``None`` and ``model`` ``"cpu"`` for a CPU run)."""

    cpus: tuple[int, ...]
    gpu: int | None
    model: str

    @property
    def device(self) -> str:
        """``"cuda"`` (the one visible GPU) or ``"cpu"``."""
        return "cpu" if self.gpu is None else "cuda"


@dataclass(frozen=True)
class Use:
    """A slot in use, with the RAM and GPU memory its run may take."""

    slot: Slot
    ram_gb: float
    gpu_mem_gb: float


def physical_cores(
    sys_cpu: Path = Path("/sys/devices/system/cpu"),
) -> list[tuple[int, ...]]:
    """The logical CPUs of each physical core, by ``thread_siblings_list``
    (``"0,16"`` or ``"0-1"``), sorted; one core per CPU where it cannot be read."""
    cores: set[tuple[int, ...]] = set()
    for f in sys_cpu.glob("cpu[0-9]*/topology/thread_siblings_list"):
        cpus: list[int] = []
        for part in f.read_text().strip().split(","):
            lo, _, hi = part.partition("-")
            cpus.extend(range(int(lo), int(hi or lo) + 1))
        cores.add(tuple(sorted(cpus)))
    if not cores:
        return [(i,) for i in range(os.cpu_count() or 1)]
    return sorted(cores)


def units(conf: SlotsConf, cores: Sequence[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """The CPU units for runs: the last ``conf.cpu`` physical ``cores`` (the
    lowest are left to the worker), split into their threads when ``smt``."""
    pool = list(cores)[-conf.cpu :] if conf.cpu > 0 else []
    if conf.smt:
        return [(c,) for core in pool for c in core]
    return pool


def allocate(
    conf: SlotsConf,
    cores: Sequence[tuple[int, ...]],
    used: Sequence[Use],
    *,
    res_class: str,
    ram_gb: float,
    gpu_mem_gb: float = 0.0,
    cordoned: Collection[str] = (),
) -> Slot | None:
    """A free slot for a run of ``res_class`` (``"cpu"``, ``"any_gpu"`` or a GPU
    model) that needs ``ram_gb`` and ``gpu_mem_gb``, or ``None``.

    The lowest free CPU unit; for a GPU run, the eligible GPU with the most
    memory left (the lowest index on a tie) whose count and memory stay within
    budget. A class in ``cordoned`` (or ``"all"``) is not given.
    """
    if ALL in cordoned:
        return None
    if sum(u.ram_gb for u in used) + ram_gb > conf.ram_gb:
        return None
    busy = {c for u in used for c in u.slot.cpus}
    cpus = next((u for u in units(conf, cores) if busy.isdisjoint(u)), None)
    if cpus is None:
        return None
    if res_class == CPU:
        return None if CPU in cordoned else Slot(cpus, None, CPU)
    best: tuple[float, int, str] | None = None
    for g in conf.gpus:
        if g.model in cordoned or res_class not in (GPU_ANY, g.model):
            continue
        on = [u for u in used if u.slot.gpu == g.index]
        left = g.mem_gb - sum(u.gpu_mem_gb for u in on)
        if len(on) >= g.jobs or left < gpu_mem_gb:
            continue
        if best is None or left > best[0]:
            best = (left, g.index, g.model)
    return None if best is None else Slot(cpus, best[1], best[2])
