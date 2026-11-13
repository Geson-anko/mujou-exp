"""``fleet.toml``: the hosts, the paths and the slots of the fleet."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["FleetConf", "Gpu", "SlotsConf", "default_path"]


@dataclass(frozen=True, kw_only=True)
class Gpu:
    """One GPU of a host: CUDA ``index``, ``model`` (a resource class such as
    ``"4090"``), the memory runs may take, and how many runs it takes at most."""

    index: int
    model: str
    mem_gb: float
    jobs: int

    def matches(self, name: str) -> bool:
        """Whether ``name`` (nvidia-smi's, such as ``"NVIDIA GeForce RTX 5070
        Ti"``) is of this model: ``model`` in it, without spaces or case."""
        return self.model.lower() in name.replace(" ", "").lower()


@dataclass(frozen=True, kw_only=True)
class SlotsConf:
    """What one host gives to runs: ``cpu`` physical cores (the others are left
    to the worker), both SMT threads of a core as two slots when ``smt``, a RAM
    budget, and the GPUs."""

    cpu: int
    smt: bool
    ram_gb: float
    gpus: tuple[Gpu, ...]

    @property
    def models(self) -> tuple[str, ...]:
        """The resource classes of the GPUs, in index order."""
        return tuple(g.model for g in self.gpus)


@dataclass(frozen=True, kw_only=True)
class FleetConf:
    """The fleet. ``root`` is host-local (bind-mounted at the same path in the
    container), ``nas`` is shared by all hosts; ``head`` answers the state
    queries and composes the configs at submission. ``slots`` has every host:
    the ``[slots]`` table, with the keys of ``[slots.<host>]`` replacing its own
    for that host."""

    nas: Path
    root: Path
    checkout: Path
    container: str
    user: str
    head: str
    hosts: tuple[str, ...]
    slots: Mapping[str, SlotsConf]

    @property
    def models(self) -> tuple[str, ...]:
        """The resource classes of the GPUs of all hosts, each once, in host
        order."""
        return tuple(dict.fromkeys(m for s in self.slots.values() for m in s.models))

    @classmethod
    def load(cls, path: Path) -> FleetConf:
        """Read ``path`` (TOML)."""
        with path.open("rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FleetConf:
        """The ``FleetConf`` of the parsed TOML ``d``; ``KeyError`` names a
        missing key."""
        hosts = tuple(str(h) for h in d["hosts"])
        if d["head"] not in hosts:
            raise ValueError(f"head {d['head']!r} is not one of the hosts {hosts}")
        table = dict(d["slots"])
        own = {k: table.pop(k) for k, v in list(table.items()) if isinstance(v, dict)}
        if unknown := sorted(set(own) - set(hosts)):
            raise ValueError(f"[slots.<host>] of {unknown}: not hosts {hosts}")
        slots = {h: _slots({**table, **own.get(h, {})}) for h in hosts}
        return cls(
            nas=Path(d["nas"]),
            root=Path(d["root"]),
            checkout=Path(d["checkout"]),
            container=str(d["container"]),
            user=str(d["user"]),
            head=str(d["head"]),
            hosts=hosts,
            slots=slots,
        )


def _slots(s: Mapping[str, Any]) -> SlotsConf:
    return SlotsConf(
        cpu=int(s["cpu"]),
        smt=bool(s.get("smt", False)),
        ram_gb=float(s["ram_gb"]),
        gpus=tuple(
            Gpu(
                index=int(g["index"]),
                model=str(g["model"]),
                mem_gb=float(g["mem_gb"]),
                jobs=int(g["jobs"]),
            )
            for g in s.get("gpus", [])
        ),
    )


def default_path() -> Path:
    """``fleet.toml`` at the root of the checkout or snapshot this module is in."""
    return Path(__file__).resolve().parents[3] / "fleet.toml"
