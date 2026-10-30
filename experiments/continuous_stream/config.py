"""The settings of one run, as frozen dataclasses that round-trip through JSON."""

import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "OptimConfig",
    "TimeConfig",
]


@dataclass(frozen=True, kw_only=True)
class TimeConfig:
    """The times of the stream and of PC-ALM, in seconds of the run's timer."""

    frame: float = 1 / 30  # the period of the frames (30 fps)
    h: float = 4e-3  # TimeConstants(h=, lam=, r=)
    lam: float = 2e-3
    r: float = 4e-3
    learn_interval: float | None = 25e-3  # None: learn in every update


@dataclass(frozen=True, kw_only=True)
class OptimConfig:
    """The optimizer of every node (PLAN.md, section 5).

    ``kind`` is one of ``"sgd"``, ``"adamw"`` (``pc.optim.AdamW``),
    ``"adam_direction"``, ``"whitened"`` and ``"projected"``.
    """

    kind: str = "sgd"
    tau: float = 1.0  # the time constant of the parameters
    graft: bool = False  # take the direction only, with the step size of SGD
    tau_m: float = 0.3  # the moments of AdamW
    tau_v: float = 30.0
    weight_decay: float = 0.0
    tau_l: float | None = None  # L = EMA of G G^T; None: no left factor
    tau_r: float | None = None  # R = EMA of G^T G; None: no right factor
    a: float = 0.25  # D = (L + eps)^-a G (R + eps)^-b
    b: float = 0.25
    rank: int = 4  # the eigenvectors that "projected" removes on each side
    eps: float = 1e-4  # relative to the largest eigenvalue
    refresh: float = 1.0  # seconds between two eigendecompositions


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    """The sizes of the network (PLAN.md, section 7)."""

    d_z: int = 64
    d_s: int = 256
    kappa: float = 0.05  # the weight of SIGReg
    directions: int = 256  # the directions of SIGReg, drawn anew each frame
    buffer: int = 255  # the past z SIGReg keeps


@dataclass(frozen=True, kw_only=True)
class DataConfig:
    """The stream. ``kind`` is ``"linear"`` (stage 0) or ``"video"`` (stages 1-2)."""

    kind: str = "video"
    seconds: float = 600.0  # the length of the stream, on the run's timer
    iid: bool = False  # draw the frames independently (no correlation in time)
    # The linear regression of stage 0.
    d_x: int = 32
    tau_x: float = 10.0  # the correlation time of the mean of x
    noise: float = 0.1
    # The synthetic video of stages 1-2.
    size: int = 32
    balls: int = 3
    physics_period: float = 600.0  # T of gamma(t) and e(t)
    episode_min: float = 30.0
    episode_max: float = 90.0


@dataclass(frozen=True, kw_only=True)
class Config:
    """The settings of one run."""

    name: str = "run"
    seed: int = 0
    time: TimeConfig = field(default_factory=TimeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return _from_dict(cls, d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Self:
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.from_json(path.read_text())


def _from_dict[T](cls: type[T], d: dict[str, Any]) -> T:
    """Build the dataclass ``cls`` from ``d``, its nested dataclasses too.

    Raises:
        TypeError: ``d`` has a key that is not a field of ``cls``.
    """
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = d.keys() - names
    if unknown:
        raise TypeError(f"{cls.__name__} has no fields {sorted(unknown)}")
    kwargs = {}
    for name, value in d.items():
        hint = hints[name]
        if isinstance(hint, type) and dataclasses.is_dataclass(hint):
            value = _from_dict(hint, value)
        elif hint is float or (
            isinstance(hint, types.UnionType) and float in hint.__args__
        ):
            value = value if value is None else float(value)  # JSON may give 1
        kwargs[name] = value
    return cls(**kwargs)
