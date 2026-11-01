"""What the runs of every experiment share about their settings: the type a
``Recorder`` and a sweep need, and the JSON to dataclass step."""

import dataclasses
import types
import typing
from typing import Any, Protocol

__all__ = ["RunConfig", "from_dict"]


class RunConfig(Protocol):
    """The settings of a run, as far as recording and sweeping it need."""

    @property
    def name(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


def from_dict[T](cls: type[T], d: dict[str, Any]) -> T:
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
            value = from_dict(hint, value)
        elif hint is float or (
            isinstance(hint, types.UnionType) and float in hint.__args__
        ):
            value = value if value is None else float(value)  # JSON may give 1
        kwargs[name] = value
    return cls(**kwargs)
