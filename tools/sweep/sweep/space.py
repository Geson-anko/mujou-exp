"""The search space of a study (the ``[space]`` table of its TOML, design 5.4).

Each entry is one optuna parameter::

    "optim"          = {categorical = ["sgd", "adamw", "graft_adam"]}
    "optim.lr"       = {float = [0.001, 0.3], log = true, when = "optim != 'adamw'"}
    "optim.lr@adamw" = {float = [1e-5, 1e-2], log = true, when = "optim == 'adamw'"}
    "dynamics.open_cycle" = {int = [10, 40], step = 5}

The name is the optuna parameter; the part before ``@`` is the hydra override
key, so two parameters (each with its own range and its own TPE model) may
set one key under exclusive conditions. ``when`` is a condition on the
parameters above it (``==``, ``!=``, ``in``, ``not in``, ``<``, ``and``,
``or``, ``not``; dotted names; Python literals); a parameter whose condition
is false is not suggested and gives no override. The parameters are suggested
in the order of the table (define-by-run), so a condition sees only earlier
ones.
"""

import ast
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import optuna

__all__ = ["Param", "Space", "override_value"]

_KINDS = ("categorical", "float", "int")
_KEYS = frozenset({*_KINDS, "log", "step", "when"})
_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,  # codespell:ignore
}


def override_value(value: Any) -> str:
    """``value`` as hydra's override grammar reads it back: ``true``/``false``,
    ``null``, ``repr`` of a number (the shortest text of a float that parses
    to the same float), a string as is."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return repr(value)
    return str(value)


class _Condition:
    """A ``when`` expression, checked when parsed and evaluated on the values
    of the parameters suggested before (``None`` for an inactive one)."""

    def __init__(self, text: str) -> None:
        self.text = text
        try:
            self._tree = ast.parse(text, mode="eval").body
        except SyntaxError as e:
            raise ValueError(f"when {text!r}: {e.msg}") from None
        self.names: set[str] = set()
        self._check(self._tree)

    def _check(self, node: ast.AST) -> None:
        match node:
            case ast.BoolOp(values=values):
                for v in values:
                    self._check(v)
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                self._check(operand)
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                if not all(type(op) in _COMPARE for op in ops):
                    raise ValueError(f"when {self.text!r}: an unknown comparison")
                for v in (left, *comparators):
                    self._check(v)
            case ast.Name() | ast.Attribute():
                self.names.add(_dotted(node, self.text))
            case ast.Constant():
                pass
            case ast.List(elts=elts) | ast.Tuple(elts=elts):
                for v in elts:
                    if not isinstance(v, ast.Constant):
                        raise ValueError(f"when {self.text!r}: a list of literals only")
            case _:
                raise ValueError(f"when {self.text!r}: not a condition")

    def __call__(self, values: Mapping[str, Any]) -> bool:
        return bool(self._eval(self._tree, values))

    def _eval(self, node: ast.AST, values: Mapping[str, Any]) -> Any:
        match node:
            case ast.BoolOp(op=ast.And(), values=vs):
                return all(self._eval(v, values) for v in vs)
            case ast.BoolOp(values=vs):
                return any(self._eval(v, values) for v in vs)
            case ast.UnaryOp(operand=operand):
                return not self._eval(operand, values)
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                a = self._eval(left, values)
                for op, right in zip(ops, comparators, strict=True):
                    b = self._eval(right, values)
                    if not _COMPARE[type(op)](a, b):
                        return False
                    a = b
                return True
            case ast.Name() | ast.Attribute():
                return values.get(_dotted(node, self.text))
            case ast.Constant(value=v):
                return v
            case ast.List(elts=elts) | ast.Tuple(elts=elts):
                return [self._eval(v, values) for v in elts]
        raise AssertionError("checked in _check")


def _dotted(node: ast.AST, text: str) -> str:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=value, attr=attr):
            return f"{_dotted(value, text)}.{attr}"
    raise ValueError(f"when {text!r}: not a parameter name")


@dataclass(frozen=True)
class Param:
    """One parameter: ``categorical`` over ``choices``, or ``float`` / ``int``
    in ``[low, high]`` (``log``: log-uniform; ``step``: a grid)."""

    name: str
    kind: str
    choices: tuple[Any, ...] = ()
    low: float = 0.0
    high: float = 0.0
    log: bool = False
    step: float | None = None
    when: _Condition | None = None

    @property
    def key(self) -> str:
        """The override key: the name without its ``@...`` suffix."""
        return self.name.split("@", 1)[0]

    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> Param:
        """The parameter of a ``[space]`` entry; ``ValueError`` when malformed."""
        unknown = set(d) - _KEYS
        kinds = [k for k in _KINDS if k in d]
        if unknown or len(kinds) != 1:
            raise ValueError(
                f"{name}: one of {_KINDS} and optionally log, step, when;"
                f" not {sorted(d)}"
            )
        kind = kinds[0]
        when = _Condition(str(d["when"])) if "when" in d else None
        if kind == "categorical":
            choices = tuple(d[kind])
            if not choices:
                raise ValueError(f"{name}: no choices")
            return cls(name, kind, choices=choices, when=when)
        bounds = d[kind]
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(f"{name}: {kind} = [low, high], not {bounds!r}")
        low, high = float(bounds[0]), float(bounds[1])
        log = bool(d.get("log", False))
        if not low < high or (log and low <= 0):
            raise ValueError(f"{name}: [{low}, {high}] is not a range (log: low > 0)")
        step = float(d["step"]) if "step" in d else None
        if log and step is not None and not (kind == "int" and step == 1):
            raise ValueError(f"{name}: log and step together (optuna refuses them)")
        return cls(name, kind, low=low, high=high, log=log, step=step, when=when)

    def suggest(self, trial: optuna.trial.BaseTrial) -> Any:
        """The value ``trial`` suggests for this parameter."""
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, list(self.choices))
        if self.kind == "int":
            step = 1 if self.step is None else int(self.step)
            return trial.suggest_int(
                self.name, int(self.low), int(self.high), step=step, log=self.log
            )
        return trial.suggest_float(
            self.name, self.low, self.high, step=self.step, log=self.log
        )


@dataclass(frozen=True)
class Space:
    """The parameters of a study, in the order they are suggested."""

    params: tuple[Param, ...]

    @classmethod
    def from_dict(cls, d: Mapping[str, Mapping[str, Any]]) -> Space:
        """The space of a ``[space]`` table; ``ValueError`` names what is wrong,
        including a condition on a name that is not an earlier parameter."""
        params: list[Param] = []
        for name, spec in d.items():
            p = Param.from_dict(name, spec)
            seen = {q.name for q in params}
            if p.when is not None and not p.when.names <= seen:
                missing = sorted(p.when.names - seen)
                raise ValueError(f"{name}: when names {missing}, not earlier params")
            params.append(p)
        return cls(tuple(params))

    def suggest(self, trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        """The active parameters' values, suggested by ``trial`` in order.
        ``ValueError`` when two active parameters set one override key."""
        values: dict[str, Any] = {}
        keys: dict[str, str] = {}
        for p in self.params:
            if p.when is not None and not p.when(values):
                continue
            if p.key in keys:
                raise ValueError(f"{p.name} and {keys[p.key]} both set {p.key}")
            keys[p.key] = p.name
            values[p.name] = p.suggest(trial)
        return values

    def overrides(self, params: Mapping[str, Any]) -> list[str]:
        """``key=value`` of each parameter in ``params``, in the space's order."""
        return [
            f"{p.key}={override_value(params[p.name])}"
            for p in self.params
            if p.name in params
        ]

    def check(self, params: Mapping[str, Any]) -> None:
        """``ValueError`` unless every name of ``params`` is a parameter (an
        ``[[enqueue]]`` entry)."""
        unknown = set(params) - {p.name for p in self.params}
        if unknown:
            raise ValueError(f"not parameters of the space: {sorted(unknown)}")

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """The ``[space]`` table this space was read from (normalized)."""
        out: dict[str, dict[str, Any]] = {}
        for p in self.params:
            d: dict[str, Any] = (
                {"categorical": list(p.choices)}
                if p.kind == "categorical"
                else {p.kind: [p.low, p.high], "log": p.log}
            )
            if p.step is not None:
                d["step"] = p.step
            if p.when is not None:
                d["when"] = p.when.text
            out[p.name] = d
        return out
