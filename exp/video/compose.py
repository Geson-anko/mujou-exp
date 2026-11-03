"""A thin wrap of hydra's Compose API: the overrides of a run to its
``RunConf``, and a ``RunConf`` to and from the plain dict of ``job.json``.

Only the Compose API is used (no ``@hydra.main``, no output directories, no
multirun). The code past this module sees ``RunConf`` instances only, never a
``DictConfig``.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from hydra import compose as hydra_compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf

from exp.video.schema import RunConf

__all__ = ["compose", "from_dict", "to_dict"]

CONFIG_MODULE = "exp.video.conf"


def compose(overrides: Sequence[str]) -> RunConf:
    """The ``RunConf`` of ``conf/run.yaml`` with hydra's ``overrides``, e.g.
    ``["+experiment=m14/s0", "optim=adamw", "seed=1"]``.

    Clears hydra's global state first, so that it can be called any number of
    times; not thread-safe (hydra's state is global). Raises
    ``ConfigCompositionException`` for an override hydra cannot apply, and the
    ``ValueError`` of a ``__post_init__`` for a value the schema rejects.

    A subtree that is ``None`` (``vision``, ``optim_vision``, ``offline``,
    ``data.tours``, ...) cannot take a ``key=value`` override (OmegaConf 2.3
    fails to merge into ``None``, and drops the override silently under
    ``python -O``); such an override raises ``ValueError``. Fill the subtree
    from a group or a preset first: ``vision=vit_t``,
    ``+optim@optim_vision=sgd``. A dict-valued field takes its entries one by
    one: ``+measure.criteria.roll10=0.5``, not ``measure.criteria={...}``.
    """
    overrides = list(overrides)
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        _check_no_override_into_none(overrides)
        cfg = hydra_compose(config_name="run", overrides=overrides)
    return _to_run_conf(cfg)


def to_dict(conf: RunConf) -> dict[str, Any]:
    """``conf`` as a dict of plain values (JSON-serializable)."""
    return dataclasses.asdict(conf)


def from_dict(d: Mapping[str, Any]) -> RunConf:
    """The ``RunConf`` of a dict of ``to_dict``, checked against the schema as
    ``compose`` checks it."""
    return _to_run_conf(OmegaConf.merge(OmegaConf.structured(RunConf), dict(d)))


def _check_no_override_into_none(overrides: list[str]) -> None:
    """Raise ``ValueError`` for a ``a.b.c=v`` override below a subtree that is
    ``None`` once the other overrides (the group choices among them) apply."""
    parsed = OverridesParser.create().parse_overrides(overrides)
    is_dotted = [
        "." in o.key_or_group and o.package is None and not o.is_delete()
        for o in parsed
    ]
    if not any(is_dotted):
        return
    rest = [line for line, d in zip(overrides, is_dotted, strict=True) if not d]
    cfg = hydra_compose(config_name="run", overrides=rest)
    dotted = [o for o, d in zip(parsed, is_dotted, strict=True) if d]
    for o in dotted:
        node: Any = cfg
        parts = o.key_or_group.split(".")
        for i, part in enumerate(parts[:-1]):
            if not isinstance(node, DictConfig) or part not in node:
                break  # hydra reports the unknown key
            node = node[part]
            if node is None:
                subtree = ".".join(parts[: i + 1])
                raise ValueError(
                    f"{o.input_line!r}: {subtree} is None; fill it first from a"
                    " group or a preset (e.g. vision=vit_t,"
                    " +optim@optim_vision=sgd)"
                )


def _to_run_conf(cfg: Any) -> RunConf:
    conf = OmegaConf.to_object(cfg)
    if not isinstance(conf, RunConf):
        raise TypeError(f"the config is a {type(conf).__name__}, not a RunConf")
    return conf
