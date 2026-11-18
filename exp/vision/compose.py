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
from omegaconf import OmegaConf

from exp.vision.schema import RunConf

__all__ = ["compose", "from_dict", "to_dict"]

CONFIG_MODULE = "exp.vision.conf"


def compose(overrides: Sequence[str]) -> RunConf:
    """The ``RunConf`` of ``conf/run.yaml`` with hydra's ``overrides``, e.g.
    ``["+experiment=v0/base", "optim=adamw", "seed=1"]``.

    Clears hydra's global state first, so that it can be called any number of
    times; not thread-safe (hydra's state is global). Raises
    ``ConfigCompositionException`` for an override hydra cannot apply, and the
    ``ValueError`` of a ``__post_init__`` for a value the schema rejects.

    A dict-valued field takes its entries one by one:
    ``+measure.criteria.stack_mse=0.5``, not ``measure.criteria={...}``.
    """
    overrides = list(overrides)
    GlobalHydra.instance().clear()
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        cfg = hydra_compose(config_name="run", overrides=overrides)
    return _to_run_conf(cfg)


def to_dict(conf: RunConf) -> dict[str, Any]:
    """``conf`` as a dict of plain values (JSON-serializable)."""
    return dataclasses.asdict(conf)


def from_dict(d: Mapping[str, Any]) -> RunConf:
    """The ``RunConf`` of a dict of ``to_dict``, checked against the schema as
    ``compose`` checks it."""
    return _to_run_conf(OmegaConf.merge(OmegaConf.structured(RunConf), dict(d)))


def _to_run_conf(cfg: Any) -> RunConf:
    conf = OmegaConf.to_object(cfg)
    if not isinstance(conf, RunConf):
        raise TypeError(f"the config is a {type(conf).__name__}, not a RunConf")
    return conf
