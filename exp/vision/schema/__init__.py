"""The structured configs of a run, as plain dataclasses (no torch).

A kind is a ``str`` checked in ``__post_init__`` (OmegaConf matches an Enum by
its member's name only), a sequence is a ``list``, and nothing is frozen
(overrides write to the nodes); the code does not change a config after
``OmegaConf.to_object``. Importing this package stores ``RunConf`` in hydra's
``ConfigStore`` as ``run_schema``, the first entry of the defaults of
``conf/run.yaml``.
"""

from hydra.core.config_store import ConfigStore

from exp.vision.schema.data import DataConf, ToursConf
from exp.vision.schema.measure import MeasureConf
from exp.vision.schema.model import ModelConf
from exp.vision.schema.optim import OptimConf
from exp.vision.schema.run import CtlConf, RunConf, TimeConf

__all__ = [
    "CtlConf",
    "DataConf",
    "MeasureConf",
    "ModelConf",
    "OptimConf",
    "RunConf",
    "TimeConf",
    "ToursConf",
]

ConfigStore.instance().store(name="run_schema", node=RunConf)
