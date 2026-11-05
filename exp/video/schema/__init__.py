"""The structured configs of a run, as plain dataclasses (no torch).

A kind is a ``str`` checked in ``__post_init__`` (OmegaConf matches an Enum by
its member's name only), a sequence is a ``list``, a subtree that may be absent
is ``X | None``, and nothing is frozen (overrides write to the nodes); the code
does not change a config after ``OmegaConf.to_object``. Importing this package
stores ``RunConf`` in hydra's ``ConfigStore`` as ``run_schema``, the first
entry of the defaults of ``conf/run.yaml``.
"""

from hydra.core.config_store import ConfigStore

from exp.video.schema.data import BallsConf, DataConf, PhysicsConf, ToursConf
from exp.video.schema.dyn import DynamicsConf, WindowConf
from exp.video.schema.learn import LearnerConf, OfflineConf, OptimConf
from exp.video.schema.measure import MeasureConf
from exp.video.schema.run import CtlConf, RunConf, TimeConf
from exp.video.schema.vision import VisionConf

__all__ = [
    "BallsConf",
    "CtlConf",
    "DataConf",
    "DynamicsConf",
    "LearnerConf",
    "MeasureConf",
    "OfflineConf",
    "OptimConf",
    "PhysicsConf",
    "RunConf",
    "TimeConf",
    "ToursConf",
    "VisionConf",
    "WindowConf",
]

ConfigStore.instance().store(name="run_schema", node=RunConf)
