"""The settings of the learning: the learner, the optimizers and the offline
bound."""

from dataclasses import dataclass

from exp.video.schema._check import check_choice

__all__ = ["LEARNERS", "OPTIMIZERS", "LearnerConf", "OfflineConf", "OptimConf"]

LEARNERS = ("pc", "bp", "none")
OPTIMIZERS = ("sgd", "adamw", "graft_adam")


@dataclass(kw_only=True)
class LearnerConf:
    """Who learns: PC-ALM nodes (``"pc"``), the same modules by BP in one node
    (``"bp"``), or nothing (``"none"``, the baselines only)."""

    kind: str = "bp"

    def __post_init__(self) -> None:
        check_choice("learner.kind", self.kind, LEARNERS)


@dataclass(kw_only=True)
class OptimConf:
    """The optimizer of the nodes, a ``mujou.pc.optim`` one: ``"sgd"``,
    ``"adamw"`` or ``"graft_adam"`` (the direction of Adam with the step size of
    SGD, ``exp.lib.optim.Grafted(AdamDirection)``).

    ``lr`` is the learning rate per frame; the optimizer is built with the
    time constant ``tau = frame / lr`` of the parameters.
    """

    kind: str = "sgd"
    lr: float = 0.03  # per frame
    warmup: float = 300.0  # seconds of virtual time the rate rises over; 0: none
    tau_m: float = 0.3  # the moments of Adam, in seconds
    tau_v: float = 30.0
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        check_choice("optim.kind", self.kind, OPTIMIZERS)


@dataclass(kw_only=True)
class OfflineConf:
    """The offline bound (``task="offline"``): the same modules fit by Adam on
    windows cut from the streams, through time (BPTT) or, with ``local``, with
    the input and the fed-back prediction detached (one frame, as the nodes)."""

    frames: int = 1024  # of each stream to cut the windows from
    window: int = 32  # frames per window
    steps: int = 6000
    batch: int = 128  # windows per step
    lr: float = 1e-3
    local: bool = False
    every: int = 1000  # steps between two evaluations
