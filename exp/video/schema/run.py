"""The settings of one run: the root of the config tree."""

import re
from dataclasses import dataclass, field

from omegaconf import MISSING

from exp.video.schema._check import check_choice
from exp.video.schema.data import DataConf
from exp.video.schema.dyn import DynamicsConf
from exp.video.schema.learn import LearnerConf, OfflineConf, OptimConf
from exp.video.schema.measure import MeasureConf
from exp.video.schema.vision import VisionConf

__all__ = ["PHASES", "PROGRAMS", "TASKS", "CtlConf", "RunConf", "TimeConf"]

PROGRAMS = ("dynamics", "vision", "joint")
TASKS = ("online", "offline", "baselines", "bench", "posthoc", "prep")
PHASES = ("explore", "confirm")
_DEVICE = re.compile(r"cpu|cuda(:\d+)?")


@dataclass(kw_only=True)
class TimeConf:
    """The times of the stream and of PC-ALM, in seconds of virtual time."""

    frame: float = 1 / 30  # the period of the frames (30 fps)
    h: float = 4e-3  # mujou.pc.TimeConstants(h=, lam=, r=)
    lam: float = 2e-3
    r: float = 4e-3
    learn_interval: float | None = 25e-3  # None: learn in every update


@dataclass(kw_only=True)
class CtlConf:
    """What a run records and how often, in seconds of virtual time (0: never).
    None of it changes the metrics of the validation clips, so it stays out of
    the hash of the run's group."""

    log_every: float = 1.0  # a line of stream.jsonl
    ckpt_every: float = 0.0  # an evaluation checkpoint (the modules' weights)
    resume_every: float = 300.0  # a checkpoint to resume from
    viz_every: float = 0.0  # the pictures to look at
    eval_test: bool = False  # also evaluate the test clips at each checkpoint
    # replay the vjp of the vision's pc.Layer nodes from CUDA graphs when on CUDA
    # (pc.Layer(cuda_graph=True): the same updates bit for bit, fewer launches)
    cuda_graph: bool = True


@dataclass(kw_only=True)
class RunConf:
    """One run: ``program`` (the dynamics of M14, the vision of M15, or both of
    M16) doing ``task`` on the stream ``data`` for ``seconds`` of virtual time.

    A program needs its subtrees: ``dynamics`` for ``"dynamics"`` and
    ``"joint"``; ``vision`` and ``optim_vision`` for ``"vision"`` and
    ``"joint"``; ``offline`` for ``task="offline"``. ``optim`` is the
    optimizer of the dynamics, ``optim_vision`` that of the encoder.
    """

    program: str = "dynamics"
    task: str = "online"
    name: str = MISSING  # set by the preset, e.g. "m14/s1"
    tags: list[str] = field(default_factory=list)
    seed: int = 0
    device: str = "cpu"  # "cpu" or "cuda[:i]"; the worker sets the slot's
    phase: str = "explore"
    seconds: float = 1800.0  # the length of the stream
    time: TimeConf = field(default_factory=TimeConf)
    data: DataConf = field(default_factory=DataConf)
    dynamics: DynamicsConf | None = None
    vision: VisionConf | None = None
    learner: LearnerConf = field(default_factory=LearnerConf)
    optim: OptimConf = field(default_factory=OptimConf)
    optim_vision: OptimConf | None = None
    offline: OfflineConf | None = None
    measure: MeasureConf = field(default_factory=MeasureConf)
    ctl: CtlConf = field(default_factory=CtlConf)

    def __post_init__(self) -> None:
        check_choice("program", self.program, PROGRAMS)
        check_choice("task", self.task, TASKS)
        check_choice("phase", self.phase, PHASES)
        if not _DEVICE.fullmatch(self.device):
            raise ValueError(f"device must be 'cpu' or 'cuda[:i]', not {self.device!r}")
        needs = {
            "dynamics": self.program in ("dynamics", "joint"),
            "vision": self.program in ("vision", "joint"),
            "optim_vision": self.program in ("vision", "joint"),
            "offline": self.task == "offline",
        }
        for name, needed in needs.items():
            if needed and getattr(self, name) is None:
                raise ValueError(
                    f"program {self.program!r} with task {self.task!r} needs {name}"
                )
        v, m = self.vision, self.measure
        if v is not None and v.layout == "packed" and max(m.horizons) > m.steps:
            # The windows of the evaluation have measure.steps stacks after the
            # first: the linear dynamics afterwards cannot look further.
            raise ValueError(
                f"measure.horizons {m.horizons} go beyond measure.steps {m.steps}"
            )
        window = self.dynamics.window if self.dynamics is not None else None
        if window is not None and not window.relax < window.stride * self.time.frame:
            # Else every window waits past the next: learned late, at the push
            # of the next one, not relax after its own.
            raise ValueError(
                f"dynamics.window.relax ({window.relax} s) must be shorter than "
                f"stride * time.frame ({window.stride * self.time.frame} s)"
            )
