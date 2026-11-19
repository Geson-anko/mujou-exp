"""The settings of one run: the root of the config tree."""

import re
from dataclasses import dataclass, field

from omegaconf import MISSING

from exp.vision.schema._check import check_choice
from exp.vision.schema.data import DataConf
from exp.vision.schema.measure import MeasureConf
from exp.vision.schema.model import ModelConf
from exp.vision.schema.optim import OptimConf

__all__ = [
    "PHASES",
    "PROGRAMS",
    "TASKS",
    "CtlConf",
    "RunConf",
    "TimeConf",
    "check_learn_interval",
]

PROGRAMS = ("vae",)
# The posthoc evaluation of a run is not a task: ``python -m exp.vision.eval``.
TASKS = ("online", "bench")
PHASES = ("explore", "confirm")
_DEVICE = re.compile(r"cpu|cuda(:\d+)?")


@dataclass(kw_only=True)
class TimeConf:
    """The times of the stream and of PC-ALM, in seconds of virtual time."""

    frame: float = 0.1  # the period of the frames (10 fps)
    h: float = 4e-3  # mujou.pc.TimeConstants(h=, lam=, r=)
    lam: float = 2e-3
    r: float = 4e-3
    # The time from a write to a node's learning step, and between two: 90 ms
    # makes every node learn exactly once a frame, after its credit has relaxed
    # (derived in exp.vision.pc's docstring for 0.1 s frames, 2 ms steps and
    # 8 hops: the window is (58, 100] ms; RunConf checks it: check_learn_interval).
    learn_interval: float = 0.09

    @property
    def period(self) -> float:
        """The step of the host, ``mujou.pc.TimeConstants(h, lam, r).period``
        (``h / (1 + h / r)``)."""
        return self.h / (1 + self.h / self.r)


def check_learn_interval(time: TimeConf, hops: int) -> None:
    """Check that with ``time.learn_interval`` every node of a chain of
    ``hops`` hops learns exactly once a frame, from BP's credit (derived in
    ``exp.vision.pc``'s docstring):

    - ``2 hops`` steps <= ``learn_interval``: the credit has reached the first
      node;
    - (frame + ``hops`` steps) / 2 < ``learn_interval``: the second learn of
      the last node comes after the next write has reached it;
    - ``learn_interval`` <= frame: the first node learns before the next
      write reaches it (in frame 0 too).

    At the defaults (0.1 s frames, 2 ms steps, 8 hops) the window is
    ``(0.058, 0.1]``.

    Raises:
        ValueError: ``learn_interval`` is outside the window.
    """
    li, frame, period = time.learn_interval, time.frame, time.period
    credit, once = 2 * hops * period, (frame + hops * period) / 2
    if not (credit <= li and once < li <= frame):
        raise ValueError(
            f"time.learn_interval {li} is outside ({max(credit, once):g}, "
            f"{frame:g}]: there not every node learns once a frame from BP's "
            "credit (exp.vision.pc)"
        )


@dataclass(kw_only=True)
class CtlConf:
    """What a run records and how often, in seconds of virtual time (0: never).
    None of it changes the metrics of the validation stacks, so it stays out of
    the hash of the run's group."""

    log_every: float = 1.0  # a line of stream.jsonl
    resume_every: float = 300.0  # a checkpoint to resume from
    eval_test: bool = False  # also evaluate the test stacks at each checkpoint
    # replay the vjp of the pc.Layer nodes from CUDA graphs when on CUDA
    # (the same updates bit for bit, fewer launches)
    cuda_graph: bool = True


@dataclass(kw_only=True)
class RunConf:
    """One run: ``program`` doing ``task`` on the stream ``data`` for
    ``measure.duration`` seconds of virtual time."""

    program: str = "vae"
    task: str = "online"
    name: str = MISSING  # set by the preset, e.g. "v0/base"
    tags: list[str] = field(default_factory=list)
    seed: int = 0
    device: str = "cpu"  # "cpu" or "cuda[:i]"; the worker sets the slot's
    phase: str = "explore"
    time: TimeConf = field(default_factory=TimeConf)
    data: DataConf = field(default_factory=DataConf)
    model: ModelConf = field(default_factory=ModelConf)
    optim: OptimConf = field(default_factory=OptimConf)
    measure: MeasureConf = field(default_factory=MeasureConf)
    ctl: CtlConf = field(default_factory=CtlConf)

    def __post_init__(self) -> None:
        check_choice("program", self.program, PROGRAMS)
        check_choice("task", self.task, TASKS)
        check_choice("phase", self.phase, PHASES)
        if not _DEVICE.fullmatch(self.device):
            raise ValueError(f"device must be 'cpu' or 'cuda[:i]', not {self.device!r}")
        if not self.time.frame > 0:
            raise ValueError(f"time.frame must be > 0, not {self.time.frame}")
        hops = self.model.enc_nodes + self.model.dec_nodes + 2  # latent, out
        check_learn_interval(self.time, hops)
        size = self.data.tours.size
        if self.model.arch == "vit" and size % self.model.patch:
            raise ValueError(
                f"data.tours.size {size} is not a multiple of model.patch "
                f"{self.model.patch}"
            )
        if self.model.arch == "resnet" and size % 16:
            raise ValueError(
                f"data.tours.size {size} is not a multiple of 16 (4 strides of 2)"
            )
        for t in [*self.measure.ckpt_times, self.measure.duration]:
            k = t / self.time.frame
            if abs(k - round(k)) > 1e-6:
                raise ValueError(
                    f"{t} s is not a whole number of frames of {self.time.frame} s"
                )
