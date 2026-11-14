"""The settings of the learning: the learner, the optimizers and the offline
bound."""

from dataclasses import dataclass, field

from exp.video.schema._check import check_choice

__all__ = [
    "CREDITS",
    "DECAYS",
    "FEEDBACKS",
    "LEARNERS",
    "OPTIMIZERS",
    "ORDERS",
    "LearnerConf",
    "OfflineConf",
    "OptimConf",
    "SCALES",
]

LEARNERS = ("pc", "bp", "none")
OPTIMIZERS = ("sgd", "adamw", "graft_adam", "rmsprop", "adamw_step")
DECAYS = ("constant", "cosine")  # exp.lib.optim.StepAdamW
SCALES = ("none", "fan_in")  # the learning rate per tensor (OptimConf.scale)
ORDERS = ("random", "stream")
FEEDBACKS = ("grad", "detach", "teacher")
CREDITS = ("bptt", "own")
# The moments' time constants: their defaults in OptimConf and the kinds that use them
_MOMENTS = {
    "tau_m": (0.3, ("adamw", "graft_adam")),
    "tau_v": (30.0, ("adamw", "graft_adam", "rmsprop")),
}


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
    ``"adamw"``, ``"graft_adam"`` (the direction of Adam with the step size of
    SGD, ``exp.lib.optim.Grafted(AdamDirection)``) or ``"rmsprop"`` (Adam
    without the first moment, ``exp.lib.optim.RMSPropDirection`` of ``tau_v``);
    or ``"adamw_step"``, AdamW by steps (``exp.lib.optim.StepAdamW``,
    ``torch.optim.AdamW``).

    ``lr`` is the learning rate per frame; the optimizer is built with the
    time constant ``tau = frame / lr`` of the parameters. For
    ``"adamw_step"`` it is the peak learning rate per step (a learning step
    of a node, whatever its ``elapsed``: about one per frame at
    ``time.learn_interval`` 25 ms), with ``betas``, ``eps``, the decoupled
    ``weight_decay`` (of the parameters of two or more dimensions; with
    ``weight_decay_end``, linear to it at ``decay_steps``), a linear warm-up
    over ``warmup_steps`` steps, then ``decay`` (``DECAYS``: constant, or a
    cosine to ``min_lr`` times ``lr`` at ``decay_steps``), and the clipping of
    the norm of a node's gradients at ``clip`` (0: none). Its ``warmup`` in
    seconds must be 0, and the other kinds take none of these step settings.

    ``tau_m`` (the first moment) is of ``"adamw"`` and ``"graft_adam"`` only,
    ``tau_v`` (the second moment) of those and ``"rmsprop"``: another kind
    must leave them at their defaults, as a sweep of them would change nothing.

    ``scale`` (``SCALES``) is the learning rate of each tensor: ``"none"``,
    ``lr`` for every one; ``"fan_in"``, ``lr / n`` with $n$ the tensor's
    fan-in by its role in the modules (``exp.lib.optim.fan_ins``: a linear
    map's ``in_features``, 1 for the vectors), the rate of Adam under the
    spectral condition (muP; Yang, Simon & Bernstein 2023). ``"fan_in"`` is
    of ``"adamw"`` (with ``weight_decay`` 0, for a coupled decay would then
    change per tensor) and ``"rmsprop"`` only.
    """

    kind: str = "sgd"
    lr: float = 0.03  # per frame (per step for "adamw_step")
    warmup: float = 300.0  # seconds of virtual time the rate rises over; 0: none
    tau_m: float = 0.3  # the moments of Adam, in seconds
    tau_v: float = 30.0
    weight_decay: float = 0.0
    scale: str = "none"  # "fan_in": lr / fan-in per tensor (adamw, rmsprop)
    # "adamw_step" only
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    warmup_steps: int = 0
    decay: str = "constant"
    decay_steps: int = 0
    min_lr: float = 0.0  # of lr
    weight_decay_end: float | None = None
    clip: float = 0.0

    def __post_init__(self) -> None:
        check_choice("optim.kind", self.kind, OPTIMIZERS)
        check_choice("optim.decay", self.decay, DECAYS)
        check_choice("optim.scale", self.scale, SCALES)
        if self.scale == "fan_in":
            if self.kind not in ("adamw", "rmsprop"):
                raise ValueError(
                    f"optim.scale 'fan_in' is of the kinds adamw and rmsprop, "
                    f"not {self.kind!r}"
                )
            if self.weight_decay != 0:
                raise ValueError(
                    "optim.scale 'fan_in' needs optim.weight_decay 0, "
                    f"got {self.weight_decay}"
                )
        steps = {
            "betas": (self.betas, [0.9, 0.999]),
            "eps": (self.eps, 1e-8),
            "warmup_steps": (self.warmup_steps, 0),
            "decay": (self.decay, "constant"),
            "decay_steps": (self.decay_steps, 0),
            "min_lr": (self.min_lr, 0.0),
            "weight_decay_end": (self.weight_decay_end, None),
            "clip": (self.clip, 0.0),
        }
        for name, (default, kinds) in _MOMENTS.items():
            if self.kind not in kinds and getattr(self, name) != default:
                raise ValueError(
                    f"optim.{name} is of the kinds {kinds}, not {self.kind!r}: "
                    f"leave it at {default}"
                )
        if self.kind == "adamw_step":
            if self.warmup != 0:
                raise ValueError(
                    "optim.kind 'adamw_step' warms up over optim.warmup_steps: "
                    f"set optim.warmup to 0, not {self.warmup}"
                )
        else:
            changed = [k for k, (got, base) in steps.items() if got != base]
            if changed:
                raise ValueError(
                    f"optim {', '.join(changed)} are of kind 'adamw_step', "
                    f"not {self.kind!r}"
                )


@dataclass(kw_only=True)
class OfflineConf:
    """The offline bound (``task="offline"``): the same modules fit by Adam on
    windows cut from the streams, through time (BPTT) or, with ``local``, with
    the input and the fed-back prediction detached (one frame, as the nodes).

    ``order``: ``"random"`` draws ``batch`` windows per step and steps
    ``adam(lr)``; ``"stream"`` takes at step $t$ the windows $[tS, tS + W)$ of
    every stream (``stride`` $S$, 0: ``window``; ``batch`` and ``lr`` are not
    used) and steps the optimizer of ``optim`` by $S$ frames of time (the
    program's ``make_optimizer`` mapping, as an online run that learns every
    $S$ frames). ``feedback`` (``"grad"``, ``"detach"`` or ``"teacher"``) and
    ``credit`` (``"bptt"`` or ``"own"``) are those of
    ``exp.video.dyn.offline.window_loss``; ``credit="own"`` detaches the
    feedback, so it needs ``feedback="detach"`` (or ``"teacher"``). With
    ``order="stream"`` ``steps`` must not exceed the ``(frames - window) //
    stride + 1`` steps the streams hold."""

    frames: int = 1024  # of each stream to cut the windows from
    window: int = 32  # frames per window
    steps: int = 6000
    batch: int = 128  # windows per step
    lr: float = 1e-3
    local: bool = False
    every: int = 1000  # steps between two evaluations
    order: str = "random"
    stride: int = 0  # frames between two windows of "stream"; 0: window
    feedback: str = "grad"
    credit: str = "bptt"

    def __post_init__(self) -> None:
        check_choice("offline.order", self.order, ORDERS)
        check_choice("offline.feedback", self.feedback, FEEDBACKS)
        check_choice("offline.credit", self.credit, CREDITS)
        if self.stride < 0:
            raise ValueError(f"offline.stride must be >= 0, got {self.stride}")
        if self.credit == "own" and self.feedback == "grad":
            raise ValueError(
                "offline.credit='own' detaches the feedback: "
                "set offline.feedback='detach'"
            )
        if self.order == "stream":
            stride = self.stride or self.window
            if self.window < 2 or self.frames < self.window:
                raise ValueError(
                    f"offline: windows of {self.window} frames in {self.frames}"
                )
            fits = (self.frames - self.window) // stride + 1
            if self.steps > fits:
                raise ValueError(
                    f"offline.steps={self.steps} > {fits} = (frames - window) // "
                    f"stride + 1: the streams hold no more windows"
                )
        if self.credit == "own" and self.local:
            raise ValueError(
                "offline.credit='own' and offline.local cut the credit twice"
            )
