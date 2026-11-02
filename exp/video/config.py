"""The settings of one run, as frozen dataclasses that round-trip through JSON."""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from mujou.pc import optim as pc_optim
from mujou.pc.optim import Optimizer

from exp.lib.config import from_dict
from exp.lib.optim import (
    AdamDirection,
    DirectionOptimizer,
    Grafted,
    Orthogonalized,
    Projected,
    Warmup,
    Whitened,
)
from exp.video.balls import Physics

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "OptimConfig",
    "TimeConfig",
    "make_optimizer",
]


@dataclass(frozen=True, kw_only=True)
class TimeConfig:
    """The times of the stream and of PC-ALM, in seconds of the run's timer."""

    frame: float = 1 / 30  # the period of the frames (30 fps)
    h: float = 4e-3  # TimeConstants(h=, lam=, r=)
    lam: float = 2e-3
    r: float = 4e-3
    learn_interval: float | None = 25e-3  # None: learn in every update


@dataclass(frozen=True, kw_only=True)
class OptimConfig:
    """The optimizer of every node (PLAN.md, section 5).

    ``kind`` is one of ``"sgd"``, ``"adamw"`` (``pc.optim.AdamW``),
    ``"adam_direction"``, ``"whitened"``, ``"orthogonalized"`` and
    ``"projected"``.
    """

    kind: str = "sgd"
    tau: float = 1.0  # the time constant of the parameters
    graft: bool = False  # take the direction only, with the step size of SGD
    tau_m: float = 0.3  # the moments of AdamW
    tau_v: float = 30.0
    weight_decay: float = 0.0
    tau_l: float | None = None  # L = EMA of G G^T; None: no left factor
    tau_r: float | None = None  # R = EMA of G^T G; None: no right factor
    a: float = 0.25  # D = (L + eps)^-a G (R + eps)^-b
    b: float = 0.25
    rank: int = 4  # the eigenvectors that "projected" removes on each side
    eps: float = 1e-4  # relative to the largest eigenvalue
    refresh: float = 1.0  # seconds between two eigendecompositions
    # Seconds over which the time of the optimizer rises from 0 (0: none); for
    # SGD, the learning rate (exp.lib.optim.Warmup).
    warmup: float = 0.0


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    """The sizes of the network (PLAN.md, section 7)."""

    hidden: int = 0  # stage 0: the width of the hidden layer (0: linear)
    d_z: int = 64
    d_s: int = 256
    kappa: float = 0.05  # the weight of SIGReg
    # Whether the prediction error sends no credit to its target z_k (a stop
    # gradient), so that only SIGReg shapes the encoder from its side.
    stop_target: bool = False
    # With stop_target, the fraction of that credit still sent (0: none), so
    # that the encoder also learns to make z predictable.
    target_weight: float = 0.0
    # Whether z is scaled onto the sphere of radius sqrt(D), where N(0, I_D)
    # concentrates, so that SIGReg cannot push |z| out without bound.
    z_norm: bool = False
    # The channels of the first convolution of the encoder of the synthetic
    # video (then twice and four times as many).
    width: int = 32
    # The standard deviation, in pixels, of a fixed Gaussian blur of the frame
    # before the encoder of the synthetic video (0: none), so that z varies
    # smoothly with the positions of the balls.
    blur: float = 0.0
    # The fraction of the predictions, the last ones of every open_cycle of a
    # video, that start from the model's own prediction (nodes.is_open).
    open_loop: float = 0.0
    open_cycle: int = 20
    # Whether the prediction is z_{k-1} + W_o s_k (the change from the input
    # the dynamics remembered) rather than W_o s_k; W_o starts at 0.
    residual: bool = False
    # The nodes of the encoder (the synthetic video: 2, G1 and Z; EgoDex: 1-3)
    # and of the dynamics (Dynamics and dyn_nodes - 1 DynamicsLayer on top).
    enc_nodes: int = 2
    dyn_nodes: int = 1
    # The weight beta of the reconstruction of the frame from z (M13): a node
    # with a decoder of z (model.decoder) whose energy is beta times the mean
    # squared error per pixel (nodes.Reconstruct), so that z keeps what the
    # frame shows (0: none).
    recon: float = 0.0
    directions: int = 256  # the directions of SIGReg, drawn anew each frame
    buffer: int = 255  # the past z SIGReg keeps
    buffer_every: int = 1  # SIGReg keeps the z of one frame in buffer_every


@dataclass(frozen=True, kw_only=True)
class DataConfig:
    """The stream. ``kind`` is ``"linear"`` (stage 0), ``"video"`` (stages 1-2)
    or ``"egodex"`` (stage 3, from the frames ``egodex.py`` prepared in ``path``;
    ``size`` 64)."""

    kind: str = "video"
    seconds: float = 600.0  # the length of the stream, on the run's timer
    iid: bool = False  # draw the frames independently (no correlation in time)
    # The linear regression of stage 0.
    d_x: int = 32
    tau_x: float = 10.0  # the correlation time of the mean of x
    noise: float = 0.1
    # The synthetic video of stages 1-2.
    size: int = 32
    balls: int = 3
    # The radii of the balls instead of the first ``balls`` of balls.RADII
    # (None), for streams of fewer or larger balls (M13).
    radii: tuple[float, ...] | None = None
    # The physics (balls.Physics): gamma(t) = gamma0 (1 + gamma_amp sin(2 pi t / T)),
    # e(t) = e0 (1 + e_amp cos(2 pi t / T)) and the drag.
    gamma0: float = 1.0
    gamma_amp: float = 0.5
    e0: float = 0.6
    e_amp: float = 0.5
    drag: float = 0.1
    conserve: bool = False  # the bounces keep the energy (balls.Physics)
    physics_period: float = 600.0  # T of gamma(t) and e(t)
    # About an evaluation clip (128 frames): with the drag and the loss at the
    # walls, the balls rest on the floor after about 20 s.
    episode_min: float = 3.0
    episode_max: float = 6.0
    # Whether new balls are drawn every episode_min to episode_max seconds;
    # without, one episode runs the whole stream (M13; for e = 1 and no drag,
    # or the balls come to rest).
    redraw: bool = True
    # The order of the frames: "stream" (as they come), "shuffled" (clips of
    # clip seconds in a random order, each a new episode) or "replay" (each
    # live clip followed by one drawn from a reservoir of replay frames).
    order: str = "stream"
    clip: float = 1.0
    replay: int = 4096
    # EgoDex (stage 3).
    path: str = "datasets/egodex64"
    # The number B of independent streams of the synthetic video that the network
    # sees side by side in every frame, as a batch (1: one stream, the batch
    # ``()``). Stream i is ``balls.Video(seed=seed + 1_000_000 * i)``; every node
    # learns from the mean over the B streams, except that the SIGReg term acts
    # on the parameters as the sum of B per-sample terms against the mean of the
    # prediction errors: the same kappa is a stronger SIGReg as B grows (per
    # sample, kappa * B / (1 - kappa) against the prediction). SIGReg relies on
    # the host writing mask = ~reset. Only for ``kind == "video"`` and
    # ``order == "stream"``.
    streams: int = 1
    # With streams > 1, whether stream i starts at i / streams of the period of
    # the physics (balls.Streams), so that every frame shows every phase.
    spread: bool = False

    def __post_init__(self) -> None:
        if self.radii is not None:  # JSON gives a list
            object.__setattr__(self, "radii", tuple(self.radii))
        if self.streams < 1:
            raise ValueError(f"streams must be at least 1, not {self.streams}")
        if self.streams > 1 and (self.kind != "video" or self.order != "stream"):
            raise ValueError(
                "streams > 1 is for the synthetic video in stream order, not "
                f"kind={self.kind!r}, order={self.order!r}"
            )

    def physics(self) -> Physics:
        """The physics of the synthetic video."""
        return Physics(
            gamma0=self.gamma0,
            gamma_amp=self.gamma_amp,
            e0=self.e0,
            e_amp=self.e_amp,
            drag=self.drag,
            period=self.physics_period,
            conserve=self.conserve,
        )

    @property
    def batch(self) -> tuple[int, ...]:
        """The batch of the streams: ``()`` for one, ``(streams,)`` for more."""
        return () if self.streams == 1 else (self.streams,)


@dataclass(frozen=True, kw_only=True)
class Config:
    """The settings of one run."""

    name: str = "run"
    seed: int = 0
    bp: bool = False  # the baseline: the same modules by BP in one node
    eval_interval: float = 60.0  # seconds of the run's timer between evaluations
    # Seconds of frames between two checkpoints of the modules (0: none), in
    # ckpt/ of the run (run.load_checkpoint).
    checkpoint_interval: float = 0.0
    realtime: bool = False  # mujou.run in real time, rather than stepped by hand
    device: str = "cpu"  # where every node computes, and the evaluation ("cuda")
    time: TimeConfig = field(default_factory=TimeConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        c = self.checkpoint_interval
        if c < 0 or c != round(c):  # the file of a checkpoint is named in seconds
            raise ValueError(f"checkpoint_interval must be whole seconds, not {c}")
        if c and self.realtime:  # the modules are read between steps only
            raise ValueError("no checkpoints in a real-time run")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return from_dict(cls, d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Self:
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.from_json(path.read_text())


def make_optimizer(c: OptimConfig) -> Optimizer:
    """The optimizer of the settings ``c``."""
    if c.warmup > 0:
        stepped = make_optimizer(dataclasses.replace(c, warmup=0.0))
        return Warmup(inner=stepped, seconds=c.warmup)
    if c.kind == "sgd":
        return pc_optim.SGD(tau=c.tau)
    if c.kind == "adamw":
        return pc_optim.AdamW(
            tau=c.tau, tau_m=c.tau_m, tau_v=c.tau_v, weight_decay=c.weight_decay
        )
    inner: DirectionOptimizer
    if c.kind == "adam_direction":
        inner = AdamDirection(tau=c.tau, tau_m=c.tau_m, tau_v=c.tau_v)
    elif c.kind == "whitened":
        inner = Whitened(
            tau=c.tau,
            tau_l=c.tau_l,
            tau_r=c.tau_r,
            a=c.a,
            b=c.b,
            eps=c.eps,
            refresh=c.refresh,
        )
    elif c.kind == "orthogonalized":
        inner = Orthogonalized(tau=c.tau, eps=c.eps)
    elif c.kind == "projected":
        inner = Projected(
            tau=c.tau,
            tau_l=c.tau_l,
            tau_r=c.tau_r,
            rank=c.rank,
            eps=c.eps,
            refresh=c.refresh,
        )
    else:
        raise ValueError(f"unknown optimizer {c.kind!r}")
    return Grafted(inner=inner) if c.graft else inner
