"""The settings of the streams: the balls (M14, M15, M16) and Walking Tours
(M15, M16)."""

from dataclasses import dataclass, field

from omegaconf import MISSING

from exp.video.schema._check import check_choice

__all__ = ["OBSERVATIONS", "BallsConf", "DataConf", "PhysicsConf", "ToursConf"]

# "pos_vel": x, y, vx, vy of each ball (M14 stage 0); "pos": x, y (stages 1,
# 2); "pixels": the rendered image as a vector (stage 3); "frames": the images
# to an encoder (M15, M16).
OBSERVATIONS = ("pos_vel", "pos", "pixels", "frames")


@dataclass(kw_only=True)
class PhysicsConf:
    r"""The physics of the balls in the box $[0, 1]^2$: gravity
    $\gamma(t) = \gamma_0 (1 + a_\gamma \sin(2\pi t / T))$ along $-y$, the
    restitution $e(t) = e_0 (1 + a_e \sin(2\pi t / T + \pi / 2))$ at the walls
    and a linear drag, by semi-implicit Euler in ``substeps`` steps per frame.
    The defaults are the easy physics of M13: constant gravity, $e = 1$, no
    drag, and bounces that keep the energy (``conserve``)."""

    gamma0: float = 1.0
    gamma_amp: float = 0.0
    e0: float = 1.0
    e_amp: float = 0.0
    drag: float = 0.0
    period: float = 600.0  # T, seconds
    conserve: bool = True
    substeps: int = 8


@dataclass(kw_only=True)
class BallsConf:
    """Balls that never collide with each other, in streams that never draw new
    balls; ``size`` is the side of the rendered images in pixels."""

    radii: list[float] = field(default_factory=lambda: [0.10])
    obs: str = "pos_vel"
    size: int = 32
    physics: PhysicsConf = field(default_factory=PhysicsConf)

    def __post_init__(self) -> None:
        check_choice("balls.obs", self.obs, OBSERVATIONS)
        if not self.radii:
            raise ValueError("balls.radii must name at least one ball")


@dataclass(kw_only=True)
class ToursConf:
    """The Walking Tours videos, prepared as u8 frames of ``size`` x ``size`` at
    ``fps`` (``task="prep"``); ``held_out`` is the video kept for evaluation
    only."""

    data_version: str = MISSING
    held_out: str = MISSING
    size: int = 64
    fps: float = 10.0
    # The video frames the stream moves on per frame of the run (``time.frame``;
    # M17: 3, a video of 30 fps learned at 10 fps); 1 only with vision.layout
    # "pixels".
    advance: int = 1

    def __post_init__(self) -> None:
        if self.advance < 1:
            raise ValueError(f"tours.advance must be at least 1, not {self.advance}")


@dataclass(kw_only=True)
class DataConf:
    """``streams`` streams side by side (the batch B), of exactly one of the
    balls and Walking Tours."""

    streams: int = 128
    balls: BallsConf | None = field(default_factory=BallsConf)
    tours: ToursConf | None = None

    def __post_init__(self) -> None:
        if (self.balls is None) == (self.tours is None):
            raise ValueError("data needs exactly one of balls and tours")
        if self.streams < 1:
            raise ValueError(f"streams must be at least 1, not {self.streams}")
