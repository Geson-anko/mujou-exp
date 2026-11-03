r"""The reference physics of the balls: the per-ball Python implementation of
M13 (``exp/video/balls.py`` at 433a72e), kept unchanged so that the vectorized
physics of ``exp.video.data.physics`` can be checked against it bit for bit
(stream ``i`` of a seed against ``Video(seed=seed + STREAM_SEED_STRIDE * i)``).
Not used by the experiment itself.

The synthetic video of stages 1-2 (PLAN.md, section 6): balls in a box.

The box is $[0, 1]^2$ with $y$ up. Each ball moves by gravity $-\gamma(t)$
along $y$ and a linear air drag $-c\,v$, integrated by semi-implicit Euler in
``substeps`` steps per frame, and bounces off the walls with the restitution
$e(t)$ (the balls do not collide with each other). The physics changes slowly
and periodically in time, continuously across the episodes:

$$
\gamma(t) = \gamma_0 (1 + a_\gamma \sin(2\pi t / T)),\qquad
e(t) = e_0 (1 + a_e \sin(2\pi t / T + \pi / 2)).
$$

Each episode lasts a uniform time in ``episode`` seconds; a new one draws the
positions and velocities anew (its first frame has ``reset``). With
``episode=None``, the balls are drawn once and the stream never starts anew
(M13: the frames change continuously; with $e = 1$ and no drag the balls keep
their energy and do not come to rest). A frame is a
``size`` x ``size`` grey image of Gaussian blobs, one per ball, of standard
deviation half the radius, in $[0, 1]$.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

__all__ = [
    "RADII",
    "STREAM_SEED_STRIDE",
    "Frame",
    "Physics",
    "Video",
    "advance",
    "draw_balls",
    "render",
]

RADII = (0.06, 0.08, 0.10)


@dataclass(frozen=True, kw_only=True)
class Physics:
    """The physics and how it changes in time (seconds)."""

    gamma0: float = 1.0
    gamma_amp: float = 0.5
    e0: float = 0.6
    e_amp: float = 0.5
    drag: float = 0.1
    period: float = 600.0  # T
    # Whether a bounce keeps the energy that semi-implicit Euler keeps between
    # the bounces (``_reflected``); without, each bounce off the floor or the
    # ceiling changes it by up to gamma h |v_y| (h the step), with the sign of
    # where in the step the ball hits, and with e = 1 the energy of a ball
    # drifts by up to about 10% over minutes. Off by default: the physics of
    # the runs before M13.
    conserve: bool = False

    def gamma(self, t: float) -> float:
        return self.gamma0 * (
            1 + self.gamma_amp * math.sin(2 * math.pi * t / self.period)
        )

    def restitution(self, t: float) -> float:
        phase = 2 * math.pi * t / self.period + math.pi / 2
        return self.e0 * (1 + self.e_amp * math.sin(phase))


class Frame(NamedTuple):
    """One frame, or a stack of them (a leading dimension)."""

    image: torch.Tensor  # (1, size, size)
    state: torch.Tensor  # (4 * balls,): x, y, vx, vy of each ball
    params: torch.Tensor  # (2,): gamma and e at the frame's time
    reset: bool | torch.Tensor  # the first frame of an episode (a tensor: a stack)


def _reflected(v: float, dy: float, g: float, h: float, up: bool) -> float:
    r"""The velocity along an axis after a bounce that moves the position by
    ``dy`` against the acceleration $-g$, such that the energy semi-implicit
    Euler keeps in steps of ``h``, $\tilde E = v^2 / 2 + g y - g h v / 2$, is
    the same as before: the root of $v'^2 - g h v' = v^2 - g h v - 2 g\,dy$
    that points away from the wall (``up``: toward larger coordinates)."""
    c = v * v - g * h * v - 2 * g * dy
    root = math.sqrt(max(g * g * h * h + 4 * c, 0.0))
    return (g * h + root) / 2 if up else (g * h - root) / 2


def _step(
    pos: list[list[float]],
    vel: list[list[float]],
    radii: tuple[float, ...],
    h: float,
    gamma: float,
    e: float,
    drag: float,
    conserve: bool = False,
) -> None:
    """One step of semi-implicit Euler, in place: the velocity first, then the
    position with the new velocity, then the bounces off the walls. With
    ``conserve``, the speed after a bounce is ``e`` times that of
    ``_reflected``, rather than ``e`` times that before."""
    for p, v, r in zip(pos, vel, radii, strict=True):
        v[0] += h * (-drag * v[0])
        v[1] += h * (-gamma - drag * v[1])
        p[0] += h * v[0]
        p[1] += h * v[1]
        for i, g in ((0, 0.0), (1, gamma)):
            if p[i] < r or p[i] > 1 - r:
                wall = r if p[i] < r else 1 - r
                q = wall + e * (wall - p[i])
                if conserve:
                    v[i] = e * _reflected(v[i], q - p[i], g, h, up=p[i] < r)
                else:
                    v[i] = -e * v[i]
                p[i] = q


def render(states: torch.Tensor, size: int, radii: tuple[float, ...]) -> torch.Tensor:
    """The images ``(n, 1, size, size)`` of the states ``(n, 4 * balls)``."""
    n = states.shape[0]
    xy = states.reshape(n, len(radii), 4)[..., :2]  # (n, balls, 2)
    centers = (torch.arange(size, dtype=states.dtype) + 0.5) / size
    cols = centers.view(1, 1, 1, size)  # x
    rows = (1 - centers).view(1, 1, size, 1)  # y is up; row 0 is the top
    sigma = torch.tensor(radii, dtype=states.dtype).view(1, -1, 1, 1) / 2
    x = xy[..., 0].reshape(n, -1, 1, 1)
    y = xy[..., 1].reshape(n, -1, 1, 1)
    blobs = torch.exp(-((cols - x) ** 2 + (rows - y) ** 2) / (2 * sigma**2))
    return blobs.sum(1, keepdim=True).clamp(max=1.0)


class Video:
    """The stream of frames, one per ``next()``, from ``seed``.

    Args:
        size: The side of the images in pixels.
        balls: How many balls (up to ``len(RADII)``), the first of ``RADII``.
        radii: The radii of the balls, instead of ``balls``.
        frame: The time between two frames, in seconds.
        physics: The physics.
        episode: The shortest and longest episode, in seconds; ``None``: one
            episode, the whole stream.
        substeps: The steps of the integration per frame.
        seed: Seeds every draw.
        start: The time of the first frame on the clock of the physics.
    """

    def __init__(
        self,
        *,
        size: int = 32,
        balls: int = 3,
        radii: tuple[float, ...] | None = None,
        frame: float = 1 / 30,
        physics: Physics = Physics(),
        episode: tuple[float, float] | None = (3.0, 6.0),
        substeps: int = 8,
        seed: int,
        start: float = 0.0,
    ) -> None:
        self._g = torch.Generator().manual_seed(seed)
        self._start = start
        self._size, self._frame, self._substeps = size, frame, substeps
        self.radii = RADII[:balls] if radii is None else radii
        self.physics = physics
        self._episode = episode
        self.k = 0  # the frame to come
        self._end = 0  # the first frame of the next episode
        self._pos: list[list[float]] = []
        self._vel: list[list[float]] = []

    def _new_episode(self) -> None:
        if self._episode is None:
            self._end = -1  # never again
        else:
            lo, hi = self._episode
            seconds = lo + (hi - lo) * float(torch.rand((), generator=self._g))
            self._end = self.k + max(1, round(seconds / self._frame))
        self._pos, self._vel = draw_balls(self.radii, self._g)

    def _state(self) -> torch.Tensor:
        return torch.tensor([[*p, *v] for p, v in zip(self._pos, self._vel)]).flatten()

    def next(self) -> Frame:
        """The next frame: a new episode starts at its first frame, otherwise the
        balls move by one frame first."""
        t = self._start + self.k * self._frame
        reset = self.k == self._end
        if reset:
            self._new_episode()
        else:
            advance(
                self._pos,
                self._vel,
                self.radii,
                self.physics,
                t - self._frame,
                self._frame,
                self._substeps,
            )
        self.k += 1
        state = self._state()
        image = render(state.unsqueeze(0), self._size, self.radii)[0]
        params = torch.tensor([self.physics.gamma(t), self.physics.restitution(t)])
        return Frame(image, state, params, reset)


# Stream i of a run of seed s was Video(seed=s + STREAM_SEED_STRIDE * i) (the
# old balls.Streams), clear of the small offsets the runs added to s.
STREAM_SEED_STRIDE = 1_000_000


def draw_balls(
    radii: tuple[float, ...], g: torch.Generator
) -> tuple[list[list[float]], list[list[float]]]:
    """Positions uniform in the box (off the walls) and velocities of standard
    deviation 0.5 per axis."""
    pos = [
        [r + (1 - 2 * r) * float(u) for u in torch.rand(2, generator=g)] for r in radii
    ]
    vel = [[0.5 * float(n) for n in torch.randn(2, generator=g)] for _ in radii]
    return pos, vel


def advance(
    pos: list[list[float]],
    vel: list[list[float]],
    radii: tuple[float, ...],
    physics: Physics,
    t: float,
    seconds: float,
    substeps: int,
) -> None:
    """Move the balls from time ``t`` by ``seconds`` in ``substeps`` steps, in
    place; each step takes the physics at its start."""
    h = seconds / substeps
    for i in range(substeps):
        s = t + i * h
        _step(
            pos,
            vel,
            radii,
            h,
            physics.gamma(s),
            physics.restitution(s),
            physics.drag,
            physics.conserve,
        )
