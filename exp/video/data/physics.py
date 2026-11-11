r"""The physics of the balls, vectorized: ``B`` streams of ``K`` balls in the box
$[0, 1]^2$ (y up) advanced together in torch float64.

Each ball moves by gravity $-\gamma(t)$ along $y$ and a linear air drag
$-c\,v$, integrated by semi-implicit Euler in ``substeps`` steps per frame,
and bounces off the walls with the restitution $e(t)$ (the balls do not
collide with each other). The physics changes slowly and periodically in time:

$$
\gamma(t) = \gamma_0 (1 + a_\gamma \sin(2\pi t / T)),\qquad
e(t) = e_0 (1 + a_e \sin(2\pi t / T + \pi / 2)).
$$

The streams never draw new balls. Every operation is the one of the per-ball
reference of M13 (``tests/video/data/reference_balls.py``), in the same order
and on the same IEEE doubles, so stream ``i`` of ``BallStreams(seed=s)`` is
the reference ``Video(seed=s + STREAM_SEED_STRIDE * i, episode=None)`` bit for
bit. All the streams share the clock, so $\gamma$ and $e$ are scalars of a
substep; a wall is a ``torch.where`` per ball and axis.
"""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from exp.video.schema.data import BallsConf, PhysicsConf

__all__ = [
    "STREAM_SEED_STRIDE",
    "BallState",
    "BallStreams",
    "Physics",
    "advance_",
    "draw",
    "render",
    "step_",
]

# Stream i of a run of seed s is drawn from seed s + STREAM_SEED_STRIDE * i (as
# the old balls.Streams), clear of the small seeds of everything else.
STREAM_SEED_STRIDE = 1_000_000


@dataclass(frozen=True, kw_only=True)
class Physics:
    """The physics and how it changes in time (seconds); see the module."""

    gamma0: float
    gamma_amp: float
    e0: float
    e_amp: float
    drag: float
    period: float  # T
    # Whether a bounce keeps the energy semi-implicit Euler keeps between the
    # bounces, $\tilde E = v^2 / 2 + g y - g h v / 2$ (see ``step_``).
    conserve: bool

    @classmethod
    def from_conf(cls, c: PhysicsConf) -> Physics:
        """The physics of a config (its ``substeps`` aside)."""
        return cls(
            gamma0=c.gamma0,
            gamma_amp=c.gamma_amp,
            e0=c.e0,
            e_amp=c.e_amp,
            drag=c.drag,
            period=c.period,
            conserve=c.conserve,
        )

    def gamma(self, t: float) -> float:
        return self.gamma0 * (
            1 + self.gamma_amp * math.sin(2 * math.pi * t / self.period)
        )

    def restitution(self, t: float) -> float:
        phase = 2 * math.pi * t / self.period + math.pi / 2
        return self.e0 * (1 + self.e_amp * math.sin(phase))


class BallState(NamedTuple):
    """The balls of ``B`` streams (or clips), updated in place."""

    pos: Tensor  # (B, K, 2) float64: x, y
    vel: Tensor  # (B, K, 2) float64: per second


def draw(n: int, radii: Sequence[float], gen: torch.Generator) -> BallState:
    """``n`` draws of the balls one after the other from ``gen``, each as the
    reference ``draw_balls``: for each ball a position uniform in the box off
    the walls (``torch.rand(2)``), then for each ball a velocity of standard
    deviation 0.5 per axis (``torch.randn(2)``).

    Returns:
        The state, ``pos`` and ``vel`` ``(n, K, 2)`` float64.
    """
    r = torch.tensor(radii, dtype=torch.float64).view(1, -1, 1)
    u = torch.empty(n, len(radii), 2)
    z = torch.empty(n, len(radii), 2)
    for i in range(n):  # the order of the draws of the reference
        for k in range(len(radii)):
            u[i, k] = torch.rand(2, generator=gen)
        for k in range(len(radii)):
            z[i, k] = torch.randn(2, generator=gen)
    return BallState(r + (1 - 2 * r) * u.double(), 0.5 * z.double())


def step_(
    s: BallState,
    radii: Tensor,
    h: float,
    gamma: float,
    e: float,
    drag: float,
    conserve: bool,
) -> None:
    r"""One step of semi-implicit Euler of ``h`` seconds, in place: the velocity
    first, $v \leftarrow v + h (g - c v)$ with $g = (0, -\gamma)$, then the
    position with the new velocity, $p \leftarrow p + h v$, then the bounces.

    A ball past a wall $w$ (the radius ``r`` off it) is put back at
    $q = w + e (w - p)$. Its velocity along that axis becomes $-e v$, or with
    ``conserve`` $e$ times the root of
    $v'^2 - g h v' = v^2 - g h v - 2 g (q - p)$ that points away from the
    wall: the velocity that keeps $\tilde E = v^2 / 2 + g y - g h v / 2$
    across the bounce ($g$ the gravity along the axis, 0 along $x$).

    Args:
        s: The balls, ``(B, K, 2)``.
        radii: ``(K,)`` float64.
    """
    pos, vel = s
    lo = radii.view(1, -1, 1)  # the walls a ball of radius r touches
    hi = 1 - lo
    acc = vel.new_tensor([0.0, -gamma])  # 0 - c v_x is -c v_x, as the reference
    a = acc - vel * drag
    vel.add_(a.mul_(h))
    pos.add_(vel * h)
    below = pos < lo
    hit = below | (pos > hi)
    if not bool(hit.any()):
        return
    wall = torch.where(below, lo, hi)
    q = wall + e * (wall - pos)
    if conserve:
        g = vel.new_tensor([0.0, gamma])  # the gravity of _reflected per axis
        gh, g2, gghh = g * h, 2 * g, g * g * h * h
        c = vel * vel - gh * vel - g2 * (q - pos)
        root = (gghh + 4 * c).clamp_(min=0.0).sqrt_()
        bounced = e * torch.where(below, (gh + root) / 2, (gh - root) / 2)
    else:
        bounced = -e * vel
    vel.copy_(torch.where(hit, bounced, vel))
    pos.copy_(torch.where(hit, q, pos))


def advance_(
    s: BallState,
    radii: Tensor,
    physics: Physics,
    t: float,
    seconds: float,
    substeps: int,
) -> None:
    """Move the balls from time ``t`` by ``seconds`` in ``substeps`` steps of
    ``step_``, in place; each step takes the physics at its start.

    Args:
        s: The balls, ``(B, K, 2)``.
        radii: ``(K,)`` float64.
    """
    h = seconds / substeps
    for i in range(substeps):
        ti = t + i * h
        step_(
            s,
            radii,
            h,
            physics.gamma(ti),
            physics.restitution(ti),
            physics.drag,
            physics.conserve,
        )


def _flat(s: BallState) -> Tensor:
    """``(B, 4K)``: x, y, vx, vy of each ball."""
    return torch.cat([s.pos, s.vel], -1).flatten(1)


class BallStreams:
    """``n`` streams of balls that never draw new ones: stream ``i`` starts as
    the reference ``Video(seed=seed + STREAM_SEED_STRIDE * i)`` does, and
    moves as it does, frame by frame.

    Args:
        n: The streams (the batch B).
        radii: The radii of the K balls of each stream.
        physics: The physics, on a clock that starts at 0 with the first frame.
        frame: The time between two frames, in seconds.
        substeps: The steps of ``step_`` per frame.
        seed: The seed of stream 0.
    """

    def __init__(
        self,
        n: int,
        *,
        radii: Sequence[float],
        physics: Physics,
        frame: float,
        substeps: int,
        seed: int,
    ) -> None:
        self._radii = torch.tensor(radii, dtype=torch.float64)
        self._physics, self._frame, self._substeps = physics, frame, substeps
        draws = [
            draw(1, radii, torch.Generator().manual_seed(seed + STREAM_SEED_STRIDE * i))
            for i in range(n)
        ]
        self._s = BallState(
            torch.cat([d.pos for d in draws]), torch.cat([d.vel for d in draws])
        )
        self._k = 0  # the frame to come

    @classmethod
    def of(cls, balls: BallsConf, n: int, *, frame: float, seed: int) -> BallStreams:
        """The streams of a config."""
        return cls(
            n,
            radii=balls.radii,
            physics=Physics.from_conf(balls.physics),
            frame=frame,
            substeps=balls.physics.substeps,
            seed=seed,
        )

    def next(self) -> Tensor:
        """The next frame of every stream: the drawn balls first, then the balls
        moved by one frame each time.

        Returns:
            ``(B, 4K)`` float32: x, y, vx, vy of each ball.
        """
        if self._k:
            t = self._k * self._frame  # the time of this frame
            advance_(
                self._s,
                self._radii,
                self._physics,
                t - self._frame,
                self._frame,
                self._substeps,
            )
        self._k += 1
        return _flat(self._s).float()

    def fork(self) -> BallStreams:
        """A deep copy: the same future, which advancing does not share."""
        other = copy.copy(self)
        other._s = BallState(self._s.pos.clone(), self._s.vel.clone())
        return other

    def state_dict(self) -> dict[str, Tensor]:
        """``pos`` and ``vel`` ``(B, K, 2)`` float64 and ``frame``, the number of
        frames given so far (a 0-d int64)."""
        return {
            "pos": self._s.pos.clone(),
            "vel": self._s.vel.clone(),
            "frame": torch.tensor(self._k),
        }

    def load_state_dict(self, d: dict[str, Tensor]) -> None:
        """Continue from ``d`` of ``state_dict`` (of streams of the same shape)."""
        for name in ("pos", "vel"):
            have, want = getattr(self._s, name), d[name]
            if want.shape != have.shape:
                raise ValueError(
                    f"{name} of shape {tuple(want.shape)}, the streams have "
                    f"{tuple(have.shape)}"
                )
            have.copy_(want)
        self._k = int(d["frame"])


def render(states: Tensor, size: int, radii: Sequence[float]) -> Tensor:
    """The grey images of the states: a Gaussian blob per ball of standard
    deviation half its radius, summed and clipped at 1.

    Args:
        states: ``(n, 4K)``.

    Returns:
        ``(n, 1, size, size)`` in ``[0, 1]``, row 0 at the top (y is up).
    """
    n = states.shape[0]
    xy = states.reshape(n, len(radii), 4)[..., :2]  # (n, K, 2)
    centers = (torch.arange(size, dtype=states.dtype) + 0.5) / size
    cols = centers.view(1, 1, 1, size)  # x
    rows = (1 - centers).view(1, 1, size, 1)  # y is up; row 0 is the top
    sigma = torch.tensor(radii, dtype=states.dtype).view(1, -1, 1, 1) / 2
    x = xy[..., 0].reshape(n, -1, 1, 1)
    y = xy[..., 1].reshape(n, -1, 1, 1)
    blobs = torch.exp(-((cols - x) ** 2 + (rows - y) ** 2) / (2 * sigma**2))
    return blobs.sum(1, keepdim=True).clamp(max=1.0)
