"""Streams of images for the vision (M15, M16): B streams side by side, one
frame of each per ``next()``.

``BallFrames`` renders the balls of ``BallStreams``; ``TourFrames`` reads the
prepared Walking Tours frames (``walking_tours.prepare``) from a local copy.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, Self, override

import torch
from torch import Tensor

from exp.lib.video_io import U8Reader
from exp.video.data.physics import BallStreams, Physics, render
from exp.video.data.walking_tours import videos
from exp.video.schema import BallsConf

__all__ = ["BallFrames", "FrameStreams", "TourFrames"]


class FrameStreams(Protocol):
    """B streams of images."""

    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        """The next frame of every stream: the images ``(B, C, H, W)`` float32
        in ``[0, 1]``, the true states ``(B, S)`` float32 (``S = 0`` when
        there are none) and ``reset`` ``(B,)`` bool, true where the frame does
        not follow the stream's previous one (the first frame, a cut)."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """What ``load_state_dict`` needs to continue the streams exactly."""
        ...

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Continue from ``d`` of ``state_dict``."""
        ...


class BallFrames(FrameStreams):
    """The balls of ``BallStreams`` (same arguments), rendered by ``render``
    as ``(B, 1, size, size)``; the states are ``(B, 4K)``. The streams never
    draw new balls, so ``reset`` is true at the first frame only."""

    def __init__(
        self,
        n: int,
        *,
        radii: Sequence[float],
        physics: Physics,
        frame: float,
        substeps: int,
        seed: int,
        size: int,
    ) -> None:
        self._streams = BallStreams(
            n, radii=radii, physics=physics, frame=frame, substeps=substeps, seed=seed
        )
        self._n, self._radii, self._size = n, list(radii), size
        self._k = 0  # the frame to come

    @classmethod
    def of(cls, balls: BallsConf, n: int, *, frame: float, seed: int) -> Self:
        """The frames of a config."""
        return cls(
            n,
            radii=balls.radii,
            physics=Physics.from_conf(balls.physics),
            frame=frame,
            substeps=balls.physics.substeps,
            seed=seed,
            size=balls.size,
        )

    @override
    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        states = self._streams.next()  # (B, 4K)
        images = render(states, self._size, self._radii)  # (B, 1, size, size)
        reset = torch.full((self._n,), self._k == 0)
        self._k += 1
        return images, states, reset

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"streams": self._streams.state_dict(), "k": self._k}

    @override
    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._streams.load_state_dict(d["streams"])
        self._k = d["k"]


class TourFrames(FrameStreams):
    r"""``n`` streams over the Walking Tours videos of the finalized data at
    ``root`` (a local copy), except ``held_out``.

    The videos are joined in name order into one loop of $N$ frames; stream
    $i$ starts at frame $\lfloor i N / n \rfloor$ of the loop, and its $k$-th
    frame is frame $(\lfloor i N / n \rfloor + k) \bmod N$. ``reset`` is true
    at a stream's first frame and at the first frame of a video. The images
    are ``(n, 3, size, size)`` float32 (the u8 / 255); the states ``(n, 0)``.

    Raises:
        ValueError: ``held_out`` is not in the data, or the other videos have
            no frames.
    """

    def __init__(self, root: Path, *, n: int, held_out: str) -> None:
        names = videos(root)
        if held_out not in names:
            raise ValueError(f"{held_out!r} is not in {root} ({names})")
        self._readers = [U8Reader(root / v) for v in names if v != held_out]
        lengths = torch.tensor([len(r) for r in self._readers])
        self._firsts = torch.cumsum(lengths, 0) - lengths  # (videos,)
        self._total = int(lengths.sum())  # N
        if self._total == 0:
            raise ValueError(f"no frames under {root} but those of {held_out!r}")
        self._offsets = torch.arange(n) * self._total // n  # (n,)
        self._k = 0  # the frame to come

    @override
    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        g = (self._offsets + self._k) % self._total  # (n,) frames of the loop
        v = torch.searchsorted(self._firsts, g, right=True) - 1  # (n,) videos
        local = g - self._firsts[v]
        images = torch.stack(
            [self._readers[a][b] for a, b in zip(v.tolist(), local.tolist())]
        )
        reset = (local == 0) | (self._k == 0)
        self._k += 1
        return images.float() / 255, torch.zeros(len(g), 0), reset

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"k": self._k}

    @override
    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._k = d["k"]
