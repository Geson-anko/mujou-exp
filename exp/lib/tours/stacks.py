"""Streams of stacks of Walking Tours frames.

``StackedTourFrames`` gives, for each of ``n`` streams, a stack of frames
(``StackStreams``) over the prepared Walking Tours videos
(``exp.lib.tours.walking_tours``), read from a local copy, and ``tours_root``
finds that copy.
"""

import os
from pathlib import Path
from typing import Any, Protocol, override

import torch
from torch import Tensor

from exp.lib.rundir import EnvError
from exp.lib.tours.walking_tours import videos
from exp.lib.video_io import U8Reader

__all__ = ["DATA_ENV", "LAYOUTS", "StackStreams", "StackedTourFrames", "tours_root"]

DATA_ENV = "MUJOU_EXP_DATA"
LAYOUTS = ("spread", "chunk")  # StackedTourFrames' layouts of the n streams
_REPO = Path(__file__).resolve().parents[3]


def tours_root(data_version: str) -> Path:
    """The local copy of the prepared data ``data_version``:
    ``$MUJOU_EXP_DATA/<data_version>``, or ``.fleet/data/<data_version>`` of
    this checkout.

    Raises:
        EnvError: It has no ``index.json`` (the data is not there).
    """
    base = Path(os.environ.get(DATA_ENV, _REPO / ".fleet" / "data"))
    root = base / data_version
    if not (root / "index.json").is_file():
        raise EnvError(
            f"no prepared Walking Tours at {root}: copy the data {data_version} "
            f"there or set {DATA_ENV} to its parent"
        )
    return root


class StackStreams(Protocol):
    """B streams of stacks of frames."""

    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        """The next stack of every stream: u8 ``(B, N, C, H, W)``, oldest
        first, ``ready`` ``(B,)`` bool (the stack is within one video) and
        ``reset`` ``(B,)`` bool (the first stack, or a stack of another video
        than the stream's previous one)."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """What ``load_state_dict`` needs to continue the streams exactly."""
        ...

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Continue from ``d`` of ``state_dict``."""
        ...


class StackedTourFrames(StackStreams):
    r"""``n`` streams of stacks of ``frames`` frames ``stride`` video frames
    apart, over the Walking Tours videos of the finalized data at ``root``
    except ``held_out``, laid out by ``layout``.

    The videos are joined into a loop of $L$ frames as in ``TourFrames``. At
    its $k$-th ``next()`` stream $i$ ends at frame $g = (o_i + k\,s) \bmod
    L$ of the loop, and its stack is the frames $g - j\,\mathrm{stride}$, $j
    = N - 1, \dots, 0$, of $g$'s video:

    - ``"spread"``: $o_i = \lfloor i L / n \rfloor$, $s$ = ``advance``: n
      streams evenly apart, each ``advance`` frames further at each call
    - ``"chunk"``: $o_i = i\,\mathrm{advance}$, $s = n\,\mathrm{advance}$:
      the $k$-th call shows the single stream's (``n`` = 1) calls $kn, \dots,
      kn + n - 1$ side by side, so a batch of n is n consecutive stacks of
      one stream

    A stack that would reach before the video's first frame is not ``ready``
    (its frames before the first are the first one); ``reset`` is true at the
    first ``next()`` and where the end is fewer than $s$ frames into its
    video (the stream's previous stack was of another video). The stacks are
    read from the memory-mapped frames at each call (no ring), so the state
    is ``{"k"}``.

    Raises:
        ValueError: as ``TourFrames``, a non-positive ``frames``, ``stride``
            or ``advance``, or a ``layout`` not of ``LAYOUTS``.
    """

    def __init__(
        self,
        root: Path,
        *,
        n: int,
        held_out: str,
        frames: int,
        stride: int,
        advance: int,
        layout: str = "spread",
    ) -> None:
        if frames < 1 or stride < 1 or advance < 1:
            raise ValueError(
                f"frames, stride and advance must be positive: {frames}, "
                f"{stride}, {advance}"
            )
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
        names = videos(root)
        if held_out not in names:
            raise ValueError(f"{held_out!r} is not in {root} ({names})")
        self._names = [v for v in names if v != held_out]
        self._readers = [U8Reader(root / v) for v in self._names]
        lengths = torch.tensor([len(r) for r in self._readers])
        self._firsts = torch.cumsum(lengths, 0) - lengths  # (videos,)
        self._total = int(lengths.sum())  # L
        if self._total == 0:
            raise ValueError(f"no frames under {root} but those of {held_out!r}")
        if layout == "spread":
            self._offsets = torch.arange(n) * self._total // n  # (n,): o_i
            self._step = advance  # s
        else:
            self._offsets = torch.arange(n) * advance
            self._step = n * advance
        self._back = torch.arange(frames - 1, -1, -1) * stride  # (N,): j stride
        self._k = 0  # the stack to come

    @override
    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        k = torch.full_like(self._offsets, self._k)
        stacks, ready, local = self._at(torch.arange(len(self._offsets)), k)
        reset = (local < self._step) | (self._k == 0)
        self._k += 1
        return stacks, ready, reset

    def _at(self, streams: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The stacks u8 ``(m, N, C, H, W)`` of the streams ``streams``
        ``(m,)`` at their ``k`` ``(m,)``-th ``next()``, whether each is
        ``ready``, and the frame of its video it ends at ``(m,)``."""
        g = (self._offsets[streams] + k * self._step) % self._total  # (m,)
        v = torch.searchsorted(self._firsts, g, right=True) - 1  # (m,) videos
        local = g - self._firsts[v]
        idx = local[:, None] - self._back  # (m, N), oldest first
        ready = idx[:, 0] >= 0
        stacks = torch.stack(
            [
                self._readers[a][i.clamp(min=0)]
                for a, i in zip(v.tolist(), idx, strict=True)
            ]
        )
        return stacks, ready, local

    def locate(self, k: int, stream: int = 0) -> tuple[str, int]:
        """The video and its frame that the stack of ``stream`` at its
        ``k``-th ``next()`` ends at (its newest frame)."""
        g = (int(self._offsets[stream]) + k * self._step) % self._total
        v = int(torch.searchsorted(self._firsts, torch.tensor(g), right=True)) - 1
        return self._names[v], g - int(self._firsts[v])

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"k": self._k}

    @override
    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._k = d["k"]
