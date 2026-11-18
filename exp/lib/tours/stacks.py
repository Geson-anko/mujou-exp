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

__all__ = ["DATA_ENV", "StackStreams", "StackedTourFrames", "tours_root"]

DATA_ENV = "MUJOU_EXP_DATA"
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
    except ``held_out``, ``advance`` video frames further at each ``next()``.

    The videos are joined into a loop of $L$ frames as in ``TourFrames``. At
    its $k$-th ``next()`` stream $i$ ends at frame $g = (\lfloor i L / n
    \rfloor + k\,\mathrm{advance}) \bmod L$ of the loop, and its stack is
    the frames $g - j\,\mathrm{stride}$, $j = N - 1, \dots, 0$, of $g$'s
    video. A stack that would reach before the video's first frame is not
    ``ready`` (its frames before the first are the first one); ``reset`` is
    true at the first ``next()`` and where the end is fewer than ``advance``
    frames into its video. The stacks are read from the memory-mapped frames
    at each call (no ring), so the state is ``{"k"}``.

    Raises:
        ValueError: as ``TourFrames``, or a non-positive ``frames``,
            ``stride`` or ``advance``.
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
    ) -> None:
        if frames < 1 or stride < 1 or advance < 1:
            raise ValueError(
                f"frames, stride and advance must be positive: {frames}, "
                f"{stride}, {advance}"
            )
        names = videos(root)
        if held_out not in names:
            raise ValueError(f"{held_out!r} is not in {root} ({names})")
        self._readers = [U8Reader(root / v) for v in names if v != held_out]
        lengths = torch.tensor([len(r) for r in self._readers])
        self._firsts = torch.cumsum(lengths, 0) - lengths  # (videos,)
        self._total = int(lengths.sum())  # L
        if self._total == 0:
            raise ValueError(f"no frames under {root} but those of {held_out!r}")
        self._offsets = torch.arange(n) * self._total // n  # (n,)
        self._back = torch.arange(frames - 1, -1, -1) * stride  # (N,): j stride
        self._advance = advance
        self._k = 0  # the stack to come

    @override
    def next(self) -> tuple[Tensor, Tensor, Tensor]:
        k = torch.full_like(self._offsets, self._k)
        stacks, ready, local = self._at(torch.arange(len(self._offsets)), k)
        reset = (local < self._advance) | (self._k == 0)
        self._k += 1
        return stacks, ready, reset

    def _at(self, streams: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The stacks u8 ``(m, N, C, H, W)`` of the streams ``streams``
        ``(m,)`` at their ``k`` ``(m,)``-th ``next()``, whether each is
        ``ready``, and the frame of its video it ends at ``(m,)``."""
        g = (self._offsets[streams] + k * self._advance) % self._total  # (m,)
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

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"k": self._k}

    @override
    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._k = d["k"]
