r"""The last frames of each of several streams, for a learner that learns every
$S$ frames from the last $W$ of them (a window replayed through time)."""

from collections.abc import Mapping
from typing import Any, override

import torch
from mujou.types import Device

from mujou import Stateful

__all__ = ["WindowRing"]


class WindowRing(Stateful):
    r"""A ring of the last ``window`` $W$ frames ``x`` ``(B, D)`` of ``B``
    streams pushed side by side, with the time of each push and, per stream,
    the pushes since its last reset.

    Push ``k`` (counted from 1) is frame $k$ of every stream. A window is due
    after push $k$ when $k \ge W$ and $(k - W) \bmod S = 0$ (``stride`` $S$):
    the windows end at the pushes $W, W + S, W + 2S, \dots$, so after $N$
    pushes $\lfloor (N - W) / S \rfloor + 1$ windows were due (none for $N <
    W$). ``window()`` is the last $W$ frames, oldest first, and their mask.

    A stream's window is in the mask iff

    - all its $W$ frames are of one video: the stream reset at none of the
      last $W - 1$ pushes and was pushed at least $W$ times (a push with
      ``reset`` is the first frame of a new video, so it may be the oldest
      frame of a window);
    - the $W - 1$ gaps between the times of the last $W$ pushes (shared by
      all streams) are all in $(0.5 g, 1.5 g)$, $g$ the lower median of
      these gaps: no frame missed or repeated (a gap of 0 or of 2 frames
      leaves it). The median, not the frame of the stream, is the reference,
      so that the ring needs no frame; frames that come at another steady
      period pass.

    ``elapsed()`` is the time the window covers that the one before did not:
    the time of its last frame minus that of the push $S$ before (the last
    frame of the window before, when there is one). When the ring has no push
    $S$ before (the first window with $S \ge W$), it is $S$ times the lower
    median gap of the pushes held. With frames of period ``frame`` it is $S$
    ``frame``.

    It is a ``mujou.Stateful``: a node that holds it keeps the frames, the
    times, the counts and the pushes in its state, restored exactly.

    Args:
        batch: ``B``.
        window: ``W`` >= 2.
        d: ``D``.
        stride: ``S`` >= 1.
        device: Where the frames are held (the times stay on the CPU).
        dtype: Of the frames.

    Raises:
        ValueError: ``window`` < 2 or ``stride`` < 1.
    """

    def __init__(
        self,
        batch: int,
        window: int,
        d: int,
        *,
        stride: int,
        device: Device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if window < 2 or stride < 1:
            raise ValueError(
                f"a ring needs window >= 2 and stride >= 1, got {window} and {stride}"
            )
        self.size, self.stride = window, stride
        self._x = torch.zeros(batch, window, d, device=device, dtype=dtype)
        # The times of the last max(W, S + 1) pushes, in ns, so that the one S
        # before the last is held.
        self._times = torch.zeros(max(window, stride + 1), dtype=torch.int64)
        self._since = torch.zeros(batch, dtype=torch.int64, device=device)
        self.k = 0  # the pushes so far

    @torch.no_grad()
    def push_(self, x: torch.Tensor, time_ns: int, reset: torch.Tensor) -> None:
        """Push the frame ``x`` ``(B, D)`` of every stream, written at
        ``time_ns``; the streams where ``reset`` ``(B,)`` holds start a new
        video at it."""
        self._x[:, self.k % self.size] = x
        self._times[self.k % len(self._times)] = time_ns
        self._since.copy_(torch.where(reset, 1, self._since + 1))
        self.k += 1

    def due(self) -> bool:
        """Whether a window ends at the last push."""
        k, w = self.k, self.size
        return k >= w and (k - w) % self.stride == 0

    def _last_times(self, n: int) -> torch.Tensor:
        """The times of the last ``n`` pushes (``n`` <= those held), oldest
        first, ``(n,)`` int64."""
        idx = torch.arange(self.k - n, self.k) % len(self._times)
        return self._times[idx]

    def window(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The last ``W`` frames of every stream, oldest first, ``(B, W, D)``
        (a copy), and the mask ``(B,)`` bool of the class docstring.

        Raises:
            ValueError: Fewer than ``W`` pushes so far.
        """
        w = self.size
        if self.k < w:
            raise ValueError(f"a window of {w} frames after {self.k} pushes")
        x = torch.roll(self._x, -(self.k % w), dims=1)
        gaps = torch.diff(self._last_times(w))
        g = int(gaps.median())
        steady = g > 0 and bool(((gaps > 0.5 * g) & (gaps < 1.5 * g)).all())
        return x, (self._since >= w) & steady

    def elapsed(self) -> float:
        """The seconds of the class docstring, at the last push.

        Raises:
            ValueError: Fewer than 2 pushes so far.
        """
        if self.k < 2:
            raise ValueError(f"no time elapsed after {self.k} pushes")
        s = self.stride
        if self.k > s:
            times = self._last_times(s + 1)
            return int(times[-1] - times[0]) / 1e9
        held = min(self.k, len(self._times))
        return s * int(torch.diff(self._last_times(held)).median()) / 1e9

    @override
    def state_dict(self) -> dict[str, Any]:
        """The frames, times and counts (sharing memory with the ring's) and
        the pushes."""
        return {"x": self._x, "times": self._times, "since": self._since, "k": self.k}

    @override
    def load_state_dict(self, state: Mapping[str, Any], /) -> None:
        """Copy ``state`` into this ring (of the same sizes; any device)."""
        with torch.no_grad():
            self._x.copy_(state["x"])
            self._times.copy_(state["times"])
            self._since.copy_(state["since"])
        self.k = int(state["k"])
