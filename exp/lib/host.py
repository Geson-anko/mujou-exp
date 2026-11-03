"""The host's loop over a stream: write a frame, step the run through it, read.

The run is a ``mujou.testing.run``: the timer moves only by the steps, so the
result does not depend on how long the computation takes. Each step moves the
timer by ``period`` and updates every node once, in the order of the tree.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, override

from mujou.testing import ManualRun
from mujou.time import Timestamp

from mujou import MultiPort, Port, Stateful

__all__ = ["Host"]


class Host(Stateful):
    """Shows a run one frame each ``frame`` seconds, by steps of ``period``.

    Frame ``k`` spans ``[k frame, (k + 1) frame)`` rounded down to the steps,
    so that the frames stay on the grid of ``frame`` however it divides
    ``period`` (a frame of 1/30 s is 16 or 17 steps of 2 ms).

    Its state is the count of the frames shown (``frames``): a host resumed
    with it on a run restored to the same time steps the same grid on.
    """

    def __init__(self, run: ManualRun, *, frame: float, period: float) -> None:
        self.run = run
        self._steps = frame / period  # per frame, maybe not a whole number
        self._period = period
        self.frames = 0  # the frames shown so far

    def _boundary(self, k: int) -> int:
        """The steps before frame ``k`` (the margin absorbs the rounding of
        ``frame / period``)."""
        return math.floor(k * self._steps + 1e-6)

    def show(
        self,
        writes: Mapping[Port[Any, Any], Any],
        watch: Port[Any, Any] | MultiPort[Any, Any],
    ) -> tuple[Any, Timestamp]:
        """Write ``writes``, step through the frame and return the first message
        of ``watch`` made from this write (its ``"time"`` is the write's): the
        prediction before learning from the frame (prequential).

        Raises:
            RuntimeError: ``watch`` sent nothing of this write within the frame.
        """
        first, written = self.show_all(writes, [watch])
        return first[0], written

    def show_all(
        self,
        writes: Mapping[Port[Any, Any], Any],
        watches: Sequence[Port[Any, Any] | MultiPort[Any, Any]],
    ) -> tuple[list[Any], Timestamp]:
        """``show`` for several ports: the first message made from the write of
        each of ``watches``, in order.

        Raises:
            RuntimeError: One of ``watches`` sent nothing of this write within
                the frame.
        """
        k = self.frames
        steps = self._boundary(k + 1) - self._boundary(k)
        written = self.run.write(writes)
        first: list[Any] = [None] * len(watches)
        for _ in range(steps):
            self.run.step(self._period)
            for i, watch in enumerate(watches):
                if first[i] is None:
                    msg = self.run.read(watch)
                    if msg["time"] == written:
                        first[i] = msg
        self.frames += 1
        if any(m is None for m in first):
            raise RuntimeError("the frame did not reach the watched port")
        return first, written

    @override
    def state_dict(self) -> dict[str, int]:
        return {"frames": self.frames}

    @override
    def load_state_dict(self, state: Mapping[str, int], /) -> None:
        self.frames = int(state["frames"])
