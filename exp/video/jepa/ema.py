r"""The target encoder of I-JEPA (M17, ``vision.ema``): an exponential moving
average of the encoder's parameters, without gradient, whose tokens of the
whole stack are the prediction's targets.

After each learning step of the encoder, $\bar\theta \leftarrow m \bar\theta
+ (1 - m) \theta$ (``ema_``), $m$ constant per step (I-JEPA ramps it to 1
over a fixed number of epochs, which a continual stream does not have). The
optimizer of the node that learns the encoder makes the move, right after its
step (``Following``): ``JEPABP``'s for the whole encoder, and in PC-ALM each
encoder node's for its own stage.

In PC-ALM the encoder is cut into nodes, one ``Stage`` each. Each node holds
the copy of its own stage (``TargetStage``) and moves it after each step of
its own optimizer (``Following``): nothing of another node is read, and no
barrier holds the nodes together. The node's activity carries the target
encoder's path along the packed tokens, its $T$ tokens of the whole stack
after them, so the head receives the targets of a stack with the tokens of
the same stack, however the nodes are scheduled. The path is detached in
each node: no credit flows along it and the copies get no gradient.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, override

import torch
from mujou.pc.optim import BoundOptimizer, Optimizer
from torch import Tensor, nn

from exp.video.jepa.vit import Stage

__all__ = ["Following", "TargetStage", "ema_"]


@torch.no_grad()
def ema_(target: nn.Module, online: nn.Module, momentum: float) -> None:
    r"""$\bar\theta \leftarrow m \bar\theta + (1 - m) \theta$ in place for the
    parameters $\bar\theta$ of ``target`` and $\theta$ of ``online`` (of the
    same structure), $m$ = ``momentum``: $m = 1$ leaves ``target`` as it is,
    $m = 0$ makes it ``online``'s (``torch.lerp`` is exact at both ends)."""
    for t, p in zip(target.parameters(), online.parameters(), strict=True):
        t.lerp_(p, 1 - momentum)


class TargetStage(nn.Module):
    """The module of an encoder node of PC-ALM with a target encoder: its
    ``stage`` on the packed tokens and the copy ``target`` of the stage on the
    target encoder's path.

    The first node (``first``) receives the packed sequence ``(B, L, P + 1)``
    and starts the path from its target view, the first $T$ = ``segments[0]``
    rows; the others receive ``(B, L + T, D)``, the packed tokens and then the
    path. It returns ``(B, L + T, D')``: ``stage(x, segments)`` of the packed
    rows, then ``target(path, (T,))`` of the path detached (no credit of the
    path reaches the node before, and ``target``, whose parameters require no
    gradient, is not learned by the node). Its parameters are the stage's and
    the copy's (the copy's in the node's state)."""

    def __init__(
        self, stage: Stage, target: Stage, segments: Sequence[int], *, first: bool
    ) -> None:
        super().__init__()
        self.stage, self.target = stage, target
        self._segments = tuple(segments)
        self._first = first

    @override
    def forward(self, x: Tensor) -> Tensor:
        t = self._segments[0]
        packed, path = (x, x[:, :t]) if self._first else (x[:, :-t], x[:, -t:])
        online = self.stage(packed, self._segments)
        return torch.cat([online, self.target(path.detach(), (t,))], 1)


class Following(Optimizer):
    """``inner``, and after each of its steps ``ema_(target, online,
    momentum)``: how the node that learns ``online`` (an encoder node with a
    ``TargetStage``, or ``JEPABP``) moves its parameters. Bind it to the
    node's parameters that learn, ``online``'s among them; the state of the
    bound optimizer is ``inner``'s (``target`` is in the node's state as a
    module of its own)."""

    def __init__(
        self, inner: Optimizer, *, online: nn.Module, target: nn.Module, momentum: float
    ) -> None:
        self._inner, self._online, self._target = inner, online, target
        self._momentum = momentum

    @override
    def bind(self, params: Iterable[Tensor]) -> BoundOptimizer:
        bound = self._inner.bind(params)
        return _BoundFollowing(bound, self._online, self._target, self._momentum)


class _BoundFollowing(BoundOptimizer):
    def __init__(
        self,
        inner: BoundOptimizer,
        online: nn.Module,
        target: nn.Module,
        momentum: float,
    ) -> None:
        self._inner, self._online, self._target = inner, online, target
        self._momentum = momentum

    @override
    def step_(self, grads: Sequence[Tensor], elapsed: float) -> None:
        self._inner.step_(grads, elapsed)
        ema_(self._target, self._online, self._momentum)

    @override
    def state_dict(self) -> dict[str, Any]:
        return self._inner.state_dict()

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._inner.load_state_dict(state)
