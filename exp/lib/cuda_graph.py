"""A function of tensors of fixed shapes, recorded in a CUDA graph at its first
call with each key and replayed after: the same kernels on the same inputs (the
result is that of the eager call, bit for bit) with far fewer launches from the
CPU, which bound an update of a node with a small batch on a GPU. The vision's
``pc.Layer`` nodes have it from mujou (``pc.Layer(cuda_graph=True)``); this is
the same for the nodes of the experiments.

The function must run the same kernels on every input of its shapes: no
synchronization with the host (``.item()``, ``bool(t)``, a branch on a value)
and parameters updated in place (as ``pc.optim`` does). Its random numbers, if
any, must come in as inputs.
"""

from collections.abc import Callable, Hashable, Sequence

import torch
from torch import Tensor

__all__ = ["Replayed", "graphed_vjp"]

_WARMUP = 3  # eager runs on a side stream before the capture, as torch advises


class Replayed:
    """``fn(key, *args) -> [tensors]`` replayed from a CUDA graph per ``key``
    (a hashable that chooses the computation, such as whether it learns).

    A call copies ``args`` into the graph's buffers, replays it and returns
    clones of its outputs, so a caller may keep them past the next replay (a
    node that compares what it sends with what it sent last does)."""

    def __init__(self, fn: Callable[..., Sequence[Tensor]]) -> None:
        self._fn = fn
        self._graphs: dict[
            Hashable, tuple[torch.cuda.CUDAGraph, list[Tensor], list[Tensor]]
        ] = {}

    def __call__(self, key: Hashable, *args: Tensor) -> list[Tensor]:
        if key not in self._graphs:
            self._graphs[key] = self._record(key, args)
        graph, inputs, outputs = self._graphs[key]
        for buf, a in zip(inputs, args, strict=True):
            buf.copy_(a)
        graph.replay()
        return [o.clone() for o in outputs]

    def _record(
        self, key: Hashable, args: Sequence[Tensor]
    ) -> tuple[torch.cuda.CUDAGraph, list[Tensor], list[Tensor]]:
        inputs = [a.clone() for a in args]
        device = inputs[0].device
        current = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(_WARMUP):
                self._fn(key, *inputs)
        current.wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        # Other nodes launch work from other threads while this one records.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            outputs = list(self._fn(key, *inputs))
        return graph, inputs, outputs


def graphed_vjp(
    fn: Callable[[Tensor], Tensor], params: Sequence[Tensor]
) -> Callable[..., tuple[Tensor, Tensor | None, list[Tensor]]]:
    r"""``vjp(x, delta, *, credit, learn) -> (fn(x), g_in, g_theta)`` with
    $g_\text{in} = -J^\top \delta$ (with ``credit``, else None) and
    $g_\theta = -J_\theta^\top \delta$ for ``params`` (with ``learn``, else
    none), as ``pc.functional.vjp`` gives them from the detached ``x``,
    replayed from a CUDA graph per ``(credit, learn)``."""

    def compute(key: tuple[bool, bool], x: Tensor, delta: Tensor) -> list[Tensor]:
        credit, learn = key
        with torch.enable_grad():
            leaf = x.detach().requires_grad_(credit)
            out = fn(leaf)
            wrt = [leaf] if credit else []
            wrt += list(params) if learn else []
            if not wrt:
                return [out.detach()]
            # The gradient of the node's own function, from a detached leaf.
            grads = torch.autograd.grad(  # mujou: local-autograd
                out, wrt, grad_outputs=-delta, materialize_grads=True
            )
        return [out.detach(), *grads]

    replayed = Replayed(compute)

    def vjp(
        x: Tensor, delta: Tensor, *, credit: bool, learn: bool
    ) -> tuple[Tensor, Tensor | None, list[Tensor]]:
        out, *grads = replayed((credit, learn), x, delta)
        if credit:
            return out, grads[0], grads[1:]
        return out, None, grads

    return vjp
