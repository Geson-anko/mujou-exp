r"""The baseline of backpropagation: the modules of the dynamics learned by
autograd inside one node, with everything else as in ``DynamicsBlock``."""

import math
from typing import cast, override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity, rows
from mujou.pc.optim import Optimizer
from mujou.types import Device
from torch import nn

from exp.lib.mingru import MinGRUTrace, ResidualMinGRU
from exp.video.dyn.modules import DynamicsModules, Top
from exp.video.dyn.nodes import compare_energy, open_mask, prediction_spec
from exp.video.schema.dyn import DynamicsConf
from mujou import DictSpec, Node, Port, TensorSpec

__all__ = ["DynamicsBP"]


class DynamicsBP(Node):
    r"""The dynamics learned by BP in one node: the loss
    $L_k = \frac1B \sum_b \frac1D \lVert \hat x_{k,b} - x_{k,b} \rVert^2$ over
    the samples that have a prediction, with $\hat x_k = W_o\,
    \mathrm{Top}(\mathrm{step}(x_{k-1}, s_{k-1}))$ (plus $x_{k-1}$ with
    ``residual``), as ``DynamicsBlock`` has it in its nodes.

    It has the same memory of $x_{k-1}, s_{k-1}$ (detached: the gradient
    stops at one frame, as in PC-ALM), the same ``LearnClock``, and receives
    on ``prev``, a port that only receives (no credit to the upstream). With
    ``open_loop``, it remembers its own prediction for the predictions that
    ``nodes.is_open`` says, as ``Dynamics`` does; with ``trace``, it learns the
    minGRU of each layer by the gradient of $L_k$ with respect to the layer's
    output times the layer's eligibility traces (``MinGRUTrace``, moved in the
    update that first sees a frame, with the layer's inputs held), as the nodes
    do.

    In the update that first sees a frame, it remembers the frame before and
    sends the prequential prediction (a ``pc.msg.Prediction``) on ``next`` and
    its error on ``error``. When its clock says, it learns all the modules from
    the gradient of $L_k$ (one optimizer bound to the parameters of ``step``
    and then of ``readout``), and then computes again the $s_k$ and $\hat x_k$
    it remembers for the next frame, at the weights it learned, as the nodes
    of PC-ALM carry theirs (the prediction it sent stays the prequential
    one). Its own prediction is remembered for the open loop only from a
    frame that predicted (``mask``), as in ``Dynamics``.

    Args:
        mods: The modules, on ``device``.
        act_spec: The spec of what the upstream sends, ``h`` ``(*batch,
            d_in)``.
        conf: The dynamics; ``d_s`` and ``nodes`` are those of ``mods``.
        tau: The time constants of PC-ALM (the period and ``max_dt``).
        optimizer: The optimizer of all the parameters.
        learn_interval: As in ``pc.clock.LearnClock``.
        device: Where the node computes.
    """

    def __init__(
        self,
        mods: DynamicsModules,
        *,
        act_spec: DictSpec[Activity],
        conf: DynamicsConf,
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self.mods = mods
        self.step, self.readout = mods.step, mods.readout
        self._top = Top(mods.step.d_s)
        self._open_loop, self._open_cycle = conf.open_loop, conf.open_cycle
        self._residual = conf.residual
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = [*self.step.parameters(), *self.readout.parameters()]
        self._n_step = len(list(self.step.parameters()))
        self.optimizer = optimizer.bind(self._params)
        batch = act_spec.field("mask", TensorSpec).shape
        self._samples = math.prod(batch)  # B
        self._traces: list[MinGRUTrace] = []
        if conf.trace:  # each an attribute of its own, so that it is state
            cells = [self.step.first] + [
                cast(ResidualMinGRU, layer).cell for layer in self.step.rest
            ]
            for i, cell in enumerate(cells):
                trace = MinGRUTrace(cell, tuple(batch))
                setattr(self, f"trace{i + 1}", trace)
                self._traces.append(trace)
        x = act_spec.field("h", TensorSpec)
        s = TensorSpec((*batch, self.step.d_state))
        dev = self.device
        self.z_prev, self.s_prev = x.zeros(device=dev), s.zeros(device=dev)
        self.z_last, self.s_last = x.zeros(device=dev), s.zeros(device=dev)
        self.z_hat_last = x.zeros(device=dev)
        # Per stream: whether z_prev is of this video, and the predictions of it.
        self.has_prev = torch.zeros(batch, dtype=torch.bool, device=dev)
        self.register_state("has_prev")
        self.predictions = torch.zeros(batch, dtype=torch.int64, device=dev)
        self.register_state("predictions")
        # Whether z_hat_last is a prediction of the frame z_last is of.
        self.z_hat_ok = torch.zeros(batch, dtype=torch.bool, device=dev)
        self.register_state("z_hat_ok")
        self.prev = Port(recv=act_spec)  # receives only: no credit to the input
        self.next = Port(send=prediction_spec(act_spec))
        self.error = Port(send=TensorSpec(()), initial=torch.tensor(math.nan))

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            self._new_frame(act)
        mask = act["mask"] & self.has_prev
        if tick.learn and bool(mask.any()):
            with torch.enable_grad():
                grads = self._grads(act["h"], mask)
            self.optimizer.step_(grads, tick.elapsed)
            # What the next frame starts from is of the weights after learning,
            # as the nodes of PC-ALM recompute theirs in every update.
            self.s_last, self.z_hat_last = self._predict()

    def loss(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
        ys: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        r"""$L_k$ at the current weights from the memory (a constant), of the
        ``target`` $x_k$ ``(*batch, D)`` of the samples in ``mask``
        (``batch``). With ``ys``, it appends the output $y_i$ ``(*batch,
        d_s)`` of each layer of the dynamics to it, in the graph of the loss.
        """
        st = self.step
        parts = self.s_prev.split(st.d_s, -1)
        y = st.first(self.z_prev, parts[0])
        out = [y]
        if ys is not None:
            ys.append(y)
        for layer, h in zip(st.rest, parts[1:], strict=True):
            y_next = layer(y, h)
            out.append(y_next - y)
            y = y_next
            if ys is not None:
                ys.append(y)
        s = torch.cat(out, -1)
        base = self.z_prev if self._residual else None
        energy = compare_energy(lambda s: self.readout(self._top(s)), s, target, base)
        return torch.where(mask, energy, 0.0).sum() / self._samples

    def _new_frame(self, act: Activity) -> None:
        # The streams that reset start a new video: s := 0, and no x before.
        reset = act["reset"]
        predictions = torch.where(self.has_prev, self.predictions + 1, 1)
        opened = self.z_hat_ok & open_mask(
            predictions, self._open_loop, self._open_cycle
        )
        z_prev = torch.where(rows(opened, self.z_last), self.z_hat_last, self.z_last)
        self.z_prev = torch.where(rows(reset, z_prev), self.z_prev, z_prev)
        self.s_last = torch.where(rows(reset, self.s_last), 0.0, self.s_last)
        self.s_prev = self.s_last
        self.has_prev = ~reset
        self.predictions = predictions.masked_fill(reset, 0)
        if self._traces:  # to s_k; a video that starts has none
            with torch.no_grad():
                self._step_traces(reset)
        s, z_hat = self._predict()
        x = act["h"]
        self.z_last, self.s_last, self.z_hat_last = x, s, z_hat
        mask = act["mask"] & self.has_prev
        self.z_hat_ok = mask
        self.next.send({"y_hat": z_hat, "y": x, "mask": mask, "time": act["time"]})
        if bool(mask.any()):
            self.error.send(((z_hat - x) ** 2)[mask].mean())

    @torch.no_grad()
    def _predict(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""$s_k$ ``(*batch, L d_s)`` and $\hat x_k$ ``(*batch, D)`` from the
        memory at the current weights ($s_k = 0$ where there is no
        $x_{k-1}$)."""
        has_prev = rows(self.has_prev, self.s_prev)
        s = self.s_prev
        if bool(self.has_prev.any()):
            s = torch.where(has_prev, self.step(self.z_prev, self.s_prev), s)
        z_hat = self.readout(self._top(s))
        if self._residual:
            z_hat = torch.where(has_prev, self.z_prev + z_hat, z_hat)
        return s, z_hat

    def _step_traces(self, reset: torch.Tensor) -> None:
        """Move the trace of each layer to the state of this frame, from the
        remembered $x_{k-1}$ and state."""
        st = self.step
        parts = self.s_prev.split(st.d_s, -1)
        self._traces[0].step_(self.z_prev, parts[0])
        y = st.first(self.z_prev, parts[0])
        for trace, layer, h in zip(self._traces[1:], st.rest, parts[1:], strict=True):
            trace.step_(nn.functional.gelu(y), h)
            y = layer(y, h)
        for trace in self._traces:
            trace.reset_(reset)

    def _grads(self, target: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        """The gradients of $L_k$ for the parameters; with traces, those of the
        dynamics are the traces times the gradient for each layer's output."""
        if not self._traces:
            loss = self.loss(target, mask)
            # BP inside this node, through its own modules from its memory.
            return list(
                torch.autograd.grad(  # mujou: local-autograd
                    loss, self._params, materialize_grads=True
                )
            )
        ys: list[torch.Tensor] = []
        loss = self.loss(target, mask, ys)
        readout = self._params[self._n_step :]
        g = torch.autograd.grad(  # mujou: local-autograd
            loss, [*readout, *ys], materialize_grads=True
        )
        g_readout, g_ys = g[: len(readout)], g[len(readout) :]
        g_step = [
            grad
            for trace, d in zip(self._traces, g_ys, strict=True)
            for grad in trace.grads(d)
        ]
        return [*g_step, *g_readout]
