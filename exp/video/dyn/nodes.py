r"""The PC-ALM nodes of the dynamics, written with the public API of
``mujou.pc`` (``pc.functional``, ``pc.clock.LearnClock``, ``pc.msg``).

``Source`` shows the host's samples downstream; ``Dynamics`` predicts the
minGRU state $s_k$ of the frame from the remembered $x_{k-1}, s_{k-1}$;
``DynamicsLayer`` stacks a layer of the same frame on it; ``Compare`` reads
$\hat x_k = W_o y_L$ out and compares it with $x_k$. ``DynamicsBlock`` is
``Dynamics -> DynamicsLayer x (nodes - 1) -> Compare``, fed by one upstream
port that only sends.

No credit reaches the input of the dynamics, by the types of the connections
rather than by a promise to send zeros: ``Source.out`` only sends, and
``Dynamics.z`` and ``Compare.target`` only receive, so a port that sends
credit cannot be connected to them (``Port.connect`` raises ``TypeError``).
The input $x_{k-1}$ and the target $x_k$ are constants of the energy.

The nodes take one stream (the batch is ``()``) or ``B`` independent streams
side by side (``(B,)``, that of the ``act_spec`` they are given): each memory
below is then one per sample, and each sample resets, counts its predictions
and carries its credit on its own. A node sends the credit of each sample (the
gradient of that sample's energy), and learns from the mean over the batch
(``g_theta / B``, as ``pc.Layer`` does). Each node differentiates only its own
computation, from detached inputs, and sends the gradients of its energy back
as messages.
"""

import math
from collections.abc import Callable
from typing import cast, override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity, rows
from mujou.pc.optim import Optimizer
from mujou.time import Timestamp
from mujou.types import Device, Size
from torch import nn

import mujou
from exp.lib.mingru import MinGRU, MinGRUTrace, ResidualMinGRU
from exp.video.dyn.modules import DynamicsModules
from exp.video.schema.dyn import DynamicsConf
from mujou import DictSpec, MultiPort, Node, Port, TensorSpec, TimestampSpec

__all__ = [
    "Compare",
    "Dynamics",
    "DynamicsBlock",
    "DynamicsLayer",
    "Source",
    "compare_energy",
    "is_open",
    "open_mask",
    "prediction_spec",
]


def prediction_spec(x_spec: DictSpec[Activity]) -> DictSpec[pc.msg.Prediction]:
    r"""The spec of ``Compare.next``: $\hat x_k$, $x_k$, ``mask`` and ``time``."""
    x = x_spec.field("h", TensorSpec)
    return DictSpec(
        pc.msg.Prediction,
        y_hat=x,
        y=x,
        mask=x_spec.field("mask", TensorSpec),
        time=x_spec.field("time", TimestampSpec),
    )


def is_open(count: int, open_loop: float, cycle: int) -> bool:
    """Whether the ``count``-th prediction of a stream's video (1: the first,
    from its first frame) starts from the model's own prediction rather than
    the observed input: the last ``open_loop`` of every ``cycle`` predictions."""
    observed = round(cycle * (1 - open_loop))
    return open_loop > 0 and (count - 1) % cycle >= observed


def open_mask(counts: torch.Tensor, open_loop: float, cycle: int) -> torch.Tensor:
    """``is_open`` of each of ``counts`` (the number of predictions of each
    stream's video so far, ``(*batch,)``), as a bool tensor."""
    observed = round(cycle * (1 - open_loop))
    if open_loop <= 0:
        return torch.zeros_like(counts, dtype=torch.bool)
    return (counts - 1) % cycle >= observed


# --- Source --------------------------------------------------------------------


class Source(Node):
    """Shows the host's samples downstream and takes no credit back.

    The host writes a ``pc.msg.Sample`` to ``prev``: ``x`` ``(*batch, *x)``,
    ``y`` unused zeros ``(*batch, 1)`` (the target of the dynamics is ``x``
    itself), ``mask`` and ``reset`` ``batch``. In each ``on_update`` the node
    sends from ``out`` an ``Activity`` whose ``h`` is ``x``, with the time the
    host wrote it, as ``pc.Input`` does. ``out`` only sends: it connects to any
    number of ports that only receive, and to none that sends credit back.

    Args:
        x: The spec of ``x``, ``(*batch, *x)``.
        period: The period of the node in seconds.
        batch: The batch ``x`` starts with: ``()`` or ``B``.
        device: Where the node computes.
    """

    def __init__(
        self, *, x: TensorSpec, period: float, batch: Size, device: Device = "cpu"
    ) -> None:
        super().__init__(device=device, period=period)
        y = TensorSpec((*pc.msg.batch_shape(batch), 1))
        self.act_spec = pc.msg.activity_spec(x, y, batch)
        mask = self.act_spec.field("mask", TensorSpec)
        self.prev = Port(
            recv=DictSpec(pc.msg.Sample, x=x, y=y, mask=mask, reset=mask)  # the host
        )
        self.out = Port(send=self.act_spec)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, written_at = self.prev.recv_with_time()
        self.out.send(
            {
                "h": sample["x"],
                "y": sample["y"],
                "mask": sample["mask"],
                "reset": sample["reset"],
                "time": Timestamp(ns=0) if written_at is None else written_at,
            }
        )


# --- Dynamics ------------------------------------------------------------------


class Dynamics(Node):
    r"""$s_k = \mathrm{minGRU}(x_{k-1}, s_{k-1})$: a node of PC-ALM whose input
    is the past.

    It receives $x$ on ``z`` (the observation in M14, an encoder's $z$ in
    M16), a port that only receives. In the update that first sees a new frame
    (a new ``time``), it remembers the $x$ it received last (that of the frame
    before) as $x_{k-1}$ and the ``mu`` it sent last as $s_{k-1}$, and sets
    ``r`` and ``lam`` to zero. Both are constants for the frame: the credit
    stops here. On the first frame of a video (``reset``), $s := 0$ and there
    is no $x_{k-1}$: it sends ``mu = 0`` with ``mask`` False, predicting
    nothing.

    Otherwise it relaxes ``r`` and ``lam`` by the ``g`` that the next nodes
    send and learns as ``pc.Layer`` does, from ``delta = r + lam``. The state
    carried to the next frame is ``mu``, not ``mu + r``.

    With ``open_loop`` > 0, it also receives ``Compare``'s prediction on
    ``z_hat`` and, for the predictions that ``is_open`` says, remembers its own
    last $\hat x_{k-1}$ instead of $x_{k-1}$: it learns to predict on from its
    own predictions, as a rollout runs. The target stays the observed $x_k$.

    With ``residual``, it also sends the $x_{k-1}$ it remembered on ``base``,
    for ``Compare`` to predict $\hat x_k = x_{k-1} + W_o y_L$.

    With ``trace``, it learns through time: in the update that first sees a
    frame, it moves the eligibility traces of its minGRU to $s_k$
    (``MinGRUTrace``, with the $x$ it remembered held) and learns by
    $-\delta^\top \partial s_k / \partial \theta$ (the $g_\theta$ of
    ``pc.functional.vjp``) with them, the gradient through $s_{k-1}, s_{k-2},
    \dots$ since the video began.

    With a batch of streams, ``has_prev`` and ``predictions`` (the count that
    ``is_open`` reads) are per stream, and a stream that resets does not touch
    the memory of the others.
    """

    def __init__(
        self,
        module: MinGRU,
        *,
        act_spec: DictSpec[Activity],
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        open_loop: float = 0.0,
        open_cycle: int = 20,
        residual: bool = False,
        trace: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self._tau = tau
        self._open_loop, self._open_cycle = open_loop, open_cycle
        self._residual = residual
        self.module = module
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(module.parameters())
        self.optimizer = optimizer.bind(self._params)
        batch = act_spec.field("mask", TensorSpec).shape
        self._samples = math.prod(batch)  # B
        self.trace = MinGRUTrace(module, tuple(batch)) if trace else None
        x = act_spec.field("h", TensorSpec)
        s = TensorSpec((*batch, module.gate.out_features))
        dev = self.device
        self.r, self.lam = s.zeros(device=dev), s.zeros(device=dev)
        self.z_prev, self.s_prev = x.zeros(device=dev), s.zeros(device=dev)  # kept
        self.z_last, self.mu = x.zeros(device=dev), s.zeros(device=dev)  # seen, sent
        # Per stream: whether z_prev is of this video, and the predictions of it.
        self.has_prev = torch.zeros(batch, dtype=torch.bool, device=dev)
        self.register_state("has_prev")
        self.predictions = torch.zeros(batch, dtype=torch.int64, device=dev)
        self.register_state("predictions")
        self.z_hat_last = x.zeros(device=dev)  # the prediction of the frame
        # Whether z_hat_last is the prediction of the frame z_last is of.
        self.z_hat_ok = torch.zeros(batch, dtype=torch.bool, device=dev)
        self.register_state("z_hat_ok")
        self.act_spec = pc.msg.activity_spec(s, act_spec.field("y", TensorSpec), batch)
        self.z = Port(recv=act_spec)  # receives only: no credit to the past
        self.next = MultiPort(send=self.act_spec, recv=s)
        if open_loop > 0:
            self.z_hat = Port(recv=prediction_spec(act_spec))
        if residual:  # the x_{k-1} it remembered, as the base of the prediction
            self.base = Port(send=act_spec)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.z.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            self._new_frame(act["reset"])
        mask = act["mask"] & self.has_prev
        learn = tick.learn and bool(mask.any())
        if self._open_loop > 0:
            p = self.z_hat.recv()
            if p["time"] == act["time"]:
                got = p["mask"]
                self.z_hat_last = torch.where(
                    rows(got, p["y_hat"]), p["y_hat"], self.z_hat_last
                )
                self.z_hat_ok = self.z_hat_ok | got
        if bool(self.has_prev.any()):
            delta = torch.where(rows(mask, self.r), self.r + self.lam, 0.0)
            s_prev = self.s_prev
            direct = learn and self.trace is None
            mu, _, g_theta = pc.functional.vjp(
                lambda z: self.module(z, s_prev),
                self.z_prev,
                lambda _: delta,
                self._params if direct else (),
            )
            if learn and self.trace is not None:
                # dE/dtheta = -J^T delta, as vjp's; the trace gives +J^T delta.
                g_theta = self.trace.grads(-delta)
            mu = torch.where(rows(self.has_prev, mu), mu, 0.0)
        else:
            mu, g_theta = torch.zeros_like(self.r), []
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        pc.functional.relax_(self.r, self.lam, g, tick.dt, self._tau)
        self.mu = mu
        self.z_last = act["h"]
        self.next.send({**act, "h": mu, "mask": mask})
        if self._residual:
            self.base.send({**act, "h": self.z_prev, "mask": mask})
        if learn:
            # theta moves by the mean gradient of the batch, g_theta / B.
            self.optimizer.step_([g / self._samples for g in g_theta], tick.elapsed)

    def _new_frame(self, reset: torch.Tensor) -> None:
        self.r.zero_()
        self.lam.zero_()
        # The streams that reset start a new video: s := 0, and no x before.
        predictions = torch.where(self.has_prev, self.predictions + 1, 1)
        # Its own prediction of the frame before, if it came in that frame.
        opened = self.z_hat_ok & open_mask(
            predictions, self._open_loop, self._open_cycle
        )
        z_prev = torch.where(rows(opened, self.z_last), self.z_hat_last, self.z_last)
        self.z_prev = torch.where(rows(reset, z_prev), self.z_prev, z_prev)
        self.mu = torch.where(rows(reset, self.mu), 0.0, self.mu)
        self.s_prev = self.mu
        if self.trace is not None:  # to s_k; a video that starts has none
            self.trace.step_(self.z_prev, self.s_prev)
            self.trace.reset_(reset)
        self.has_prev = ~reset
        self.predictions = predictions.masked_fill(reset, 0)
        self.z_hat_ok = torch.zeros_like(self.z_hat_ok)


class DynamicsLayer(Node):
    r"""$y_k = x_k + h_k$ with $h_k = \mathrm{minGRU}(\mathrm{GELU}(x_k),
    h_{k-1})$ (``ResidualMinGRU``): a node of PC-ALM stacked on ``Dynamics`` or
    on another of it, whose input $x_k$ is of the same frame.

    In the update that first sees a new frame (a new ``time`` of its input), it
    remembers the state $h$ it computed last as $h_{k-1}$ (0 on the first frame
    of a video, ``reset``) and sets ``r`` and ``lam`` to zero. From the second
    frame of a video on, it sends $y_k$; while the frame has a target
    (``mask``), it also sends the credit of $x_k$ back on ``prev`` (``g_in``:
    the input is of this frame, so the credit goes on), relaxes ``r`` and
    ``lam`` by the ``g`` from ``next`` and learns as ``pc.Layer`` does, from
    ``delta = r + lam``. On the first frame, it passes $x_k$ (0) on and keeps
    its state at 0. With a batch of streams, ``first`` is per stream. With
    ``trace``, it learns through time as ``Dynamics`` does, by the traces of
    its cell on $\mathrm{GELU}(x_k)$ (as $\partial y_k / \partial \theta =
    \partial h_k / \partial \theta$), moved when it first sees the frame.
    """

    def __init__(
        self,
        module: ResidualMinGRU,
        *,
        act_spec: DictSpec[Activity],
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        trace: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self._tau = tau
        self.module = module
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(module.parameters())
        self.optimizer = optimizer.bind(self._params)
        s = act_spec.field("h", TensorSpec)
        batch = act_spec.field("mask", TensorSpec).shape
        self._samples = math.prod(batch)  # B
        self.trace = MinGRUTrace(module.cell, tuple(batch)) if trace else None
        dev = self.device
        self.r, self.lam = s.zeros(device=dev), s.zeros(device=dev)
        self.h_prev, self.h_last = s.zeros(device=dev), s.zeros(device=dev)
        # Per stream: the first frame of a video, with nothing to predict from.
        self.first = torch.ones(batch, dtype=torch.bool, device=dev)
        self.register_state("first")
        self.act_spec = act_spec
        self.prev = Port(send=s, recv=act_spec)
        self.next = MultiPort(send=act_spec, recv=s)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            self.r.zero_()
            self.lam.zero_()
            self.first = act["reset"]
            self.h_last = torch.where(rows(self.first, self.h_last), 0.0, self.h_last)
            self.h_prev = self.h_last
            if self.trace is not None:
                self.trace.step_(nn.functional.gelu(act["h"]), self.h_prev)
                self.trace.reset_(self.first)
        x = act["h"]
        mask = act["mask"] & ~self.first
        learn = tick.learn and bool(mask.any())
        if bool((~self.first).any()):
            delta = torch.where(rows(mask, self.r), self.r + self.lam, 0.0)
            h_prev = self.h_prev
            direct = learn and self.trace is None
            mu, g_in, g_theta = pc.functional.vjp(
                lambda x: self.module(x, h_prev),
                x,
                lambda _: delta,
                self._params if direct else (),
            )
            if learn and self.trace is not None:
                # dE/dtheta = -J^T delta, as vjp's; the trace gives +J^T delta.
                g_theta = self.trace.grads(-delta)
            first = rows(self.first, x)
            mu = torch.where(first, x, mu)  # the first frame passes x on
            self.h_last = torch.where(first, self.h_last, mu - x)
        else:
            mu, g_in, g_theta = x, torch.zeros_like(x), []
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        pc.functional.relax_(self.r, self.lam, g, tick.dt, self._tau)
        self.prev.send(g_in)
        self.next.send({**act, "h": mu})
        if learn:
            self.optimizer.step_([g / self._samples for g in g_theta], tick.elapsed)


# --- Compare -------------------------------------------------------------------


def compare_energy(
    readout: Callable[[torch.Tensor], torch.Tensor],
    s: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""$\frac{1}{D} \lVert \hat x - x \rVert^2$ of each sample, ``(*batch,)``:
    the prediction $\hat x = W_o s$ (``readout(s)``, ``s`` ``(*batch, d_s)``),
    or $\hat x = b + W_o s$ from a ``base`` $b$ ``(*batch, D)`` (the residual
    prediction), against ``target`` $x$ ``(*batch, D)``, averaged over the last
    axis."""
    x_hat = readout(s) if base is None else base + readout(s)
    return ((x_hat - target) ** 2).mean(-1)


class Compare(Node):
    r"""The output node of the prediction: $\hat x_k = W_o y_k$ against $x_k$.

    Its energy is ``compare_energy`` of the top output $y_k$ received on ``s``
    and the target $x_k$ received on ``target``; it sends its gradient with
    respect to $y_k$ back on ``s`` and learns $W_o$ by the one with respect to
    $W_o$, when both inputs are of the same frame and the dynamics predicted
    (their ``mask``). ``target`` only receives: the target is a constant of the
    energy. With ``residual``, it predicts $\hat x_k = x_{k-1} + W_o y_k$ from
    the $x_{k-1}$ that ``Dynamics`` sends on ``base`` (a constant too). It sends
    the prediction ``y_hat`` $= \hat x_k$, the target ``y`` $= x_k$, their
    ``mask`` and ``time`` (a ``pc.msg.Prediction``) on ``next``, in every
    update whose inputs are of the same frame (none while ``s`` is still of
    the frame before), and the prediction error $\frac{1}{D} \lVert \hat x_k -
    x_k \rVert^2$ on ``error``. The first prediction of a frame on ``next`` is
    the prequential one; the later ones are of the weights as they learn (what
    ``Dynamics`` remembers with ``open_loop``). ``error`` is sent once a frame,
    from the first prediction: the prequential error.

    With a batch of streams, the energy is the sum of the samples that have a
    prediction (so each sample gets the gradient of its own energy), $W_o$
    learns from the mean over the batch, and ``error`` is the mean over the
    samples that have one.
    """

    def __init__(
        self,
        readout: nn.Module,
        *,
        s_spec: DictSpec[Activity],
        target_spec: DictSpec[Activity],
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        residual: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self.module = readout
        self._residual = residual
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(readout.parameters())
        self.optimizer = optimizer.bind(self._params)
        self._samples = math.prod(target_spec.field("mask", TensorSpec).shape)  # B
        self.s = Port(send=s_spec.field("h", TensorSpec), recv=s_spec)
        self.target = Port(recv=target_spec)  # receives only: a constant
        self.next = Port(send=prediction_spec(target_spec))
        if residual:  # the x_{k-1} that Dynamics remembered
            self.base = Port(recv=target_spec)
        self.error = Port(send=TensorSpec(()), initial=torch.tensor(math.nan))
        self.error_sent = False  # the error of the frame of the target was sent
        self.register_state("error_sent")

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        s, x = self.s.recv(), self.target.recv()
        tick = self.clock.tick(dt, x["time"], self.timer.now())
        if tick.new:
            self.error_sent = False
        synced = s["time"] == x["time"]  # the inputs are of the same frame
        mask = s["mask"] & x["mask"]
        base = None
        if self._residual:
            b = self.base.recv()
            synced = synced and b["time"] == x["time"]
            mask = mask & b["mask"]
            base = b["h"]
        if not synced:
            mask = torch.zeros_like(mask)
        any_mask = bool(mask.any())
        learn = tick.learn and any_mask
        with torch.no_grad():
            y_hat = self.module(s["h"])
            if base is not None:
                y_hat = base + y_hat
        if any_mask:
            g_s, g_theta = self._vjp(s["h"], x["h"], mask, learn, base)
            self.s.send(g_s)
        else:
            self.s.send(torch.zeros_like(s["h"]))
            g_theta = []
        if synced:  # no prediction of x_k from the s of another frame
            y = x["h"]
            self.next.send({"y_hat": y_hat, "y": y, "mask": mask, "time": x["time"]})
        if any_mask and not self.error_sent:  # the prequential error, once
            self.error.send(((y_hat - x["h"]) ** 2)[mask].mean())
            self.error_sent = True
        if learn:
            self.optimizer.step_([g / self._samples for g in g_theta], tick.elapsed)

    def _vjp(
        self,
        s: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        learn: bool,
        base: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        params = self._params if learn else []
        with torch.enable_grad():
            s = s.detach().requires_grad_()
            energy = compare_energy(self.module, s, target, base)
            energy = torch.where(mask, energy, 0.0).sum()
            # The gradient of this node's own energy, from a detached leaf.
            g_s, *g_theta = torch.autograd.grad(  # mujou: local-autograd
                energy, [s, *params]
            )
        return g_s, g_theta


# --- The block ----------------------------------------------------------------


class DynamicsBlock(mujou.Network):
    """``Dynamics -> DynamicsLayer x (conf.nodes - 1) -> Compare`` of the
    modules ``mods``, fed by one upstream port that only sends.

    The nodes are ``dynamics``, ``dynamics2``, ... (``names``) and
    ``compare``. ``connect(upstream)`` connects the upstream to
    ``dynamics.z`` (the input) and to ``compare.target`` (the target); with
    ``conf.open_loop`` > 0, ``compare.next`` also feeds ``dynamics.z_hat``,
    and with ``conf.residual``, ``dynamics.base`` feeds ``compare.base``.
    ``next`` is ``compare.next`` (the prequential prediction) and ``error``
    ``compare.error``. Every node binds its own copy of ``optimizer``.

    Args:
        mods: The modules, on ``device``.
        act_spec: The spec of what the upstream sends, ``h`` ``(*batch,
            d_in)``.
        conf: The dynamics; ``d_s`` and ``nodes`` are those of ``mods``.
        tau: The time constants of PC-ALM.
        optimizer: The optimizer of every node.
        learn_interval: As in ``pc.clock.LearnClock``.
        device: Where the nodes compute.
    """

    dynamics2: DynamicsLayer  # set by name: those of conf.nodes only
    dynamics3: DynamicsLayer

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
        super().__init__()
        self.mods = mods
        self.dynamics = Dynamics(
            mods.step.first,
            act_spec=act_spec,
            tau=tau,
            optimizer=optimizer,
            learn_interval=learn_interval,
            open_loop=conf.open_loop,
            open_cycle=conf.open_cycle,
            residual=conf.residual,
            trace=conf.trace,
            device=device,
        )
        self.names = ("dynamics",) + tuple(
            f"dynamics{i + 2}" for i in range(len(mods.step.rest))
        )
        top: Dynamics | DynamicsLayer = self.dynamics
        for name, layer in zip(self.names[1:], mods.step.rest, strict=True):
            node = DynamicsLayer(
                cast(ResidualMinGRU, layer),
                act_spec=top.act_spec,
                tau=tau,
                optimizer=optimizer,
                learn_interval=learn_interval,
                trace=conf.trace,
                device=device,
            )
            setattr(self, name, node)
            top.next.connect(node.prev)
            top = node
        self.compare = Compare(
            mods.readout,
            s_spec=top.act_spec,
            target_spec=act_spec,
            tau=tau,
            optimizer=optimizer,
            learn_interval=learn_interval,
            residual=conf.residual,
            device=device,
        )
        top.next.connect(self.compare.s)
        if conf.open_loop > 0:
            self.compare.next.connect(self.dynamics.z_hat)
        if conf.residual:
            self.dynamics.base.connect(self.compare.base)
        self.next, self.error = self.compare.next, self.compare.error

    def connect(self, upstream: Port[Activity]) -> None:
        """Feed the block from ``upstream`` (``Source.out`` in M14, the
        encoder's output in M16): the input and the target of the dynamics."""
        upstream.connect(self.dynamics.z)
        upstream.connect(self.compare.target)
