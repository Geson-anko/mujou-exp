r"""The nodes of PLAN.md, sections 3-4, written with the public API of
``mujou.pc`` (``pc.functional``, ``pc.clock.LearnClock``, ``pc.msg``).

``Dynamics`` predicts the minGRU state $s_k$ of the frame from the remembered
$z_{k-1}, s_{k-1}$; ``Compare`` reads $\hat z_k = W_o s_k$ out and compares it
with $z_k$ (``SIGReg``, in ``exp.lib.sigreg``, keeps $z$ isotropic and Gaussian).
One stream: the batch is ``()``.

Each node differentiates only its own computation, from detached inputs
(``# mujou: local-autograd``), and sends the gradients of its energy back as
messages.
"""

import math
from typing import override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity
from mujou.pc.optim import Optimizer
from mujou.types import Device
from torch import nn

from exp.lib.mingru import MinGRU, ResidualMinGRU
from mujou import DictSpec, MultiPort, Node, Port, TensorSpec, TimestampSpec

__all__ = [
    "Compare",
    "Dynamics",
    "DynamicsLayer",
    "compare_energy",
    "is_open",
    "prediction_spec",
]


def prediction_spec(z_spec: DictSpec[Activity]) -> DictSpec[pc.msg.Prediction]:
    r"""The spec of ``Compare.next``: $\hat z_k$, $z_k$, ``mask`` and ``time``."""
    z = z_spec.field("h", TensorSpec)
    return DictSpec(
        pc.msg.Prediction,
        y_hat=z,
        y=z,
        mask=z_spec.field("mask", TensorSpec),
        time=z_spec.field("time", TimestampSpec),
    )


def is_open(count: int, open_loop: float, cycle: int) -> bool:
    """Whether the ``count``-th prediction of a video (1: the first, from its
    first frame) starts from the model's own prediction rather than the
    observed $z$: the last ``open_loop`` of every ``cycle`` predictions."""
    observed = round(cycle * (1 - open_loop))
    return open_loop > 0 and (count - 1) % cycle >= observed


# --- Dynamics ----------------------------------------------------------------


class Dynamics(Node):
    r"""$s_k = \mathrm{minGRU}(z_{k-1}, s_{k-1})$: a node of PC-ALM whose input
    is the past.

    It receives $z$ from ``Z``. In the update that first sees a new frame (a new
    ``time``), it remembers the $z$ it received last (that of the frame before)
    as $z_{k-1}$ and the ``mu`` it sent last as $s_{k-1}$, and sets ``r`` and
    ``lam`` to zero. Both are constants for the frame: the credit stops here, and
    the node sends no ``g_in`` back to ``Z`` (the port stays at its zeros). On
    the first frame of a video (``reset``), $s := 0$ and there is no $z_{k-1}$:
    it sends ``mu = 0`` with ``mask`` False, predicting nothing.

    Otherwise it relaxes ``r`` and ``lam`` by the ``g`` that ``Compare`` sends
    and learns as ``pc.Layer`` does, from ``delta = r + lam``.

    With ``open_loop`` > 0, it also receives ``Compare``'s prediction on
    ``z_hat`` and, for the predictions that ``is_open`` says, remembers its own
    last $\hat z_{k-1}$ instead of $z_{k-1}$: it learns to predict on from its
    own predictions, as a rollout runs. The target stays the observed $z_k$.

    With ``residual``, it also sends the $z_{k-1}$ it remembered on ``base``,
    for ``Compare`` to predict $\hat z_k = z_{k-1} + W_o s_k$.
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
        z = act_spec.field("h", TensorSpec)
        d_s = module.gate.out_features
        s = TensorSpec((d_s,))
        dev = self.device
        self.r, self.lam = s.zeros(device=dev), s.zeros(device=dev)
        self.z_prev, self.s_prev = (
            z.zeros(device=dev),
            s.zeros(device=dev),
        )  # remembered
        self.z_last, self.mu = (
            z.zeros(device=dev),
            s.zeros(device=dev),
        )  # received, sent
        self.has_prev = False  # whether z_prev is of this video
        self.register_state("has_prev")
        self.predictions = 0  # of this video so far
        self.register_state("predictions")
        self.z_hat_last = z.zeros(device=dev)  # the prediction of the frame
        self.z_hat_ns = -1  # the time of the frame z_hat_last predicts
        self.register_state("z_hat_ns")
        self.seen_ns = -1  # the time of the frame z_last is of
        self.register_state("seen_ns")
        if open_loop > 0:
            self.z_hat = Port(recv=prediction_spec(act_spec))
        if residual:  # the z_{k-1} it remembered, as the base of the prediction
            self.base = Port(send=act_spec)
        self.act_spec = pc.msg.activity_spec(s, act_spec.field("y", TensorSpec))
        self.z = Port(send=z, recv=act_spec)  # sends nothing back: zeros
        self.next = MultiPort(send=self.act_spec, recv=s)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.z.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            self._new_frame(bool(act["reset"]))
        mask = bool(act["mask"]) and self.has_prev
        learn = tick.learn and mask
        if self._open_loop > 0:
            p = self.z_hat.recv()
            if bool(p["mask"]) and p["time"] == act["time"]:
                self.z_hat_last, self.z_hat_ns = p["y_hat"], p["time"].ns
        if self.has_prev:
            delta = self.r + self.lam if mask else torch.zeros_like(self.r)
            s_prev = self.s_prev
            mu, _, g_theta = pc.functional.vjp(
                lambda z: self.module(z, s_prev),
                self.z_prev,
                lambda _: delta,
                self._params if learn else (),
            )
        else:
            mu, g_theta = torch.zeros_like(self.r), []
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        pc.functional.relax_(self.r, self.lam, g, tick.dt, self._tau)
        self.mu = mu
        self.z_last = act["h"]
        self.seen_ns = act["time"].ns
        self.next.send({**act, "h": mu, "mask": torch.tensor(mask)})
        if self._residual:
            self.base.send({**act, "h": self.z_prev, "mask": torch.tensor(mask)})
        if learn:
            self.optimizer.step_(g_theta, tick.elapsed)

    def _new_frame(self, reset: bool) -> None:
        self.r.zero_()
        self.lam.zero_()
        if reset:  # a new video: s := 0, and no z before
            self.s_prev = torch.zeros_like(self.s_prev)
            self.mu = torch.zeros_like(self.mu)
            self.has_prev = False
            self.predictions = 0
            return
        self.predictions = self.predictions + 1 if self.has_prev else 1
        # Its own prediction of the frame before, if it came in that frame (from
        # the second prediction of a video on, in a run stepped by hand).
        opened = self.z_hat_ns == self.seen_ns and is_open(
            self.predictions, self._open_loop, self._open_cycle
        )
        self.z_prev = self.z_hat_last if opened else self.z_last
        self.s_prev = self.mu
        self.has_prev = True


class DynamicsLayer(Node):
    r"""$y_k = x_k + h_k$ with $h_k = \mathrm{minGRU}(\mathrm{GELU}(x_k),
    h_{k-1})$ (``ResidualMinGRU``): a node of PC-ALM stacked on ``Dynamics`` or
    on another of it, whose input $x_k$ is of the same frame.

    In the update that first sees a new frame (a new ``time`` of its input), it
    remembers the state $h$ it computed last as $h_{k-1}$ (0 on the first frame
    of a video, ``reset``) and sets ``r`` and ``lam`` to zero. From the second
    frame of a video on, it sends $y_k$, as ``Dynamics`` moves its state from
    then on; while the frame has a target (``mask``), it also sends the credit
    of $x_k$ back (``g_in``: the input is of this frame, so the credit goes on),
    relaxes ``r`` and ``lam`` by the ``g`` from ``next`` and learns as
    ``pc.Layer`` does, from ``delta = r + lam``. On the first frame, it passes
    $x_k$ (0) on and keeps its state at 0.
    """

    def __init__(
        self,
        module: ResidualMinGRU,
        *,
        act_spec: DictSpec[Activity],
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
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
        dev = self.device
        self.r, self.lam = s.zeros(device=dev), s.zeros(device=dev)
        self.h_prev, self.h_last = s.zeros(device=dev), s.zeros(device=dev)
        self.first = True  # the first frame of a video: nothing to predict from
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
            self.first = bool(act["reset"])
            if self.first:
                self.h_last = torch.zeros_like(self.h_last)
            self.h_prev = self.h_last
        x = act["h"]
        mask = bool(act["mask"])
        learn = tick.learn and mask
        if not self.first:
            delta = self.r + self.lam if mask else torch.zeros_like(self.r)
            h_prev = self.h_prev
            mu, g_in, g_theta = pc.functional.vjp(
                lambda x: self.module(x, h_prev),
                x,
                lambda _: delta,
                self._params if learn else (),
            )
            self.h_last = mu - x
        else:
            mu, g_in, g_theta = x, torch.zeros_like(x), []
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        pc.functional.relax_(self.r, self.lam, g, tick.dt, self._tau)
        self.prev.send(g_in)
        self.next.send({**act, "h": mu})
        if learn:
            self.optimizer.step_(g_theta, tick.elapsed)


# --- Compare -------------------------------------------------------------------


def compare_energy(
    readout: nn.Module,
    s: torch.Tensor,
    z: torch.Tensor,
    kappa: float,
    base: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""$(1 - \kappa) \frac{1}{D} \lVert \hat z - z \rVert^2$ with
    $\hat z = W_o s$, or $\hat z = b + W_o s$ from a ``base`` $b$ (a constant:
    the residual prediction)."""
    z_hat = readout(s) if base is None else base + readout(s)
    return (1 - kappa) * ((z_hat - z) ** 2).mean(-1)


class Compare(Node):
    r"""The output node of the prediction: $\hat z_k = W_o s_k$ against $z_k$.

    Its energy is ``compare_energy``; it sends its gradient with respect to
    $s$ to ``Dynamics`` and with respect to $z$ to ``Z``, and learns $W_o$ by
    the one with respect to $W_o$, when both inputs are of the same frame and
    ``Dynamics`` predicted (its ``mask``). With ``residual``, it predicts
    $\hat z_k = z_{k-1} + W_o s_k$ from the $z_{k-1}$ that ``Dynamics`` sends on
    ``base`` (a constant of the energy). It scales the gradient it sends to
    ``Z`` by ``target_weight``: 0 makes the target a constant of the energy (a
    stop gradient), a fraction lets the encoder learn to make $z$ predictable
    by that much. It sends the prediction ``y_hat``
    $= \hat z_k$, the target ``y`` $= z_k$, their ``mask`` and ``time`` (a
    ``pc.msg.Prediction``), and the prediction error ``loss`` $\frac{1}{D}
    \lVert \hat z_k - z_k \rVert^2$ on ``error``.
    """

    def __init__(
        self,
        readout: nn.Linear,
        *,
        s_spec: DictSpec[Activity],
        z_spec: DictSpec[Activity],
        kappa: float,
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        target_weight: float = 1.0,
        residual: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self.module = readout
        self._kappa = kappa
        self._target_weight = target_weight
        self._residual = residual
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(readout.parameters())
        self.optimizer = optimizer.bind(self._params)
        s, z = s_spec.field("h", TensorSpec), z_spec.field("h", TensorSpec)
        self.s = Port(send=s, recv=s_spec)
        self.z = Port(send=z, recv=z_spec)
        self.next = Port(send=prediction_spec(z_spec))
        if residual:  # the z_{k-1} that Dynamics remembered
            self.base = Port(recv=z_spec)
        self.error = Port(send=TensorSpec(()), initial=torch.tensor(math.nan))

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        s, z = self.s.recv(), self.z.recv()
        tick = self.clock.tick(dt, z["time"], self.timer.now())
        mask = bool(s["mask"]) and bool(z["mask"]) and s["time"] == z["time"]
        base = None
        if self._residual:
            b = self.base.recv()
            mask = mask and bool(b["mask"]) and b["time"] == z["time"]
            base = b["h"]
        learn = tick.learn and mask
        with torch.no_grad():
            y_hat = self.module(s["h"])
            if base is not None:
                y_hat = base + y_hat
        if mask:
            g_s, g_z, g_theta = self._vjp(s["h"], z["h"], learn, base)
            self.s.send(g_s)
            self.z.send(self._target_weight * g_z)
        else:
            self.s.send(torch.zeros_like(s["h"]))
            self.z.send(torch.zeros_like(z["h"]))
            g_theta = []
        self.next.send(
            {"y_hat": y_hat, "y": z["h"], "mask": torch.tensor(mask), "time": z["time"]}
        )
        if mask:
            self.error.send(((y_hat - z["h"]) ** 2).mean())
        if learn:
            self.optimizer.step_(g_theta, tick.elapsed)

    def _vjp(
        self, s: torch.Tensor, z: torch.Tensor, learn: bool, base: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        params = self._params if learn else []
        with torch.enable_grad():
            s = s.detach().requires_grad_()
            z = z.detach().requires_grad_()
            energy = compare_energy(self.module, s, z, self._kappa, base)
            g_s, g_z, *g_theta = torch.autograd.grad(  # mujou: local-autograd
                energy, [s, z, *params]
            )
        return g_s, g_z, g_theta
