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

from exp.lib.mingru import MinGRU
from mujou import DictSpec, MultiPort, Node, Port, TensorSpec, TimestampSpec

__all__ = [
    "Compare",
    "Dynamics",
    "compare_energy",
]


# --- minGRU and Dynamics -----------------------------------------------------


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
    """

    def __init__(
        self,
        module: MinGRU,
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
        self.next.send({**act, "h": mu, "mask": torch.tensor(mask)})
        if learn:
            self.optimizer.step_(g_theta, tick.elapsed)

    def _new_frame(self, reset: bool) -> None:
        self.r.zero_()
        self.lam.zero_()
        if reset:  # a new video: s := 0, and no z before
            self.s_prev = torch.zeros_like(self.s_prev)
            self.mu = torch.zeros_like(self.mu)
            self.has_prev = False
            return
        self.z_prev = self.z_last
        self.s_prev = self.mu
        self.has_prev = True


# --- Compare -------------------------------------------------------------------


def compare_energy(
    readout: nn.Module, s: torch.Tensor, z: torch.Tensor, kappa: float
) -> torch.Tensor:
    r"""$(1 - \kappa) \frac{1}{D} \lVert W_o s - z \rVert^2$."""
    return (1 - kappa) * ((readout(s) - z) ** 2).mean(-1)


class Compare(Node):
    r"""The output node of the prediction: $\hat z_k = W_o s_k$ against $z_k$.

    Its energy is ``compare_energy``; it sends its gradient with respect to
    $s$ to ``Dynamics`` and with respect to $z$ to ``Z``, and learns $W_o$ by
    the one with respect to $W_o$, when both inputs are of the same frame and
    ``Dynamics`` predicted (its ``mask``). With ``stop_target``, it sends
    zeros to ``Z`` instead: the target is a constant of the energy, as a stop
    gradient. It sends the prediction ``y_hat``
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
        stop_target: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self.module = readout
        self._kappa = kappa
        self._stop_target = stop_target
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(readout.parameters())
        self.optimizer = optimizer.bind(self._params)
        s, z = s_spec.field("h", TensorSpec), z_spec.field("h", TensorSpec)
        self.s = Port(send=s, recv=s_spec)
        self.z = Port(send=z, recv=z_spec)
        mask = z_spec.field("mask", TensorSpec)
        self.next = Port(
            send=DictSpec(
                pc.msg.Prediction,
                y_hat=z,
                y=z,
                mask=mask,
                time=z_spec.field("time", TimestampSpec),
            )
        )
        self.error = Port(send=TensorSpec(()), initial=torch.tensor(math.nan))

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        s, z = self.s.recv(), self.z.recv()
        tick = self.clock.tick(dt, z["time"], self.timer.now())
        mask = bool(s["mask"]) and bool(z["mask"]) and s["time"] == z["time"]
        learn = tick.learn and mask
        with torch.no_grad():
            y_hat = self.module(s["h"])
        if mask:
            g_s, g_z, g_theta = self._vjp(s["h"], z["h"], learn)
            self.s.send(g_s)
            self.z.send(torch.zeros_like(g_z) if self._stop_target else g_z)
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
        self, s: torch.Tensor, z: torch.Tensor, learn: bool
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        params = self._params if learn else []
        with torch.enable_grad():
            s = s.detach().requires_grad_()
            z = z.detach().requires_grad_()
            energy = compare_energy(self.module, s, z, self._kappa)
            g_s, g_z, *g_theta = torch.autograd.grad(  # mujou: local-autograd
                energy, [s, z, *params]
            )
        return g_s, g_z, g_theta
