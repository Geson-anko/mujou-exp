r"""SIGReg (``exp/video/PLAN.md``, section 3): a node that keeps $z$ isotropic
and Gaussian, by the Epps-Pulley statistic on random projections of $z$ and the
past $z$."""

import math
from typing import override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity
from mujou.types import Device

from mujou import DictSpec, Node, Port, TensorSpec

__all__ = ["SIGReg", "T_POINTS", "epps_pulley", "insert_", "sigreg"]

T_POINTS = torch.linspace(-5, 5, 17)


def epps_pulley(x: torch.Tensor, t: torch.Tensor = T_POINTS) -> torch.Tensor:
    r"""The Epps-Pulley statistic of each column of ``x`` (``(N, M)``: ``N``
    samples of ``M`` projections) against $N(0, 1)$:
    $N \int \lvert \hat\varphi(t) - e^{-t^2/2} \rvert^2 e^{-t^2/2}\, dt$ by the
    trapezoid rule on ``t``. Returns ``(M,)``."""
    n = x.shape[0]
    t = t.to(x)
    tx = x.unsqueeze(-1) * t  # (N, M, T)
    re = torch.cos(tx).mean(0)  # (M, T): the empirical characteristic function
    im = torch.sin(tx).mean(0)
    phi = torch.exp(-(t**2) / 2)  # of N(0, 1), and the weight
    err = ((re - phi) ** 2 + im**2) * phi
    return n * torch.trapezoid(err, t, dim=-1)


def sigreg(
    z: torch.Tensor, buffer: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    r"""$\frac{1}{M} \sum_m \mathrm{EP}(\{a_m^\top z_n\})$ over $z$ ``(D,)`` and
    ``buffer`` ``(N - 1, D)``, along ``directions`` ``(M, D)`` (unit rows)."""
    samples = torch.cat([z.unsqueeze(0), buffer])  # (N, D)
    return epps_pulley(samples @ directions.T).mean()


def insert_(
    buffer: torch.Tensor, count: int, z: torch.Tensor, g: torch.Generator
) -> int:
    """Put ``z`` into ``buffer`` (``(capacity, D)``, of which ``count`` rows
    are filled) in place: the next empty row until it is full, then a row drawn
    uniformly. Returns the new count."""
    capacity = buffer.shape[0]
    if count < capacity:
        buffer[count] = z
        return count + 1
    buffer[int(torch.randint(capacity, (), generator=g, device=g.device))] = z
    return count


class SIGReg(Node):
    r"""$\kappa\,\mathrm{SIGReg}$ of $z_k$ with the past $z$
    (``exp/video/PLAN.md``, section 3).

    It holds no parameters: a buffer of ``capacity`` past $z$ (detached), and
    ``directions`` unit directions, drawn anew with each frame from the node's
    ``generator``. In the update that first sees a new frame, it puts the $z$
    it received last (of the frame before) into the buffer (``insert_``) and
    draws the directions. Each update, it sends $\kappa\,\partial
    \mathrm{SIGReg} / \partial z_k$ back to ``Z``, with ``buffer`` held
    constant, once the buffer holds ``min_samples - 1`` rows and the frame has a
    target (``mask``); zeros otherwise.
    """

    def __init__(
        self,
        *,
        act_spec: DictSpec[Activity],
        kappa: float,
        directions: int,
        capacity: int,
        tau: pc.TimeConstants,
        min_samples: int = 16,
        seed: int = 0,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period, seed=seed)
        self._kappa, self._min = kappa, min_samples
        z = act_spec.field("h", TensorSpec)
        (d,) = z.shape
        self.buffer = torch.zeros(capacity, d, device=self.device)
        self.count = 0
        self.register_state("count")
        self.directions = torch.zeros(directions, d, device=self.device)
        self.z_last = z.zeros(device=self.device)
        self.seen = -1  # the time (ns) of the frame seen last
        self.register_state("seen")
        # The g_in sent last and the z it was computed for, valid within a frame:
        # z changes only when the nodes before learn, so most updates resend it.
        self.g, self.g_of = z.zeros(device=self.device), z.zeros(device=self.device)
        self.cached = False
        self.register_state("cached")
        self.prev = Port(send=z, recv=act_spec)
        self.value = Port(send=TensorSpec(()), initial=torch.tensor(math.nan))

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        if act["time"].ns != self.seen:
            if self.seen >= 0:
                self.count = insert_(
                    self.buffer, self.count, self.z_last, self.generator
                )
            self.seen = act["time"].ns
            a = torch.randn(
                self.directions.shape, generator=self.generator, device=self.device
            )
            self.directions = a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
            self.cached = False
        self.z_last = act["h"]
        if not bool(act["mask"]) or self.count + 1 < self._min:
            self.prev.send(torch.zeros_like(act["h"]))
            return
        if not (self.cached and torch.equal(act["h"], self.g_of)):
            with torch.enable_grad():
                z = act["h"].detach().requires_grad_()
                value = sigreg(z, self.buffer[: self.count], self.directions)
                (g,) = torch.autograd.grad(value, [z])  # mujou: local-autograd
            self.g, self.g_of, self.cached = self._kappa * g, act["h"], True
            self.value.send(value.detach())
        self.prev.send(self.g)
