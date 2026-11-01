r"""SIGReg (``exp/video/PLAN.md``, section 3): a node that keeps $z$ isotropic
and Gaussian, by the Epps-Pulley statistic on random projections of $z$ and the
past $z$."""

import math
from typing import override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity, rows
from mujou.types import Device

from mujou import DictSpec, Node, Port, TensorSpec

__all__ = [
    "SIGReg",
    "T_POINTS",
    "ecf_sums",
    "epps_pulley",
    "insert_",
    "sigreg",
    "sigreg_from_sums",
]

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
    r"""$\frac{1}{M} \sum_m \mathrm{EP}(\{a_m^\top z_n\})$ over $z$ and
    ``buffer`` ``(N - B, D)``, along ``directions`` ``(M, D)`` (unit rows). $z$
    is one sample ``(D,)`` or a batch of them ``(B, D)``: the statistic of the
    ``N = B + len(buffer)`` samples together."""
    samples = torch.cat([z.reshape(-1, z.shape[-1]), buffer])  # (N, D)
    return epps_pulley(samples @ directions.T).mean()


def ecf_sums(
    buffer: torch.Tensor, directions: torch.Tensor, t: torch.Tensor = T_POINTS
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""$\sum_n \cos(t\, a_m^\top z_n)$ and $\sum_n \sin(t\, a_m^\top z_n)$
    over the rows of ``buffer`` (``(N - B, D)``), each ``(M, T)``: the part of
    the empirical characteristic function that stays the same while only the
    current $z$ change."""
    tx = (buffer @ directions.T).unsqueeze(-1) * t.to(buffer)  # (N - B, M, T)
    return torch.cos(tx).sum(0), torch.sin(tx).sum(0)


def sigreg_from_sums(
    z: torch.Tensor,
    sums: tuple[torch.Tensor, torch.Tensor],
    n: int,
    directions: torch.Tensor,
    t: torch.Tensor = T_POINTS,
) -> torch.Tensor:
    """``sigreg(z, buffer, directions)`` from ``sums = ecf_sums(buffer,
    directions)`` and ``n = len(buffer) + B`` ($B$ = 1 for $z$ of ``(D,)``, else
    the rows of ``(B, D)``): the same value, at the cost of $z$ alone."""
    t = t.to(z)
    if z.dim() == 1:
        tx = (directions @ z).unsqueeze(-1) * t  # (M, T)
        cos, sin = torch.cos(tx), torch.sin(tx)
    else:
        tx = (z @ directions.T).unsqueeze(-1) * t  # (B, M, T)
        cos, sin = torch.cos(tx).sum(0), torch.sin(tx).sum(0)
    re = (sums[0] + cos) / n
    im = (sums[1] + sin) / n
    phi = torch.exp(-(t**2) / 2)
    err = ((re - phi) ** 2 + im**2) * phi
    return (n * torch.trapezoid(err, t, dim=-1)).mean()


def insert_(
    buffer: torch.Tensor, count: int, z: torch.Tensor, g: torch.Generator
) -> int:
    """Put ``z`` (``(D,)`` or the rows of ``(B, D)``) into ``buffer``
    (``(capacity, D)``, of which ``count`` rows are filled) in place: the next
    empty rows until it is full, then rows drawn uniformly: one row for one
    ``z``, and for several, distinct rows (without replacement, so the result
    does not depend on the order a scatter writes duplicates in; more rows than
    ``capacity`` keep the last ``capacity``). Returns the new count."""
    rows = z.reshape(-1, z.shape[-1])
    capacity = buffer.shape[0]
    fill = min(len(rows), capacity - count)
    buffer[count : count + fill] = rows[:fill]
    rest = rows[fill:]
    rest = rest[-capacity:]
    if len(rest) == 1:
        drawn = torch.randint(capacity, (1,), generator=g, device=g.device)
        buffer[drawn] = rest
    elif len(rest) > 1:
        drawn = torch.randperm(capacity, generator=g, device=g.device)[: len(rest)]
        buffer[drawn] = rest
    return count + fill


class SIGReg(Node):
    r"""$\kappa\,\mathrm{SIGReg}$ of $z_k$ with the past $z$
    (``exp/video/PLAN.md``, section 3).

    It holds no parameters: a buffer of ``capacity`` past $z$ (detached), and
    ``directions`` unit directions, drawn anew with each frame from the node's
    ``generator``. In the update that first sees a new frame, it puts the $z$
    it received last (of the frame before) into the buffer (``insert_``; that of
    one frame in ``every``, to spread the buffer over a longer time),
    draws the directions and sums the buffer's part of the characteristic
    function (``ecf_sums``), which stay the same through the frame. Each
    update, it sends $\kappa\,\partial \mathrm{SIGReg} / \partial z_k$ back to
    ``Z``, with ``buffer`` held
    constant, once the buffer holds ``min_samples - B`` rows and the frame has a
    target (``mask``); zeros otherwise.

    With a batch of ``B`` streams (the batch of ``act_spec``), the statistic is
    that of the ``B`` current $z$ and the buffer together, one number $S$, and
    the batch is inserted into the buffer row by row. The loss of the batch is
    the mean over the samples of the per-sample losses, whose SIGReg term is $S$,
    so each sample's credit is $\kappa B\,\partial S / \partial z_b$ (a node
    sends the credit of one sample, and the nodes before it divide the
    parameters' gradient by $B$); it is zero for a sample without a target.
    Mind the scale: on the parameters, the SIGReg term thus acts as the sum of
    ``B`` per-sample terms against the mean of the prediction errors, so the
    same ``kappa`` is a stronger SIGReg as ``B`` grows (per sample,
    $\kappa B / (1 - \kappa)$ against the prediction). The node reads ``mask``
    as "this sample is not the first frame of its video", so the host must write
    ``mask = ~reset`` (``WorldBP`` uses ``mask & has_prev``).
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
        every: int = 1,
        seed: int = 0,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period, seed=seed)
        self._kappa, self._min, self._every = kappa, min_samples, every
        z = act_spec.field("h", TensorSpec)
        batch = act_spec.field("mask", TensorSpec).shape
        self._samples = math.prod(batch)  # B
        d = z.shape[-1]
        self.buffer = torch.zeros(capacity, d, device=self.device)
        self.count = 0
        self.register_state("count")
        self.directions = torch.zeros(directions, d, device=self.device)
        # ecf_sums of the buffer along the directions, made with each frame.
        self.sum_cos, self.sum_sin = ecf_sums(self.buffer[:0], self.directions)
        self.z_last = z.zeros(device=self.device)
        self.seen = -1  # the time (ns) of the frame seen last
        self.register_state("seen")
        self.frames = 0  # the frames seen
        self.register_state("frames")
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
            if self.seen >= 0 and self.frames % self._every == 0:
                self.count = insert_(
                    self.buffer, self.count, self.z_last, self.generator
                )
            self.frames += self.seen >= 0
            self.seen = act["time"].ns
            a = torch.randn(
                self.directions.shape, generator=self.generator, device=self.device
            )
            self.directions = a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
            self.sum_cos, self.sum_sin = ecf_sums(
                self.buffer[: self.count], self.directions
            )
            self.cached = False
        self.z_last = act["h"]
        mask = act["mask"]
        if not bool(mask.any()) or self.count + self._samples < self._min:
            self.prev.send(torch.zeros_like(act["h"]))
            return
        if not (self.cached and torch.equal(act["h"], self.g_of)):
            with torch.enable_grad():
                z = act["h"].detach().requires_grad_()
                sums = (self.sum_cos, self.sum_sin)
                n = self.count + self._samples
                value = sigreg_from_sums(z, sums, n, self.directions)
                (g,) = torch.autograd.grad(value, [z])  # mujou: local-autograd
            g = torch.where(rows(mask, g), self._samples * g, 0.0)
            self.g, self.g_of, self.cached = self._kappa * g, act["h"], True
            self.value.send(value.detach())
        self.prev.send(self.g)
