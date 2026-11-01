"""The baselines of backpropagation: the same modules learned by autograd inside
one node.

A ``pc.Output`` differentiates its module by autograd from the loss, so a chain
of one ``nn.Sequential`` of all the modules is BP in one node, with the same
``LearnClock`` and the same optimizers as the chain of PC-ALM (``bp_chain``).
``WorldBP`` is the world model of ``model.Network`` in one node.
"""

import math
from collections.abc import Callable
from typing import Any, override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity
from mujou.pc.optim import Optimizer
from mujou.types import Device
from torch import nn

import mujou
from exp.lib.sigreg import insert_, sigreg
from exp.video.config import Config, make_optimizer
from exp.video.model import (
    DynamicsStack,
    Top,
    calibration_images,
    dynamics_modules,
    encoder_stages,
    image_shape,
    standardize_,
    target_weight,
)
from exp.video.nodes import compare_energy, is_open
from mujou import DictSpec, Node, Port, TensorSpec, TimestampSpec

__all__ = ["WorldBP", "WorldBPNetwork", "bp_chain"]


def bp_chain(
    *modules: nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    **kwargs: Any,
) -> pc.Chain:
    """``pc.Chain(nn.Sequential(*modules), ...)``: ``Input -> Output``, the
    output learning all of ``modules`` by BP. ``kwargs`` are those of
    ``pc.Chain``."""
    return pc.Chain(nn.Sequential(*modules), loss=loss, **kwargs)


class WorldBP(Node):
    r"""The world model learned by BP in one node: the same loss
    $L_k = (1 - \kappa) \frac1D \lVert \hat z_k - z_k \rVert^2 + \kappa\,
    \mathrm{SIGReg}(z_k; \mathcal B_k)$, the same memory of $z_{k-1}, s_{k-1}$
    (detached: the gradient stops at one frame, as in PC-ALM), the same buffer
    and directions as the nodes of ``model.Network``, and the same
    ``LearnClock``. The gradient through $z_k$ of the first term is scaled by
    ``target_weight`` (0: detached), as ``Compare`` scales its credit; with
    ``open_loop``, it remembers its own prediction for the predictions that
    ``nodes.is_open`` says, as ``Dynamics`` does; with ``residual``, it
    predicts $z_{k-1} + W_o s_k$, as ``Compare`` does.

    In the update that first sees a frame, it remembers the frame before,
    updates the buffer and the directions, and sends the prequential
    prediction (a ``pc.msg.Prediction``) and its ``error``. When its clock
    says, it learns all the modules from the gradient of $L_k$.
    """

    def __init__(
        self,
        modules: tuple[nn.Module, DynamicsStack, nn.Linear],
        *,
        act_spec: DictSpec[Activity],
        kappa: float,
        directions: int,
        capacity: int,
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        min_samples: int = 16,
        every: int = 1,
        seed: int = 0,
        target_weight: float = 1.0,
        open_loop: float = 0.0,
        open_cycle: int = 20,
        residual: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period, seed=seed)
        encoder, step, readout = modules
        self._target_weight = target_weight
        self._open_loop, self._open_cycle = open_loop, open_cycle
        self.residual = residual
        self.encoder = encoder
        self.step = step
        self.readout = nn.Sequential(Top(step.d_s), readout)
        self._kappa, self._min, self._every = kappa, min_samples, every
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = [
            *self.encoder.parameters(),
            *step.parameters(),
            *readout.parameters(),
        ]
        self.optimizer = optimizer.bind(self._params)
        d_z, d_s = readout.out_features, step.d_state
        dev = self.device
        self.buffer = torch.zeros(capacity, d_z, device=dev)
        self.count = 0
        self.register_state("count")
        self.directions = torch.zeros(directions, d_z, device=dev)
        self.z_prev, self.s_prev = (
            torch.zeros(d_z, device=dev),
            torch.zeros(d_s, device=dev),
        )
        self.z_last, self.s_last = (
            torch.zeros(d_z, device=dev),
            torch.zeros(d_s, device=dev),
        )
        self.has_prev = False
        self.register_state("has_prev")
        self.predictions = 0  # of this video so far
        self.register_state("predictions")
        self.z_hat_last = torch.zeros(d_z, device=dev)
        self.started = False  # whether a frame came before
        self.register_state("started")
        self.frames = 0  # the frames before the current one
        self.register_state("frames")
        x = act_spec.field("h", TensorSpec)
        z_spec = TensorSpec((d_z,))
        self.prev = Port(send=x, recv=act_spec)
        self.next = Port(
            send=DictSpec(
                pc.msg.Prediction,
                y_hat=z_spec,
                y=z_spec,
                mask=act_spec.field("mask", TensorSpec),
                time=act_spec.field("time", TimestampSpec),
            )
        )
        self.error = Port(send=TensorSpec(()), initial=torch.tensor(math.nan))

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            self._new_frame(act)
        mask = bool(act["mask"]) and self.has_prev
        if tick.learn and mask:
            with torch.enable_grad():
                loss = self._loss(act["h"])
                grads = torch.autograd.grad(  # mujou: local-autograd
                    loss, self._params, materialize_grads=True
                )
            self.optimizer.step_(grads, tick.elapsed)

    def _new_frame(self, act: Activity) -> None:
        if self.started and self.frames % self._every == 0:
            self.count = insert_(self.buffer, self.count, self.z_last, self.generator)
        self.frames += self.started
        self.started = True
        a = torch.randn(
            self.directions.shape, generator=self.generator, device=self.device
        )
        self.directions = a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
        if bool(act["reset"]):
            self.s_prev = torch.zeros_like(self.s_prev)
            self.s_last = torch.zeros_like(self.s_last)
            self.has_prev = False
            self.predictions = 0
        else:
            self.predictions = self.predictions + 1 if self.has_prev else 1
            opened = self.predictions > 1 and is_open(
                self.predictions, self._open_loop, self._open_cycle
            )
            self.z_prev = self.z_hat_last if opened else self.z_last
            self.s_prev = self.s_last
            self.has_prev = True
        with torch.no_grad():
            z = self.encoder(act["h"])
            s = self.step(self.z_prev, self.s_prev) if self.has_prev else self.s_prev
            z_hat = self.readout(s)
            if self.residual and self.has_prev:
                z_hat = self.z_prev + z_hat
        self.z_last, self.s_last, self.z_hat_last = z, s, z_hat
        mask = bool(act["mask"]) and self.has_prev
        self.next.send(
            {"y_hat": z_hat, "y": z, "mask": torch.tensor(mask), "time": act["time"]}
        )
        if mask:
            self.error.send(((z_hat - z) ** 2).mean())

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        """$L_k$ at the current weights (the memory is constant)."""
        z = self.encoder(x)
        s = self.step(self.z_prev, self.s_prev)
        w = self._target_weight  # the value is z; its gradient is w times
        target = w * z + (1 - w) * z.detach()
        base = self.z_prev if self.residual else None
        loss = compare_energy(self.readout, s, target, self._kappa, base)
        if self.count + 1 >= self._min:
            loss = loss + self._kappa * sigreg(
                z, self.buffer[: self.count], self.directions
            )
        return loss


class WorldBPNetwork(mujou.Network):
    """``Input -> WorldBP``, made from ``config`` as ``model.Network`` is (the
    same initial weights for the same seed)."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        m, t, d = config.model, config.time, config.data
        tau = pc.TimeConstants(h=t.h, lam=t.lam, r=t.r)
        self.tau = tau
        torch.manual_seed(config.seed)
        encoder = nn.Sequential(*encoder_stages(config))
        standardize_(encoder, calibration_images(config))
        first, layers = dynamics_modules(m)
        readout = nn.Linear(m.d_s, m.d_z)
        if m.residual:  # the prediction starts as z_{k-1}, the change as 0
            nn.init.zeros_(readout.weight)
            nn.init.zeros_(readout.bias)
        step = DynamicsStack(first, layers)
        dev = config.device
        encoder, step, readout = encoder.to(dev), step.to(dev), readout.to(dev)
        self.input = pc.Input(
            x=TensorSpec(image_shape(d)),
            y=TensorSpec((1,)),
            period=tau.period,
            device=dev,
        )
        self.world = WorldBP(
            (encoder, step, readout),
            act_spec=self.input.act_spec,
            kappa=m.kappa,
            directions=m.directions,
            capacity=m.buffer,
            every=m.buffer_every,
            tau=tau,
            optimizer=make_optimizer(config.optim),
            learn_interval=t.learn_interval,
            seed=config.seed,
            target_weight=target_weight(m),
            open_loop=m.open_loop,
            open_cycle=m.open_cycle,
            residual=m.residual,
            device=dev,
        )
        self.input.next.connect(self.world.prev)
        self.prev, self.next = self.input.prev, self.world.next
