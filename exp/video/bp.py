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
from mujou.pc.msg import Activity, rows
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
    decoder,
    dynamics_modules,
    encoder_stages,
    image_shape,
    standardize_,
    target_weight,
)
from exp.video.nodes import compare_energy, open_mask, recon_energy
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
    \mathrm{SIGReg}(z_k; \mathcal B_k)$ (plus ``recon_energy`` with a
    ``decoder``, as ``Reconstruct`` has), the same memory of $z_{k-1}, s_{k-1}$
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

    With a batch of ``B`` streams, the memories are per stream, and the loss is
    the mean over the samples of the per-sample prediction error plus
    $\kappa\,\mathrm{SIGReg}$ of the ``B`` current $z$ with the buffer (the
    same loss as the nodes of ``model.Network`` have, with the credit that
    ``SIGReg`` sends); a sample without a prediction adds nothing.
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
        decoder: nn.Module | None = None,
        recon: float = 0.0,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period, seed=seed)
        encoder, step, readout = modules
        self.decoder, self._recon = decoder, recon
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
            *([] if decoder is None else decoder.parameters()),
        ]
        self.optimizer = optimizer.bind(self._params)
        d_z, d_s = readout.out_features, step.d_state
        dev = self.device
        batch = act_spec.field("mask", TensorSpec).shape
        self._samples = math.prod(batch)  # B
        self.buffer = torch.zeros(capacity, d_z, device=dev)
        self.count = 0
        self.register_state("count")
        self.directions = torch.zeros(directions, d_z, device=dev)
        self.z_prev, self.s_prev = (
            torch.zeros(*batch, d_z, device=dev),
            torch.zeros(*batch, d_s, device=dev),
        )
        self.z_last, self.s_last = (
            torch.zeros(*batch, d_z, device=dev),
            torch.zeros(*batch, d_s, device=dev),
        )
        # Per stream: whether z_prev is of this video, and the predictions of it.
        self.has_prev = torch.zeros(batch, dtype=torch.bool, device=dev)
        self.register_state("has_prev")
        self.predictions = torch.zeros(batch, dtype=torch.int64, device=dev)
        self.register_state("predictions")
        self.z_hat_last = torch.zeros(*batch, d_z, device=dev)
        self.started = False  # whether a frame came before
        self.register_state("started")
        self.frames = 0  # the frames before the current one
        self.register_state("frames")
        x = act_spec.field("h", TensorSpec)
        z_spec = TensorSpec((*batch, d_z))
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
        mask = act["mask"] & self.has_prev
        if tick.learn and bool(mask.any()):
            with torch.enable_grad():
                loss = self._loss(act["h"], mask)
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
        # The streams that reset start a new video: s := 0, and no z before.
        reset = act["reset"]
        predictions = torch.where(self.has_prev, self.predictions + 1, 1)
        opened = (predictions > 1) & open_mask(
            predictions, self._open_loop, self._open_cycle
        )
        z_prev = torch.where(rows(opened, self.z_last), self.z_hat_last, self.z_last)
        self.z_prev = torch.where(rows(reset, z_prev), self.z_prev, z_prev)
        self.s_last = torch.where(rows(reset, self.s_last), 0.0, self.s_last)
        self.s_prev = self.s_last
        self.has_prev = ~reset
        self.predictions = predictions.masked_fill(reset, 0)
        with torch.no_grad():
            z = self.encoder(act["h"])
            has_prev = rows(self.has_prev, self.s_prev)
            s = self.s_prev
            if bool(self.has_prev.any()):
                s = torch.where(has_prev, self.step(self.z_prev, self.s_prev), s)
            z_hat = self.readout(s)
            if self.residual:
                z_hat = torch.where(has_prev, self.z_prev + z_hat, z_hat)
        self.z_last, self.s_last, self.z_hat_last = z, s, z_hat
        mask = act["mask"] & self.has_prev
        self.next.send({"y_hat": z_hat, "y": z, "mask": mask, "time": act["time"]})
        if bool(mask.any()):
            self.error.send(((z_hat - z) ** 2)[mask].mean())

    def _loss(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """$L_k$ at the current weights (the memory is constant): the mean over
        the samples of the prediction errors of those in ``mask`` (and, with a
        decoder, of their ``recon_energy``), plus SIGReg of the current $z$ of
        those."""
        z = self.encoder(x)
        s = self.step(self.z_prev, self.s_prev)
        w = self._target_weight  # the value is z; its gradient is w times
        target = w * z + (1 - w) * z.detach()
        base = self.z_prev if self.residual else None
        energy = compare_energy(self.readout, s, target, self._kappa, base)
        if self.decoder is not None:
            energy = energy + recon_energy(self.decoder, z, x, self._recon)
        loss = torch.where(mask, energy, 0.0).sum() / self._samples
        if self.count + self._samples >= self._min:
            # A sample without a prediction sends SIGReg no credit.
            z_sig = torch.where(rows(mask, z), z, z.detach())
            loss = loss + self._kappa * sigreg(
                z_sig, self.buffer[: self.count], self.directions
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
        dec = decoder(m.d_z, image_shape(d)) if m.recon > 0 else None
        dev = config.device
        encoder, step, readout = encoder.to(dev), step.to(dev), readout.to(dev)
        self.input = pc.Input(
            x=TensorSpec((*d.batch, *image_shape(d))),
            y=TensorSpec((*d.batch, 1)),
            period=tau.period,
            batch=d.batch,
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
            decoder=None if dec is None else dec.to(dev),
            recon=m.recon,
            device=dev,
        )
        self.input.next.connect(self.world.prev)
        self.prev, self.next = self.input.prev, self.world.next
