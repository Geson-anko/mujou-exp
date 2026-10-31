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
from experiments.continuous_stream.config import Config
from experiments.continuous_stream.model import encoder_modules, standardize_
from experiments.continuous_stream.nodes import MinGRU, compare_energy, insert_, sigreg
from experiments.continuous_stream.optim import from_config
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
    ``LearnClock``. With ``stop_target``, $z_k$ of the first term is detached.

    In the update that first sees a frame, it remembers the frame before,
    updates the buffer and the directions, and sends the prequential
    prediction (a ``pc.msg.Prediction``) and its ``error``. When its clock
    says, it learns all the modules from the gradient of $L_k$.
    """

    def __init__(
        self,
        modules: tuple[nn.Module, nn.Module, MinGRU, nn.Linear],
        *,
        act_spec: DictSpec[Activity],
        kappa: float,
        directions: int,
        capacity: int,
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        min_samples: int = 16,
        seed: int = 0,
        stop_target: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period, seed=seed)
        g1, z, step, readout = modules
        self._stop_target = stop_target
        self.encoder = nn.Sequential(g1, z)
        self.step, self.readout = step, readout
        self._kappa, self._min = kappa, min_samples
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = [
            *self.encoder.parameters(),
            *step.parameters(),
            *readout.parameters(),
        ]
        self.optimizer = optimizer.bind(self._params)
        d_z, d_s = readout.out_features, readout.in_features
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
        self.started = False  # whether a frame came before
        self.register_state("started")
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
        if self.started:
            self.count = insert_(self.buffer, self.count, self.z_last, self.generator)
        self.started = True
        a = torch.randn(
            self.directions.shape, generator=self.generator, device=self.device
        )
        self.directions = a / torch.linalg.vector_norm(a, dim=1, keepdim=True)
        if bool(act["reset"]):
            self.s_prev = torch.zeros_like(self.s_prev)
            self.s_last = torch.zeros_like(self.s_last)
            self.has_prev = False
        else:
            self.z_prev, self.s_prev = self.z_last, self.s_last
            self.has_prev = True
        with torch.no_grad():
            z = self.encoder(act["h"])
            s = self.step(self.z_prev, self.s_prev) if self.has_prev else self.s_prev
            z_hat = self.readout(s)
        self.z_last, self.s_last = z, s
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
        target = z.detach() if self._stop_target else z
        loss = compare_energy(self.readout, s, target, self._kappa)
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
        g1, z = encoder_modules(d.size, m.d_z)
        standardize_(g1, z, d.size, seed=config.seed + 10_000)
        step, readout = MinGRU(m.d_z, m.d_s), nn.Linear(m.d_s, m.d_z)
        dev = config.device
        g1, z, step, readout = g1.to(dev), z.to(dev), step.to(dev), readout.to(dev)
        self.input = pc.Input(
            x=TensorSpec((1, d.size, d.size)),
            y=TensorSpec((1,)),
            period=tau.period,
            device=dev,
        )
        self.world = WorldBP(
            (g1, z, step, readout),
            act_spec=self.input.act_spec,
            kappa=m.kappa,
            directions=m.directions,
            capacity=m.buffer,
            tau=tau,
            optimizer=from_config(config.optim),
            learn_interval=t.learn_interval,
            seed=config.seed,
            stop_target=m.stop_target,
            device=dev,
        )
        self.input.next.connect(self.world.prev)
        self.prev, self.next = self.input.prev, self.world.next
