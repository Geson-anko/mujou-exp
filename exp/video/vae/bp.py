r"""The autoencoder learned by BP in one node (M18, stage 1), and its network.

The host writes the stacks to ``AEBP.prev`` as a ``pc.msg.Sample`` and reads
the loss on ``stats``; the encoder and the decoder learn together, by autograd
inside the node, as ``jepa.bp.JEPABP`` learns the JEPA."""

import math
from typing import TypedDict, override

import mujou.pc as pc
import torch
from mujou.pc.optim import Optimizer
from mujou.time import Timestamp
from mujou.types import Device

import mujou
from exp.video.dyn.nets import make_optimizer
from exp.video.jepa.vit import image_shape
from exp.video.schema.run import RunConf
from exp.video.vae.loss import ae_loss, batch_loss, target
from exp.video.vae.modules import AEModules, build_ae
from mujou import DictSpec, Node, Port, TensorSpec, TimestampSpec

__all__ = ["AE_STATS", "AEBP", "AENet", "AEStats", "build_ae_net"]


class AEStats(TypedDict):
    """What ``AEBP`` sends on ``stats`` for the host: $L$ of the frame's
    stacks at the weights before it learned from them (0 when no stack
    counts), the number ``n`` of stacks that count (0-d floats), and the
    frame's ``time``."""

    loss: torch.Tensor
    n: torch.Tensor
    time: Timestamp


_SCALAR = TensorSpec(())
AE_STATS = DictSpec(AEStats, loss=_SCALAR, n=_SCALAR, time=TimestampSpec())


class AEBP(Node):
    r"""The autoencoder ``mods`` learned by BP in one node.

    It receives on ``prev`` the host's ``pc.msg.Sample``: the stacks ``x`` u8
    ``(B, N, C, H, W)``, unused zeros ``y`` ``(B, 1)``, ``mask`` (the stacks
    that count: within one video) and ``reset`` (unused). In the update that
    first sees a write it computes $L$ (``loss.batch_loss`` of
    ``loss.ae_loss``) at the current weights and sends it on ``stats``
    (prequential); when its clock says (``pc.clock.LearnClock``, as the nodes
    of PC-ALM) and a stack counts, it learns all the parameters of ``mods``
    (the encoder's, then the decoder's) by $\partial L / \partial \theta$
    with one optimizer, stepped by the clock's ``elapsed``. The graph of the
    frame's first $L$ is kept for its first learning (the weights and the
    stacks are the same until then: the clock never learns in the update that
    first sees a write), so a frame costs one forward and one backward; a
    second learning in the same frame computes $L$ again.

    Args:
        mods: The modules, on ``device``.
        stack: The spec of the stacks, ``(B, N, C, H, W)`` u8.
        tau: The time constants of PC-ALM (the period and ``max_dt``).
        optimizer: The optimizer of all the parameters.
        learn_interval: As in ``pc.clock.LearnClock``.
        device: Where the node computes.
    """

    def __init__(
        self,
        mods: AEModules,
        *,
        stack: TensorSpec,
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period)
        self.mods = mods
        self.encoder = mods.vision.encoder  # attributes, so that they are in the state
        self.decoder = mods.decoder
        self._params = mods.parameters()
        self.optimizer = optimizer.bind(self._params)
        self._learns = learn_interval is None or not math.isinf(learn_interval)
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        # L of the frame at the current weights, in its graph, until a step
        # changes the weights. Not in the state: a resume computes it again.
        self._pending: list[torch.Tensor] = []
        b = stack.shape[0]
        mask = TensorSpec((b,), torch.bool)
        self.prev = Port(
            recv=DictSpec(
                pc.msg.Sample, x=stack, y=TensorSpec((b, 1)), mask=mask, reset=mask
            )
        )
        self.stats = Port(send=AE_STATS)

    def loss(self, stacks: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """$L$ of the stacks u8 ``(B, N, C, H, W)`` over ``valid`` ``(B,)``
        at the current weights, in the graph."""
        return batch_loss(ae_loss(target(stacks), self.mods(stacks)), valid)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, written = self.prev.recv_with_time()
        if written is None:
            return
        tick = self.clock.tick(dt, written, self.timer.now())
        x, valid = sample["x"], sample["mask"]
        if tick.new:
            with torch.set_grad_enabled(self._learns):
                loss = self.loss(x, valid)
            self._pending = [loss] if self._learns else []
            self.stats.send(
                {
                    "loss": loss.detach(),
                    "n": valid.sum().to(loss.dtype),
                    "time": written,
                }
            )
        if tick.learn and bool(valid.any()):
            with torch.enable_grad():
                loss = self._pending[0] if self._pending else self.loss(x, valid)
                # BP inside this node, through its own modules from its input.
                grads = torch.autograd.grad(  # mujou: local-autograd
                    loss, self._params, materialize_grads=True
                )
            self._pending = []
            self.optimizer.step_(list(grads), tick.elapsed)


class AENet(mujou.Network):
    """The autoencoder of ``conf`` in one ``AEBP`` node: the host writes a
    ``pc.msg.Sample`` to ``prev`` and reads ``stats``.

    Args:
        mods: The modules, on ``conf.device``.
        conf: The run: its vision, time, streams and device.
        optimizer: The optimizer of all the parameters (``optim_vision``).
    """

    def __init__(self, mods: AEModules, *, conf: RunConf, optimizer: Optimizer) -> None:
        super().__init__()
        v, t = conf.vision, conf.time
        assert v is not None  # build_ae checks
        self.mods = mods
        self.tau = pc.TimeConstants(h=t.h, lam=t.lam, r=t.r)
        channels, size = image_shape(conf)
        self.ae = AEBP(
            mods,
            stack=TensorSpec(
                (conf.data.streams, v.frames, channels, size, size), torch.uint8
            ),
            tau=self.tau,
            optimizer=optimizer,
            learn_interval=t.learn_interval,
            device=conf.device,
        )
        self.prev, self.stats = self.ae.prev, self.ae.stats


def build_ae_net(conf: RunConf) -> AENet:
    """The autoencoder of ``conf``: the modules of ``build_ae`` (seed
    ``conf.seed``) learned by BP with the optimizer of ``conf.optim_vision``
    (``scale`` ``"fan_in"`` by the roles in the encoder and the decoder).

    Raises:
        ValueError: ``conf`` has no ``optim_vision``, its learner is not
            ``"bp"`` (PC-ALM comes later), or ``build_ae`` raises.
    """
    if conf.optim_vision is None:
        raise ValueError(f"the run {conf.name!r} needs optim_vision")
    if conf.learner.kind != "bp":
        raise ValueError(
            f"the autoencoder learns by learner=bp, not {conf.learner.kind!r}"
        )
    mods = build_ae(conf, seed=conf.seed)
    optimizer = make_optimizer(
        conf.optim_vision, conf.time.frame, list(mods.named().values())
    )
    return AENet(mods, conf=conf, optimizer=optimizer)
