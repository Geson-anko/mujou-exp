r"""The baseline of backpropagation: the encoder and the predictor of the vision
learned by autograd inside one node, from the same ``Chunker`` as the PC-ALM
version."""

import math
from typing import override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity
from mujou.pc.optim import Optimizer
from mujou.types import Device

from exp.video.jepa.loss import JEPATerms, jepa_loss
from exp.video.jepa.nodes import STATS, draw_directions, stats_message, z_spec
from exp.video.jepa.vit import VisionModules
from mujou import DictSpec, Node, Port, TensorSpec

__all__ = ["JEPABP"]


class JEPABP(Node):
    r"""The vision learned by BP in one node: the loss $L$ of
    ``loss.jepa_loss`` of the views it receives, through the encoder and the
    predictor of ``mods``, as ``Enc1 .. EncL -> JEPAHead`` has it in its nodes.

    In the update that first sees a frame, it draws the directions of SIGReg
    from its ``generator`` (as ``JEPAHead`` does, so the same seed draws the
    same), computes $L$ at the current weights and sends its terms on
    ``stats`` (prequential) and $z$ of the target views on ``z_out`` (a port
    that only sends). When its clock says (``pc.clock.LearnClock``, as the
    nodes of PC-ALM), it learns all the parameters by $\partial L / \partial
    \theta$ with one optimizer (the encoder's, then the predictor's) and sends
    $z$ again at the weights it learned. The graph of the frame's first $L$ is
    kept for the first learning of the frame (the weights, the views and the
    directions are the same until then: the clock never learns in the update
    that first sees a frame), so a frame costs one forward of the views, its
    backward, and one forward of the target views after the step; a second
    learning in the same frame computes $L$ again. ``prev`` receives the
    views; it sends no credit to the ``Chunker`` (which drops it anyway).

    Args:
        mods: The modules, on ``device``.
        act_spec: The spec of what the ``Chunker`` sends.
        views: $V$.
        lam: $\lambda$, the weight of SIGReg.
        sg: Stop the prediction's gradient at the target view.
        directions: The directions of SIGReg drawn per frame.
        sigreg_on: What SIGReg sees (``loss.jepa_loss``).
        tau: The time constants of PC-ALM (the period and ``max_dt``).
        optimizer: The optimizer of all the parameters.
        learn_interval: As in ``pc.clock.LearnClock``.
        seed: The seed of ``generator``.
        device: Where the node computes.
    """

    def __init__(
        self,
        mods: VisionModules,
        *,
        act_spec: DictSpec[Activity],
        views: int,
        lam: float,
        sg: bool,
        directions: int,
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        seed: int,
        sigreg_on: str = "z",
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period, seed=seed)
        self.mods = mods
        self.encoder, self.predictor = mods.encoder, mods.predictor
        self._v, self._lam, self._sg, self._sigreg_on = views, lam, sg, sigreg_on
        self._learns = learn_interval is None or not math.isinf(learn_interval)
        # L of the frame at the current weights, in its graph, until a step
        # changes the weights. In a list, not the state: a resume computes it
        # again, to the same value.
        self._pending: list[torch.Tensor] = []
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = [*self.encoder.parameters(), *self.predictor.parameters()]
        self.optimizer = optimizer.bind(self._params)
        h = act_spec.field("h", TensorSpec)
        n = h.shape[0]
        self._b = n // views
        d = self.predictor.out.out_features
        self.directions = torch.zeros(directions, d, device=self.device)
        self.prev = Port(send=h, recv=act_spec)  # sends nothing
        self.z_out = Port(send=z_spec(self._b, d))
        self.stats = Port(send=STATS)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        valid = act["mask"].view(self._b, self._v)[:, 0]
        if tick.new:
            m, d = self.directions.shape
            self.directions = draw_directions(m, d, self.generator)
            with torch.set_grad_enabled(self._learns):
                terms = self.terms(act)
            self._pending = [terms.loss] if self._learns else []
            self.stats.send(stats_message(terms, act["time"]))
            self._send_z(act, terms.z)
        if tick.learn and bool(valid.any()):
            with torch.enable_grad():
                loss = self._pending[0] if self._pending else self.terms(act).loss
                # BP inside this node, through its own modules from its input.
                grads = torch.autograd.grad(  # mujou: local-autograd
                    loss, self._params, materialize_grads=True
                )
            self._pending = []
            self.optimizer.step_(list(grads), tick.elapsed)
            with torch.no_grad():
                h = act["h"][:: self._v]  # the target views
                self._send_z(act, self.encoder.pool(self.encoder(h)))

    def terms(
        self, act: Activity, outputs: list[torch.Tensor] | None = None
    ) -> JEPATerms:
        r"""$L$ and its terms of the views of ``act`` at the current weights and
        ``directions``. With ``outputs``, it appends the output ``(B V, E + T,
        D)`` of each stage of the encoder to it, in the graph of the loss."""
        x = act["h"]
        for stage in self.encoder.stages:
            x = stage(x)
            if outputs is not None:
                outputs.append(x)
        b, v = self._b, self._v
        return jepa_loss(
            x.view(b, v, *x.shape[1:]),
            act["y"].view(b, v, -1)[:, 1:],
            act["mask"].view(b, v)[:, 0],
            self.directions,
            self.predictor,
            extra=self.encoder.extra,
            z_pool=self.encoder.z_pool,
            lam=self._lam,
            sg=self._sg,
            sigreg_on=self._sigreg_on,
        )

    def _send_z(self, act: Activity, z: torch.Tensor) -> None:
        v = self._v
        self.z_out.send(
            {
                "h": z.detach(),
                "y": torch.zeros(self._b, 1, device=self.device),
                "mask": act["mask"][::v],
                "reset": act["reset"][::v],
                "time": act["time"],
            }
        )
