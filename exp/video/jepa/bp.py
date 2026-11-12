r"""The baseline of backpropagation: the encoder and the predictor of the vision
learned by autograd inside one node, from the same ``Chunker`` (or
``StackChunker``, the packed layout) as the PC-ALM version."""

import copy
import math
from typing import override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity
from mujou.pc.optim import Optimizer
from mujou.types import Device

from exp.video.jepa.loss import (
    JEPATerms,
    Reconstruction,
    jepa_loss,
    jepa_loss_packed,
    split_y,
)
from exp.video.jepa.nodes import (
    STATS,
    PastTokens,
    draw_directions,
    draw_packed_directions,
    stats_message,
    z_spec,
)
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
    $z$ again at the weights it learned. With ``mods.decoder`` and ``recon``
    $\mu > 0$ the loss has the reconstruction term of ``JEPAHead`` (read from
    the ``y`` the ``Chunker`` carries), and the decoder is learned by the same
    optimizer (after the predictor). The graph of the frame's first $L$ is
    kept for the first learning of the frame (the weights, the views and the
    directions are the same until then: the clock never learns in the update
    that first sees a frame), so a frame costs one forward of the views, its
    backward, and one forward of the target views after the step; a second
    learning in the same frame computes $L$ again. ``prev`` receives the
    views; it sends no credit to the ``Chunker`` (which drops it anyway).

    With modules of the packed layout (``mods.layout == "packed"``) it
    receives the packed sequences of the ``StackChunker``, the loss is
    ``loss.jepa_loss_packed`` with the stages attending within the views'
    segments (``mods.segments``), the projection learns with the predictor
    (after it), and the directions are those of ``draw_packed_directions``
    (``PackedJEPAHead``'s), then those of $u$ with SIGReg of the views. The
    projector and the predictor of $u$ of the space-time completion
    (``mods.projector``, ``mods.zpred``) learn after the projection. ``views``
    is then unused. With ``token_buffer`` > 0 the SIGReg of the tokens sees
    the past target tokens of a ``PastTokens`` as ``PackedJEPAHead`` has them
    (``past``; the same draws from the same seed): it keeps the target tokens
    of the frame's first $L$ and of each step's new $z$, and the frame's first
    update, after drawing the directions, inserts those of the frame before.

    Args:
        mods: The modules, on ``device``.
        act_spec: The spec of what the ``Chunker`` sends.
        views: $V$.
        lam: $\lambda$, the weight of SIGReg.
        sg: Stop the prediction's gradient at the target view.
        directions: The directions of SIGReg drawn per frame.
        sigreg_on: What SIGReg sees (``loss.jepa_loss``).
        token_sigreg: The SIGReg of the tokens (packed,
            ``loss.jepa_loss_packed``).
        token_weight: $\beta$ (packed).
        z_weight: $\alpha$ (packed).
        target_norm: The prediction's targets through a LayerNorm without
            affine (packed).
        ema: The momentum of an EMA of the encoder whose target tokens are
            the prediction's targets (packed; a control, 0: none). It moves
            after each step: $\bar\theta \leftarrow m \bar\theta + (1 - m)
            \theta$.
        token_buffer: The capacity of the past target tokens (packed; 0:
            none).
        token_insert: The tokens of a frame inserted into them.
        recon: $\mu$, the weight of the reconstruction (needs ``mods.decoder``).
        pixel_mean: The data's mean pixel (the standardization of the target).
        pixel_std: The data's pixel std.
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
        recon: float = 0.0,
        pixel_mean: float = 0.0,
        pixel_std: float = 1.0,
        pred_loss: str = "l2",
        token_sigreg: str = "scaled",
        token_weight: float = 1.0,
        z_weight: float = 0.0,
        target_norm: bool = False,
        ema: float = 0.0,
        token_buffer: int = 0,
        token_insert: int = 64,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period, seed=seed)
        self.mods = mods
        self._packed = mods.layout == "packed"
        self._pred_loss, self._token_sigreg = pred_loss, token_sigreg
        self._weights = token_weight, z_weight
        self._target_norm, self._ema = target_norm, ema
        self.proj = mods.proj  # an attribute, so that it is in the state
        self.projector, self.zpred = mods.projector, mods.zpred  # so are these
        self._recon = recon, pixel_mean, pixel_std
        self.encoder, self.predictor = mods.encoder, mods.predictor
        self.decoder = mods.decoder  # an attribute, so that it is in the state
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
        if mods.decoder is not None:
            self._params += mods.decoder.parameters()
        for extra in (mods.proj, mods.projector, mods.zpred):
            if extra is not None:
                self._params += extra.parameters()
        self.optimizer = optimizer.bind(self._params)
        # The EMA of the encoder (a control, packed): not learned, in the state.
        self.ema_encoder: torch.nn.Module | None = None
        if ema:
            self.ema_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        h = act_spec.field("h", TensorSpec)
        n = h.shape[0]
        self._b = n if self._packed else n // views
        d = self.predictor.out.out_features
        d_z = d if mods.proj is None else mods.proj.out_features
        tokens = sigreg_on in ("tokens", "both", "views_tokens")
        self._two = self._packed and tokens and d_z != d
        self._views = self._packed and sigreg_on in ("views", "views_tokens")
        self._d_u = d_z if mods.projector is None else mods.projector.out_features
        self.directions = torch.zeros(directions, d_z, device=self.device)
        self.directions_t: torch.Tensor | None = None
        self.directions_u: torch.Tensor | None = None
        self.past: PastTokens | None = None  # the past target tokens (state)
        if token_buffer:
            self.past = PastTokens(
                capacity=token_buffer,
                insert=token_insert,
                streams=self._b,
                tokens=mods.segments[0],
                dim=d,
                directions=directions,
                device=self.device,
            )
        self.prev = Port(send=h, recv=act_spec)  # sends nothing
        self.z_out = Port(send=z_spec(self._b, d_z))
        self.stats = Port(send=STATS)

    def _valid(self, act: Activity) -> torch.Tensor:
        if self._packed:
            return act["mask"]
        return act["mask"].view(self._b, self._v)[:, 0]

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        valid = self._valid(act)
        if tick.new:
            m, d = self.directions.shape
            if self._packed:
                width = self.predictor.out.out_features
                self.directions, self.directions_t = draw_packed_directions(
                    m, d, width, tokens=self._two, generator=self.generator
                )
                if self._views:
                    self.directions_u = draw_directions(m, self._d_u, self.generator)
                if self.past is not None:
                    on = self.directions_t  # those of the tokens, or of z
                    on = self.directions if on is None else on
                    self.past.next_frame(on, self.generator)
            else:
                self.directions = draw_directions(m, d, self.generator)
            with torch.set_grad_enabled(self._learns):
                terms = self.terms(act)
            if self.past is not None:
                assert terms.target is not None  # of the packed loss
                self.past.keep(terms.target, valid)
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
                if self.ema_encoder is not None:
                    online = self.encoder.parameters()
                    for e, p in zip(self.ema_encoder.parameters(), online, strict=True):
                        e.lerp_(p, 1 - self._ema)
                self._send_z(act, self._target_z(act))

    def _target_z(self, act: Activity) -> torch.Tensor:
        """$z$ of the target views of ``act`` at the current weights (packed:
        their tokens are kept for the past tokens)."""
        if self._packed:
            t = self.mods.segments[0]
            tokens = self.encoder(act["h"][:, :t], (t,))
            if self.past is not None:
                self.past.keep(tokens, act["mask"])
            return self.mods.z_of(tokens)
        h = act["h"][:: self._v]  # the target views
        return self.encoder.pool(self.encoder(h))

    def terms(
        self, act: Activity, outputs: list[torch.Tensor] | None = None
    ) -> JEPATerms:
        r"""$L$ and its terms of the views of ``act`` at the current weights and
        ``directions``. With ``outputs``, it appends the output ``(B V, E + T,
        D)`` of each stage of the encoder to it, in the graph of the loss."""
        x = act["h"]
        segments = self.mods.segments if self._packed else None
        for stage in self.encoder.stages:
            x = stage(x, segments)
            if outputs is not None:
                outputs.append(x)
        if self._packed:
            goal = None
            if self.ema_encoder is not None:
                t = self.mods.segments[0]
                with torch.no_grad():
                    goal = self.ema_encoder(act["h"][:, :t], (t,))
            return jepa_loss_packed(
                x,
                act["y"],
                act["mask"],
                self.directions,
                self.directions_t,
                self.predictor,
                keeps=list(self.mods.segments[1:]),
                proj=self.mods.proj,
                lam=self._lam,
                sg=self._sg,
                sigreg_on=self._sigreg_on,
                pred_loss=self._pred_loss,
                token_sigreg=self._token_sigreg,
                token_weight=self._weights[0],
                z_weight=self._weights[1],
                projector=self.projector,
                zpred=self.zpred,
                directions_u=self.directions_u,
                target_norm=self._target_norm,
                goal=goal,
                past=None if self.past is None else self.past.sums,
            )
        b, v = self._b, self._v
        decoder = self.decoder
        masks, frames = split_y(
            act["y"], b, v, None if decoder is None else decoder.shape
        )
        recon = None
        if decoder is not None:
            assert frames is not None, "the Chunker must carry the frame"
            mu, mean, std = self._recon
            recon = Reconstruction(mu, decoder, mean, std, frames)
        return jepa_loss(
            x.view(b, v, *x.shape[1:]),
            masks,
            act["mask"].view(b, v)[:, 0],
            self.directions,
            self.predictor,
            extra=self.encoder.extra,
            z_pool=self.encoder.z_pool,
            lam=self._lam,
            sg=self._sg,
            sigreg_on=self._sigreg_on,
            recon=recon,
        )

    def _send_z(self, act: Activity, z: torch.Tensor) -> None:
        v = 1 if self._packed else self._v
        self.z_out.send(
            {
                "h": z.detach(),
                "y": torch.zeros(self._b, 1, device=self.device),
                "mask": act["mask"][::v],
                "reset": act["reset"][::v],
                "time": act["time"],
            }
        )
