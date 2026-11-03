r"""The PC-ALM nodes of the vision around the encoder's ``pc.Layer`` nodes:
``Chunker`` (the host's frames as views of chunks of $N$ frames) and
``JEPAHead`` (the predictor and the JEPA loss).

::

    host -Sample-> Chunker -(B V samples)-> Enc1 -> ... -> EncL -> JEPAHead -z_out->
                   (no params)              pc.Layer: ViT stages  predictor + SIGReg

Time is a dimension of a sample: the chunk of frame $k$ is frames
$k - N + 1 \dots k$ of its stream, one frame further at every frame. The views
of the chunks are the samples of the encoder: ``B`` streams of ``V`` views are
``B V`` samples, sample ``b V + v`` view ``v`` of stream ``b`` (0 the target,
the others the context), each encoded on its own. The context masks of the
tokens travel in the activities' ``y``, which every ``pc.Layer`` passes on with
the ``h`` it was computed from, so the head pairs each embedding with its
masks, however the nodes are scheduled.

Only ``JEPAHead`` holds a loss across the samples (the prediction pairs the
views of a stream, SIGReg the target views of all streams). It differentiates
it inside the node, from the embeddings it received (detached), and sends each
sample's credit back as ``pc.Layer`` expects: the gradient of the head's energy
$E = B V L$ ($L$ of ``loss.jepa_loss``, the mean over the streams). The
encoder's nodes divide their parameters' gradient by their batch ``B V``, so
they move by $\partial L / \partial \theta$, as ``JEPABP`` does.
"""

from typing import TypedDict, cast, override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity, rows
from mujou.pc.optim import Optimizer
from mujou.time import Timestamp
from mujou.types import Device

from exp.video.jepa.loss import JEPATerms, Reconstruction, jepa_loss, pool, split_y
from exp.video.jepa.masks import block_masks, make_views
from exp.video.jepa.vit import PixelDecoder, Predictor
from mujou import DictSpec, Node, Port, Stateful, TensorSpec, TimestampSpec

__all__ = [
    "STATS",
    "Chunker",
    "FrameRing",
    "JEPAHead",
    "JEPAStats",
    "draw_directions",
    "stats_message",
    "z_spec",
]


class JEPAStats(TypedDict):
    """What the JEPA heads send on ``stats`` for the host: the terms of $L$ of
    the frame (``loss.JEPATerms``: 0 when no stream has a chunk), the number of
    streams with a chunk ``n`` (0-d floats), and the ``time`` of the frame."""

    loss: torch.Tensor
    pred: torch.Tensor
    sigreg: torch.Tensor
    recon: torch.Tensor
    n: torch.Tensor
    time: Timestamp


_SCALAR = TensorSpec(())
STATS = DictSpec(
    JEPAStats,
    loss=_SCALAR,
    pred=_SCALAR,
    sigreg=_SCALAR,
    recon=_SCALAR,
    n=_SCALAR,
    time=TimestampSpec(),
)


def stats_message(terms: JEPATerms, time: Timestamp) -> JEPAStats:
    """The message of ``STATS`` of ``terms`` (detached)."""
    return {
        "loss": terms.loss.detach(),
        "pred": terms.pred.detach(),
        "sigreg": terms.sigreg.detach(),
        "recon": terms.recon.detach(),
        "n": terms.valid.detach(),
        "time": time,
    }


def z_spec(streams: int, dim: int) -> DictSpec[Activity]:
    """The spec of ``z_out``: an ``Activity`` whose ``h`` is $z$ ``(B, D)``,
    with unused zeros ``y`` ``(B, 1)`` (as ``dyn.nodes.Source``)."""
    return pc.msg.activity_spec(
        TensorSpec((streams, dim)), TensorSpec((streams, 1)), streams
    )


def draw_directions(m: int, d: int, generator: torch.Generator) -> torch.Tensor:
    """``m`` unit directions ``(m, d)`` of SIGReg, normal draws from
    ``generator`` (on its device) normalized."""
    a = torch.randn(m, d, generator=generator, device=generator.device)
    return a / torch.linalg.vector_norm(a, dim=1, keepdim=True)


# --- the frames ------------------------------------------------------------------


class FrameRing(Stateful):
    """The last ``frames`` frames of each of ``B`` streams, and the frames
    since each stream's reset.

    ``push_`` appends a frame of each stream (after clearing the streams that
    reset); ``chunk`` is ``(B, N, C, H, W)``, oldest first, zeros before a
    stream's first frames; ``ready`` says which streams have ``N`` frames
    since their reset. Its state is both tensors.

    Args:
        image: The spec of a frame of every stream, ``(B, C, H, W)``.
        frames: $N$.
        device: Where the tensors are.
    """

    def __init__(self, image: TensorSpec, frames: int, device: Device) -> None:
        b, *shape = image.shape
        self._n = frames
        self.chunk = torch.zeros(b, frames, *shape, device=device)
        self.count = torch.zeros(b, dtype=torch.int64, device=device)

    def push_(self, x: torch.Tensor, reset: torch.Tensor) -> None:
        """Append ``x`` ``(B, C, H, W)``; the streams with ``reset`` ``(B,)``
        start anew from it."""
        kept = torch.where(rows(reset, self.chunk), 0.0, self.chunk)
        self.chunk = torch.cat([kept[:, 1:], x[:, None].to(kept)], 1)
        self.count = (torch.where(reset, 0, self.count) + 1).clamp(max=self._n)

    @property
    def ready(self) -> torch.Tensor:
        """``(B,)`` bool: the streams with ``N`` frames since their reset."""
        return self.count >= self._n

    @override
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"chunk": self.chunk, "count": self.count}

    @override
    def load_state_dict(self, state: dict[str, torch.Tensor], /) -> None:
        self.chunk = state["chunk"].to(self.chunk).clone()
        self.count = state["count"].to(self.count).clone()


class Chunker(Node):
    r"""Shows the host's frames to the encoder as views of chunks of $N$ frames.

    It holds no parameters. The host writes a ``pc.msg.Sample``: the frames
    ``x`` ``(B, C, H, W)``, unused zeros ``y`` ``(B, 1)``, ``mask`` and
    ``reset`` ``(B,)``. In the update that first sees a write, the node pushes
    the frames into its ``FrameRing`` (``ring``), draws ``V - 1`` context masks
    per stream from its ``generator`` (``masks.block_masks``) and sends, once,
    the views of each stream's chunk as ``B V`` samples
    (``masks.make_views``): ``h`` ``(B V, N, C + 1, H, W)``, ``y`` the masks of
    the tokens ``(B V, T)`` (0 for the target view), and ``mask`` the host's
    ``mask`` of the stream while it has ``N`` frames since its ``reset``
    (False before: nothing learns from a chunk that is not full), ``reset``
    the host's. The masks are drawn at every frame, whether the chunk is full
    or not. The encoder nodes keep their ``r`` and ``lam`` from frame to frame
    (a ``reset`` clears them), so a frame's first updates start from the
    credit of the context masks of the frame before, as they start from that
    of the frame before in any stream of ``pc.Layer``. ``next`` receives the
    credit of the first encoder node and drops it: the frames are constants.

    With ``carry_frame`` (the reconstruction term of the head) each row of
    ``y`` has, after the ``T`` masks, the ``C H W`` pixels of the stream's
    last (raw) frame in the target view's row and zeros in the others
    (``loss.split_y``): the head reads it from the message it receives, as it
    reads the masks, and the encoder nodes pass it on with the tokens.

    Args:
        image: The spec of the frames, ``(B, C, H, W)``.
        frames: $N$.
        views: $V$.
        patch: The side of a patch.
        ratio: The ``ratio`` of ``block_masks``.
        blocks: The ``blocks`` of ``block_masks``.
        period: The period of the node in seconds.
        seed: The seed of ``generator``.
        carry_frame: Send the last frame in ``y`` (module docstring).
        device: Where the node computes.
    """

    def __init__(
        self,
        *,
        image: TensorSpec,
        frames: int,
        views: int,
        patch: int,
        ratio: float,
        blocks: int,
        period: float,
        seed: int,
        carry_frame: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=period, seed=seed)
        b, c, h, w = image.shape
        self._b, self._v, self._patch = b, views, patch
        self._carry = carry_frame
        self._grid = (h // patch, w // patch)
        self._ratio, self._blocks = ratio, blocks
        y = TensorSpec((b, 1))
        mask = TensorSpec((b,), torch.bool)
        self.prev = Port(
            recv=DictSpec(pc.msg.Sample, x=image, y=y, mask=mask, reset=mask)
        )
        n = b * views
        h_spec = TensorSpec((n, frames, c + 1, h, w))
        width = self._grid[0] * self._grid[1] + (c * h * w if carry_frame else 0)
        tokens = TensorSpec((n, width))
        self.act_spec = pc.msg.activity_spec(h_spec, tokens, n)
        self.next = Port(send=self.act_spec, recv=h_spec)  # the credit is dropped
        self.ring = FrameRing(image, frames, self.device)
        self.seen = -1  # the time (ns) of the write seen last
        self.register_state("seen")

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, written = self.prev.recv_with_time()
        if written is None or written.ns == self.seen:
            return
        self.seen = written.ns
        self.ring.push_(sample["x"], sample["reset"])
        b, v = self._b, self._v
        masks = block_masks(
            b * (v - 1),
            self._grid,
            ratio=self._ratio,
            blocks=self._blocks,
            generator=self.generator,
        ).unflatten(0, (b, v - 1))
        h, y = make_views(self.ring.chunk, masks, self._patch)
        if self._carry:
            last = self.ring.chunk[:, -1].flatten(1).to(y)  # (B, C H W)
            frames = last.new_zeros(b, v, last.shape[1])
            frames[:, 0] = last
            y = torch.cat([y, frames.flatten(0, 1)], 1)
        mask = sample["mask"] & self.ring.ready
        self.next.send(
            {
                "h": h,
                "y": y,
                "mask": mask.repeat_interleave(v),
                "reset": sample["reset"].repeat_interleave(v),
                "time": written,
            }
        )


# --- the head --------------------------------------------------------------------


class JEPAHead(Node):
    r"""The predictor and the JEPA loss $L$ (``loss.jepa_loss``) on the tokens
    of the last encoder node, as a node of PC-ALM.

    It receives on ``prev`` the tokens of the ``B V`` views ``(B V, E + T, D)``
    with the context masks in ``y``. In the update that first sees a frame it
    draws ``directions`` unit directions of SIGReg from its ``generator``
    (``draw_directions``), the same for the whole frame. Each update it sends
    back on ``prev`` the credit of each sample, $\partial E / \partial h$ of
    $E = B V L$ (module docstring; 0 for the views of a stream without a full
    chunk), from one autograd of its own computation from the detached tokens;
    $L$ depends on the tokens, the predictor and the directions only, so the
    credit is computed again only when one of them changed (the last credit,
    and the tokens it is of, are kept). It sends $z$ of the target views on
    ``z_out`` (an ``Activity``: ``h`` ``(B, D)``, the ``mask`` and ``reset`` of
    the stream, the frame's ``time``), a port that only sends: what reads it
    sends no credit back (M16's dynamics). It sends the terms of $L$ on
    ``stats`` (``STATS``): the first of a frame are those before any node
    learned from it (prequential). When its clock says, it learns the
    predictor by $\frac{1}{BV} \partial E / \partial \theta = \partial L /
    \partial \theta$.

    With a ``decoder`` and ``recon`` $\mu > 0$ the loss has the term $\mu
    L_{rec}$ (``loss.jepa_loss``): the head decodes $z$ of the target views and
    compares it with the target view's last frame, which it reads from the
    ``y`` it receives (``Chunker`` with ``carry_frame``). The decoder is the
    head's own parameters, learned with the predictor's by the same
    optimizer, from the same energy. Its credit reaches the encoder through
    $z$ of the target view as that of SIGReg does, also with ``sg`` (which only
    stops the prediction's).

    Args:
        predictor: The predictor, on ``device``.
        act_spec: The spec of what the last encoder node sends.
        views: $V$.
        extra: The tokens before the patch tokens.
        z_pool: How $z$ is pooled (``loss.pool``).
        lam: $\lambda$, the weight of SIGReg.
        sg: Stop the prediction's gradient at the target view.
        directions: The directions of SIGReg drawn per frame.
        sigreg_on: What SIGReg sees (``loss.jepa_loss``).
        decoder: The decoder of the reconstruction, on ``device`` (None: no
            reconstruction).
        recon: $\mu$, its weight.
        pixel_mean: The data's mean pixel (the standardization of the target).
        pixel_std: The data's pixel std.
        tau: The time constants of PC-ALM.
        optimizer: The optimizer of the predictor and the decoder.
        learn_interval: As in ``pc.clock.LearnClock``.
        seed: The seed of ``generator``.
        device: Where the node computes.
    """

    def __init__(
        self,
        predictor: Predictor,
        *,
        act_spec: DictSpec[Activity],
        views: int,
        extra: int,
        z_pool: str,
        lam: float,
        sg: bool,
        directions: int,
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        seed: int,
        sigreg_on: str = "z",
        decoder: PixelDecoder | None = None,
        recon: float = 0.0,
        pixel_mean: float = 0.0,
        pixel_std: float = 1.0,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period, seed=seed)
        self.module = predictor
        self.decoder = decoder
        self._recon = recon, pixel_mean, pixel_std
        self._v, self._extra, self._pool = views, extra, z_pool
        self._lam, self._sg, self._sigreg_on = lam, sg, sigreg_on
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(predictor.parameters())
        if decoder is not None:
            self._params += decoder.parameters()
        self.optimizer = optimizer.bind(self._params)
        h = act_spec.field("h", TensorSpec)
        n = h.shape[0]
        self._samples, self._b = n, n // views  # B V, B
        d = h.shape[-1]
        dev = self.device
        self.directions = torch.zeros(directions, d, device=dev)
        # The credit sent last and the tokens it is of; stale after a new frame
        # or a step of the predictor.
        self.g, self.g_of = h.zeros(device=dev), h.zeros(device=dev)
        self.cached = False
        self.register_state("cached")
        self.values = torch.zeros(5, device=dev)  # loss, pred, sigreg, recon, n of g
        self.z = torch.zeros(self._b, d, device=dev)
        self.prev = Port(send=h, recv=act_spec)
        self.z_out = Port(send=z_spec(self._b, d))
        self.stats = Port(send=STATS)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            m, d = self.directions.shape
            self.directions = draw_directions(m, d, self.generator)
            self.cached = False
        b, v = self._b, self._v
        valid = act["mask"].view(b, v)[:, 0]
        # The two reads of the device: whether to learn (on the clock's ticks
        # only) and whether the tokens changed since the credit was computed.
        learn = tick.learn and bool(valid.any())
        g_theta: list[torch.Tensor] = []
        if learn or not (self.cached and torch.equal(act["h"], self.g_of)):
            g_theta = self._credit(act, valid, learn)
        self.prev.send(self.g)
        target = slice(0, None, v)  # the rows of view 0
        self.z_out.send(
            {
                "h": self.z,
                "y": torch.zeros(b, 1, device=self.device),
                "mask": valid,
                "reset": act["reset"][target],
                "time": act["time"],
            }
        )
        loss, pred, sig, rec, n = self.values
        self.stats.send(
            {
                "loss": loss,
                "pred": pred,
                "sigreg": sig,
                "recon": rec,
                "n": n,
                "time": act["time"],
            }
        )
        if learn:
            # theta moves by dL/dtheta = (1 / B V) dE/dtheta.
            self.optimizer.step_([g / self._samples for g in g_theta], tick.elapsed)
            self.cached = False

    def _credit(
        self, act: Activity, valid: torch.Tensor, learn: bool
    ) -> list[torch.Tensor]:
        """Set the credit ``g`` of ``act``, $z$ and the terms of $L$; returns
        the predictor's gradient of $E$ when it ``learn``\\ s."""
        b, v = self._b, self._v
        h = act["h"]
        params = self._params if learn else []
        shape = None if self.decoder is None else self.decoder.shape
        masks, frames = split_y(act["y"], b, v, shape)
        recon = None
        if self.decoder is not None:
            assert frames is not None, "the Chunker must carry the frame"
            mu, mean, std = self._recon
            recon = Reconstruction(mu, self.decoder, mean, std, frames)
        with torch.enable_grad():
            leaf = h.detach().requires_grad_()
            terms = jepa_loss(
                leaf.view(b, v, *h.shape[1:]),
                masks,
                valid,
                self.directions,
                self.module,
                extra=self._extra,
                z_pool=self._pool,
                lam=self._lam,
                sg=self._sg,
                sigreg_on=self._sigreg_on,
                recon=recon,
            )
            energy = self._samples * terms.loss  # 0, of no gradient, with no chunk
            # The gradient of this node's own energy, from a detached leaf.
            g, *g_theta = torch.autograd.grad(  # mujou: local-autograd
                energy, [leaf, *params], materialize_grads=True
            )
        self.g, self.g_of, self.cached = g, h, True
        self.z = pool(
            h.view(b, v, *h.shape[1:])[:, 0], z_pool=self._pool, extra=self._extra
        )
        self.values = torch.stack(
            [
                terms.loss.detach(),
                terms.pred.detach(),
                terms.sigreg.detach(),
                terms.recon.detach(),
                terms.valid,
            ]
        )
        return cast(list[torch.Tensor], g_theta)
