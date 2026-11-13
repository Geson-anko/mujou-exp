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

from typing import Any, TypedDict, cast, override

import mujou.pc as pc
import torch
from mujou.pc.msg import Activity, rows
from mujou.pc.optim import Optimizer
from mujou.time import Timestamp
from mujou.types import Device

from exp.lib.cuda_graph import Replayed
from exp.lib.gauss_kl import GaussMoments, buffer_moments, min_rows, off
from exp.lib.sigreg import T_POINTS, ecf_sums, insert_
from exp.video.jepa.loss import (
    JEPATerms,
    PastSums,
    Reconstruction,
    jepa_loss,
    jepa_loss_packed,
    pool,
    split_y,
)
from exp.video.jepa.masks import block_masks, make_views, multiblock_masks, pack_views
from exp.video.jepa.vit import (
    BroadcastDecoder,
    PackedPredictor,
    PixelDecoder,
    Predictor,
    Projector,
    ZPredictor,
)
from exp.video.schema.vision import MaskConf
from mujou import DictSpec, Node, Port, Stateful, TensorSpec, TimestampSpec

__all__ = [
    "STATS",
    "Chunker",
    "FrameRing",
    "JEPAHead",
    "JEPAStats",
    "PackedJEPAHead",
    "PastPoints",
    "PastTokens",
    "StackChunker",
    "draw_directions",
    "draw_packed_directions",
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
    zpred: torch.Tensor
    n: torch.Tensor
    time: Timestamp


_SCALAR = TensorSpec(())
STATS = DictSpec(
    JEPAStats,
    loss=_SCALAR,
    pred=_SCALAR,
    sigreg=_SCALAR,
    recon=_SCALAR,
    zpred=_SCALAR,
    n=_SCALAR,
    time=TimestampSpec(),
)


def stats_message(terms: JEPATerms, time: Timestamp) -> JEPAStats:
    """The message of ``STATS`` of ``terms`` (detached; ``zpred`` 0 without
    one)."""
    zpred = terms.zpred if terms.zpred is not None else torch.zeros_like(terms.loss)
    return {
        "loss": terms.loss.detach(),
        "pred": terms.pred.detach(),
        "sigreg": terms.sigreg.detach(),
        "recon": terms.recon.detach(),
        "zpred": zpred.detach(),
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


def draw_packed_directions(
    m: int, d_z: int, d: int, *, tokens: bool, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The directions of SIGReg of the packed layout: ``(m, d_z)`` on $z$, then,
    when SIGReg also sees the tokens (``tokens``) and $z$ is not of their width
    ``d``, ``(m, d)`` on the tokens (else None: the same as on $z$)."""
    dz = draw_directions(m, d_z, generator)
    dt = draw_directions(m, d, generator) if tokens and d != d_z else None
    return dz, dt


class PastTokens(Stateful):
    r"""The past target tokens of the SIGReg of the tokens (``vision.token_buffer``,
    ``loss.PastSums``), held by the node that computes the loss.

    Its sizes are per stream: of ``streams`` streams it holds ``streams x
    capacity`` rows and inserts ``streams x insert`` tokens per frame, so the
    ages in the buffer do not depend on the streams (with all of them valid,
    a row survives a frame with probability $1 - \mathrm{insert} /
    \mathrm{capacity}$, and a row's age, in frames since the frame of its
    tokens, is geometric of mean ``capacity / insert``).
    ``keep`` holds the target tokens of a frame ``(B, T, D)`` (detached) and
    the streams that have a stack ``(B,)``: the last kept of a frame are
    those of its last update. ``next_frame``, in the update that first sees a
    frame, after its directions are drawn, puts ``streams x insert`` of the
    tokens kept into ``rows`` (``streams x capacity`` x ``D``, ``count``
    filled), drawn uniformly without replacement by ``generator`` from the
    tokens of the streams that had a stack (all of them when fewer; at
    ``insert`` = $T$ every token of each valid stream), each into the next
    empty row or, once full, a uniformly drawn one (``exp.lib.sigreg.insert_``:
    random replacement). It then sums the characteristic function of the
    filled rows along the frame's directions (``exp.lib.sigreg.ecf_sums``):
    ``sums``, the same through the frame. A frame without a stack inserts
    nothing, and one stream alone ($B = 1$) fills the buffer from its own
    past. Its state is all of these.

    Raises:
        ValueError: ``capacity`` or ``insert`` below 1, or ``insert`` above
            the tokens of a stack $T$; in ``load_state_dict``, a state of
            other shapes (other sizes per stream, streams or tokens: a state
            from before the sizes were per stream holds ``capacity`` rows in
            all).
    """

    def __init__(
        self,
        *,
        capacity: int,
        insert: int,
        streams: int,
        tokens: int,
        dim: int,
        directions: int,
        device: Device,
    ) -> None:
        if capacity < 1 or not 1 <= insert <= tokens:
            raise ValueError(
                f"the past tokens need capacity >= 1 and 1 <= insert <= the "
                f"{tokens} tokens of a stack, per stream: not {capacity} and {insert}"
            )
        self._insert = streams * insert
        self.rows = torch.zeros(streams * capacity, dim, device=device)
        self.count = 0
        self.kept = torch.zeros(streams, tokens, dim, device=device)
        self.kept_valid = torch.zeros(streams, dtype=torch.bool, device=device)
        sums = torch.zeros(directions, len(T_POINTS), device=device)
        self.cos, self.sin = sums, sums.clone()
        self.n = torch.zeros((), device=device)

    def keep(self, target: torch.Tensor, valid: torch.Tensor) -> None:
        """Hold the target tokens ``(B, T, D)`` and the streams with a stack
        ``(B,)`` of this update."""
        self.kept, self.kept_valid = target.detach(), valid

    def next_frame(self, directions: torch.Tensor, generator: torch.Generator) -> None:
        """Insert the tokens kept, then sum the rows along ``directions``
        ``(M, D)`` (class docstring)."""
        d = self.rows.shape[1]
        pool = self.kept[self.kept_valid].reshape(-1, d)  # the valid streams' tokens
        drawn = torch.randperm(len(pool), generator=generator, device=generator.device)
        self.count = insert_(
            self.rows, self.count, pool[drawn[: self._insert]], generator
        )
        self.kept_valid = torch.zeros_like(self.kept_valid)  # inserted once
        self.cos, self.sin = ecf_sums(self.rows[: self.count], directions)
        self.n = torch.tensor(float(self.count), device=self.rows.device)

    @property
    def sums(self) -> PastSums:
        """The ``PastSums`` of the filled rows along the frame's directions."""
        return PastSums(self.cos, self.sin, self.n)

    @override
    def state_dict(self) -> dict[str, Any]:
        names = ("rows", "count", "kept", "kept_valid", "cos", "sin", "n")
        return {name: getattr(self, name) for name in names}

    @override
    def load_state_dict(self, state: dict[str, Any], /) -> None:
        names = ("rows", "kept", "kept_valid", "cos", "sin", "n")
        for name in names:
            have, got = getattr(self, name).shape, state[name].shape
            if got != have:
                raise ValueError(f"the past tokens' {name} is {have}, not {got}")
        for name in names:
            setattr(self, name, state[name].to(getattr(self, name)).clone())
        self.count = int(state["count"])


class PastPoints(Stateful):
    r"""The past data points of the SIGReg of the data points
    (``vision.datapoint_buffer``, ``loss.jepa_loss_packed`` with ``sigreg_on``
    ``"datapoints"``), held by the node that computes the loss.

    A data point is one row of ``width`` values of ``dtype``: the
    representation of a stack (its target tokens flattened) or its input (the
    target view's pixels, u8). Of ``streams`` streams it holds ``streams x
    capacity`` rows. ``keep`` holds the data points of an update ``(B,
    width)`` (detached) and the streams that have a stack ``(B,)``: the last
    kept of a frame are those of its last update. ``next_frame``, in the
    update that first sees frame $j$ (of this node, from 0), puts the data
    points kept (of frame $j - 1$, of the streams that had a stack) into the
    rows when $j$ is a multiple of ``every`` (``exp.lib.sigreg.insert_``: the
    next empty rows, then distinct rows drawn uniformly by ``generator``),
    and, given ``directions`` ``(M, width)``, sums the characteristic function
    of the filled rows along them (``exp.lib.sigreg.ecf_sums``): ``sums``,
    constants through the frame. Once full, an insertion of the $B$ streams
    replaces $B$ of the $B N$ rows ($N$ = ``capacity``), so a row survives it
    with probability $1 - 1/N$, and $s$ frames ($s$ a multiple of $k$ =
    ``every``) with probability $(1 - 1/N)^{s/k}$: a row's age is geometric
    of mean $N k$ frames, whatever $B$. Its state is all of these and the
    frames seen.

    With ``moments`` (``vision.datapoint_law`` "gauss") an insertion of at
    least one row also makes the ``exp.lib.gauss_kl.buffer_moments`` of the
    filled rows (``moments``, constants until the next insertion; off while
    fewer than ``min_rows(width)`` = ``width + 5`` rows are filled), and they
    are in the state. Filled rows of a singular scatter give moments of
    $\tilde F = +\infty$ and a NaN precision (``buffer_moments``).

    Raises:
        ValueError: ``capacity`` or ``every`` below 1, or with ``moments``
            fewer than ``width + 5`` rows in all (the term would never be
            on); in ``load_state_dict``, a state of other shapes.
    """

    def __init__(
        self,
        *,
        capacity: int,
        every: int,
        streams: int,
        width: int,
        dtype: torch.dtype,
        directions: int,
        device: Device,
        moments: bool = False,
    ) -> None:
        if capacity < 1 or every < 1:
            raise ValueError(
                f"the past data points need capacity >= 1 and every >= 1, not "
                f"{capacity} and {every}"
            )
        if moments and streams * capacity < min_rows(width):
            raise ValueError(
                f"the moments of data points of {width} numbers need at least "
                f"{min_rows(width)} rows, not {streams} x {capacity}"
            )
        self._every = every
        self._moments = moments
        self.rows = torch.zeros(streams * capacity, width, dtype=dtype, device=device)
        self.count = 0
        self.frames = 0  # the frames seen
        self.kept = torch.zeros(streams, width, dtype=dtype, device=device)
        self.kept_valid = torch.zeros(streams, dtype=torch.bool, device=device)
        sums = torch.zeros(directions, len(T_POINTS), device=device)
        self.cos, self.sin = sums, sums.clone()
        self.n = torch.zeros((), device=device)
        # the moments (with moments; of width 0 without)
        self.mean, self.precision, self.kl, self.on = (
            off(width, device, dtype) if moments else off(0, device)
        )

    def keep(self, points: torch.Tensor, valid: torch.Tensor) -> None:
        """Hold the data points ``(B, width)`` and the streams with a stack
        ``(B,)`` of this update."""
        self.kept, self.kept_valid = points.detach(), valid

    def next_frame(
        self, directions: torch.Tensor | None, generator: torch.Generator
    ) -> None:
        """Insert the data points kept when due (and, with ``moments``,
        make their moments), then sum the rows along ``directions`` (None: no
        sums; class docstring)."""
        if self.frames % self._every == 0:
            new = self.kept[self.kept_valid]
            self.count = insert_(self.rows, self.count, new, generator)
            if self._moments and len(new):
                m = buffer_moments(self.filled)
                self.mean, self.precision, self.kl, self.on = m
        self.frames += 1
        self.kept_valid = torch.zeros_like(self.kept_valid)  # inserted once
        if directions is not None:
            self.cos, self.sin = ecf_sums(self.filled.float(), directions)
            self.n = torch.tensor(float(self.count), device=self.rows.device)

    @property
    def filled(self) -> torch.Tensor:
        """The filled rows ``(count, width)``."""
        return self.rows[: self.count]

    @property
    def moments(self) -> GaussMoments:
        """The ``GaussMoments`` of the filled rows at the last insertion."""
        return GaussMoments(self.mean, self.precision, self.kl, self.on)

    def _names(self) -> tuple[str, ...]:
        """The tensors of the state."""
        names = ("rows", "kept", "kept_valid", "cos", "sin", "n")
        return names + (("mean", "precision", "kl", "on") if self._moments else ())

    @property
    def sums(self) -> PastSums:
        """The ``PastSums`` of the filled rows along the frame's directions."""
        return PastSums(self.cos, self.sin, self.n)

    @override
    def state_dict(self) -> dict[str, Any]:
        names = ("count", "frames", *self._names())
        return {name: getattr(self, name) for name in names}

    @override
    def load_state_dict(self, state: dict[str, Any], /) -> None:
        names = self._names()
        for name in names:
            have, got = getattr(self, name).shape, state[name].shape
            if got != have:
                raise ValueError(f"the past data points' {name} is {have}, not {got}")
        for name in names:
            setattr(self, name, state[name].to(getattr(self, name)).clone())
        self.count, self.frames = int(state["count"]), int(state["frames"])


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


class StackChunker(Node):
    r"""Packs the host's stacks of frames into the sequences of the encoder
    (the packed layout, M17).

    It holds no parameters. The host writes a ``pc.msg.Sample``: the stacks
    ``x`` u8 ``(B, N, C, H, W)`` (``StackedTourFrames``), unused zeros ``y``
    ``(B, 1)``, ``mask`` (the streams whose stack is within one video) and
    ``reset`` ``(B,)``. In the update that first sees a write, it draws the
    mask of each context view (in the order of ``masks``, ``B`` each, from its
    ``generator``; ``masks.multiblock_masks`` with ``keeps``) and sends, once,
    ``masks.pack_views``: ``h`` ``(B, L, P + 1)``, ``y`` ``(B, (V - 1) T)``,
    and the host's ``mask`` and ``reset``. ``next`` receives the credit of the
    first encoder node and drops it: the frames are constants.

    Args:
        stack: The spec of the stacks, ``(B, N, C, H, W)`` u8.
        patch: The side of a patch.
        tubelet: The frames of a tubelet.
        masks: The context views.
        keeps: The visible patches of a time slot of each
            (``masks.spatial_keep``).
        period: The period of the node in seconds.
        seed: The seed of ``generator``.
        device: Where the node computes.
    """

    def __init__(
        self,
        *,
        stack: TensorSpec,
        patch: int,
        tubelet: int,
        masks: list[MaskConf],
        keeps: list[int],
        period: float,
        seed: int,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=period, seed=seed)
        b, f, c, h, w = stack.shape
        self._patch, self._tubelet = patch, tubelet
        self._masks, self._keeps = list(masks), list(keeps)
        self._grid = (h // patch, w // patch)
        slots = f // tubelet
        t = slots * self._grid[0] * self._grid[1]
        length = t + sum(
            k * (slots - m.future) for m, k in zip(masks, keeps, strict=True)
        )
        y = TensorSpec((b, 1))
        mask = TensorSpec((b,), torch.bool)
        self.prev = Port(
            recv=DictSpec(pc.msg.Sample, x=stack, y=y, mask=mask, reset=mask)
        )
        h_spec = TensorSpec((b, length, tubelet * c * patch * patch + 1))
        self.act_spec = pc.msg.activity_spec(h_spec, TensorSpec((b, len(masks) * t)), b)
        self.next = Port(send=self.act_spec, recv=h_spec)  # the credit is dropped
        self.seen = -1  # the time (ns) of the write seen last
        self.register_state("seen")

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, written = self.prev.recv_with_time()
        if written is None or written.ns == self.seen:
            return
        self.seen = written.ns
        b = sample["x"].shape[0]
        views = [
            multiblock_masks(b, self._grid, m, keep=k, generator=self.generator)
            for m, k in zip(self._masks, self._keeps, strict=True)
        ]
        h, y = pack_views(
            sample["x"],
            views,
            keeps=self._keeps,
            patch=self._patch,
            tubelet=self._tubelet,
            futures=[m.future for m in self._masks],
        )
        self.next.send(
            {
                "h": h,
                "y": y,
                "mask": sample["mask"],
                "reset": sample["reset"],
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
        cuda_graph: Replay the credit and the gradients from CUDA graphs
            (``exp.lib.cuda_graph``; on CUDA only), the same bit for bit.
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
        decoder: PixelDecoder | BroadcastDecoder | None = None,
        recon: float = 0.0,
        pixel_mean: float = 0.0,
        pixel_std: float = 1.0,
        device: Device = "cpu",
        cuda_graph: bool = False,
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
        on_cuda = cuda_graph and self.device.type == "cuda"
        self._replay = Replayed(self._terms) if on_cuda else None  # not state
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
                "zpred": torch.zeros_like(loss),
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
        h = act["h"]
        if self._replay is not None:
            g, z, values, *g_theta = self._replay(
                learn, h, act["y"], valid, self.directions
            )
        else:
            g, z, values, *g_theta = self._terms(
                learn, h, act["y"], valid, self.directions
            )
        self.g, self.g_of, self.cached = g, h, True
        self.z, self.values = z, values
        return g_theta

    def _terms(
        self,
        learn: bool,
        h: torch.Tensor,
        y: torch.Tensor,
        valid: torch.Tensor,
        directions: torch.Tensor,
    ) -> list[torch.Tensor]:
        """``[g, z, values, *g_theta]`` of ``_credit`` (``g_theta`` only when
        it learns), a function of tensors only: the one the CUDA graph
        replays."""
        b, v = self._b, self._v
        params = self._params if learn else []
        shape = None if self.decoder is None else self.decoder.shape
        masks, frames = split_y(y, b, v, shape)
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
                directions,
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
        z = pool(h.view(b, v, *h.shape[1:])[:, 0], z_pool=self._pool, extra=self._extra)
        values = torch.stack(
            [
                terms.loss.detach(),
                terms.pred.detach(),
                terms.sigreg.detach(),
                terms.recon.detach(),
                terms.valid,
            ]
        )
        return [g, z, values, *cast(list[torch.Tensor], g_theta)]


class PackedJEPAHead(Node):
    r"""The predictor, the projection and the JEPA loss of the packed layout
    (``loss.jepa_loss_packed``) on the tokens of the last encoder node, as a
    node of PC-ALM (M17); ``JEPAHead`` otherwise.

    It receives on ``prev`` the packed tokens ``(B, L, D)`` with the context
    masks in ``y``. In the update that first sees a frame it draws the
    directions of SIGReg (``draw_packed_directions``). Each update it sends
    back the credit of each sample, $\partial E / \partial h$ of $E = B L$
    (0 for a stream whose ``mask`` is False), from one autograd of its own
    computation from the detached tokens, computed again only when the tokens,
    the parameters or the directions changed. It sends $z$ (after the
    projection, ``(B, D_z)``) on ``z_out`` (send-only) and the terms of $L$ on
    ``stats`` (the first of a frame prequential; ``recon`` 0). When its clock
    says, it learns the predictor and the projection by $\frac1B \partial E /
    \partial \theta = \partial L / \partial \theta$.

    With ``token_buffer`` > 0 the SIGReg of the tokens also sees the past
    target tokens of a ``PastTokens`` of that capacity per stream (``past``),
    the node's own: it keeps the target tokens it receives in every update,
    and in the update that first sees a frame, after drawing the directions,
    inserts ``token_insert`` per stream of those of the frame before (from
    ``generator``) and sums them along the directions of the tokens. Nothing
    of another node goes into it.

    With ``sigreg_on`` ``"datapoints"`` the directions of SIGReg are of the
    flattened target tokens ($T D$), and with ``datapoint_buffer`` > 0 the
    samples include the past data points of a ``PastPoints`` of that capacity
    per stream (``points``; their representations, constants), the node's
    own: it keeps the flattened target tokens it receives in every update,
    and in the update that first sees a frame, after drawing the directions,
    inserts those of the frame before once every ``datapoint_every`` frames
    and sums them along the directions. (Stored inputs encoded anew need the
    encoder, which the head does not hold: ``JEPABP`` only.) With
    ``datapoint_law`` ``"gauss"`` the term of the data points is the
    Gaussian moment matching (``loss.jepa_loss_packed``): the buffer makes
    the moments of its rows at each insertion (``PastPoints.moments``)
    instead of the sums, and the credit of the current data points is their
    $\lambda \hat g_b / D$ (``exp.lib.gauss_kl``), the head's own: the
    regularizer stays within the node, as SIGReg does.

    With ``target`` (I-JEPA's target encoder, ``vision.ema``) it receives
    ``(B, L + T, D)``: the packed tokens, then the $T$ tokens of the target
    encoder's path (``ema.TargetStage``), the prediction's targets. They are
    constants of $L$: their credit is 0.

    Args:
        predictor: The predictor, on ``device``.
        proj: The projection to $z$, on ``device`` (None: $z$ is the mean).
        act_spec: The spec of what the last encoder node sends.
        keeps: $K_v$ of each context view (tokens).
        lam: $\lambda$, the weight of SIGReg.
        sg: Stop the prediction's gradient at the target view.
        sigreg_on: What SIGReg sees.
        pred_loss: ``"l2"`` or ``"l1"``.
        directions: The directions of SIGReg drawn per frame.
        tau: The time constants of PC-ALM.
        optimizer: The optimizer of the predictor and the projection.
        learn_interval: As in ``pc.clock.LearnClock``.
        seed: The seed of ``generator``.
        token_sigreg: The SIGReg of the tokens (``loss.jepa_loss_packed``).
        token_weight: $\beta$ of the prediction of the tokens.
        z_weight: $\alpha$ of the completion of $u_g$.
        projector: $h$ of $u$, on ``device`` (None: $u = z$), learned with the
            predictor.
        zpred: $q$ of the completion (None: none), learned with them.
        target_norm: The prediction's targets through a LayerNorm without
            affine.
        target: The last $T$ rows it receives are the target encoder's
            tokens, the prediction's targets.
        token_buffer: The capacity of the past target tokens per stream (0:
            none).
        token_insert: The tokens of a frame inserted into them per stream.
        datapoint_buffer: The capacity of the past data points per stream
            (0: none).
        datapoint_every: The frames between two insertions into them.
        datapoint_law: The term of the data points, ``"sigreg"`` or
            ``"gauss"``.
        device: Where the node computes.
        cuda_graph: Replay the credit and the gradients from CUDA graphs (on
            CUDA only), the same bit for bit.
    """

    def __init__(
        self,
        predictor: PackedPredictor,
        *,
        proj: torch.nn.Linear | None,
        act_spec: DictSpec[Activity],
        keeps: list[int],
        lam: float,
        sg: bool,
        sigreg_on: str,
        pred_loss: str,
        directions: int,
        tau: pc.TimeConstants,
        optimizer: Optimizer,
        learn_interval: float | None,
        seed: int,
        token_sigreg: str = "scaled",
        token_weight: float = 1.0,
        z_weight: float = 0.0,
        projector: Projector | None = None,
        zpred: ZPredictor | None = None,
        target_norm: bool = False,
        target: bool = False,
        token_buffer: int = 0,
        token_insert: int = 64,
        datapoint_buffer: int = 0,
        datapoint_every: int = 1,
        datapoint_law: str = "sigreg",
        device: Device = "cpu",
        cuda_graph: bool = False,
    ) -> None:
        super().__init__(device=device, period=tau.period, seed=seed)
        self.module = predictor
        self._target = target
        self._law = datapoint_law
        self.proj = proj
        self.projector, self.zpred = projector, zpred  # in the state
        self._token_sigreg = token_sigreg
        self._weights = token_weight, z_weight
        self._keeps = list(keeps)
        self._lam, self._sg, self._sigreg_on = lam, sg, sigreg_on
        self._pred_loss = pred_loss
        self._target_norm = target_norm
        self.clock = pc.clock.LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval
        )
        self._params = list(predictor.parameters())
        for extra in (proj, projector, zpred):
            if extra is not None:
                self._params += extra.parameters()
        self.optimizer = optimizer.bind(self._params)
        on_cuda = cuda_graph and self.device.type == "cuda"
        self._replay = Replayed(self._terms) if on_cuda else None  # not state
        h = act_spec.field("h", TensorSpec)
        self._b, _, d = h.shape
        d_z = d if proj is None else proj.out_features
        d_u = d_z if projector is None else projector.out_features
        dev = self.device
        self._two = sigreg_on in ("tokens", "both", "views_tokens") and d_z != d
        self._views = sigreg_on in ("views", "views_tokens")
        self._t = act_spec.field("y", TensorSpec).shape[1] // len(self._keeps)  # T
        # the directions of the representation SIGReg sees: z, or a data point
        width = self._t * d if sigreg_on == "datapoints" else d_z
        self.directions = torch.zeros(directions, width, device=dev)
        self.directions_t = torch.zeros(directions if self._two else 0, d, device=dev)
        self.directions_u = torch.zeros(
            directions if self._views else 0, d_u, device=dev
        )
        self.g, self.g_of = h.zeros(device=dev), h.zeros(device=dev)
        self.cached = False
        self.register_state("cached")
        # loss, pred, sigreg, recon, n, zpred of g
        self.values = torch.zeros(6, device=dev)
        self.z = torch.zeros(self._b, d_z, device=dev)
        self.past: PastTokens | None = None  # the past target tokens (state)
        if token_buffer:
            self.past = PastTokens(
                capacity=token_buffer,
                insert=token_insert,
                streams=self._b,
                tokens=self._t,
                dim=d,
                directions=directions,
                device=dev,
            )
        self.points: PastPoints | None = None  # the past data points (state)
        if datapoint_buffer:
            self.points = PastPoints(
                capacity=datapoint_buffer,
                every=datapoint_every,
                streams=self._b,
                width=width,
                dtype=torch.float32,
                directions=directions,
                device=dev,
                moments=datapoint_law == "gauss",
            )
        self.prev = Port(send=h, recv=act_spec)
        self.z_out = Port(send=z_spec(self._b, d_z))
        self.stats = Port(send=STATS)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        if tick.new:
            m, d_z = self.directions.shape
            dz, dt_ = draw_packed_directions(
                m,
                d_z,
                self.directions_t.shape[1],
                tokens=self._two,
                generator=self.generator,
            )
            self.directions = dz
            if dt_ is not None:
                self.directions_t = dt_
            if self._views:
                self.directions_u = draw_directions(
                    m, self.directions_u.shape[1], self.generator
                )
            if self.past is not None:
                tokens = self.directions_t if self._two else self.directions
                self.past.next_frame(tokens, self.generator)
            if self.points is not None:
                gauss = self._law == "gauss"  # the moments, not the sums
                self.points.next_frame(
                    None if gauss else self.directions, self.generator
                )
            self.cached = False
        valid = act["mask"]
        if self.past is not None:
            self.past.keep(act["h"][:, : self._t], valid)
        if self.points is not None:
            self.points.keep(act["h"][:, : self._t].flatten(1), valid)
        learn = tick.learn and bool(valid.any())
        g_theta: list[torch.Tensor] = []
        if learn or not (self.cached and torch.equal(act["h"], self.g_of)):
            g_theta = self._credit(act, valid, learn)
        self.prev.send(self.g)
        self.z_out.send(
            {
                "h": self.z,
                "y": torch.zeros(self._b, 1, device=self.device),
                "mask": valid,
                "reset": act["reset"],
                "time": act["time"],
            }
        )
        loss, pred, sig, rec, n, zp = self.values
        self.stats.send(
            {
                "loss": loss,
                "pred": pred,
                "sigreg": sig,
                "recon": rec,
                "zpred": zp,
                "n": n,
                "time": act["time"],
            }
        )
        if learn:
            # theta moves by dL/dtheta = (1 / B) dE/dtheta.
            self.optimizer.step_([g / self._b for g in g_theta], tick.elapsed)
            self.cached = False

    def _credit(
        self, act: Activity, valid: torch.Tensor, learn: bool
    ) -> list[torch.Tensor]:
        h = act["h"]
        past: tuple[torch.Tensor, ...] = ()
        if self.past is not None:
            past = self.past.sums
        elif self.points is not None:
            gauss = self._law == "gauss"
            past = self.points.moments if gauss else self.points.sums
        args = (
            h,
            act["y"],
            valid,
            self.directions,
            self.directions_t,
            self.directions_u,
            *past,
        )
        if self._replay is not None:
            g, z, values, *g_theta = self._replay(learn, *args)
        else:
            g, z, values, *g_theta = self._terms(learn, *args)
        self.g, self.g_of, self.cached = g, h, True
        self.z, self.values = z, values
        return g_theta

    def _terms(
        self,
        learn: bool,
        h: torch.Tensor,
        y: torch.Tensor,
        valid: torch.Tensor,
        directions: torch.Tensor,
        directions_t: torch.Tensor,
        directions_u: torch.Tensor,
        *past: torch.Tensor,
    ) -> list[torch.Tensor]:
        """``[g, z, values, *g_theta]`` (``g_theta`` only when it learns), a
        function of tensors only: the one the CUDA graph replays (``past``:
        the ``PastSums`` of the past tokens or data points, the
        ``GaussMoments`` of the data points, or nothing)."""
        gauss = self._law == "gauss"
        params = self._params if learn else []
        token_weight, z_weight = self._weights
        with torch.enable_grad():
            leaf = h.detach().requires_grad_()
            tokens, goal = leaf, None
            if self._target:  # the target encoder's tokens after the packed ones
                tokens, goal = leaf[:, : -self._t], leaf[:, -self._t :]
            terms = jepa_loss_packed(
                tokens,
                y,
                valid,
                directions,
                directions_t if self._two else None,
                self.module,
                keeps=self._keeps,
                proj=self.proj,
                lam=self._lam,
                sg=self._sg,
                sigreg_on=self._sigreg_on,
                pred_loss=self._pred_loss,
                token_sigreg=self._token_sigreg,
                token_weight=token_weight,
                z_weight=z_weight,
                projector=self.projector,
                zpred=self.zpred,
                directions_u=directions_u if self._views else None,
                target_norm=self._target_norm,
                goal=goal,
                past=PastSums(*past) if past and not gauss else None,
                datapoint_law=self._law,
                moments=GaussMoments(*past) if past and gauss else None,
            )
            energy = self._b * terms.loss
            # The gradient of this node's own energy, from a detached leaf.
            g, *g_theta = torch.autograd.grad(  # mujou: local-autograd
                energy, [leaf, *params], materialize_grads=True
            )
        assert terms.zpred is not None  # of the packed loss
        values = torch.stack(
            [
                terms.loss.detach(),
                terms.pred.detach(),
                terms.sigreg.detach(),
                terms.recon.detach(),
                terms.valid,
                terms.zpred.detach(),
            ]
        )
        return [g, terms.z.detach(), values, *cast(list[torch.Tensor], g_theta)]
