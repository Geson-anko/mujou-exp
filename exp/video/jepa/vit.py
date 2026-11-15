r"""The modules of the vision: a ViT on chunks of $N$ frames and the predictor
of the JEPA loss, shared by the PC-ALM nodes, their BP baseline, the
evaluation and the decoder fit afterwards.

A chunk is ``(n, N, C', H, W)``: the $N$ frames of each sample, with
$C' = C + 1$ channels, the last of which marks the masked patches
(``masks.make_views``). A token is one patch of $p \times p$ pixels through
all $N$ frames (a tube), so there are $T = (H / p)(W / p)$ patch tokens
whatever $N$ is, in row-major order of the grid. ``extra`` tokens (a class
token for ``z_pool="cls"``, ``registers`` tokens for ``"registers"``, none
for ``"mean"``) come first: a stage maps ``(n, E + T, D)`` to the same shape,
the first stage maps the chunk to it.

The encoder is cut into ``nodes`` stages of ``depth / nodes`` pre-norm
transformer blocks; the first also embeds the patches (the pixels
standardized by the data's constants, masked patches at 0 after it; a linear
map of the tube, plus fixed 2-D sine-cosine positions), the last ends with a
``LayerNorm``. Each stage is one PC-ALM node (``pc.Layer``). Every sample is
computed on its own: attention is within a sample, LayerNorm within a token,
and there is no dropout and no batch norm.

The ``"packed"`` layout (M17, V-JEPA's): a stack of $N$ frames is cut into
tubelets of $\tau$ frames x $p \times p$ pixels (``patchify_tubelets``),
$T = (N / \tau)(H / p)(W / p)$ tokens in the order (time slot, row, column),
a token's id its index in that order. A sample is one sequence ``(L, P + 1)``
of the target view's $T$ tokens and the visible tokens of each context view
(``masks.pack_views``), the last column the token's id; ``PackedEmbed`` maps
each to the width and adds its fixed 3-D position, and the blocks attend
within each view's segment only (``Block``'s ``segments``), so the views of a
sample do not mix. $z$ is the mean of the target view's tokens, through a
linear map (``proj``) when ``vision.z_dim`` is not the width. With
``vision.token_dim`` the last stage maps each token to ``token_dim``
dimensions after its ``LayerNorm`` (``Stage``'s ``out``): the tokens the
encoder sends are the representation, and the predictor maps them to its
width and back (``PackedPredictor``).
"""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from exp.video.jepa.loss import pool
from exp.video.jepa.masks import make_views
from exp.video.schema.run import RunConf
from exp.video.schema.vision import VisionConf

__all__ = [
    "Block",
    "BroadcastDecoder",
    "Embed",
    "Encoder",
    "PackedEmbed",
    "PackedPredictor",
    "PixelDecoder",
    "Predictor",
    "Projector",
    "Segmented",
    "Stage",
    "VisionModules",
    "ZPredictor",
    "build_vision",
    "extra_tokens",
    "image_shape",
    "patchify",
    "patchify_tubelets",
    "sincos_2d",
    "sincos_3d",
]

MLP_RATIO = 4
INIT_STD = 0.02


def patchify(x: Tensor, patch: int) -> Tensor:
    """The tubes of ``x`` ``(n, N, C, H, W)``: ``(n, T, N C p^2)`` with
    $T = (H / p)(W / p)$ in row-major order of the grid, each the flattened
    ``(N, C, p, p)`` of its patch."""
    n, f, c, h, w = x.shape
    gh, gw = h // patch, w // patch
    x = x.reshape(n, f, c, gh, patch, gw, patch)
    return x.permute(0, 3, 5, 1, 2, 4, 6).reshape(n, gh * gw, f * c * patch * patch)


def patchify_tubelets(x: Tensor, patch: int, tubelet: int) -> Tensor:
    r"""The tubelets of ``x`` ``(n, N, C, H, W)``: ``(n, T, \tau C p^2)`` with
    $T = (N / \tau)(H / p)(W / p)$ in the order (time slot, row, column), each
    the flattened ``(tau, C, p, p)`` of its frames and patch. With
    ``tubelet = N`` it is ``patchify``."""
    n, f, c, h, w = x.shape
    gt, gh, gw = f // tubelet, h // patch, w // patch
    x = x.reshape(n, gt, tubelet, c, gh, patch, gw, patch)
    x = x.permute(0, 1, 4, 6, 2, 3, 5, 7)  # (n, gt, gh, gw, tau, C, p, p)
    return x.reshape(n, gt * gh * gw, tubelet * c * patch * patch)


def _sincos_1d(pos: Tensor, dim: int) -> Tensor:
    r"""``(len(pos), dim)``: $\sin$ and $\cos$ of ``pos`` times
    $10000^{-i / (dim / 2)}$, $i < dim / 2$ (``sincos_2d``'s code of an axis)."""
    half = dim // 2
    omega = 1.0 / 10000 ** (torch.arange(half, dtype=torch.float64) / half)
    a = pos.double().reshape(-1, 1) * omega
    return torch.cat([a.sin(), a.cos()], 1)


def sincos_3d(grid: tuple[int, int, int], dim: int) -> Tensor:
    """The fixed positions of the tubelet tokens, ``(gt gh gw, dim)`` in the
    order (time slot, row, column): ``dim / 4`` dimensions encode the slot,
    ``3 dim / 8`` the row and ``3 dim / 8`` the column, each by
    ``_sincos_1d`` (V-JEPA's 3-D sine-cosine; ``dim % 16 == 0``)."""
    gt, gh, gw = grid
    dt, ds = dim // 4, 3 * dim // 8
    t, r, c = torch.meshgrid(
        torch.arange(gt), torch.arange(gh), torch.arange(gw), indexing="ij"
    )
    parts = [_sincos_1d(t, dt), _sincos_1d(r, ds), _sincos_1d(c, ds)]
    return torch.cat(parts, 1).float()


def sincos_2d(grid: tuple[int, int], dim: int) -> Tensor:
    r"""The fixed positions of the patch tokens, ``(gh gw, dim)``: half of the
    dimensions encode the row, half the column, each as $\sin$ and $\cos$ of
    the index times $10000^{-2i / (dim / 2)}$ (as in MAE)."""
    gh, gw = grid
    quarter = dim // 4
    omega = 1.0 / 10000 ** (torch.arange(quarter, dtype=torch.float64) / quarter)
    rows, cols = torch.meshgrid(
        torch.arange(gh, dtype=torch.float64),
        torch.arange(gw, dtype=torch.float64),
        indexing="ij",
    )

    def code(pos: Tensor) -> Tensor:  # (T,) -> (T, dim / 2)
        a = pos.reshape(-1, 1) * omega
        return torch.cat([a.sin(), a.cos()], 1)

    return torch.cat([code(rows), code(cols)], 1).float()


def extra_tokens(conf: VisionConf) -> int:
    """The tokens before the patch tokens: 1 for ``"cls"``, ``registers`` for
    ``"registers"``, 0 for ``"mean"``."""
    return {"cls": 1, "registers": conf.registers, "mean": 0}[conf.z_pool]


def image_shape(conf: RunConf) -> tuple[int, int]:
    """The channels and the side of the frames of ``conf``'s stream: 1 and
    ``balls.size`` for the balls, 3 and ``tours.size`` for Walking Tours."""
    if conf.data.balls is not None:
        return 1, conf.data.balls.size
    assert conf.data.tours is not None  # DataConf has exactly one
    return 3, conf.data.tours.size


class Block(nn.Module):
    r"""A pre-norm transformer block on ``(n, T, D)``:
    $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$, then
    $x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x))$, with ``heads`` heads of
    softmax attention within each sample and an MLP of ``4 D`` with GELU.
    With ``segments`` (lengths that sum to ``T``) the attention is within each
    segment of consecutive tokens only (block diagonal); the rest is per
    token, so a segment's output depends on its own tokens alone."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self._heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, MLP_RATIO * dim)
        self.fc2 = nn.Linear(MLP_RATIO * dim, dim)

    @override
    def forward(self, x: Tensor, segments: Sequence[int] | None = None) -> Tensor:
        n, t, d = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(n, t, 3, self._heads, d // self._heads)
            .permute(2, 0, 3, 1, 4)
        )  # each (n, heads, T, d_head)
        scale = math.sqrt(d // self._heads)
        if segments is None:
            a = (q @ k.transpose(-2, -1) / scale).softmax(-1)
            out = a @ v
        else:
            outs = []
            for qs, ks, vs in zip(
                q.split(list(segments), 2),
                k.split(list(segments), 2),
                v.split(list(segments), 2),
                strict=True,
            ):
                outs.append((qs @ ks.transpose(-2, -1) / scale).softmax(-1) @ vs)
            out = torch.cat(outs, 2)
        x = x + self.proj(out.transpose(1, 2).reshape(n, t, d))
        return x + self.fc2(nn.functional.gelu(self.fc1(self.norm2(x))))


class Embed(nn.Module):
    r"""The chunk ``(n, N, C', H, W)`` as tokens ``(n, E + T, D)``: the pixels
    standardized, $(x - \mu) / \sigma$ with ``mean`` and ``std``, and 0 where
    the last channel marks a masked patch (the mean of the data), then a linear
    map of each tube plus its fixed position, after ``extra`` learned tokens.
    """

    def __init__(
        self,
        *,
        frames: int,
        channels: int,
        patch: int,
        grid: tuple[int, int],
        dim: int,
        extra: int,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        super().__init__()
        self._patch, self._mean, self._std = patch, mean, std
        self.proj = nn.Linear(frames * channels * patch * patch, dim)
        self.register_buffer("pos", sincos_2d(grid, dim), persistent=False)
        self.extra = nn.Parameter(torch.zeros(extra, dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        pixels, mask = x[:, :, :-1], x[:, :, -1:]  # (n, N, C, H, W), (n, N, 1, H, W)
        pixels = (pixels - self._mean) / self._std * (1 - mask)
        x = torch.cat([pixels, mask], 2)
        tokens = self.proj(patchify(x, self._patch)) + self.pos  # (n, T, D)
        extra = self.extra.expand(len(x), -1, -1)
        return torch.cat([extra, tokens], 1)


class PackedEmbed(nn.Module):
    r"""The packed sequences ``(n, L, P + 1)`` (``masks.pack_views``: the
    pixels of each tubelet in $[0, 1]$, the last column the token's id) as
    tokens ``(n, L, D)``: a linear map of the standardized pixels
    $(x - \mu) / \sigma$ plus the fixed 3-D position of the id
    (``sincos_3d`` of ``grid3``). The id column is read as a constant."""

    def __init__(
        self,
        *,
        patch_dim: int,
        dim: int,
        grid3: tuple[int, int, int],
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        super().__init__()
        self._mean, self._std = mean, std
        self.proj = nn.Linear(patch_dim, dim)
        self.register_buffer("pos", sincos_3d(grid3, dim), persistent=False)
        self.pos: Tensor

    @override
    def forward(self, x: Tensor) -> Tensor:
        pixels, ids = x[..., :-1], x[..., -1].detach().long()  # (n, L, P), (n, L)
        return self.proj((pixels - self._mean) / self._std) + self.pos[ids]


class Stage(nn.Module):
    """The module of one encoder node: the embedding (the first stage only),
    blocks, and the final ``LayerNorm`` (the last stage only), then, with
    ``out`` (the last stage, ``vision.token_dim``), a linear map of each token
    to ``out`` dimensions; ``segments`` goes to every block (``Block``)."""

    def __init__(
        self,
        blocks: list[Block],
        dim: int,
        *,
        embed: Embed | PackedEmbed | None = None,
        norm: bool = False,
        out: int | None = None,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim) if norm else None
        self.out = nn.Linear(dim, out) if out is not None else None

    @override
    def forward(self, x: Tensor, segments: Sequence[int] | None = None) -> Tensor:
        if self.embed is not None:
            x = self.embed(x)
        for block in self.blocks:
            x = block(x, segments)
        if self.norm is not None:
            x = self.norm(x)
        return x if self.out is None else self.out(x)


class Segmented(nn.Module):
    """A ``Stage`` with fixed ``segments`` (the module of a ``pc.Layer`` of the
    packed layout, which calls it with ``h`` alone): ``stage(x, segments)``.
    Its parameters are the stage's."""

    def __init__(self, stage: Stage, segments: Sequence[int]) -> None:
        super().__init__()
        self.stage = stage
        self._segments = tuple(segments)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.stage(x, self._segments)


class Encoder(nn.Module):
    """The stages in order: the chunk ``(n, N, C', H, W)`` (or the packed
    sequences ``(n, L, P + 1)``) to the tokens ``(n, E + T, D)`` (``(n, L,
    D)``); ``pool`` reads $z$ ``(n, D)`` out of them."""

    def __init__(self, stages: list[Stage], *, extra: int, z_pool: str) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.extra, self.z_pool = extra, z_pool

    @override
    def forward(self, x: Tensor, segments: Sequence[int] | None = None) -> Tensor:
        for stage in self.stages:
            x = stage(x, segments)
        return x

    def pool(self, tokens: Tensor) -> Tensor:
        """$z$ ``(n, D)`` of the tokens ``(n, E + T, D)`` (``loss.pool``)."""
        return pool(tokens, z_pool=self.z_pool, extra=self.extra)


class Predictor(nn.Module):
    """The predictor of the JEPA loss: blocks, a ``LayerNorm`` and a linear map,
    from the context view's tokens ``(n, E + T, D)`` to the predicted target
    tokens of the same shape (the loss reads the masked patch tokens)."""

    def __init__(self, blocks: list[Block], dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


class PackedPredictor(nn.Module):
    """The predictor of the packed layout (V-JEPA's): the context view's tokens
    ``ctx`` ``(n, K, D)`` of the ids ``ctx_ids`` ``(n, K)`` and a learned mask
    token at each id of ``tgt_ids`` ``(n, M)``, each plus its fixed 3-D
    position, through blocks, a ``LayerNorm`` and a linear map; the
    prediction of the tokens at ``tgt_ids`` is the last ``M`` outputs
    ``(n, M, D)``. With a ``width`` other than ``dim`` (V-JEPA's narrow
    predictor) a linear map first takes the context to the width, where the
    mask token, the positions and the blocks are; ``out`` maps back."""

    def __init__(
        self,
        blocks: list[Block],
        dim: int,
        grid3: tuple[int, int, int],
        width: int | None = None,
    ) -> None:
        super().__init__()
        w = dim if width is None else width
        self.embed = nn.Linear(dim, w) if w != dim else None
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(w)
        self.out = nn.Linear(w, dim)
        self.mask_token = nn.Parameter(torch.zeros(w))
        self.register_buffer("pos", sincos_3d(grid3, w), persistent=False)
        self.pos: Tensor

    @override
    def forward(self, ctx: Tensor, ctx_ids: Tensor, tgt_ids: Tensor) -> Tensor:
        m = tgt_ids.shape[1]
        if self.embed is not None:
            ctx = self.embed(ctx)
        x = torch.cat([ctx + self.pos[ctx_ids], self.mask_token + self.pos[tgt_ids]], 1)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x[:, -m:]))


class Projector(nn.Module):
    r"""LeJEPA's projector of the space-time completion (``jepa.loss``): $z$
    ``(n, D_z)`` to $u$ ``(n, widths[-1])`` through ``Linear - LayerNorm -
    GELU`` to each width but the last, then a ``Linear`` to it. Its weights
    are LeCun's (as the projection of $z$, so that $u$ keeps the scale of
    $z$), the biases 0."""

    def __init__(self, d_in: int, widths: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for w in widths[:-1]:
            layers += [nn.Linear(d_in, w), nn.LayerNorm(w), nn.GELU()]
            d_in = w
        layers.append(nn.Linear(d_in, widths[-1]))
        self.net = nn.Sequential(*layers)
        self.out_features = widths[-1]
        for m in self.net:
            if isinstance(m, nn.Linear):
                std = 1 / math.sqrt(m.in_features)
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
                nn.init.zeros_(m.bias)

    @override
    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class ZPredictor(nn.Module):
    r"""The predictor of the space-time completion: $u_g$ from a context
    view's $u_v$ ``(n, d)`` and its kind $k$ (the index of the context view),
    $q(u_v, k) = u_v + W_2\, \mathrm{GELU}(W_1 [u_v, e_k] + b_1) + b_2$ with
    $e_k$ one-hot of ``kinds``. $W_2$ and $b_2$ start at 0 (at first
    $q(u_v, k) = u_v$), $W_1$ LeCun."""

    def __init__(self, d: int, kinds: int, hidden: int) -> None:
        super().__init__()
        self._kinds = kinds
        self.fc1 = nn.Linear(d + kinds, hidden)
        self.fc2 = nn.Linear(hidden, d)
        std = 1 / math.sqrt(d + kinds)
        nn.init.trunc_normal_(self.fc1.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @override
    def forward(self, u: Tensor, kind: int) -> Tensor:
        e = u.new_zeros(len(u), self._kinds)
        e[:, kind] = 1.0
        return u + self.fc2(nn.functional.gelu(self.fc1(torch.cat([u, e], 1))))


class PixelDecoder(nn.Module):
    """The decoder of the reconstruction term (``loss.Reconstruction``): an MLP
    of one hidden layer from $z$ ``(n, D)`` to the standardized frames
    ``(n, C, H, W)`` of ``shape = (C, H, W)``."""

    def __init__(self, dim: int, hidden: int, shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.shape = shape
        # PyTorch's default init (not ``_init_``): a plain MLP head, not a ViT block.
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, math.prod(shape))
        )

    @override
    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).unflatten(1, self.shape)


class BroadcastDecoder(nn.Module):
    """The spatial broadcast decoder (Watters et al. 2019, arXiv:1901.07017) of
    the reconstruction term: $z$ ``(n, D)`` is tiled over the ``H x W`` grid,
    the pixel's coordinates $(x, y) \\in [-1, 1]^2$ (pixel centers) are
    appended, and the same MLP ``Linear(D + 2, hidden) - ReLU - Linear(hidden,
    hidden) - ReLU - Linear(hidden, C)`` maps each pixel's input to its
    channels. The output ``(n, C, H, W)`` is of the standardized frames, as
    that of ``PixelDecoder``; a pixel depends on $z$ and on its own
    coordinates only. The intent (a hypothesis to test, not a result) is that a
    position of an object becomes a function of $z$ shared by all pixels,
    rather than one of the $C H W$ separate output units of an MLP."""

    def __init__(self, dim: int, hidden: int, shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.shape = shape
        _, h, w = shape
        ys = (torch.arange(h) + 0.5) / h * 2 - 1
        xs = (torch.arange(w) + 0.5) / w * 2 - 1
        grid = torch.stack(
            [xs.expand(h, w).reshape(-1), ys[:, None].expand(h, w).reshape(-1)], 1
        )
        self.register_buffer("coords", grid, persistent=False)  # (H W, 2): x, y
        self.coords: Tensor
        # PyTorch's default init, as ``PixelDecoder``.
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, shape[0]),
        )

    def at(self, z: Tensor, coords: Tensor) -> Tensor:
        """The channels ``(n, P, C)`` at ``coords`` ``(P, 2)`` for $z$ ``(n, D)``."""
        n, p = len(z), len(coords)
        x = torch.cat([z[:, None].expand(n, p, -1), coords.expand(n, p, 2)], 2)
        return self.net(x)

    @override
    def forward(self, z: Tensor) -> Tensor:
        c, h, w = self.shape
        return self.at(z, self.coords).transpose(1, 2).reshape(len(z), c, h, w)


@dataclass
class VisionModules:
    """The encoder and the predictor of a vision, and the shape of its chunks.

    ``chunk`` is ``(N, C, H, W)`` of the frames (without the mask channel),
    ``patch`` the side of a patch. The packed layout (``layout="packed"``) has
    tubelets of ``tubelet`` frames, the lengths ``segments`` of the views of
    a packed sequence (the target view's $T$ first) and, with ``vision.z_dim``
    not the width, the projection ``proj`` of the pooled tokens to $z$. With
    ``vision.ema`` it has the ``target`` encoder (I-JEPA): a copy of
    ``encoder`` whose parameters require no gradient, moved by the learner
    after each step (``jepa.ema.ema_``), whose tokens are the prediction's
    targets.
    """

    encoder: Encoder
    predictor: Predictor | PackedPredictor
    chunk: tuple[int, int, int, int]
    patch: int
    decoder: PixelDecoder | BroadcastDecoder | None = None  # with ``vision.recon`` > 0
    proj: nn.Linear | None = None
    layout: str = "pixels"
    # the space-time completion (packed, vision.projector / zpred_hidden)
    projector: Projector | None = None
    zpred: ZPredictor | None = None
    tubelet: int = 0
    segments: tuple[int, ...] = ()
    target: Encoder | None = None  # the target encoder (packed, vision.ema)

    def named(self) -> dict[str, nn.Module]:
        """The modules by the names of the checkpoints (the decoder and the
        projection only when there are)."""
        out: dict[str, nn.Module] = {
            "encoder": self.encoder,
            "predictor": self.predictor,
        }
        if self.decoder is not None:
            out["decoder"] = self.decoder
        if self.proj is not None:
            out["proj"] = self.proj
        if self.projector is not None:
            out["projector"] = self.projector
        if self.zpred is not None:
            out["zpred"] = self.zpred
        if self.target is not None:
            out["target"] = self.target
        return out

    @property
    def grid3(self) -> tuple[int, int, int]:
        """The tubelets along the time, the height and the width (packed)."""
        n = self.chunk[0]
        return (n // self.tubelet, *self.grid)

    @property
    def tokens_per_view(self) -> int:
        """$T$ of the packed layout: the tokens of a whole stack."""
        return math.prod(self.grid3)

    def z_of(self, tokens: Tensor) -> Tensor:
        """$z$ ``(n, D_z)`` of the target view's tokens ``(n, T, D)`` of the
        packed layout: their mean, through ``proj`` when there is one."""
        z = tokens.mean(1)
        return z if self.proj is None else self.proj(z)

    def tokens(self, chunks: Tensor) -> Tensor:
        """The target view's tokens ``(n, T, D)`` of the stacks ``chunks``
        ``(n, N, C, H, W)`` (u8, or float in ``[0, 1]``) of the packed
        layout, at the weights as they are now."""
        return self.encoder(self._whole(chunks), (self.tokens_per_view,))

    def target_tokens(self, chunks: Tensor) -> Tensor:
        """The prediction's targets ``(n, T, D)`` of the stacks ``chunks``:
        the tokens of the ``target`` encoder, or ``tokens`` without one."""
        if self.target is None:
            return self.tokens(chunks)
        return self.target(self._whole(chunks), (self.tokens_per_view,))

    def _whole(self, chunks: Tensor) -> Tensor:
        """The packed sequence ``(n, T, P + 1)`` of the stacks' target view."""
        x = chunks.float() / 255 if chunks.dtype == torch.uint8 else chunks.float()
        t = self.tokens_per_view
        pixels = patchify_tubelets(x, self.patch, self.tubelet)
        ids = torch.arange(t, device=x.device, dtype=x.dtype).expand(len(x), t)
        return torch.cat([pixels, ids[..., None]], 2)

    @property
    def grid(self) -> tuple[int, int]:
        """The patches along the height and the width."""
        _, _, h, w = self.chunk
        return h // self.patch, w // self.patch

    def encode(self, chunks: Tensor) -> Tensor:
        """$z$ ``(n, D)`` of the target views (no mask) of ``chunks`` ``(n, N,
        C, H, W)``, at the weights as they are now (packed: ``z_of`` of
        ``tokens``)."""
        if self.layout == "packed":
            return self.z_of(self.tokens(chunks))
        n = len(chunks)
        none = torch.zeros(n, 0, *self.grid, dtype=torch.bool, device=chunks.device)
        h, _ = make_views(chunks, none, self.patch)
        return self.encoder.pool(self.encoder(h))


def _init_(module: nn.Module) -> None:
    r"""Normal weights truncated at 2 std and zero biases for the linear maps,
    unit LayerNorms, and the extra tokens as the weights (std 0.02), in the
    order of ``module.modules()``. The patch embedding's weights are of std
    $1 / \sqrt{\text{fan-in}}$ (LeCun), the others of std 0.02: a tube of
    standardized pixels then gives a token of about the scale of its
    position. With std 0.02 the content is a few hundredths of the position,
    and the pooled $z$ of every image is nearly the same before anything
    learns, a collapse that SIGReg on $z$ cannot undo (its gradient is the
    same for every sample there). So are those of a stage's map of the tokens
    to ``vision.token_dim`` (``Stage.out``): the tokens of the representation
    keep the unit scale of the ``LayerNorm``'s output, as the projection of
    $z$ does. And so is the predictor's map of the context's tokens to its
    width (``PackedPredictor.embed``): with std 0.02 and a few (2 to 16)
    inputs of unit scale, the context's content is a few hundredths of the
    position added to it (the mask tokens are their positions), and the
    predictor starts nearly blind to the context; with LeCun's it is of about
    the position's scale."""
    lecun = {id(m.proj) for m in module.modules() if isinstance(m, Embed | PackedEmbed)}
    lecun |= {
        id(m.out)
        for m in module.modules()
        if isinstance(m, Stage) and m.out is not None
    }
    lecun |= {
        id(m.embed)
        for m in module.modules()
        if isinstance(m, PackedPredictor) and m.embed is not None
    }
    for m in module.modules():
        if isinstance(m, nn.Linear):
            std = 1 / math.sqrt(m.in_features) if id(m) in lecun else INIT_STD
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, Embed) and m.extra.numel():
            nn.init.trunc_normal_(m.extra, std=INIT_STD)
        elif isinstance(m, PackedPredictor):
            nn.init.trunc_normal_(m.mask_token, std=INIT_STD)


def build_vision(conf: RunConf, *, seed: int) -> VisionModules:
    """The modules of ``conf.vision`` for the frames of ``conf.data``: after
    seeding the CPU's global RNG with ``seed`` (not the CUDA ones), the
    encoder's stages, the predictor, (packed, ``z_dim`` not the width) the
    projection, and (``vision.projector``, ``zpred_hidden``) the projector and
    the predictor of $u$, built and initialized in this order on the CPU (so that the initial weights do not depend on the
    device), then (``vision.ema``) the target encoder as a copy of the
    encoder, all moved to ``conf.device``. The CPU's global RNG is restored
    afterwards. The one place that draws the vision's initial weights, for
    PC, BP and the decoder fit afterwards.

    Raises:
        ValueError: ``conf`` has no vision, or ``vision.patch`` does not divide
            the frames' side.
    """
    v = conf.vision
    if v is None:
        raise ValueError(f"the run {conf.name!r} has no vision")
    channels, size = image_shape(conf)
    if size % v.patch:
        raise ValueError(f"vision.patch {v.patch} does not divide the side {size}")
    grid = (size // v.patch, size // v.patch)
    extra = extra_tokens(v)
    per = v.depth // v.nodes
    packed = v.layout == "packed"
    grid3 = (v.frames // max(v.tubelet, 1), *grid)
    with torch.random.fork_rng(devices=[]):  # the global RNG is left as it was
        torch.default_generator.manual_seed(seed)
        stages = []
        for i in range(v.nodes):
            embed: Embed | PackedEmbed | None = None
            if i == 0 and packed:
                embed = PackedEmbed(
                    patch_dim=v.tubelet * channels * v.patch**2,
                    dim=v.dim,
                    grid3=grid3,
                    mean=v.pixel_mean,
                    std=v.pixel_std,
                )
            elif i == 0:
                embed = Embed(
                    frames=v.frames,
                    channels=channels + 1,
                    patch=v.patch,
                    grid=grid,
                    dim=v.dim,
                    extra=extra,
                    mean=v.pixel_mean,
                    std=v.pixel_std,
                )
            blocks = [Block(v.dim, v.heads) for _ in range(per)]
            last = i == v.nodes - 1
            out = v.token_dim if last else None
            stage = Stage(blocks, v.dim, embed=embed, norm=last, out=out)
            _init_(stage)
            stages.append(stage)
        width, heads = v.pred_dim or v.dim, v.pred_heads or v.heads
        pred_blocks = [Block(width, heads) for _ in range(v.pred_depth)]
        predictor: Predictor | PackedPredictor
        if packed:
            predictor = PackedPredictor(
                pred_blocks, v.token_width, grid3, v.pred_dim or v.dim
            )
        else:
            predictor = Predictor(pred_blocks, v.dim)
        _init_(predictor)
        proj = None
        if packed and v.projected:
            # LeCun, as the patch embedding: z keeps the scale of the pooled
            # tokens (a z of std 0.02 of it would start SIGReg near a collapse).
            proj = nn.Linear(v.dim, v.z_width)
            std = 1 / math.sqrt(v.dim)
            nn.init.trunc_normal_(proj.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(proj.bias)
        projector = Projector(v.z_width, v.projector) if v.projector else None
        zpred = None
        if v.zpred_hidden:
            zpred = ZPredictor(v.u_width, len(v.context_masks), v.zpred_hidden)
        decoder = None  # built last: the others' weights do not depend on it
        if v.recon > 0:
            shape = (channels, size, size)
            if v.recon_decoder == "broadcast":
                decoder = BroadcastDecoder(v.dim, v.decoder_hidden, shape)
            else:
                decoder = PixelDecoder(v.dim, v.decoder_hidden, shape)
    encoder = Encoder(stages, extra=extra, z_pool=v.z_pool)
    target = None  # a copy: the other weights do not depend on it
    if v.ema:  # 0: none (and none in the checkpoints of the runs before)
        target = copy.deepcopy(encoder).requires_grad_(False).to(conf.device)
    mods = VisionModules(
        encoder=encoder.to(conf.device),
        predictor=predictor.to(conf.device),
        chunk=(v.frames, channels, size, size),
        patch=v.patch,
        decoder=None if decoder is None else decoder.to(conf.device),
        proj=None if proj is None else proj.to(conf.device),
        layout=v.layout,
        tubelet=v.tubelet,
        projector=None if projector is None else projector.to(conf.device),
        zpred=None if zpred is None else zpred.to(conf.device),
        target=target,
    )
    if packed:  # imported here: masks imports this module
        from exp.video.jepa.masks import token_keep

        t = mods.tokens_per_view
        keeps = [token_keep(grid, grid3[0], m) for m in v.context_masks]
        mods.segments = (t, *keeps)
    return mods
