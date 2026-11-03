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
"""

import math
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
    "Embed",
    "Encoder",
    "PixelDecoder",
    "Predictor",
    "Stage",
    "VisionModules",
    "build_vision",
    "extra_tokens",
    "image_shape",
    "patchify",
    "sincos_2d",
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
    softmax attention within each sample and an MLP of ``4 D`` with GELU."""

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
    def forward(self, x: Tensor) -> Tensor:
        n, t, d = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .reshape(n, t, 3, self._heads, d // self._heads)
            .permute(2, 0, 3, 1, 4)
        )  # each (n, heads, T, d_head)
        a = (q @ k.transpose(-2, -1) / math.sqrt(d // self._heads)).softmax(-1)
        x = x + self.proj((a @ v).transpose(1, 2).reshape(n, t, d))
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


class Stage(nn.Module):
    """The module of one encoder node: the embedding (the first stage only),
    blocks, and the final ``LayerNorm`` (the last stage only)."""

    def __init__(
        self,
        blocks: list[Block],
        dim: int,
        *,
        embed: Embed | None = None,
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim) if norm else None

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.embed is not None:
            x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return x if self.norm is None else self.norm(x)


class Encoder(nn.Module):
    """The stages in order: the chunk ``(n, N, C', H, W)`` to the tokens
    ``(n, E + T, D)``; ``pool`` reads $z$ ``(n, D)`` out of them."""

    def __init__(self, stages: list[Stage], *, extra: int, z_pool: str) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.extra, self.z_pool = extra, z_pool

    @override
    def forward(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x = stage(x)
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


@dataclass
class VisionModules:
    """The encoder and the predictor of a vision, and the shape of its chunks.

    ``chunk`` is ``(N, C, H, W)`` of the frames (without the mask channel),
    ``patch`` the side of a patch.
    """

    encoder: Encoder
    predictor: Predictor
    chunk: tuple[int, int, int, int]
    patch: int
    decoder: PixelDecoder | None = None  # with ``vision.recon`` > 0

    def named(self) -> dict[str, nn.Module]:
        """The modules by the names of the checkpoints (the decoder only when
        there is one)."""
        out: dict[str, nn.Module] = {
            "encoder": self.encoder,
            "predictor": self.predictor,
        }
        if self.decoder is not None:
            out["decoder"] = self.decoder
        return out

    @property
    def grid(self) -> tuple[int, int]:
        """The patches along the height and the width."""
        _, _, h, w = self.chunk
        return h // self.patch, w // self.patch

    def encode(self, chunks: Tensor) -> Tensor:
        """$z$ ``(n, D)`` of the target views (no mask) of ``chunks`` ``(n, N,
        C, H, W)``, at the weights as they are now."""
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
    same for every sample there)."""
    patch = {id(m.proj) for m in module.modules() if isinstance(m, Embed)}
    for m in module.modules():
        if isinstance(m, nn.Linear):
            std = 1 / math.sqrt(m.in_features) if id(m) in patch else INIT_STD
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, Embed) and m.extra.numel():
            nn.init.trunc_normal_(m.extra, std=INIT_STD)


def build_vision(conf: RunConf, *, seed: int) -> VisionModules:
    """The modules of ``conf.vision`` for the frames of ``conf.data``: after
    seeding the CPU's global RNG with ``seed`` (not the CUDA ones), the
    encoder's stages and then the predictor, built and initialized in this
    order on the CPU (so that the initial weights do not depend on the
    device), then moved to ``conf.device``. The CPU's global RNG is restored
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
    with torch.random.fork_rng(devices=[]):  # the global RNG is left as it was
        torch.default_generator.manual_seed(seed)
        stages = []
        for i in range(v.nodes):
            embed = None
            if i == 0:
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
            stage = Stage(blocks, v.dim, embed=embed, norm=i == v.nodes - 1)
            _init_(stage)
            stages.append(stage)
        predictor = Predictor(
            [Block(v.dim, v.heads) for _ in range(v.pred_depth)], v.dim
        )
        _init_(predictor)
        decoder = None  # built last: the others' weights do not depend on it
        if v.recon > 0:
            decoder = PixelDecoder(v.dim, v.recon_hidden, (channels, size, size))
    encoder = Encoder(stages, extra=extra, z_pool=v.z_pool)
    return VisionModules(
        encoder=encoder.to(conf.device),
        predictor=predictor.to(conf.device),
        chunk=(v.frames, channels, size, size),
        patch=v.patch,
        decoder=None if decoder is None else decoder.to(conf.device),
    )
