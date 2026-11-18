r"""The modules of the VAE of V0 (``plans/v0-foundation.md`` 「モデル」): the
encoder (a ViT or a ResNet) cut into exactly ``ENC_NODES`` = 4 node modules,
the light decoder cut into ``DEC_NODES`` = 2 node modules and a readout, and
the likelihood's learned $\ln \sigma$.

A stack $x \in \{0..255\}^{n \times 3 \times H \times W}$ enters as $x / 255$
with its frames stacked on the channels (``early_fusion``: $3n$ channels, the
channel $3f + c$ is channel $c$ of frame $f$), the same input for both
encoders. Each encoder ends on a grid of $(H / 16) \times (W / 16)$ (4 x 4 at
64 px), flattened into one ``Linear`` to $h = (\mu, \ell)$ ``(B, 2D)``, $\ell
= \ln s^2$; the rows of $\ell$ and their bias start at 0 ($s^2 = 1$ for every
stack, as M18's ``build_vae``).

- ViT (``ViTPart``): node 1 is the patch embedding (``model.patch``, fixed
  2-D sine-cosine positions) and $k$ pre-norm blocks of ``model.width``,
  nodes 2-3 are $k$ blocks, node 4 is $k$ blocks, a ``LayerNorm`` and the
  head. Initialized by M17's rule (``_init_vit``).
- ResNet (``ResNetPart``): node 1 is a 3 x 3 stem of stride 2 and $k$ basic
  blocks, nodes 2-4 are $k$ basic blocks whose first has stride 2, node 4
  then the head. ``GroupNorm`` (``_groups``), since a stream of $B = 1$
  cannot hold batch statistics. Torch's default initialization.

$k$ = ``ModelConf.blocks`` (S, M, L = 1, 2, 4). The decoder is the same for
every encoder: node 1 maps $z$ ``(B, D)`` by a ``Linear`` to
``decoder_channels[0]`` channels on the encoder's last grid and doubles it
twice, node 2 doubles it twice more (``UpStage``: nearest x2, 3 x 3 conv,
``GroupNorm``, GELU), and the readout (a 3 x 3 conv, in ``GaussOut``) maps it
to $\hat y$ ``(B, 3n, H, W)``. Its initial weights are drawn from ``seed +
DECODER_SEED`` alone, so they do not depend on the encoder.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from exp.vision.schema.model import ARCHS, BLOCKS, DEC_NODES, ENC_NODES, ModelConf
from exp.vision.schema.run import RunConf

__all__ = [
    "DECODER_SEED",
    "BasicBlock",
    "Block",
    "Decoder",
    "DecoderPart",
    "Encoder",
    "LogSigma",
    "PatchEmbed",
    "ResNetPart",
    "UpStage",
    "VAEModules",
    "ViTPart",
    "build_vae",
    "early_fusion",
    "param_table",
    "sincos_2d",
]

MLP_RATIO = 4
INIT_STD = 0.02
DOWN = 16  # the encoders' total stride: 4 halvings
DECODER_SEED = 11  # added to the run's seed: the decoder's initial weights
_GROUPS = 32  # the most groups of a GroupNorm


def early_fusion(stacks: Tensor) -> Tensor:
    """The stacks ``(B, n, C, H, W)`` (u8, or floats in ``[0, 1]``) as images
    ``(B, n C, H, W)`` of ``x / 255`` (u8) or ``x`` (floats)."""
    b, n, c, h, w = stacks.shape
    x = stacks.reshape(b, n * c, h, w)
    return x.float() / 255 if x.dtype == torch.uint8 else x


def sincos_2d(grid: tuple[int, int], dim: int) -> Tensor:
    r"""The fixed positions of the patch tokens, ``(gh gw, dim)`` in row-major
    order: half of the dimensions encode the row, half the column, each as
    $\sin$ and $\cos$ of the index times $10000^{-i / (dim / 4)}$ (MAE's)."""
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


def _groups(channels: int) -> int:
    """The groups of a ``GroupNorm`` of ``channels``: the largest divisor of
    ``channels`` up to 32."""
    return math.gcd(channels, _GROUPS)


# -- ViT ------------------------------------------------------------------------


class Block(nn.Module):
    r"""A pre-norm transformer block on ``(B, T, w)``:
    $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$, then
    $x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x))$, ``heads`` heads of softmax
    attention within each sample, an MLP of $4w$ with GELU."""

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


class PatchEmbed(nn.Module):
    """The images ``(B, C, H, W)`` as tokens ``(B, T, w)``: a linear map of
    each ``patch`` x ``patch`` patch (flattened ``(C, p, p)``), plus its fixed
    position (``sincos_2d``); $T = (H / p)(W / p)$ in row-major order."""

    def __init__(self, *, channels: int, patch: int, size: int, dim: int) -> None:
        super().__init__()
        self._patch = patch
        self.proj = nn.Linear(channels * patch * patch, dim)
        grid = size // patch
        self.register_buffer("pos", sincos_2d((grid, grid), dim), persistent=False)
        self.pos: Tensor

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        p = self._patch
        x = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        return self.proj(x.reshape(b, (h // p) * (w // p), c * p * p)) + self.pos


class ViTPart(nn.Module):
    """One encoder node of the ViT: the embedding (node 1), ``blocks``, then
    (node 4) a ``LayerNorm`` and ``head``, a ``Linear`` of the flattened tokens
    ``(B, T w)`` to ``(B, 2D)``."""

    def __init__(
        self,
        blocks: list[Block],
        *,
        dim: int,
        embed: PatchEmbed | None = None,
        head: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = None if head is None else nn.LayerNorm(dim)
        self.head = head

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.embed is not None:
            x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        if self.head is None:
            return x
        assert self.norm is not None
        return self.head(self.norm(x).flatten(1))


def _init_vit(module: nn.Module) -> None:
    r"""M17's rule (``exp/video/jepa/vit.py``): truncated normal weights (at
    2 std) and zero biases of the linear maps, of std $1 / \sqrt{\text{fan-in}}$
    (LeCun) for the patch embedding and the head, 0.02 for the others; unit
    ``LayerNorm``\ s."""
    lecun = {id(m.proj) for m in module.modules() if isinstance(m, PatchEmbed)}
    lecun |= {
        id(m.head)
        for m in module.modules()
        if isinstance(m, ViTPart) and m.head is not None
    }
    for m in module.modules():
        if isinstance(m, nn.Linear):
            std = 1 / math.sqrt(m.in_features) if id(m) in lecun else INIT_STD
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _vit_parts(m: ModelConf, *, channels: int, size: int) -> list[nn.Module]:
    k, w = m.blocks, m.width
    tokens = (size // m.patch) ** 2
    parts: list[nn.Module] = []
    for i in range(ENC_NODES):
        blocks = [Block(w, m.heads) for _ in range(k)]
        embed = (
            PatchEmbed(channels=channels, patch=m.patch, size=size, dim=w)
            if i == 0
            else None
        )
        head = nn.Linear(tokens * w, 2 * m.latent) if i == ENC_NODES - 1 else None
        parts.append(ViTPart(blocks, dim=w, embed=embed, head=head))
    for part in parts:
        _init_vit(part)
    return parts


# -- ResNet ---------------------------------------------------------------------


class BasicBlock(nn.Module):
    r"""ResNet's basic block with ``GroupNorm``: $y = \mathrm{ReLU}(F(x) +
    S(x))$, $F$ = conv 3 x 3 (``stride``), GN, ReLU, conv 3 x 3, GN; $S$ the
    identity, or a 1 x 1 conv (``stride``) and GN when the shape changes."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(c_out), c_out)
        self.short: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.GroupNorm(_groups(c_out), c_out),
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        y = nn.functional.relu(self.norm1(self.conv1(x)))
        return nn.functional.relu(self.norm2(self.conv2(y)) + self.short(x))


class ResNetPart(nn.Module):
    """One encoder node of the ResNet: the stem (node 1: conv 3 x 3 of stride
    2, GN, ReLU), ``blocks``, then (node 4) ``head``, a ``Linear`` of the
    flattened map ``(B, C g^2)`` to ``(B, 2D)``."""

    def __init__(
        self,
        blocks: list[BasicBlock],
        *,
        stem: nn.Sequential | None = None,
        head: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.stem = stem
        self.blocks = nn.Sequential(*blocks)
        self.head = head

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.stem is not None:
            x = self.stem(x)
        x = self.blocks(x)
        return x if self.head is None else self.head(x.flatten(1))


def _resnet_parts(m: ModelConf, *, channels: int, size: int) -> list[nn.Module]:
    k, cs = m.blocks, m.resnet_channels
    grid = size // DOWN
    stem = nn.Sequential(
        nn.Conv2d(channels, cs[0], 3, 2, 1, bias=False),
        nn.GroupNorm(_groups(cs[0]), cs[0]),
        nn.ReLU(),
    )
    parts: list[nn.Module] = [
        ResNetPart([BasicBlock(cs[0], cs[0]) for _ in range(k)], stem=stem)
    ]
    for i in range(1, ENC_NODES):
        blocks = [BasicBlock(cs[i - 1], cs[i], stride=2)]
        blocks += [BasicBlock(cs[i], cs[i]) for _ in range(k - 1)]
        head = None
        if i == ENC_NODES - 1:
            head = nn.Linear(cs[i] * grid * grid, 2 * m.latent)
        parts.append(ResNetPart(blocks, head=head))
    return parts


class Encoder(nn.Module):
    """The ``ENC_NODES`` node modules of the encoder (``parts``), in order:
    images ``(B, 3n, H, W)`` to $h = (\\mu, \\ell)$ ``(B, 2D)``."""

    def __init__(self, parts: Sequence[nn.Module]) -> None:
        super().__init__()
        self.parts = nn.ModuleList(parts)

    @override
    def forward(self, x: Tensor) -> Tensor:
        for part in self.parts:
            x = part(x)
        return x

    @property
    def head(self) -> nn.Linear:
        """The last map, to $\\mu$ (the first $D$ rows) and $\\ell$."""
        head = getattr(self.parts[-1], "head")
        assert isinstance(head, nn.Linear)
        return head


# -- the decoder ----------------------------------------------------------------


class UpStage(nn.Module):
    """``(B, c_in, g, g)`` to ``(B, c_out, 2g, 2g)``: nearest x2, conv 3 x 3,
    ``GroupNorm``, GELU. The conv has no bias, which the norm would remove
    (its gradient is 0), as in ``BasicBlock``."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False)
        self.norm = nn.GroupNorm(_groups(c_out), c_out)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return nn.functional.gelu(self.norm(self.conv(x)))


class DecoderPart(nn.Module):
    """One decoder node: (node 1) a ``Linear`` of $z$ ``(B, D)`` to ``(B, C,
    g, g)``, then two ``UpStage``\\ s."""

    def __init__(
        self, stages: list[UpStage], *, embed: nn.Linear | None = None, grid: int = 0
    ) -> None:
        super().__init__()
        self.embed = embed
        self._grid = grid
        self.stages = nn.Sequential(*stages)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.embed is not None:
            x = self.embed(x).reshape(len(x), -1, self._grid, self._grid)
        return self.stages(x)


class Decoder(nn.Module):
    """The light decoder: ``DEC_NODES`` node modules (``parts``) and the
    readout (``readout``, conv 3 x 3 to $3n$ channels, in ``GaussOut``): $z$
    ``(B, D)`` to $\\hat y$ ``(B, 3n, H, W)``."""

    def __init__(self, parts: Sequence[DecoderPart], readout: nn.Conv2d) -> None:
        super().__init__()
        self.parts = nn.ModuleList(parts)
        self.readout = readout

    @override
    def forward(self, z: Tensor) -> Tensor:
        for part in self.parts:
            z = part(z)
        return self.readout(z)


def _decoder(m: ModelConf, *, channels: int, size: int) -> Decoder:
    c = m.decoder_channels
    grid = size // DOWN
    embed = nn.Linear(m.latent, c[0] * grid * grid)
    first = DecoderPart(
        [UpStage(c[0], c[1]), UpStage(c[1], c[2])], embed=embed, grid=grid
    )
    second = DecoderPart([UpStage(c[2], c[3]), UpStage(c[3], c[4])])
    assert DEC_NODES == 2  # the cut above
    return Decoder([first, second], nn.Conv2d(c[4], channels, 3, 1, 1))


class LogSigma(nn.Module):
    r"""The likelihood's one learned $\ln \sigma$ (``log_sigma``, 0-d), from
    $\frac12 \ln$ ``sigma0_sq``."""

    def __init__(self, sigma0_sq: float) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(0.5 * math.log(sigma0_sq)))


@dataclass
class VAEModules:
    r"""The VAE: ``encoder`` (images to $h = (\mu, \ell)$ ``(B, 2D)``),
    ``decoder`` ($z$ ``(B, D)`` to $\hat y$ ``(B, 3n, H, W)``, the mean of
    the dequantized target $y = (x + u) / 256$) and ``likelihood`` ($\ln
    \sigma$). The code of the evaluations is $\mu$ (``mu``); it is the
    ``exp.vision.eval.VaeCode`` of the run (``modules``, ``mu``, ``decode``)."""

    encoder: Encoder
    decoder: Decoder
    likelihood: LogSigma
    frames: int  # n, the frames of a stack

    def modules(self) -> dict[str, nn.Module]:
        """The learned modules by the names of the checkpoints."""
        return {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "likelihood": self.likelihood,
        }

    def parameters(self) -> list[nn.Parameter]:
        """The encoder's, the decoder's, then $\\ln \\sigma$."""
        return [p for m in self.modules().values() for p in m.parameters()]

    @property
    def log_sigma(self) -> Tensor:
        r"""$\ln \sigma$ (0-d)."""
        return self.likelihood.log_sigma

    @property
    def latent(self) -> int:
        """$D$."""
        return self.encoder.head.out_features // 2

    def posterior(self, stacks: Tensor) -> tuple[Tensor, Tensor]:
        r"""$\mu$ and $\ell = \ln s^2$ ``(B, D)`` of the stacks ``(B, n, 3, H,
        W)`` (u8, or floats in ``[0, 1]``) at the weights as they are now."""
        h = self.encoder(early_fusion(stacks).to(self.log_sigma.dtype))
        d = h.shape[1] // 2
        return h[:, :d], h[:, d:]

    def mu(self, stacks: Tensor) -> Tensor:
        r"""$\mu$ ``(B, D)`` of the stacks."""
        return self.posterior(stacks)[0]

    def decode(self, z: Tensor) -> Tensor:
        r"""The reconstruction ``(B, n, 3, H, W)`` of the codes ``z`` ``(B, D)``
        in the units of $x / 255$: $\hat y$ is the mean of $y = (x + u) /
        256$, $E[u] = 1/2$, so the mean of $x / 255$ is $(256 \hat y - 1/2)
        / 255$."""
        y = (256 * self.decoder(z) - 0.5) / 255
        b, c, h, w = y.shape
        return y.reshape(b, self.frames, c // self.frames, h, w)


def build_vae(conf: RunConf, *, seed: int) -> VAEModules:
    r"""The VAE of ``conf.model`` for the stacks of ``conf.data.tours``, on
    ``conf.device``: the encoder drawn from ``seed``, the decoder from ``seed +
    DECODER_SEED`` (the global CPU RNG, forked so that it is left as it was),
    built on the CPU, so that the initial weights do not depend on the device.
    The head's rows of $\ell$ and their bias are then set to 0."""
    m, t = conf.model, conf.data.tours
    channels, size = 3 * t.frames, t.size
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        build = _vit_parts if m.arch == "vit" else _resnet_parts
        encoder = Encoder(build(m, channels=channels, size=size))
        torch.manual_seed(seed + DECODER_SEED)
        decoder = _decoder(m, channels=channels, size=size)
    with torch.no_grad():
        encoder.head.weight[m.latent :].zero_()
        encoder.head.bias[m.latent :].zero_()
    dev = torch.device(conf.device)
    return VAEModules(
        encoder.to(dev),
        decoder.to(dev),
        LogSigma(m.sigma0_sq).to(dev),
        frames=t.frames,
    )


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def param_table(conf: RunConf | None = None) -> list[dict[str, int | str]]:
    """The parameters of the encoder of every arch and size (with the rest of
    ``conf``, ``RunConf()`` by default: 8 frames of 64 px), and of the
    decoder and $\\ln \\sigma$ (the same for all): rows ``{"arch", "size",
    "encoder", "decoder"}``."""
    conf = RunConf(name="params") if conf is None else conf
    t = conf.data.tours
    channels, size = 3 * t.frames, t.size
    rows: list[dict[str, int | str]] = []
    for arch in ARCHS:
        for s in BLOCKS:
            m = ModelConf(**{**conf.model.__dict__, "arch": arch, "size": s})
            with torch.device("meta"):
                build = _vit_parts if arch == "vit" else _resnet_parts
                encoder = Encoder(build(m, channels=channels, size=size))
                decoder = _decoder(m, channels=channels, size=size)
            rows.append(
                {
                    "arch": arch,
                    "size": s,
                    "encoder": _count(encoder),
                    "decoder": _count(decoder) + 1,  # + ln sigma
                }
            )
    return rows
